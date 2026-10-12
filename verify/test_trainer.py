"""In-browser trainer (content/train-in-browser.md): local checks.

The JavaScript trainer (web/widgets/_tinygpt.js) is checked three ways:
  * gradient checks against finite differences, layer by layer (node --test tests/js/trainer-ops.test.mjs);
  * parity with DLFS6's own codebot.model.GPT in PyTorch (fixtures from verify/_trainer_export.py);
  * the recorded runs shown on the page replay exactly (tests/js/trainer-runs.test.mjs).
The rest checks facts the page states about the data and about the PyTorch side.

    uv run pytest verify/test_trainer.py
"""
import base64
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from _trainer_data import CORPUS, EOT, OUT, build_slice
from _trainer_export import FIXTURES, build_default, build_erf, build_tiny

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(4)


def run_node_tests(*files: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run([node, "--test", *files], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    return proc.stdout


def read_slice():
    train = (OUT / "tinycodes-train.txt").read_text(encoding="utf-8")
    val = (OUT / "tinycodes-val.txt").read_text(encoding="utf-8")
    return train, val


def char_ids(text: str, chars: list[str]) -> np.ndarray:
    """The page's character tokenizer: sorted characters, <|endoftext|> gets the last id."""
    stoi = {c: i for i, c in enumerate(chars)}
    out = []
    for k, piece in enumerate(text.split(EOT)):
        if k:
            out.append(len(chars))
        out.extend(stoi[c] for c in piece)
    return np.array(out, dtype=np.int64)


def tiny_gpt(dlfs6_path, dropout=0.0):
    from codebot.model import GPT
    return GPT(vocab_size=96, max_context_len=64, embed_dim=64, n_head=4, n_layer=2, ff_dim=192,
               dropout_rate=dropout)


# ----------------------------------------------------------------- data
def test_data_slice_is_reproducible(dlfs6_path):
    if not CORPUS.is_file():
        pytest.skip("codebot/tiny_codes.txt is missing")
    corpus = CORPUS.read_text(encoding="utf-8")
    assert (CORPUS.stat().st_size, len(corpus.split(EOT))) == (6_487_033, 21_836)
    train, val = build_slice()
    shipped_train, shipped_val = read_slice()
    assert (train, val) == (shipped_train, shipped_val)
    assert (len(train), len(val)) == (220_437, 26_171) and len(train) + len(val) < 250_000
    corpus_docs = set(corpus.split(EOT))
    docs = train.split(EOT) + val.split(EOT)
    assert len(docs) == 778 + 86 and all(d in corpus_docs for d in docs)  # whole documents of the book's corpus
    size = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file())
    assert size <= 300_000, f"web/data/train-in-browser holds {size:,} bytes"


def test_bpe_sequences_are_shorter(dlfs6_path):
    """CodeBot's BPE tokenizer (chapter 1) needs 2.36x fewer tokens than characters for the held-out slice."""
    from codebot.tokenizer import BPETokenizer
    _, val = read_slice()
    tok = BPETokenizer.load_from(str(dlfs6_path / "codebot" / "merge_rules.pkl"))
    assert tok.end_token_id == 999  # <|endoftext|> is the last id, as in the character tokenizer
    n_bpe = len(tok.encode(val))
    n_char = len(val) - val.count(EOT) * (len(EOT) - 1)
    assert (n_bpe, n_char) == (10_659, 25_151)
    assert abs(n_char / n_bpe - 2.36) < 0.005


# ----------------------------------------------------------------- the JavaScript trainer
def test_js_gradient_checks():
    """Every layer's hand-written backward agrees with central finite differences (float64)."""
    out = run_node_tests("tests/js/trainer-ops.test.mjs")
    assert "fail 0" in out


def test_parity_fixtures_come_from_dlfs6_gpt(dlfs6_path):
    """Rebuilding the fixtures with DLFS6's GPT reproduces the committed JSON (to 1e-12)."""
    for name, build in (("parity-tiny.json", build_tiny), ("parity-default.json", build_default)):
        saved = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        fresh = build()
        assert saved["names"] == fresh["names"] and saved["ids"] == fresh["ids"]
        assert abs(saved["loss"] - fresh["loss"]) < 1e-12
        if "grads" in saved:
            for n in saved["names"]:
                assert saved["weights"][n] == fresh["weights"][n]
                a, b = (np.frombuffer(base64.b64decode(f["grads"][n]), "<f8") for f in (saved, fresh))
                assert np.allclose(a, b, rtol=1e-12, atol=1e-15), n
        else:
            assert saved["param_count"] == fresh["param_count"] == 93_408
    assert json.loads((FIXTURES / "erf.json").read_text(encoding="utf-8")) == build_erf()


def test_js_matches_dlfs6_gpt():
    """JS loss, every gradient and three AdamW steps equal DLFS6's GPT + torch.optim.AdamW."""
    out = run_node_tests("tests/js/trainer-parity.test.mjs")
    assert "fail 0" in out


def test_param_counts(dlfs6_path):
    from codebot.model import GPT
    mini = tiny_gpt(dlfs6_path)
    codebot = GPT(1000, 256, 384, 6, 6, 4 * 384, 0.1)
    count = lambda m: sum(p.numel() for p in m.parameters())  # the shared embedding counts once
    assert (count(mini), count(codebot)) == (93_408, 11_121_640)
    assert mini.embed.weight is mini.unembed.weight


# ----------------------------------------------------------------- the PyTorch side of the page's claims
def windows(ids: np.ndarray, n: int, T: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(ids) - T - 1, size=n)
    w = np.stack([ids[s:s + T + 1] for s in starts])
    return torch.from_numpy(np.ascontiguousarray(w[:, :-1])), torch.from_numpy(np.ascontiguousarray(w[:, 1:]))


def test_initial_loss_is_below_ln_v_because_of_tied_embeddings(dlfs6_path):
    """At initialization: random tokens give ln V + sigma^2/2 (above ln V); real code gives less than ln V,
    because the tied embedding favours "the next character equals this one" (8.6% of TinyCodes)."""
    train, _ = read_slice()
    chars = sorted(set(train.replace(EOT, "")))
    ids = char_ids(train, chars)
    assert abs((ids[1:] == ids[:-1]).mean() - 0.0858) < 5e-4
    torch.manual_seed(0)
    model = tiny_gpt(dlfs6_path).eval()
    rng = np.random.default_rng(0)
    x, y = windows(ids, 64, 64, rng)
    ce = lambda xx, yy: F.cross_entropy(model(xx).reshape(-1, 96), yy.reshape(-1)).item()
    with torch.no_grad():
        real = ce(x, y)
        lg = model(x)  # the logit of "next = current character", above the mean logit: about 0.6, below the toy's 0.88
        boost = (lg.gather(-1, x.unsqueeze(-1)).squeeze(-1) - lg.mean(-1)).mean().item()
        r = torch.from_numpy(rng.integers(0, 96, size=(64, 65)))
        rand = ce(r[:, :-1], r[:, 1:])
        sigma = model(r[:, :-1]).std().item()
        model.unembed.weight = torch.nn.Parameter(0.02 * torch.randn(96, 64))  # untie
        untied = ce(x, y)
    ln_v = math.log(96)
    assert real < ln_v - 0.02                       # about 4.52
    assert abs(rand - (ln_v + sigma ** 2 / 2)) < 0.01
    assert untied > ln_v - 0.01                     # without tying, back to about ln V
    assert 0.55 < boost < 0.7


def test_dlfs6_gpt_overfits_one_batch(dlfs6_path):
    """The same "overfit one batch" check in PyTorch: DLFS6's GPT at the trainer's size, AdamW lr 3e-3,
    8 windows of 64 characters; the training loss falls below 0.05 within 150 steps."""
    train, _ = read_slice()
    ids = char_ids(train, sorted(set(train.replace(EOT, ""))))
    torch.manual_seed(0)
    model = tiny_gpt(dlfs6_path)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = windows(ids, 8, 64, np.random.default_rng(1))
    losses = []
    for _ in range(150):
        loss = F.cross_entropy(model(x).reshape(-1, 96), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 4.4 and min(losses) < 0.05


def test_recorded_runs_replay():
    """The runs shown on the page (web/data/train-in-browser/runs.json) replay step for step, and the
    milestones the page quotes hold in them."""
    out = run_node_tests("tests/js/trainer-runs.test.mjs")
    assert "fail 0" in out


# ----------------------------------------------------------------- the page itself
def test_page_js_excerpts_are_verbatim():
    """Every ```js block on the page is copied from web/widgets/_tinygpt.js (pieces separated by '// …')."""
    page = (ROOT / "content" / "train-in-browser.md").read_text(encoding="utf-8")
    source = (ROOT / "web" / "widgets" / "_tinygpt.js").read_text(encoding="utf-8")
    flat = lambda s: re.sub(r"\s+", " ", s).strip()
    blocks = re.findall(r"^```js\n(.*?)^```", page, re.S | re.M)
    assert blocks, "the page should quote the trainer's code"
    for block in blocks:
        for piece in re.split(r"^\s*// ….*$", block, flags=re.M):
            if piece.strip():
                assert flat(piece) in flat(source), f"not in _tinygpt.js:\n{piece}"
