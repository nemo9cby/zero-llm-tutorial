"""Local checks for content/playground.md, the in-browser CodeBot workbench.

    uv run pytest verify/test_playground.py          (CPU only; about two minutes)

The browser runs the int8 exports in web/data/playground/<name>/, written by tools/export_codebot.py from the
tutorial's own run of the book's ch03 scripts (runs/codebot-3060ti). These tests tie the exports back to those
checkpoints and to the book's own codebot.model.GPT; the JavaScript side is checked against the same references
by tests/js/playground-*.test.mjs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from _playground_addition import book_reward, build as build_addition, load_export, OUT as ADDITION
from _playground_cells import run_cells_upto

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "data" / "playground"
NAMES = ["pretrain", "sft", "grpo"]
CONFIG_KEYS = ("vocab_size", "max_context_len", "embed_dim", "n_head", "n_layer", "ff_dim")
OFFICIAL_URL = "https://huggingface.co/datasets/koki0702/zero-llm-data/blob/main/codebot/model_grpo.pt"

sys.path.insert(0, str(ROOT / "tools"))
import export_codebot as exporter  # noqa: E402  (tools/ is not a package)

torch.set_num_threads(4)


def manifest_of(name: str) -> dict:
    return json.loads((DATA / name / "manifest.json").read_text(encoding="utf-8"))


def ref_of(name: str) -> dict:
    return json.loads((DATA / name / "ref.json").read_text(encoding="utf-8"))


def checkpoint_of(name: str) -> Path:
    path = ROOT / manifest_of(name)["provenance"]["checkpoint"]
    if not path.is_file():
        pytest.skip(f"{path.relative_to(ROOT)} is missing: run gpu/run_local.py --run codebot-3060ti first")
    return path


def book_model(state_dict: dict, cfg: dict):
    from codebot.model import GPT
    model = GPT(**cfg, dropout_rate=0.0)
    model.load_state_dict(state_dict)
    return model.eval()


def load_both(name: str):
    """(float32 model from the checkpoint, model with the int8 weights the browser reconstructs)."""
    ckpt = torch.load(checkpoint_of(name), map_location="cpu", weights_only=True)
    cfg = {k: int(ckpt[k]) for k in CONFIG_KEYS}
    return book_model(ckpt["model_state_dict"], cfg), book_model(load_export(name), cfg), ckpt, cfg


# ----------------------------------------------------------------------------- P.2 the exports
def test_quantize_cell_matches_exporter():
    ns = run_cells_upto("P.2-b")
    rng = np.random.default_rng(3)
    w = rng.normal(0, 0.05, size=(64, 384)).astype(np.float32)
    w[5] = 0                                       # an all-zero row
    w[9] *= 40                                     # an outlier row
    q1, s1 = ns["quantize_rows"](w)
    q2, s2 = exporter.quantize_rows(w)
    assert np.array_equal(q1, q2) and np.array_equal(s1, s2)
    assert np.array_equal(ns["dequantize_rows"](q1, s1), exporter.dequantize_rows(q2, s2))


@pytest.mark.parametrize("name", NAMES)
def test_exports_match_checkpoints(name, dlfs6_path):
    manifest = manifest_of(name)
    prov = manifest["provenance"]
    path = checkpoint_of(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov["checkpoint_sha256"]
    # provenance: the step of the run report that wrote this checkpoint
    report = json.loads((ROOT / "web" / prov["run_report"]).read_text(encoding="utf-8"))
    step = [s for s in report["steps"] if prov["checkpoint"].split("/work/")[1] in s["produced"]][-1]
    assert step["script"] == prov["run_step"]["script"] and step["final"] == prov["run_step"]["final"]
    assert step["exit_code"] == 0 and prov["dlfs6_commit"] == report["dlfs6_commit"]
    # shards: sizes and hashes as recorded, at most 2 MB each (STYLE.md §4)
    blob = b""
    for s in manifest["shards"]:
        part = (DATA / name / s["file"]).read_bytes()
        assert len(part) == s["bytes"] <= 2_000_000
        assert hashlib.sha256(part).hexdigest() == s["sha256"]
        blob += part
    assert len(blob) == manifest["total_bytes"] == 11_277_120
    assert hashlib.sha256(blob).hexdigest() == manifest["sha256"]
    # weights: every 2-D row within half a scale of the checkpoint, every 1-D parameter bit-exact
    _, _, ckpt, cfg = load_both(name)
    sd = ckpt["model_state_dict"]
    recon = load_export(name)
    n_params = sum(p.numel() for p in book_model(sd, cfg).parameters())    # parameters() counts the tied matrix once
    assert n_params == 11_121_640 == sum(int(np.prod(t["shape"])) for t in manifest["tensors"])
    assert torch.equal(sd["embed.weight"], sd["unembed.weight"])
    for t in manifest["tensors"]:
        w, r = sd[t["name"]].float().numpy(), recon[t["name"]].numpy()
        if t["dtype"] == "q8":
            amax = np.abs(w).max(axis=1, keepdims=True)
            half = amax / 127 / 2                  # exact arithmetic: |q·s − w| ≤ s/2; float32 adds an ulp or two
            assert np.all(np.abs(r - w) <= half + 2 * np.spacing(amax)), t["name"]
        else:
            assert np.array_equal(r, w), t["name"]


@pytest.mark.parametrize("name", NAMES)
def test_int8_logit_shift_over_windows(name, dlfs6_path):
    """P.2.3: the 0.29 of 验证 P.2-d is measured at the last position of the 12 reference prompts only. Here: every
    position of the four reference prompts, and 8 fixed 256-token windows of tiny_codes.bin (the pretraining data)."""
    f32, q8, _, _ = load_both(name)
    data = np.fromfile(dlfs6_path / "codebot" / "tiny_codes.bin", dtype=np.uint16)
    starts = np.random.default_rng(0).integers(0, len(data) - 256, size=8)
    with torch.no_grad():
        prompts = max(float((f32(torch.tensor([p["ids"]]))[0] - q8(torch.tensor([p["ids"]]))[0]).abs().max())
                      for p in ref_of(name)["prompts"])
        worst, flips, total = 0.0, 0, 0
        for s in starts:
            ids = torch.from_numpy(data[s:s + 256].astype(np.int64))[None]
            a, b = f32(ids)[0], q8(ids)[0]
            worst = max(worst, float((a - b).abs().max()))
            flips += int((a.argmax(-1) != b.argmax(-1)).sum())     # positions where int8 changes the top-1 token
            total += a.shape[0]
    print(f"{name}: max|Δlogit| over every position of the reference prompts {prompts:.3f}; over 8 TinyCodes "
          f"windows {worst:.3f}, top-1 changes at {flips}/{total} positions ({flips / total:.2%})")
    assert prompts < 0.4 and worst < 0.5 and flips / total < 0.02


# ----------------------------------------------------------------------------- P.4, P.5, P.9 the reference
@torch.no_grad()
def rerun(model, ids: list[int], new_tokens: int, attention: bool) -> dict:
    """Independently of the exporter: last logits, logit lens (top-1 and its probability), greedy tokens with the
    book's window crop (no stop at <|endoftext|>), attention weights."""
    caps, hooks = {}, []
    hooks.append(model.dropout.register_forward_hook(lambda m, i, o: caps.__setitem__(0, o[0, -1].clone())))
    for l, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l + 1, o[0, -1].clone())))
        if attention:
            hooks.append(block.attn.attention_dropout.register_forward_hook(
                lambda m, i, o, l=l: caps.__setitem__(("attn", l), o[0].clone())))
    try:
        logits = model(torch.tensor([ids]))[0, -1]
    finally:
        for hk in hooks:
            hk.remove()
    lens = [torch.softmax(model.unembed(model.norm(caps[l])), -1) for l in range(len(model.blocks) + 1)]
    seq, greedy = list(ids), []
    for _ in range(new_tokens):
        nxt = int(model(torch.tensor([seq[-model.max_context_len:]]))[0, -1].argmax())
        greedy.append(nxt)
        seq.append(nxt)
    att = torch.stack([caps[("attn", l)] for l in range(len(model.blocks))]).numpy() if attention else None
    return {"logits": logits.numpy(), "lens": lens, "greedy": greedy, "attention": att}


@pytest.mark.parametrize("name", NAMES)
def test_reference_rederived_with_torch(name, dlfs6_path):
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from(dlfs6_path / "codebot" / "merge_rules.pkl")
    f32, q8, _, _ = load_both(name)
    ref = ref_of(name)
    for p in ref["prompts"]:
        assert tok.encode(p["text"]) == p["ids"], p["name"]
        for variant, model in [("f32", f32), ("q8", q8)]:
            want = p[variant]
            got = rerun(model, p["ids"], p["new_tokens"], "attention" in want)
            where = f"{name}/{p['name']}/{variant}"
            assert np.abs(got["logits"] - np.array(want["logits"])).max() < 2e-5, where   # ref.json keeps 6 decimals
            assert got["greedy"] == want["greedy"], where
            for l, rec in enumerate(want["lens"]):
                assert int(got["lens"][l].argmax()) == rec["ids"][0], f"{where}: lens layer {l}"
                assert abs(float(got["lens"][l][rec["ids"][0]]) - rec["probs"][0]) < 1e-5, f"{where}: lens layer {l}"
            if got["attention"] is not None:
                assert np.abs(got["attention"] - np.array(want["attention"])).max() < 1e-5, where


@pytest.mark.parametrize("name", NAMES)
def test_layer0_lens_self_prediction(name, dlfs6_path):
    """P.4.2: the layer-0 logit lens (final LayerNorm + tied unembedding of token + position embedding) for each of the
    1,000 ids at each of the 256 positions, with the int8 weights the browser runs. Ids that occur in tiny_codes.bin
    (the pretraining data) are always their own top-1; every id that never occurs loses the top spot somewhere."""
    sd = {k: v.double().numpy() for k, v in load_export(name).items()}
    emb, pos, g, b, ub = (sd[k] for k in ["embed.weight", "pos_embed.weight", "norm.weight", "norm.bias", "unembed.bias"])
    V, T = emb.shape[0], pos.shape[0]
    norm = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * g + b
    seen = np.bincount(np.fromfile(dlfs6_path / "codebot" / "tiny_codes.bin", dtype=np.uint16), minlength=V) > 0
    always, p_min = np.ones(V, bool), np.ones(V)
    for t in range(T):
        z = norm(emb + pos[t]) @ emb.T + ub                    # row i: the lens's logits for id i at position t
        always &= z.argmax(1) == np.arange(V)
        z -= z.max(1, keepdims=True)
        p_min = np.minimum(p_min, np.exp(np.diag(z)) / np.exp(z).sum(1))
    print(f"{name}: {int(seen.sum())} ids occur in tiny_codes.bin, {int((~seen).sum())} never do "
          f"({int((~seen[128:256]).sum())} of them non-ASCII bytes); own top-1 at all {T} positions: seen ids "
          f"{int(always[seen].sum())}/{int(seen.sum())} (lowest p(self) {p_min[seen].min():.2f}), never-seen ids "
          f"{int(always[~seen].sum())}/{int((~seen).sum())}")
    assert seen.sum() == 839 and not seen[128:256].any()       # TinyCodes is pure ASCII: no byte >= 0x80 occurs
    assert always[seen].all() and p_min[seen].min() > 0.5
    assert not always[~seen].any()                             # every never-seen id loses the top spot somewhere
    # the page's example: 不 is the UTF-8 bytes e4 b8 8d = ids 228, 184, 141; at position 2 the lens is not 141
    ids = list("不".encode())
    top = int(np.argmax(norm(emb[ids[-1]] + pos[len(ids) - 1]) @ emb.T + ub))
    assert ids == [228, 184, 141] and top != 141, top


@pytest.mark.parametrize("name", NAMES)
def test_previous_token_head(name, dlfs6_path):
    f32, _, _, _ = load_both(name)
    long = next(p for p in ref_of(name)["prompts"] if p["name"] == "long")
    att = rerun(f32, long["ids"], 0, attention=True)["attention"]          # (layer, head, query, key)
    t = np.arange(1, att.shape[-1])
    prev = att[:, :, t, t - 1].mean(-1)
    uniform = float(np.mean(1 / (t + 1)))
    print(f"{name}: {att.shape[-1]} tokens; layer 1 head 1 gives the previous token {prev[0, 0]:.3f} "
          f"(uniform {uniform:.3f}); the strongest head is {tuple(int(i) + 1 for i in np.unravel_index(prev.argmax(), prev.shape))}")
    assert uniform < 0.03
    assert 0.45 < prev[0, 0] < 0.49                                           # the page: 46% to 48%


@torch.no_grad()
def test_for_i_in_prefers_range(dlfs6_path):
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from(dlfs6_path / "codebot" / "merge_rules.pkl")
    ids = torch.tensor([tok.encode("for i in")])
    for variant, model in zip(["float32", "int8"], load_both("pretrain")[:2]):
        p = torch.softmax(model(ids)[0, -1].double(), -1)
        top = int(p.argmax())
        entropy = float(-(p * torch.log(p)).sum())                # nats; figure P-5 shows it for the int8 weights
        print(f"pretrain ({variant}): after 'for i in' -> {tok.decode([top])!r} with p = {float(p[top]):.3f}; "
              f"entropy {entropy:.3f} nats")
        assert tok.decode([top]) == " range" and 0.75 < float(p[top]) < 0.85     # the page: about 80%
        if variant == "int8":
            assert round(entropy, 2) == 1.23


# ----------------------------------------------------------------------------- P.2.4 the exporter on official weights
def test_exporter_handles_official_checkpoint(tmp_path, dlfs6_path):
    src = dlfs6_path / "codebot" / "model_grpo.pt"
    if not src.is_file():
        pytest.skip("official checkpoints not fetched: uv run python tools/fetch_data.py")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "-1"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "export_codebot.py"), "--ckpt", str(src), "--name", "official-grpo",
         "--label", "GRPO（官方）", "--template", "alpaca", "--order", "12", "--source", OFFICIAL_URL,
         "--out", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    m = json.loads((tmp_path / "official-grpo" / "manifest.json").read_text(encoding="utf-8"))
    prov = m["provenance"]
    assert prov["checkpoint"] == "third_party/dlfs6/codebot/model_grpo.pt" and prov["source"] == OFFICIAL_URL
    assert prov["run"] is None and prov["run_step"] is None
    assert prov["checkpoint_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert m["total_bytes"] == 11_277_120 and len(m["shards"]) == 6
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert [(c["name"], c["template"]) for c in index["checkpoints"]] == [("official-grpo", "alpaca")]
    # the exporter's reference and the addition record (two separate code paths) give the same answer to 3+4=
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from(dlfs6_path / "codebot" / "merge_rules.pkl")
    math = next(p for p in json.loads((tmp_path / "official-grpo" / "ref.json").read_text(encoding="utf-8"))["prompts"]
                if p["name"] == "math")
    answer = tok.decode(math["f32"]["greedy"][: math["f32"]["greedy"].index(tok.end_token_id)])
    record = json.loads(ADDITION.read_text(encoding="utf-8"))
    assert answer == record["models"]["official-grpo-f32"]["results"]["train"]["responses"][record["sets"]["train"].index("3+4=")]


def test_exporter_refuses_untied_weights(tmp_path, dlfs6_path):
    src = dlfs6_path / "codebot" / "model_grpo.pt"
    if not src.is_file():
        pytest.skip("official checkpoints not fetched: uv run python tools/fetch_data.py")
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    ckpt["model_state_dict"]["unembed.weight"] = ckpt["model_state_dict"]["unembed.weight"].clone() + 1.0
    torch.save(ckpt, tmp_path / "untied.pt")
    with pytest.raises(SystemExit, match="tied"):
        exporter.load_checkpoint(tmp_path / "untied.pt")


# ----------------------------------------------------------------------------- P.8 the addition record
def test_addition_record_is_current(dlfs6_path):
    reward = book_reward()
    assert reward(7, "7") == 1.0 and reward(7, "3+4=7") == 1.0 and reward(7, "17") == 0.0 and reward(7, "") == 0.0
    recorded = json.loads(ADDITION.read_text(encoding="utf-8"))
    fresh = build_addition()
    assert fresh["sets"] == recorded["sets"] and fresh["answers"] == recorded["answers"]
    for name, m in recorded["models"].items():
        assert fresh["models"][name]["checkpoint_sha256"] == m["checkpoint_sha256"], name
        for s, r in m["results"].items():
            got = fresh["models"][name]["results"][s]
            assert got["responses"] == r["responses"] and got["correct"] == r["correct"], f"{name}/{s}"
            assert all(r["eos"]), f"{name}/{s}: an answer hit the {recorded['max_new_tokens']}-token cap"
    n = {name: m["results"]["train"]["n_correct"] for name, m in recorded["models"].items()}
    assert n == {"ours-sft-f32": 44, "ours-sft-q8": 44, "ours-grpo-f32": 70, "ours-grpo-q8": 70,
                 "official-sft-f32": 44, "official-grpo-f32": 67}
    report = json.loads((ROOT / "web/data/runs/codebot-3060ti/report.json").read_text(encoding="utf-8"))
    acc = next(s for s in report["steps"] if s["script"] == "ch03/09_grpo.py")["final"]["acc"]
    # The progress bar's last value (86.4) is the evaluation at i = 490 (09_grpo.py evaluates when i % 10 == 0);
    # the checkpoint is saved after i = 499, nine update iterations later. So this is two measurements that agree,
    # not one measurement read twice.
    assert round(100 * n["ours-grpo-f32"] / 81, 1) == acc


# ----------------------------------------------------------------------------- P.9 the JavaScript side
PARITY_ROW = re.compile(r"^\s+(\w+)/(\w+)\s+max\|Δlogit\| (\S+) \(int8\) \S+ \(float32\); greedy (\d+)/(\d+) \(int8\)", re.M)
PLAIN_ROW = re.compile(r"^\s+(\w+)/(\w+)\s+plain JS: max\|Δlogit\| (\S+) \(int8\); greedy (\d+)/(\d+) \(int8\)", re.M)


def test_javascript_matches_pytorch():
    """The workbench's JavaScript (web/widgets/_gpt.js and the engine of the page), run by Node on the three exports.
    The Node tests compare it with ref.json (PyTorch with the same int8 weights) and with addition.json; this test
    runs them and checks the numbers they report, so the page's claims about the JavaScript rest on pytest too."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    tests = ["tests/js/playground-gpt.test.mjs", "tests/js/playground-parity.test.mjs", "tests/js/playground-engine.test.mjs"]
    proc = subprocess.run([node, "--test", "--test-reporter=spec", *tests], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert re.search(r"^ℹ fail 0$", proc.stdout, re.M)
    simd, plain = PARITY_ROW.findall(proc.stdout), PLAIN_ROW.findall(proc.stdout)
    assert len(simd) == 12 and len(plain) == 9, proc.stdout[-4000:]   # 3 checkpoints x 4 prompts; plain JS skips 'long'
    greedy = {}
    for name, prompt, _, same, total in simd:
        assert same == total, f"{name}/{prompt}: greedy tokens differ"
        greedy[name] = greedy.get(name, 0) + int(total)
    assert all(same == total for *_, same, total in plain)
    worst, worst_plain = max(float(r[2]) for r in simd), max(float(r[2]) for r in plain)
    print(f"WASM SIMD: max|Δlogit| {worst:.1e} over 12 prompts, greedy tokens per checkpoint {greedy}; "
          f"plain JS: max|Δlogit| {worst_plain:.1e}")
    assert worst < 2e-5 and worst_plain < 2e-5
    assert greedy == {"pretrain": 112, "sft": 112, "grpo": 112}
    # the engine re-asked all 181 addition problems for SFT and GRPO and got PyTorch's int8 answers token for token
    assert proc.stdout.count("every addition answer equals PyTorch with the same int8 weights (181 problems)") == 2
