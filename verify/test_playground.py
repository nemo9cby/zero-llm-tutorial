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
    assert prev[0, 0] > 0.4


@torch.no_grad()
def test_for_i_in_prefers_range(dlfs6_path):
    from codebot.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from(dlfs6_path / "codebot" / "merge_rules.pkl")
    ids = torch.tensor([tok.encode("for i in")])
    for variant, model in zip(["float32", "int8"], load_both("pretrain")[:2]):
        p = torch.softmax(model(ids)[0, -1], -1)
        top = int(p.argmax())
        print(f"pretrain ({variant}): after 'for i in' -> {tok.decode([top])!r} with p = {float(p[top]):.3f}")
        assert tok.decode([top]) == " range" and float(p[top]) > 0.5


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
    assert round(100 * n["ours-grpo-f32"] / 81, 1) == acc            # the progress bar's last value: 86.4
