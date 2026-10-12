"""Named GPU jobs, shared by gpu/run_local.py (this PC, RunPod, Lambda) and gpu/modal_app.py (Modal).

A job is a list of book scripts that run in order in one workspace, the data files they read, the
checkpoints they may start from, and, only where a job cannot exist otherwise, exact text patches.
Patches never touch third_party/dlfs6: they are applied to the workspace copy of a script, must match
exactly once, and are recorded in the run report next to the step they changed.

    uv run python gpu/jobs.py webbot --nproc 8        # print the resolved plan (scripts, data, patches)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

HF = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"

STORY_DATA = ["storybot/tiny_stories_train.bin", "storybot/tiny_stories_valid.bin", "storybot/merge_rules.pkl"]
WEB_DATA = ["webbot/owt_train.bin", "webbot/owt_valid.bin"]
WEBBOT = "ch09/01_pretrain.py"

# How a missing checkpoint is obtained when the run has not produced it itself.
#   official        : the book's own checkpoint from Hugging Face (tools/fetch_data.py / Modal `prepare`)
#   official-keyfix : the same, with StoryBot's `blocks.N.mlp.*` keys renamed to `blocks.N.ffn.*`
#                     (the HF files predate the rename in storybot/model.py; load_state_dict fails otherwise)
SEED_HOW = {"official", "official-keyfix"}


@dataclass(frozen=True)
class Patch:
    script: str
    old: str
    new: str
    why: str


@dataclass
class Plan:
    job: str
    bot: str
    scripts: list[str]
    data: list[str] = field(default_factory=list)
    seeds: dict[str, str] = field(default_factory=dict)
    patches: list[Patch] = field(default_factory=list)
    nproc: int = 1
    modal_gpu: str = "L4"
    overrides: dict[str, int] = field(default_factory=dict)   # new hyperparameter values the patches set
    notes: list[str] = field(default_factory=list)

    def patches_for(self, script: str) -> list[Patch]:
        return [p for p in self.patches if p.script == script]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan(**{**d, "patches": [Patch(**p) for p in d.get("patches", [])]})


JOBS: dict[str, dict] = {
    "codebot": dict(bot="codebot", scripts=["ch03/01_pretrain.py", "ch03/02_generate.py",
                                            "ch03/03_sft.py", "ch03/09_grpo.py"], modal_gpu="L4"),
    "codebot-pretrain": dict(bot="codebot", scripts=["ch03/01_pretrain.py"], modal_gpu="L4"),
    "codebot-generate": dict(bot="codebot", scripts=["ch03/02_generate.py"], modal_gpu="L4",
                             seeds={"codebot/model_pretrain.pt": "official"}),
    "codebot-sft": dict(bot="codebot", scripts=["ch03/03_sft.py"], modal_gpu="L4",
                        seeds={"codebot/model_pretrain.pt": "official"}),
    "codebot-grpo": dict(bot="codebot", scripts=["ch03/09_grpo.py"], modal_gpu="L4",
                         seeds={"codebot/model_sft.pt": "official"}),
    "storybot": dict(bot="storybot", scripts=["ch06/05_pretrain.py", "ch06/06_generate.py", "ch06/08_dpo.py"],
                     data=STORY_DATA, modal_gpu="H100"),
    "storybot-pretrain": dict(bot="storybot", scripts=["ch06/05_pretrain.py"], data=STORY_DATA, modal_gpu="H100"),
    "storybot-generate": dict(bot="storybot", scripts=["ch06/06_generate.py"], data=["storybot/merge_rules.pkl"],
                              seeds={"storybot/model_pretrain.pt": "official-keyfix"}, modal_gpu="L4"),
    "storybot-dpo": dict(bot="storybot", scripts=["ch06/08_dpo.py"], data=["storybot/merge_rules.pkl"],
                         seeds={"storybot/model_pretrain.pt": "official-keyfix"}, modal_gpu="L4"),
    "webbot-probe": dict(bot="webbot", scripts=[WEBBOT], data=WEB_DATA, modal_gpu="H100"),
    "webbot": dict(bot="webbot", scripts=[WEBBOT, "ch09/02_generate.py"],
                   data=WEB_DATA + ["webbot/merge_rules.pkl"], modal_gpu="H100"),
}

# Hyperparameter lines of ch09/01_pretrain.py that the WebBot variants rewrite (each occurs exactly once).
WEB_LINES = {"max_iters": "max_iters = 100000", "warmup_iters": "warmup_iters = 500",
             "micro_batch_size": "micro_batch_size = 32", "accumulation_steps": "accumulation_steps = 4"}
PROBE_LINE = "pbar = tqdm(range(max_iters), disable=not is_main)"


def probe_patch(steps: int = 200) -> Patch:
    """Stop after the first `steps` steps of the *real* run: same LR schedule, same seed, same data order."""
    return Patch(WEBBOT, PROBE_LINE, f"pbar = tqdm(range({steps}), disable=not is_main)  # zero-llm probe",
                 f"计时探针：只跑真实训练的前 {steps} 步（学习率调度、随机种子都不变），用 it/s 推算全程")


WEB_DEFAULTS = {"max_iters": 100000, "warmup_iters": 500, "micro_batch_size": 32, "accumulation_steps": 4}


def webbot_settings(nproc: int = 1, max_iters: int | None = None, scale: str = "iters") -> dict[str, int]:
    """New values for ch09/01_pretrain.py's hyperparameters (only the ones that change).

    ch09/01_pretrain.py fixes max_iters = 100000 and lets every rank draw its own 32×4×1024-token
    micro-batches, so `torchrun --nproc-per-node=N` *unchanged* trains N times the tokens for N times
    the money in the same wall time. Two ways to keep the 1-GPU budget:
      scale="iters": divide max_iters and warmup_iters by N (same LR-vs-tokens curve, N× bigger steps)
      scale="batch": divide the per-GPU batch by N (same global batch, same steps; needs 128 % N == 0)
    """
    if nproc < 1:
        raise ValueError("nproc must be >= 1")
    base = max_iters or WEB_DEFAULTS["max_iters"]
    if scale == "iters":
        if base % nproc:
            raise ValueError(f"max_iters {base} is not divisible by nproc {nproc}")
        new = {"max_iters": base // nproc, "warmup_iters": max(1, int(500 / nproc + 0.5))}
    elif scale == "batch":
        if 128 % nproc:
            raise ValueError(f"global batch 128 is not divisible by nproc {nproc}")
        seqs = 128 // nproc                      # sequences each GPU contributes per optimizer step
        micro, accum = (32, seqs // 32) if seqs >= 32 else (seqs, 1)
        new = {"max_iters": base, "micro_batch_size": micro, "accumulation_steps": accum}
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return {k: v for k, v in new.items() if v != WEB_DEFAULTS[k]}


def webbot_patches(nproc: int = 1, max_iters: int | None = None, scale: str = "iters") -> list[Patch]:
    """The patches that realize webbot_settings(), each with the reason recorded in the report."""
    s = webbot_settings(nproc, max_iters, scale)
    base = max_iters or WEB_DEFAULTS["max_iters"]
    why = {
        "max_iters": (f"{nproc} 卡时每步处理 {nproc} 倍 token：{base:,} ÷ {nproc} = {s.get('max_iters', 0):,} 步，总 token 不变"
                      if nproc > 1 and scale == "iters" else f"只跑 {s.get('max_iters', 0):,} 步"),
        "warmup_iters": f"warmup 也除以 {nproc}，学习率随 token 的变化曲线保持不变",
        "micro_batch_size": f"全局 batch 保持 128 条序列：每卡每步 {128 // nproc} 条",
        "accumulation_steps": f"全局 batch 保持 128 条序列：每卡每步 {128 // nproc} 条",
    }
    return [Patch(WEBBOT, WEB_LINES[k], f"{k} = {v}", why[k]) for k, v in s.items()]


def plan(job: str, nproc: int = 1, max_iters: int | None = None, scale: str = "iters",
         probe_steps: int = 200) -> Plan:
    if job not in JOBS:
        raise KeyError(f"unknown job {job!r}; choose from {', '.join(JOBS)}")
    spec = JOBS[job]
    p = Plan(job=job, bot=spec["bot"], scripts=list(spec["scripts"]), data=list(spec.get("data", [])),
             seeds=dict(spec.get("seeds", {})), modal_gpu=spec.get("modal_gpu", "L4"), nproc=nproc)
    if spec["bot"] != "webbot" and (nproc != 1 or max_iters):
        raise ValueError("--nproc/--max-iters only apply to WebBot (ch09 is the only DDP script)")
    if job == "webbot-probe":
        if nproc != 1:
            raise ValueError("the probe measures one GPU; multiply by N yourself")
        p.patches.append(probe_patch(probe_steps))
        p.overrides = {"steps": probe_steps}
    elif job == "webbot":
        p.patches += webbot_patches(nproc, max_iters, scale)
        p.overrides = webbot_settings(nproc, max_iters, scale)
    if nproc > 1:
        p.notes.append(f"torchrun --standalone --nproc-per-node={nproc}：评估只在 rank 0 上跑，其余 {nproc - 1} 张卡空等（照样计费）")
    return p


def files_for_bots(bots: list[str], official: bool = True) -> list[str]:
    """What a fresh machine (or Modal's `prepare`) must download for these bots' jobs:
    every data file they read, plus the official checkpoints the jobs can start from."""
    known = {spec["bot"] for spec in JOBS.values()}
    if set(bots) - known:
        raise KeyError(f"unknown bot(s) {sorted(set(bots) - known)}; choose from {sorted(known)}")
    out: list[str] = []
    for spec in JOBS.values():
        if spec["bot"] in bots:
            for rel in spec.get("data", []) + (list(spec.get("seeds", {})) if official else []):
                if rel not in out:
                    out.append(rel)
    return out


def apply_patches(text: str, patches: list[Patch]) -> str:
    """Apply exact, single-occurrence replacements; refuse anything ambiguous."""
    for p in patches:
        n = text.count(p.old)
        if n != 1:
            raise ValueError(f"patch for {p.script}: {p.old!r} occurs {n} times (need exactly 1)")
        text = text.replace(p.old, p.new)
    return text


def fix_storybot_keys(src, dst) -> int:
    """Copy a Hugging Face StoryBot checkpoint, renaming `.mlp.` -> `.ffn.`; return how many keys changed."""
    import torch  # local import: jobs.py itself must stay importable without torch

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]
    renamed = {k.replace(".mlp.", ".ffn."): v for k, v in sd.items()}
    changed = sum(k != k.replace(".mlp.", ".ffn.") for k in sd)
    torch.save({**ckpt, "model_state_dict": renamed}, dst)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Show what a GPU job runs.")
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--scale", choices=["iters", "batch"], default="iters")
    a = ap.parse_args()
    print(json.dumps(plan(a.job, a.nproc, a.max_iters, a.scale).to_dict(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
