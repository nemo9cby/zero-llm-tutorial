"""Build web/data/glossary/sources.json: the evidence behind the glossary's naming tables.

The glossary page states which name each code base uses for the same quantity (vocab size, residual
width, heads, ...) and which metric names RL frameworks log. Those claims are checked by verify
cells against this file, and this file is built from primary sources:

  * DLFS6: constructor signatures parsed from third_party/dlfs6 (re-checked by verify/test_reference.py)
  * CS336 A1: adapter signatures parsed from tests/adapters.py of the pinned A1 clone
  * CS336 A5 (2026): the "log at least the following metrics" lines of the handout, its clip-fraction
    request, and the lines that say how a GRPO batch is counted (rollout_batch_size, train_batch_size, ...)
  * Hugging Face: config.json of openai-community/gpt2, allenai/OLMo-2-0425-1B, Qwen/Qwen2.5-Math-1.5B,
    Qwen/Qwen3-0.6B (its own fetch date: it states head_dim separately), and GPT2Config.attribute_map from
    transformers' configuration_gpt2.py
  * nanoGPT: the GPTConfig dataclass in model.py
  * veRL / tinker-cookbook: the source lines that create each metric key and say how it is averaged, plus the
    batch-size settings; `blobs` records each fetched file's git blob id (equal to the repository tree's entry
    at the pinned commit, so the excerpts are from exactly that version)

    uv run python verify/_glossary_sources.py --ext <dir with the fetched files> --a1 <A1 clone> \
        --a5-text <cs336_spring2026_assignment5_alignment.txt>

The fetched files were downloaded on the date in FETCHED; URLs and commits are recorded per source.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
OUT = ROOT / "web" / "data" / "glossary" / "sources.json"
FETCHED = "2026-10-11"

COMMITS = {
    "cs336_a1": "a158843b20107949f1a8d7df1b05cd33b9166712",
    "cs336_a5": "c2734a26308710949fe13226960a1e8cece94b7e",
    "nanogpt_model_py": "f08abb45bd2285627d17da16daea14dda7e7253e",
    "transformers": "55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e",
    "verl": "1d367a4f459a43768f76312c0d3430a99ec46b8c",
    "tinker_cookbook": "70767c738ba5f523b66210a70e50b0dc14c9fc1a",
}


def signature(path: Path, cls: str | None, func: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if cls:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == func)
    return [a.arg for a in fn.args.args if a.arg != "self"]


def dlfs6_signatures() -> dict:
    return {f"{bot}/model.py GPT.__init__": signature(DLFS6 / bot / "model.py", "GPT", "__init__")
            for bot in ("codebot", "storybot", "webbot")}


def matching_lines(path: Path, pattern: str, limit: int = 40) -> list[str]:
    rx = re.compile(pattern)
    out = [f"{i}: {line.strip()}" for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if rx.search(line)]
    return out[:limit]


def lines_in_function(path: Path, func: str, pattern: str) -> list[str]:
    """Lines matching `pattern` inside the body of `def func(` (until the next top-level statement)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {func}("))
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
                and not lines[i].startswith(")")), len(lines))
    rx = re.compile(pattern)
    return [f"{i + 1}: {lines[i].strip()}" for i in range(start, end) if rx.search(lines[i])]


def git_blob_sha(path: Path) -> str:
    """The git blob id of a fetched file; equal to the `sha` of its entry in the repository tree at the commit."""
    import hashlib

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def text_lines(lines: list[str], patterns: list[str]) -> list[str]:
    """The first line of a handout text that contains each pattern, stripped (pdftotext bullets become •)."""
    return [next(l for l in lines if p in l).replace("�", "•").strip() for p in patterns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", type=Path, required=True)
    ap.add_argument("--a1", type=Path, required=True)
    ap.add_argument("--a5-text", type=Path, required=True)
    a = ap.parse_args()
    ext = a.ext
    a1_names = ["run_transformer_lm", "run_transformer_block", "run_multihead_self_attention_with_rope",
                "run_rope", "run_get_batch", "run_rmsnorm", "run_swiglu"]
    a5_lines = a.a5_text.read_text(encoding="utf-8", errors="replace").splitlines()  # pdftotext bullets are not UTF-8
    start = next(i for i, l in enumerate(a5_lines) if "Please log at least the following" in l)
    plots = next(i for i, l in enumerate(a5_lines) if "Log the following metrics, and produce plots" in l)
    clean = lambda lines: [l.replace("�", "•").strip() for l in lines]
    data = {
        "fetched": FETCHED,
        "dlfs6": {"commit": "c9b6e2e", "signatures": dlfs6_signatures()},
        "cs336_a1": {
            "url": "https://github.com/stanford-cs336/assignment1-basics/blob/"
                   f"{COMMITS['cs336_a1']}/tests/adapters.py",
            "signatures": {n: signature(a.a1 / "tests" / "adapters.py", None, n) for n in a1_names},
        },
        "cs336_a5_2026": {
            "url": f"https://github.com/stanford-cs336/assignment5-alignment/tree/{COMMITS['cs336_a5']}",
            "handout": "cs336_spring2026_assignment5_alignment.pdf",
            "required_metrics_excerpt": clean(a5_lines[start:start + 9]),   # + the pass@k suggestion
            "plotted_metrics_excerpt": clean(a5_lines[plots:plots + 9]),
            # how the handout counts a GRPO batch (suggested hyperparameters, on- and off-policy)
            "batch_excerpt": text_lines(a5_lines, [
                "rollout_batch_size = n_prompts_per_rollout_batch * group_size.",
                "rollout_batch_size = train_batch_size = 256", "group_size = 8", "gradient_accumulation_steps = 32",
                "Note that rollout_batch_size and train_batch_size count responses, not prompts. So",
                "rollout_batch_size = train_batch_size = 256 means 32 prompts with 8 rollouts each.",
                "32x off policy (so we take 32 training steps for each inference batch):",
                "train_batch_size = 8"]),
            "clip_fraction_excerpt": text_lines(a5_lines, [
                "In addition to the metrics you're already logging, make sure to also log the clip fraction."]),
        },
        "hf": {
            "gpt2": {"url": "https://huggingface.co/openai-community/gpt2/blob/main/config.json",
                     "config": json.loads((ext / "hf-gpt2" / "config.json").read_text(encoding="utf-8"))},
            "olmo2_0425_1b": {"url": "https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json",
                              "config": json.loads((ext / "hf-olmo2" / "config.json").read_text(encoding="utf-8"))},
            "qwen25_math_1p5b": {"url": "https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/config.json",
                                 "config": json.loads((ext / "hf-qwen25math" / "config.json").read_text(encoding="utf-8"))},
            # a newer config that states head_dim separately, and not as hidden_size / heads (fetched later)
            "qwen3_0p6b": {"url": "https://huggingface.co/Qwen/Qwen3-0.6B/blob/"
                                  "c1899de289a04d12100db370d81485cdf75e47ca/config.json", "fetched": "2026-10-12",
                           "config": json.loads((ext / "hf-qwen3-0p6b" / "config.json").read_text(encoding="utf-8"))},
            "gpt2_attribute_map": {
                "url": "https://github.com/huggingface/transformers/blob/"
                       f"{COMMITS['transformers']}/src/transformers/models/gpt2/configuration_gpt2.py",
                "lines": matching_lines(ext / "hf-transformers" / "configuration_gpt2.py",
                                        r'^\s+"(hidden_size|max_position_embeddings|num_attention_heads|num_hidden_layers)": "|n_inner: '),
                "inner_dim_line": matching_lines(ext / "hf-transformers" / "modeling_gpt2.py", r"inner_dim = config\.n_inner"),
                "tying_lines": (matching_lines(ext / "hf-transformers" / "configuration_gpt2.py", r"tie_word_embeddings: bool")
                                + matching_lines(ext / "hf-transformers" / "modeling_gpt2.py", r"_tied_weights_keys = ")[:1]),
            },
        },
        "nanogpt": {
            "url": f"https://github.com/karpathy/nanoGPT/blob/{COMMITS['nanogpt_model_py']}/model.py",
            "gptconfig_lines": matching_lines(ext / "nanogpt" / "model.py",
                                              r"^\s+(block_size|vocab_size|n_layer|n_head|n_embd|dropout|bias): "),
            "head_size_lines": matching_lines(ext / "nanogpt" / "model.py", r"assert config\.n_embd % config\.n_head|# \(B, nh, T, hs\)")[:2],
            "other_lines": matching_lines(ext / "nanogpt" / "model.py",
                                          r"self\.c_fc +=|F\.layer_norm\(input|self\.transformer\.wte\.weight = self\.lm_head\.weight"),
        },
        "verl": {
            "repo": "https://github.com/verl-project/verl", "commit": COMMITS["verl"],
            "lines": {
                "verl/trainer/ppo/metric_utils.py": matching_lines(
                    ext / "verl" / "metric_utils.py",
                    # every metric key, plus the branch that decides whether the value-network keys exist, plus
                    # how rewards (one sum per response) and advantages (every response token) are averaged
                    r'^\s+"(critic|response_length)/[a-z_/]+":|^\s+if use_critic:|Response mask is all False'
                    r'|critic_value_metrics = \{\}|sequence_reward = batch\.batch\["token_level_rewards"\]\.sum\(-1\)'
                    r'|reward_mean = torch\.mean\(non_aborted_sequence_reward\)'
                    r'|valid_adv = torch\.masked_select\(advantages, response_mask\)|adv_mean = torch\.mean\(valid_adv\)'),
                "verl/workers/utils/losses.py": matching_lines(ext / "verl" / "losses.py", r'"actor/pg_loss"'),
                "verl/trainer/ppo/core_algos.py": matching_lines(ext / "verl" / "core_algos.py",
                                                                 r'"actor/(pg_clipfrac|ppo_kl)"')[:2]
                + lines_in_function(ext / "verl" / "core_algos.py", "compute_grpo_outcome_advantage",
                                    r"scores\[i\] = scores\[i\] - id2mean|scores = scores\.unsqueeze\(-1\) \* response_mask"),
                "verl/trainer/ppo/ray_trainer.py": matching_lines(
                    ext / "verl" / "ray_trainer.py",
                    r'rename_dict\(actor_output, "actor/"\)|"actor/entropy"|"actor/grad_norm"'
                    r'|core_var = "acc" if "acc" in var2metric2val else "reward"|metric_sec = "val-(core|aux)"'
                    r'|pfx = f"\{metric_sec\}/\{data_source\}/\{var_name\}/\{metric_name\}"'
                    r'|or self\.config\.data\.train_batch_size|batch\.repeat\(repeat_times=self\.config\.actor_rollout_ref\.rollout\.n'
                    r'|ppo_mini_batch_size = ppo_mini_batch_size \* self\.config\.actor_rollout_ref\.rollout\.n'),
                "verl/workers/engine_workers.py": matching_lines(
                    ext / "verl" / "engine_workers.py",
                    r'lr = metrics\.pop\("lr", None\)|final_metrics\["lr"\] = lr|output\["metrics"\]\["lr"\] = lr'),
            },
            "blobs": {name: git_blob_sha(ext / "verl" / Path(name).name) for name in (
                "verl/trainer/ppo/metric_utils.py", "verl/workers/utils/losses.py", "verl/trainer/ppo/core_algos.py",
                "verl/trainer/ppo/ray_trainer.py", "verl/workers/engine_workers.py")},
        },
        "tinker_cookbook": {
            "repo": "https://github.com/thinking-machines-lab/tinker-cookbook", "commit": COMMITS["tinker_cookbook"],
            "lines": {
                "tinker_cookbook/rl/metric_util.py": matching_lines(ext / "tinker" / "metric_util.py",
                                                                    r'f"env/all/\{k\}"|"reward/total"|"by_group/frac_(mixed|all_good|all_bad)": n_|"ac_tokens_per_turn"'),
                "tinker_cookbook/rl/metrics.py": matching_lines(
                    ext / "tinker" / "metrics.py",
                    r'"optim/(entropy|kl_sample_train_v1)"|Compares the logprobs from when trajectories were sampled'
                    r'|logprob_diff = sampling_logprobs_actions - training_logprobs_actions'
                    r'|kl_sample_train_v1 = flat_diffs\.mean\(\)\.item\(\)'),
                "tinker_cookbook/rl/train.py": matching_lines(ext / "tinker" / "train.py", r'"optim/lr"')[:1],
            },
            "blobs": {name: git_blob_sha(ext / "tinker" / Path(name).name) for name in (
                "tinker_cookbook/rl/metric_util.py", "tinker_cookbook/rl/metrics.py", "tinker_cookbook/rl/train.py")},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
