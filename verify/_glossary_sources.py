"""Build web/data/glossary/sources.json: the evidence behind the glossary's naming tables.

The glossary page states which name each code base uses for the same quantity (vocab size, residual
width, heads, ...) and which metric names RL frameworks log. Those claims are checked by verify
cells against this file, and this file is built from primary sources:

  * DLFS6: constructor signatures parsed from third_party/dlfs6 (re-checked by verify/test_reference.py)
  * CS336 A1: adapter signatures parsed from tests/adapters.py of the pinned A1 clone
  * CS336 A5 (2026): the "log at least the following metrics" lines of the handout
  * Hugging Face: config.json of openai-community/gpt2, allenai/OLMo-2-0425-1B, Qwen/Qwen2.5-Math-1.5B,
    and GPT2Config.attribute_map from transformers' configuration_gpt2.py
  * nanoGPT: the GPTConfig dataclass in model.py
  * veRL / tinker-cookbook: the source lines that create each metric key

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
        },
        "hf": {
            "gpt2": {"url": "https://huggingface.co/openai-community/gpt2/blob/main/config.json",
                     "config": json.loads((ext / "hf-gpt2" / "config.json").read_text(encoding="utf-8"))},
            "olmo2_0425_1b": {"url": "https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json",
                              "config": json.loads((ext / "hf-olmo2" / "config.json").read_text(encoding="utf-8"))},
            "qwen25_math_1p5b": {"url": "https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/config.json",
                                 "config": json.loads((ext / "hf-qwen25math" / "config.json").read_text(encoding="utf-8"))},
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
                "verl/trainer/ppo/metric_utils.py": matching_lines(ext / "verl" / "metric_utils.py",
                                                                   r'^\s+"(critic|response_length)/[a-z_/]+":'),
                "verl/workers/utils/losses.py": matching_lines(ext / "verl" / "losses.py", r'"actor/pg_loss"'),
                "verl/trainer/ppo/core_algos.py": matching_lines(ext / "verl" / "core_algos.py",
                                                                 r'"actor/(pg_clipfrac|ppo_kl)"')[:2],
                "verl/trainer/ppo/ray_trainer.py": matching_lines(ext / "verl" / "ray_trainer.py",
                                                                  r'rename_dict\(actor_output, "actor/"\)|"actor/entropy"|"actor/grad_norm"'),
            },
        },
        "tinker_cookbook": {
            "repo": "https://github.com/thinking-machines-lab/tinker-cookbook", "commit": COMMITS["tinker_cookbook"],
            "lines": {
                "tinker_cookbook/rl/metric_util.py": matching_lines(ext / "tinker" / "metric_util.py",
                                                                    r'f"env/all/\{k\}"|"reward/total"|"by_group/frac_(mixed|all_good|all_bad)": n_|"ac_tokens_per_turn"'),
                "tinker_cookbook/rl/metrics.py": matching_lines(ext / "tinker" / "metrics.py",
                                                                r'"optim/(entropy|kl_sample_train_v1)"'),
                "tinker_cookbook/rl/train.py": matching_lines(ext / "tinker" / "train.py", r'"optim/lr"')[:1],
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
