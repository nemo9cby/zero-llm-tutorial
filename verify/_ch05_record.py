"""第 5 章：用 HF 上发布的、训练好的 StoryBot 重现 KV cache 的三个现象，并把结果记录下来。

    uv run python verify/_ch05_record.py          # 重新生成 web/data/ch05/storybot_samples.json

页面直接读取这份记录；verify/test_ch05.py 会用同一个函数重新计算一遍，并要求与记录逐字相同。
只用 CPU，约半分钟。
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True       # 不往 third_party/dlfs6 里写 __pycache__

from _ch05_util import DLFS6, ROOT, greedy_recompute, greedy_with_cache, load_defs, load_storybot  # noqa: E402

OUT = ROOT / "web" / "data" / "ch05" / "storybot_samples.json"
PROMPT = "Once upon a time, there was a little girl named Lily. She"
CKPT = DLFS6 / "storybot" / "model_pretrain.pt"
BUG = {"if not use_cache or (use_cache and is_first_call):": "if not use_cache:"}   # 8b3911f 之前的条件


def storybot_samples(n_new: int = 60, total_beyond: int = 300) -> dict:
    import torch

    sys.path.insert(0, str(DLFS6))
    try:
        from storybot.model import GPT
        from storybot.tokenizer import BPETokenizer
        from storybot.utils import generate
    finally:
        sys.path.remove(str(DLFS6))

    tok = BPETokenizer.load_from(str(DLFS6 / "storybot" / "merge_rules.pkl"))
    model = load_storybot(CKPT, GPT)
    buggy = load_storybot(CKPT, load_defs("storybot/model.py", replace=BUG)["GPT"])
    prompt = torch.tensor([tok.encode(PROMPT)])

    # ① 有无 KV cache：贪心生成的 token 完全相同；storybot.utils.generate(temperature=0) 也给出同一段文字
    no_cache = greedy_recompute(model, prompt, n_new)
    cache = greedy_with_cache(model, prompt, n_new)
    via_utils = generate(model, tok, PROMPT, n_new, temperature=0)

    # ② 旧 bug：prefill 时不加 causal mask
    bad = greedy_with_cache(buggy, prompt, n_new)
    diff = (cache != bad).nonzero()

    # ③ 超过 max_context_len = 256：cache 版（RoPE 位置被钳在 255）与“只看最近 256 个 token、每步重算”分道扬镳。
    #    两条路在分叉之前完全相同，所以沿 cache 版的 token 逐个检查“滑动窗口重算会不会给出同一个 token”，
    #    第一个不同的位置就是分叉点（结果与两条路各自完整生成再比较相同，但快得多）。
    C = model.max_context_len
    long_cache = greedy_with_cache(model, prompt, total_beyond - prompt.shape[1])
    first_diff_long = None
    with torch.no_grad():
        for t in range(C - 8, total_beyond):
            window_pred = model(long_cache[:, max(0, t - C):t])[0, -1].argmax().item()
            if window_pred != long_cache[0, t].item():
                first_diff_long = t
                break

    return {
        "vocab_size": tok.vocab_size,
        "n_merges": len(tok.merge_rules),
        "end_token_id": tok.end_token_id,
        "prompt": PROMPT,
        "prompt_ids": prompt[0].tolist(),
        "n_new": n_new,
        "same_with_and_without_cache": bool(torch.equal(no_cache, cache)),
        "utils_generate_same_text": via_utils == tok.decode(cache[0].tolist()),
        "correct": tok.decode(cache[0].tolist()),
        "buggy": tok.decode(bad[0].tolist()),
        "buggy_first_diff_token": int(diff[0, 1]) if len(diff) else None,
        "max_context_len": C,
        "beyond_total_tokens": int(long_cache.shape[1]),
        "beyond_first_diff_position": first_diff_long,
        "source": "third_party/dlfs6/storybot/model_pretrain.pt（HF koki0702/zero-llm-data，键名 mlp→ffn 改名后加载）",
    }


def main() -> None:
    rec = storybot_samples()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
