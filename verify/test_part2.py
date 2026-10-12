"""第 II 部扉页（StoryBot）：本地验证。

TinyStories 验证集的统计在全量验证集上重算；终点预览用官方的三个预训练检查点（iter 500 / 5000 / 40000）
在 CPU 上以固定种子重新生成，与 web/data/part2/preview.json 逐字比较。
"""
import json

import pytest

import _parts_preview as pp


def recorded(rel):
    return json.loads((pp.WEB_DATA / rel).read_text(encoding="utf-8"))


def test_tinystories_valid_stats(dlfs6_path):
    stats, head = pp.tinystories_stats()
    rec = recorded("part2/doclen.json")
    for k in ("text_bytes", "non_ascii_bytes", "segments", "separators", "first_segment_head", "tokens",
              "summary", "lengths"):
        assert stats[k] == rec[k], k
    assert (stats["segments"], stats["tokens"]) == (27_631, 5_465_873)
    assert stats["summary"]["over_ctx"] == 3_002
    assert (pp.WEB_DATA / "part2/stories_head.txt").read_text(encoding="utf-8") == head
    assert head.count(pp.EOT) == 199                               # 200 篇完整的故事
    text = (pp.DLFS6 / "storybot/tiny_stories_valid.txt").read_text(encoding="utf-8")
    assert pp.EOT + head in text                                   # 样本原样取自验证集


def test_storybot_checkpoints_need_key_rename(dlfs6_path):
    import torch
    from storybot.model import GPT
    ck = torch.load(pp.DLFS6 / "storybot/model_pretrain.pt", map_location="cpu")
    keys = ck["model_state_dict"].keys()
    assert any(".mlp." in k for k in keys) and not any(".ffn." in k for k in keys)
    model = GPT(ck["vocab_size"], ck["max_context_len"], ck["embed_dim"], ck["n_head"], ck["n_layer"],
                ck["ff_dim"], ck["theta"])
    with pytest.raises(RuntimeError, match="ffn"):
        model.load_state_dict(ck["model_state_dict"])             # 原样加载会失败
    pp.load_storybot("storybot/model_pretrain.pt")                 # 改名后可以加载


def test_storybot_preview_matches_record(dlfs6_path):
    rec = recorded("part2/preview.json")["checkpoints"]
    new = pp.storybot_preview()["checkpoints"]
    assert [c["iters"] for c in new] == [500, 5000, 40000]
    for n, r in zip(new, rec):
        assert n["samples"] == r["samples"], n["file"]
        assert abs(n["val_loss_subset"] - r["val_loss_subset"]) < 1e-3, n["file"]
    losses = [c["val_loss_subset"] for c in rec]
    assert losses[0] > losses[1] > losses[2]
    assert rec[1]["tokens_seen"] == 40_960_000                      # CS336 低资源版的 token 预算
    assert all(s["tokens"] <= 256 for c in rec for s in c["samples"])   # 没有越过上下文长度
