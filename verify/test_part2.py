"""第 II 部扉页（StoryBot）：本地验证。

TinyStories 验证集的统计在全量验证集上重算；终点预览用官方的三个预训练 checkpoint（iter 500 / 5000 / 40000）
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
    assert all(s["tokens"] <= 256 for c in rec for s in c["samples"])   # 没有越过上下文长度
    # 文件名里的 500、5000 是从 0 数起的循环变量 i：保存发生在第 i 次循环的 optimizer.step() 之后
    src = (pp.DLFS6 / "ch06/05_pretrain.py").read_text(encoding="utf-8")
    loop = src[src.index("for i in pbar:"):src.index("# Validation lossのグラフを描画")]
    assert "pbar = tqdm(range(max_iters))" in src and loop.index("optimizer.step()") < loop.index("if i in save_iters:")
    assert "model.save(model_save_path)" in src[src.index("# Validation lossのグラフを描画"):]   # 最终模型在循环之后保存
    assert "manual_seed" not in src                 # 脚本不设随机种子：本书的 H100 重跑与官方 checkpoint 不是同一次训练
    assert [c["updates"] for c in rec] == [501, 5_001, 40_000]
    assert [c["tokens_seen"] for c in rec] == [4_104_192, 40_968_192, 327_680_000]


def test_tinystories_valid_is_the_cs336_file(dlfs6_path):
    """原书的 TinyStories 与 CS336 A1 用的是同一份文件：本地验证集的 SHA-256 等于 roneneldan/TinyStories 上
    TinyStoriesV2-GPT4-valid.txt 的 LFS oid（训练集不在本地，两边文件列表里的 oid 也相同，见 _parts_preview.CS336_SHA256）。"""
    assert pp.sha256(pp.DLFS6 / "storybot/tiny_stories_valid.txt") == pp.CS336_SHA256["storybot/tiny_stories_valid.txt"]


def test_storybot_cache_keeps_the_beginning(dlfs6_path):
    """storybot/utils.generate 只把新 token 送进模型（use_cache=True），所以截断 ids 的那两行管不到模型：
    KV cache 越过 max_context_len 以后照样变长。WebBot 的模型则只保留最近 max_context_len 个。"""
    import torch
    from storybot.model import GPT as StoryGPT
    from webbot.model import GPT as WebGPT
    src = (pp.DLFS6 / "storybot/utils.py").read_text(encoding="utf-8")
    assert "ids = ids[:, -model.max_context_len:]" in src and "logits = model(next_id, use_cache=True)" in src
    assert pp.script_hparams("ch06/06_generate.py")["max_new_tokens"] == 300          # 书中生成 300 个 token
    torch.manual_seed(0)
    held = {}
    for name, model in (("storybot", StoryGPT(100, 8, 32, 4, 2, 64)), ("webbot", WebGPT(100, 8, 32, 4, 2, 2, 64))):
        model.eval()
        model.clear_cache()
        x = torch.tensor([[1]])
        with torch.no_grad():
            for _ in range(20):                                   # 20 次 decode，上下文上限只有 8
                x = model(x, use_cache=True)[:, -1:].argmax(-1)
        held[name] = model.blocks[0].attn.k_cache.size(2)
    assert held == {"storybot": 20, "webbot": 8}
