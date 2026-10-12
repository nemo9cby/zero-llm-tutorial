"""第 0 章 LLM 是如何炼成的：本地验证。

三台机器人的规模表（web/data/ch00/bots.json）要和真实代码对得上：参数量来自真正构造出来的模型，
训练预算来自训练脚本里的超参数（用 AST 读取，不运行脚本），初始 loss 来自真实语料上的一次前向。
全部在 CPU 上运行。
"""
import importlib.util
import inspect
import json
import math

import numpy as np
import pytest

import _parts_preview as pp

BOTS = json.loads((pp.WEB_DATA / "ch00" / "bots.json").read_text(encoding="utf-8"))["bots"]
BY_ID = {b["id"]: b for b in BOTS}


@pytest.fixture(scope="module")
def models(dlfs6_path):
    return pp.build_models(seed=0)


def test_param_counts_match_real_models(models):
    expected = {"codebot": 11_121_640, "storybot": 22_696_448, "webbot": 152_316_672}
    for key, model in models.items():
        total = sum(p.numel() for p in model.parameters())   # 共享的权重只算一次
        assert total == expected[key] == BY_ID[key]["params"]
        assert pp.param_split(model) == BY_ID[key]["params_split"]
        assert sum(BY_ID[key]["params_split"].values()) == total
    # CodeBot 的 embed 与 unembed 是同一个张量，所以“输出层”只多出 1000 个 bias
    assert models["codebot"].embed.weight is models["codebot"].unembed.weight
    assert BY_ID["codebot"]["params_split"]["output_extra"] == 1000


def test_training_budgets_match_book_scripts(dlfs6_path):
    for b in BOTS:
        h = pp.script_hparams(b["pretrain"]["script"])
        m, t = b["model"], b["pretrain"]
        for k in ("vocab_size", "context_len", "embed_dim", "n_layer", "n_head", "ff_dim"):
            assert h[k] == m[k], (b["id"], k)
        assert h["max_iters"] == t["max_iters"] and h["learning_rate"] == t["learning_rate"]
        assert h.get("batch_size", h.get("micro_batch_size")) == t["batch_size"]
        assert h.get("accumulation_steps", 1) == t["accumulation_steps"]
        assert t["tokens"] == t["max_iters"] * t["batch_size"] * t["accumulation_steps"] * t["context_len"]
    assert [BY_ID[k]["pretrain"]["tokens"] for k in ("codebot", "storybot", "webbot")] == \
        [163_840_000, 327_680_000, 13_107_200_000]
    assert pp.script_hparams("ch09/01_pretrain.py")["n_kv_head"] == BY_ID["webbot"]["model"]["n_kv_head"] == 4


def test_post_training_budgets_match_book_scripts(dlfs6_path):
    sft, grpo = BY_ID["codebot"]["post"]
    (dpo,) = BY_ID["storybot"]["post"]
    assert BY_ID["webbot"]["post"] == []
    for stage in (sft, grpo, dpo):
        h = pp.script_hparams(stage["script"])
        for k in ("max_iters", "batch_size", "learning_rate"):
            assert h[k] == stage[k], (stage["stage"], k)
    assert pp.script_hparams("ch03/09_grpo.py")["group_size"] == grpo["group_size"] == 8
    assert pp.script_hparams("ch06/08_dpo.py")["beta"] == dpo["beta"] == 0.1
    sft_data = json.loads((pp.DLFS6 / "codebot/tiny_codes_sft.json").read_text(encoding="utf-8"))
    dpo_data = json.loads((pp.DLFS6 / "storybot/tiny_stories_dpo.json").read_text(encoding="utf-8"))
    assert (len(sft_data), len(dpo_data)) == (sft["examples"], dpo["examples"]) == (1092, 144)
    # GRPO 的题目就是 1..9 加 1..9 的全部组合（ch03/09_grpo.py 的 GRPODataset）
    src = (pp.DLFS6 / "ch03/09_grpo.py").read_text(encoding="utf-8")
    assert "for i in range(1, 10):" in src and "for j in range(1, 10):" in src and grpo["examples"] == 81


def test_corpus_sizes(dlfs6_path):
    # 本地就有的语料：token 数 = .bin 字节数 / 2（uint16）
    assert (pp.DLFS6 / "codebot/tiny_codes.bin").stat().st_size // 2 == BY_ID["codebot"]["corpus"]["tokens"] == 2_677_398
    assert (pp.DLFS6 / "codebot/tiny_codes.txt").stat().st_size == BY_ID["codebot"]["corpus"]["text_bytes"] == 6_487_033
    # 训练集默认不下载：大小取自 Hugging Face 的文件列表，并与 tools/fetch_data.py 的清单交叉核对
    spec = importlib.util.spec_from_file_location("fetch_data", pp.ROOT / "tools" / "fetch_data.py")
    fetch_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_data)
    manifest = {rel: size for rel, size, _ in fetch_data.MANIFEST}
    for rel, size in pp.HF_SIZES.items():
        if rel in manifest:
            assert manifest[rel] == size, rel
    assert BY_ID["storybot"]["corpus"]["tokens"] == pp.HF_SIZES["storybot/tiny_stories_train.bin"] // 2 == 541_229_223
    assert BY_ID["webbot"]["corpus"]["tokens"] == pp.HF_SIZES["webbot/owt_train.bin"] // 2 == 2_649_004_800


def test_cpu_step_record(dlfs6_path):
    """web/data/ch00/cpu_step.json（0.7 节的 CPU 时间账）：配置取自原书脚本，早先的两次计时取自研究日志。
    计时本身取决于机器和负载，不在这里重测；重新测量用 `_parts_preview.py --write --only cpustep`。"""
    rec = json.loads((pp.WEB_DATA / "ch00" / "cpu_step.json").read_text(encoding="utf-8"))
    h = pp.script_hparams(rec["script"])
    assert rec["script"] == "ch03/01_pretrain.py" and rec["max_iters"] == h["max_iters"] == 20_000
    assert rec["tokens_per_step"] == h["batch_size"] * h["context_len"] == 8_192
    assert rec["params"] == BY_ID["codebot"]["params"] and len(rec["seconds_per_step"]) >= 3
    log = (pp.ROOT / "research" / "repo-ch03-train-align.md").read_text(encoding="utf-8")
    assert "torch 2.9.1+cpu（8 threads）" in log
    for e in rec["earlier"]:
        assert f"{e['s_per_step']:.3f} s/iter" in log and e["torch"] == "2.9.1+cpu" and e["threads"] == 8


def test_initial_loss_is_a_little_above_ln_vocab(models):
    for key, model in models.items():
        tokens = np.memmap(pp.DLFS6 / BY_ID[key]["init_loss_data"], dtype=np.uint16, mode="r")
        loss = pp.initial_loss(model, tokens)
        assert abs(loss - BY_ID[key]["init_loss"]) < 1e-3, (key, loss)
        excess = loss - math.log(BY_ID[key]["model"]["vocab_size"])
        assert 0 < excess < 0.2, (key, excess)


def test_embed_dim_is_the_residual_width(models):
    """embed_dim（HF: hidden_size，CS336: d_model，GPT-2: n_embd）是每个 token 向量在整个网络里的宽度。"""
    import torch
    inner = {"codebot": lambda blk: blk.ffn.layers[0].out_features,     # nn.Linear(embed_dim, ff_dim)
             "storybot": lambda blk: blk.ffn.W.out_features,             # SwiGLU 的 W
             "webbot": lambda blk: blk.mlp.W.out_features}
    for key, model in models.items():
        cfg = BY_ID[key]["model"]
        d = cfg["embed_dim"]
        assert "embed_dim" in inspect.signature(type(model).__init__).parameters
        assert tuple(model.embed.weight.shape) == (cfg["vocab_size"], d)
        x = torch.randn(1, 8, d)
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
                assert tuple(x.shape) == (1, 8, d)               # 每一层进出都是 d 维
        assert inner[key](model.blocks[0]) == cfg["ff_dim"] != d  # FFN 的中间宽度是另一个数


def test_windows_pitfalls(dlfs6_path, tmp_path):
    import contextlib
    import io
    import pickle
    from storybot.tokenizer import train_bpe                    # 第 4 章起的训练器：以二进制模式读文件
    raw = (pp.DLFS6 / "codebot/tiny_codes.txt").read_bytes()
    assert b"\r" not in raw                                      # core.autocrlf=false 的检出是纯 LF
    doc = json.loads((pp.WEB_DATA / "part1/doclen.json").read_text(encoding="utf-8"))
    assert raw.count(b"\n") == doc["line_feeds"] == 256_501     # 页面上 0.7-a 用的换行数
    crlf = tmp_path / "tiny_codes.txt"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))                # autocrlf=true 检出的样子：每行多一个 \r
    assert crlf.stat().st_size == len(raw) + doc["line_feeds"] == 6_743_534
    # 第 1 章的脚本用文本模式读（open(...).read()）：Python 把 \r\n 还原成 \n，结果与 LF 版完全相同
    assert "open(\"codebot/tiny_codes.txt\").read()" in (pp.DLFS6 / "ch01/07_tiny_codes.py").read_text(encoding="utf-8")
    with open(crlf, encoding="utf-8") as f:
        assert f.read() == raw.decode("utf-8")
    # 二进制读取看得见 \r：在完整的 CRLF 版语料上，第 2 次合并就成了 \r\n，原本第 2 的 4 个空格被挤到第 3
    with contextlib.redirect_stderr(io.StringIO()):            # tqdm 进度条
        rules = train_bpe(str(crlf), 256 + 3 + 1, num_processes=8, num_chunks=64)
    assert list(rules) == [(32, 32), (13, 10), (256, 256)]
    official = list(pickle.loads((pp.DLFS6 / "codebot/merge_rules.pkl").read_bytes()))
    assert official[:2] == [(32, 32), (256, 256)]               # LF 版：两个空格，然后四个空格
    # 原书读 SFT 数据的两个脚本用不带 encoding 的 open()；简体中文 Windows 的默认编码是 GBK
    for rel in ("ch03/03_alpaca.py", "ch03/03_sft.py"):
        src = (pp.DLFS6 / rel).read_text(encoding="utf-8")
        assert "open('codebot/tiny_codes_sft.json')" in src or "open(data_path)" in src
    data = (pp.DLFS6 / "codebot/tiny_codes_sft.json").read_bytes()
    utf8, gbk = json.loads(data.decode("utf-8")), json.loads(data.decode("gbk"))   # GBK 不报错……
    changed = [(a["response"], b["response"]) for a, b in zip(utf8, gbk) if a != b]
    assert len(changed) == 2 and all("O(n²)" in a and "O(n虏)" in b for a, b in changed)  # ……而是悄悄改字
    # 用文本模式读 TinyStories 的写法只在 ch04/02_bpe_cache.py 末尾的注释里；照着它读就会报错
    cache_src = (pp.DLFS6 / "ch04/02_bpe_cache.py").read_text(encoding="utf-8")
    assert '# file_path = "storybot/tiny_stories_train.txt"' in cache_src and "# text = open(file_path).read()" in cache_src
    with pytest.raises(UnicodeDecodeError):
        (pp.DLFS6 / "storybot/tiny_stories_valid.txt").read_bytes().decode("gbk")
