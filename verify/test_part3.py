"""第 III 部扉页（WebBot）：本地验证。

原书第 III 部只给提示，Hugging Face 上也没有 WebBot 的权重。这里验证能在 CPU 上验证的部分：
OWT 验证集的统计、三种分词器在三种数据上的压缩率、未训练的 WebBot 在原书 5 个 prompt 上的输出，
以及“哪些主题在仓库里有代码、哪些没有”。
"""
import json

import _parts_preview as pp


def recorded(rel):
    return json.loads((pp.WEB_DATA / rel).read_text(encoding="utf-8"))


def test_owt_valid_stats(dlfs6_path):
    new = pp.owt_stats()
    rec = recorded("part3/doclen.json")
    for k in ("tokens", "summary", "lengths", "bytes_per_token_first_2M"):
        assert new[k] == rec[k], k
    assert new["tokens"] == 64_496_066 and new["summary"]["n"] == 59_060
    assert new["summary"]["over_ctx"] == 18_925                       # 32.0% 的文档比上下文 1024 长


def test_tokenizer_matrix_matches_record(dlfs6_path):
    import pickle
    import codebot.tokenizer as cbt
    import storybot.tokenizer as sbt
    rec = recorded("part3/tokenizers.json")
    new = pp.tokenizer_matrix()
    assert new["matrix"] == rec["matrix"] and new["chips"] == rec["chips"] and new["sentence"] == rec["sentence"]
    # 第 1 章的逐条套用编码与第 4 章的按 rank 编码，对 CodeBot 的合并规则给出相同的结果
    merges = pickle.load(open(pp.DLFS6 / "codebot/merge_rules.pkl", "rb"))
    sample = rec["sentence"] + "\n" + pp.owt_document(pp.OWT_EXCERPT_DOC)[:1500]
    assert cbt.BPETokenizer(merges).encode(sample) == sbt.BPETokenizer(merges).encode(sample)


def test_webbot_untrained_preview_matches_record(dlfs6_path):
    rec = recorded("part3/preview.json")
    new = pp.webbot_preview()
    assert new["outputs"] == rec["outputs"] and new["prompts"] == pp.WEBBOT_PROMPTS
    assert abs(new["init_loss"] - rec["init_loss"]) < 1e-3
    bots = {b["id"]: b for b in recorded("ch00/bots.json")["bots"]}
    assert abs(rec["init_loss"] - bots["webbot"]["init_loss"]) < 1e-3   # 与第 0 章同一个随机初始化
    src = (pp.DLFS6 / "ch09/02_generate.py").read_text(encoding="utf-8")
    assert all(f'"{p}"' in src for p in pp.WEBBOT_PROMPTS) and "temperature = 0.5" in src


def test_hint_only_topics(dlfs6_path):
    """MoE、梯度检查点、ZeRO、Muon、上下文扩展在仓库里没有代码；GQA、SDPA、compile、DDP、梯度累积有。"""
    code = {p.relative_to(pp.DLFS6).as_posix(): p.read_text(encoding="utf-8") for p in pp.DLFS6.rglob("*.py")}
    everything = "\n".join(code.values())
    for absent in ("torch.utils.checkpoint", "ZeroRedundancyOptimizer", "Muon", "expert", "MoE",
                   "rope_scaling", "YaRN"):
        assert absent not in everything, absent
    web = code["webbot/model.py"]
    assert "scaled_dot_product_attention" in web and "enable_gqa=True" in web
    train = code["ch09/01_pretrain.py"]
    for present in ("torch.compile(model)", "DistributedDataParallel", "no_sync()", "accumulation_steps", "wandb.log"):
        assert present in train, present
    # 每张卡都跑满 max_iters 步：脚本从不读取卡数，所以加卡等于加 token，而不是缩短时间
    assert "max_iters = 100000" in train and "pbar = tqdm(range(max_iters)" in train
    assert "WORLD_SIZE" not in train and "world_size" not in train
    assert not (pp.DLFS6 / "ch08").exists()                         # 第 8 章没有代码目录
