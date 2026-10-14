"""第 I 部扉页（CodeBot）：本地验证。

TinyCodes 的统计在完整语料上重算；终点预览用官方 SFT / GRPO 检查点在 CPU 上贪心解码，
与 web/data/part1/preview.json 逐字比较。不运行原书脚本，只导入它们的模块。
"""
import json
import re
import subprocess

import _parts_preview as pp


def recorded(rel):
    return json.loads((pp.WEB_DATA / rel).read_text(encoding="utf-8"))


def test_tinycodes_full_corpus_stats(dlfs6_path):
    stats, head = pp.tinycodes_stats()
    rec = recorded("part1/doclen.json")
    for k in ("text_bytes", "line_feeds", "carriage_returns", "non_ascii_bytes", "programs", "separators",
              "tokens", "summary", "lengths"):
        assert stats[k] == rec[k], k
    assert (stats["programs"], stats["separators"], stats["tokens"]) == (21_836, 21_835, 2_677_398)
    assert stats["non_ascii_bytes"] == 0 and stats["summary"]["over_ctx"] == 2_113
    assert (pp.WEB_DATA / "part1/tiny_codes_head.txt").read_text(encoding="utf-8") == head
    text = (pp.DLFS6 / "codebot/tiny_codes.txt").read_text(encoding="utf-8")
    assert text.startswith(head)                                  # 附带的样本是语料的开头，一字未改
    programs = text.split(pp.EOT)
    assert pp.ast_failures(programs) == 2_426                     # 本地 Python 的解析器：11.1% 不是合法 Python
    assert pp.ast_failures(programs[:500]) == 85                  # 与浏览器里（Python 3.14）数的样本结果相同


def test_sft_data(dlfs6_path):
    shipped = (pp.WEB_DATA / "part1/tiny_codes_sft.json").read_bytes()
    assert shipped == (pp.DLFS6 / "codebot/tiny_codes_sft.json").read_bytes()
    data = json.loads(shipped.decode("utf-8"))
    assert len(data) == 1092 and all(set(x) == {"instruction", "response"} for x in data)
    assert [x["instruction"] for x in data[-81:]] == [f"{i}+{j}=" for i in range(1, 10) for j in range(1, 10)]
    assert sum(bool(re.fullmatch(r"\d\+\d=", x["instruction"])) for x in data) == 81
    assert sum("\n" in x["response"] for x in data) == 187


def test_codebot_preview_matches_record(dlfs6_path):
    rec = recorded("part1/preview.json")
    new = pp.codebot_preview()
    assert new["prompts"] == rec["prompts"] and new["template"] == rec["template"]
    for name in ("pretrain", "sft", "grpo"):
        assert new["models"][name] == rec["models"][name], name
        assert rec["models"][name]["sha256"] == pp.HF_SHA256[f"codebot/model_{name}.pt"]   # 官方 checkpoint
    base = dict(zip(pp.CHAT_PROMPTS, rec["models"]["pretrain"]["greedy"]))
    assert [base[p] for p in ("Hello", "Who are you?", "write loop", "7+8=")] == [""] * 4   # 直接输出结束记号
    assert base["make variable x"].startswith("def main():")                            # 续写出不相干的代码
    assert rec["models"]["pretrain"]["def_continuation"].startswith("def sort_list(lst):")
    assert rec["models"]["sft"]["addition_correct"] == 44            # 44/81 = 54.3%
    assert rec["models"]["grpo"]["addition_correct"] == 67           # 67/81 = 82.7%
    assert rec["models"]["grpo"]["addition_greedy"]["7+8"] == "16"


def test_pretrain_windows_cross_programs(dlfs6_path):
    """ch03/01_pretrain.py 的 TokenDataset：整份语料首尾相连，每个位置都可以是一条 256-token 训练数据的开头。"""
    import ast
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    src = (pp.DLFS6 / "ch03/01_pretrain.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "TokenDataset")
    ns = {"torch": torch, "Dataset": Dataset}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "ch03/01_pretrain.py:TokenDataset", "exec"), ns)
    assert "dataset = TokenDataset(ids, context_len)" in src and "DataLoader(dataset, batch_size=batch_size, shuffle=True)" in src
    ids = np.fromfile(pp.DLFS6 / "codebot/tiny_codes.bin", dtype=np.uint16)
    ds = ns["TokenDataset"](ids, 256)
    assert len(ds) == len(ids) - 256                              # 每个起点都是一条数据
    x, y = ds[1]
    assert torch.equal(x[1:], y[:-1])                             # y 是 x 往后错一位
    starts = np.random.default_rng(0).integers(1, len(ds), size=4000)
    seps = np.array([int((ds[int(i)][0] == 999).sum()) for i in starts])
    assert 2.0 < seps.mean() < 2.2 and (seps >= 2).mean() > 0.5  # 平均两个多分隔记号，多数窗口跨过 2 个以上
    at_boundary = np.isin(ids[starts - 1], [999]) | (ids[starts] == 999)
    assert at_boundary.mean() < 0.03                              # 开头几乎总在某个程序的中间


def test_sft_and_grpo_batches_follow_dataloader_cycle(dlfs6_path):
    """ch03/03_sft.py 与 09_grpo.py 都用 cycle(DataLoader(dataset, batch_size=32, shuffle=True))：
    DataLoader 不丢弃最后不满的 batch，cycle 从第二个 epoch 起按第一个 epoch 的顺序原样重放。"""
    from collections import Counter
    from itertools import cycle
    import torch
    from torch.utils.data import DataLoader
    for rel in ("ch03/03_sft.py", "ch03/09_grpo.py"):
        src = (pp.DLFS6 / rel).read_text(encoding="utf-8")
        assert "DataLoader(dataset, batch_size=batch_size, shuffle=True)" in src and "cycle(dataloader)" in src, rel
        assert "drop_last" not in src, rel
        h = pp.script_hparams(rel)
        assert (h["batch_size"], h["max_iters"]) == (32, 500), rel
    n_sft = len(json.loads((pp.DLFS6 / "codebot/tiny_codes_sft.json").read_text(encoding="utf-8")))

    def batches(n):
        torch.manual_seed(0)
        it = cycle(DataLoader(list(range(n)), batch_size=32, shuffle=True))
        return [next(it) for _ in range(500)]

    sft = batches(n_sft)                                         # 1,092 = 34 × 32 + 4：每个 epoch 35 个 batch
    assert Counter(len(b) for b in sft) == {32: 486, 4: 14}
    assert sum(len(b) for b in sft) == 15_608 and round(15_608 / n_sft, 1) == 14.3
    assert torch.equal(sft[0], sft[35])                          # 第二个 epoch 原样重放第一个
    grpo = batches(81)                                           # 81 = 32 + 32 + 17：每三轮过一遍
    assert [len(b) for b in grpo[:6]] == [32, 32, 17, 32, 32, 17]
    assert Counter(len(b) for b in grpo) == {32: 334, 17: 166} and sum(len(b) for b in grpo) == 13_510
    assert torch.equal(grpo[0], grpo[3])


def test_codebot_ships_data_and_tokenizer_in_git(dlfs6_path):
    tracked = subprocess.check_output(["git", "-C", str(pp.DLFS6), "ls-files", "codebot"], text=True).split()
    for f in ("codebot/tiny_codes.txt", "codebot/tiny_codes.bin", "codebot/merge_rules.pkl",
              "codebot/tiny_codes_sft.json"):
        assert f in tracked, f
    assert not any(f.endswith(".pt") for f in tracked)            # 权重在 Hugging Face 上，不在仓库里
