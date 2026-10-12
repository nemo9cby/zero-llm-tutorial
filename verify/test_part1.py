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
        assert rec["models"][name]["sha256"] == pp.HF_SHA256[f"codebot/model_{name}.pt"]   # 官方检查点
    base = dict(zip(pp.CHAT_PROMPTS, rec["models"]["pretrain"]["greedy"]))
    assert [base[p] for p in ("Hello", "Who are you?", "write loop", "7+8=")] == [""] * 4   # 直接输出结束记号
    assert base["make variable x"].startswith("def main():")                            # 续写出不相干的代码
    assert rec["models"]["pretrain"]["def_continuation"].startswith("def sort_list(lst):")
    assert rec["models"]["sft"]["addition_correct"] == 44            # 44/81 = 54.3%
    assert rec["models"]["grpo"]["addition_correct"] == 67           # 67/81 = 82.7%
    assert rec["models"]["grpo"]["addition_greedy"]["7+8"] == "16"


def test_codebot_ships_data_and_tokenizer_in_git(dlfs6_path):
    tracked = subprocess.check_output(["git", "-C", str(pp.DLFS6), "ls-files", "codebot"], text=True).split()
    for f in ("codebot/tiny_codes.txt", "codebot/tiny_codes.bin", "codebot/merge_rules.pkl",
              "codebot/tiny_codes_sft.json"):
        assert f in tracked, f
    assert not any(f.endswith(".pt") for f in tracked)            # 权重在 Hugging Face 上，不在仓库里
