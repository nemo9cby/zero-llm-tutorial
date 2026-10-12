"""共享分词器库 web/widgets/_bpe.js 的本地验证（B 层）。

页面上的分词器（_bpe.js + web/data/tokenizers/*.json）必须就是原书的分词器：
  1. JSON 里的 merge 表与 third_party/dlfs6/<bot>/merge_rules.pkl 逐条相同（顺序就是合并的优先级）；
  2. JS 的 pretokenize / encode / decode / trainBPE 与 Python 逐 id 一致。第 2 条由
     tests/js/lib-bpe.test.mjs 检查（参考输出由 tests/js/fixtures/lib/make_fixtures.py 用原书代码生成），
     这里用 node 运行它，方便各章用 localcheck 引用。
"""
import hashlib
import importlib
import json
import pickle
import shutil
import subprocess

import pytest
from conftest import ROOT

# bot -> (Python module the book uses for it, vocab size the book trains)
BOTS = {
    "codebot": ("codebot.tokenizer", 1_000),   # ch01/07
    "storybot": ("storybot.tokenizer", 10_000),  # ch04/04
    "webbot": ("storybot.tokenizer", 50_000),   # ch07/01 reuses storybot's code
}


@pytest.mark.parametrize("bot", list(BOTS))
def test_tokenizer_json_is_the_books_merge_table(dlfs6_path, bot):
    module, vocab_size = BOTS[bot]
    data = json.loads((ROOT / "web" / "data" / "tokenizers" / f"{bot}.json").read_text(encoding="utf-8"))
    raw = (dlfs6_path / bot / "merge_rules.pkl").read_bytes()
    rules = pickle.loads(raw)
    assert data["sha256"] == hashlib.sha256(raw).hexdigest()
    flat = data["merges"]
    assert list(zip(flat[0::2], flat[1::2])) == list(rules)            # same pairs, same order
    assert list(rules.values()) == list(range(256, 256 + len(rules)))  # merge i creates id 256 + i
    tok = importlib.import_module(module).BPETokenizer(rules)
    assert data["vocab_size"] == tok.vocab_size == vocab_size == len(rules) + 257
    assert data["special_tokens"] == {"<|endoftext|>": tok.end_token_id} == {"<|endoftext|>": vocab_size - 1}
    source = (dlfs6_path / module.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
    assert f'pattern = r"""{data["pattern"]}"""' in source                # the same pre-tokenizer regex


def test_bpe_js_matches_python():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run([node, "--test", "tests/js/lib-bpe.test.mjs"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "# fail 0" in proc.stdout or "fail 0" in proc.stdout
