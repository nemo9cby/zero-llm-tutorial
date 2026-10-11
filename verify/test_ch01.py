"""第 1 章 分词器：本地验证。"""
from conftest import run_dlfs6_script


def test_char_and_byte_scripts_match_book_comments():
    # 原书脚本里，每个 print 后面的注释写着预期输出；这里逐行核对。
    assert run_dlfs6_script("ch01/01_char_tokenizer.py") == [
        "['h', 'e', 'l', 'l', 'o', '世', '界', '😁']",
        "104",
        "128513",
        "h",
        "😁",
        "[104, 101, 108, 108, 111, 19990, 30028, 128513]",
        "[104, 101, 108, 108, 111, 19990, 30028, 128513]",
        "hello世界😁",
    ]
    assert run_dlfs6_script("ch01/02_byte_tokenizer.py") == [
        "b'A'",
        "[65]",
        "b'\\xe3\\x81\\x82'",
        "[227, 129, 130]",
        "A",
        "[104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]",
        "hello世界😁",
    ]
