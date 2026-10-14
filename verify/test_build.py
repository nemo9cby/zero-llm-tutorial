"""Tests for tools/build.py: callout titles, listing line numbers, the nav, hidden pages and the build gates.

    uv run pytest verify/test_build.py
"""
import html
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Loaded from its path under its own name: a plain `import build` could also find the build/ output directory.
_spec = importlib.util.spec_from_file_location("zero_llm_build", ROOT / "tools" / "build.py")
build = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build          # dataclasses look their module up here
_spec.loader.exec_module(build)


def render(md_text: str) -> str:
    state = {"commit": "c0ffee", "results": {}, "fig": 0, "chapter": "9", "sections": []}
    return build.make_md(state).render(md_text)


def callout_title(out: str) -> str:
    m = re.search(r'<div class="callout-title">(.*?)</div>', out, re.S)
    assert m, out
    return m.group(1)


def text_of(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


# ---------------------------------------------------------------- callout titles

def test_callout_title_renders_code_math_and_links():
    out = render("::: warn 用 `count_pairs` 数 $x_i$，见 [2.3 节](ch02.html#s2-3)\n正文\n:::\n")
    title = callout_title(out)
    assert '<span class="callout-kind">注意</span>' in title
    assert "<code>count_pairs</code>" in title
    assert '<span class="math-inline">x_i</span>' in title
    assert '<a href="ch02.html#s2-3">2.3 节</a>' in title
    assert "<p>正文</p>" in out


@pytest.mark.parametrize("title", [
    "embed_dim、hidden_size、d_model、n_embd 是同一个东西吗？",
    "DPO 的损失里有 π_θ(y_w)，训练时是不是要让 π_θ 先生成回答？",
    "2*3*4 与 a*b*c",
    "__init__ 与 **kwargs** 与 _x_",
    "特殊 token 注入：用户的输入里也可能出现 <|endoftext|>",
    "二进制读取会看到 \\r：Windows 的 CRLF 陷阱",
    "这些数字是谁定的：num_processes=8、num_chunks=64、4096……",
    "<b>不是标签</b> & ~~也不是删除线~~",
])
def test_callout_title_has_no_accidental_markup(title):
    title_html = callout_title(render(f"::: confusion {title}\n正文\n:::\n"))
    for tag in ("<em>", "<strong>", "<s>", "<del>", "<b>", "<img"):
        assert tag not in title_html
    assert text_of(title_html) == "你可能会困惑" + title


def test_think_and_answer_callouts_keep_their_shape():
    out = render("::: think 把 `vocab_size` 设成 300 会怎样？\n问\n:::\n::: answer\n答\n:::\n")
    assert '<aside class="callout callout-think"><div class="callout-title"><span class="callout-kind">想一想</span>' \
           '<span class="callout-text">把 <code>vocab_size</code> 设成 300 会怎样？</span></div>' in out
    assert '<details class="callout callout-answer"><summary>答案（先想一想再展开）</summary>' in out
    assert out.count("</aside>") == 1 and out.count("</details>") == 1


def test_callout_without_title():
    out = render("::: note\n正文\n:::\n")
    assert '<div class="callout-title"><span class="callout-kind">注</span></div>' in out


def test_inline_code_glues_a_leading_space_to_the_next_character():
    # Inline code keeps its spaces (site.css), and a line may break after any kept space. A leading space, or a
    # space after another space, is glued to the character after it, so ` reversible` is never split into an
    # empty box at the end of one line and the word at the start of the next.
    out = render("最后学到的是 ` reversible`、` <|endoftext|>`、`print(ids)  # 注释`、`a b` 和 `    `。\n")
    assert '<code><span class="glue"> r</span>eversible</code>' in out
    assert '<code><span class="glue"> &lt;</span>|endoftext|&gt;</code>' in out
    assert '<code>print(ids)<span class="glue">  #</span> 注释</code>' in out
    assert "<code>a b</code>" in out                       # a single inner space still lets long code wrap
    assert '<code><span class="glue">    </span></code>' in out
    codes = re.findall(r"<code>(.*?)</code>", out)
    assert [text_of(c) for c in codes] == [" reversible", " <|endoftext|>", "print(ids)  # 注释", "a b", "    "]
    title = callout_title(render("::: note 看 ` t` 与 `he`\n正文\n:::\n"))
    assert '<code><span class="glue"> t</span></code>' in title and "<code>he</code>" in title


# ---------------------------------------------------------------- inline code in table cells: <wbr> break points

def table(*cells: str) -> str:
    """A one-row Markdown table with the given cells."""
    return "| " + " | ".join(f"h{i}" for i in range(len(cells))) + " |\n" + "|---" * len(cells) + "|\n" \
        + "| " + " | ".join(cells) + " |\n"


def codes(out: str) -> list[str]:
    return re.findall(r"<code>(.*?)</code>", out)


def test_table_code_may_break_after_slash_underscore_and_dot():
    out = render(table("`critic/rewards/mean`", "`max_position_embeddings`", "`actor_rollout_ref.rollout.n`",
                       "`verify/_ch03_grpo_replicate.py`"))
    assert codes(out) == ["critic/<wbr>rewards/<wbr>mean", "max_<wbr>position_<wbr>embeddings",
                          "actor_<wbr>rollout_<wbr>ref.<wbr>rollout.<wbr>n", "verify/<wbr>_ch03_<wbr>grpo_<wbr>replicate.<wbr>py"]
    # <wbr> adds no character: the text (what copying and find-in-page see) is unchanged
    assert [text_of(c) for c in codes(out)] == ["critic/rewards/mean", "max_position_embeddings",
                                                "actor_rollout_ref.rollout.n", "verify/_ch03_grpo_replicate.py"]


@pytest.mark.parametrize("code,want", [
    ("1.5", "1.5"),                          # "." before a digit: a number stays whole
    ("1e-5", "1e-5"),
    ("3.0e-4", "3.0e-4"),
    ("__init__", "__init__"),                # "_" not between two letters or digits
    ("_ch03", "_ch03"),
    ("n_", "n_"),
    (".py", ".py"),                          # a leading "." or "/" would be left alone at a line end
    ("/v1/chat", "/v1/<wbr>chat"),
    ("a/", "a/"),                            # nothing to break before
    ("a/ b", "a/ b"),                        # a space is already a break point
    ("https://x.org/a", "https://<wbr>x.<wbr>org/<wbr>a"),   # never between the two slashes of //
    ("model.eval()", "model.<wbr>eval()"),
    ("x.T", "x.<wbr>T"),
    ("π_θ", "π_<wbr>θ"),                     # letters are Unicode letters
    ("a&lt;b_c", "a&amp;lt;b_<wbr>c"),        # escaped like any inline code
    ("val-core/<数据源>/reward/mean@N", "val-core/<wbr>&lt;数据源&gt;/<wbr>reward/<wbr>mean@N"),
])
def test_table_code_break_points(code, want):
    assert codes(render(table(f"`{code}`"))) == [want]


def test_code_outside_tables_gets_no_wbr():
    src = "`critic/rewards/mean` 与 `max_position_embeddings`"
    out = render(f"{src}\n\n- {src}\n\n## 1.1 {src}\n\n::: note {src}\n{src}\n:::\n")
    assert "<wbr>" not in out
    assert codes(out).count("max_position_embeddings") == 5


def test_table_code_wbr_composes_with_glue():
    # A leading space or a run of spaces is glued to the character after it; <wbr> goes only between non-space characters.
    out = render(table("` max_seq_len`", "`a_b  /c_d`", "`x  _y_z`"))
    assert codes(out) == ['<span class="glue"> m</span>ax_<wbr>seq_<wbr>len',
                          'a_<wbr>b<span class="glue">  /</span>c_<wbr>d',
                          'x<span class="glue">  _</span>y_<wbr>z']


def test_table_code_inside_a_nowrap_span_stays_on_one_line():
    # Chrome breaks at <wbr> even under white-space: nowrap, so code inside a nowrap span gets none.
    out = render(table('<span class="nowrap">`vocab_size`</span> = 10', '<span style="white-space:nowrap">`a/b`</span>',
                       '<span class="nowrap">x <span>`in_both`</span> `still_in`</span> `out_side`'))
    assert codes(out) == ["vocab_size", "a/b", "in_both", "still_in", "out_<wbr>side"]


def test_raw_html_code_in_tables_is_untouched():
    out = render(table("<code>rewards.<wbr>mean()</code>", "<code>a_b</code>"))
    assert "<code>rewards.<wbr>mean()</code>" in out and "<code>a_b</code>" in out


def test_callout_title_does_not_disturb_page_footnotes():
    out = render("正文[^1]\n\n::: note 标题 `x`\n内容\n:::\n\n[^1]: 脚注。\n")
    assert out.count('class="footnotes"') == 1
    assert "<code>x</code>" in callout_title(out)


# ---------------------------------------------------------------- listings

def listing_lines(out: str) -> list[str]:
    code = re.search(r"<code>(.*)</code></pre></figure>$", out, re.S).group(1)
    pieces = code.split("\n")
    assert pieces[-1] == ""            # every line, the last one too, ends with a newline
    return pieces[:-1]


def check_lines(out: str, path: str, start: int, end: int) -> None:
    src = (build.DLFS6 / path).read_text(encoding="utf-8").splitlines()[start - 1:end]
    lines = listing_lines(out)
    assert len(lines) == end - start + 1
    for piece, want in zip(lines, src):
        assert piece.startswith('<span class="line">') and piece.endswith("</span>")
        inner = piece[len('<span class="line">'):-len("</span>")]
        assert inner.count("<span") == inner.count("</span>")      # each line is balanced on its own
        assert text_of(inner) == want


def test_listing_numbers_start_at_the_cited_line():
    out = build.render_listing("ch05/05_kvcache.py#L82-L127", {}, "c0ffee")
    assert 'data-start="82"' in out
    assert '<pre class="code numbered" style="counter-reset: line 81; --ln-digits: 3">' in out
    check_lines(out, "ch05/05_kvcache.py", 82, 127)


def test_listing_stays_aligned_through_multiline_strings():
    # L38-L52 is one f-string with blank lines inside it
    check_lines(build.render_listing("ch06/07_llm_judge.py#L35-L66", {}, "c"), "ch06/07_llm_judge.py", 35, 66)


@pytest.mark.parametrize("arg,start,end", [
    ("ch06/05_pretrain.py#L150-L159", 150, 159),    # starts with a blank line
    ("ch04/03_bpe_chunk.py#L79-L101", 79, 101),     # ends with a blank line
    ("codebot/tokenizer.py#L87-L91", 87, 91),
])
def test_listing_keeps_blank_edge_lines(arg, start, end):
    path = arg.split("#")[0]
    check_lines(build.render_listing(arg, {}, "c"), path, start, end)


def test_whole_file_listing_is_numbered_from_one():
    out = build.render_listing("ch06/07_llm_judge.py", {}, "c")
    assert "counter-reset: line 0;" in out
    n = len((build.DLFS6 / "ch06/07_llm_judge.py").read_text(encoding="utf-8").splitlines())
    check_lines(out, "ch06/07_llm_judge.py", 1, n)


def test_listing_range_errors_still_raise():
    with pytest.raises(build.BuildError):
        build.render_listing("ch06/07_llm_judge.py#L5-L9999", {}, "c")
    with pytest.raises(build.BuildError):
        build.render_listing("no/such_file.py", {}, "c")


def test_hl_lines_handles_any_text():
    lines = build.hl_lines('\nx = """a\n\nb"""\ny = 1\n\n', "python")
    assert [text_of(l) for l in lines] == ["", 'x = """a', "", 'b"""', "y = 1", "", ""]
    assert all(l.count("<span") == l.count("</span>") for l in lines)


# ---------------------------------------------------------------- localcheck sources

LOCALCHECK_SRC = '''import pytest


def helper():
    return 1


@pytest.mark.parametrize("n", [
    1, 2,
])
@pytest.mark.slow
def test_decorated(n):
    assert n > 0  # 结尾的注释也要显示
# a comment at column 0 after the function belongs to what follows


def test_plain():
    x = helper()
    assert x == 1


def test_multiline_last_statement():
    assert helper() in (
        1, 2,
    )  # 注释在多行语句的最后一行


async def test_async():
    assert True  # async too


def outer():
    @pytest.mark.slow
    def test_nested():
        return 1  # nested
    return test_nested
'''


@pytest.fixture
def lc_file(tmp_path, monkeypatch):
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "test_x.py").write_text(LOCALCHECK_SRC, encoding="utf-8", newline="\n")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    return "verify/test_x.py"


def test_localcheck_source_starts_at_the_first_decorator_and_keeps_the_trailing_comment(lc_file):
    file, src = build.extract_test_source(f"{lc_file}::test_decorated")
    assert file == lc_file
    assert src == ('@pytest.mark.parametrize("n", [\n    1, 2,\n])\n@pytest.mark.slow\n'
                   'def test_decorated(n):\n    assert n > 0  # 结尾的注释也要显示')


@pytest.mark.parametrize("name,want", [
    ("test_plain", "def test_plain():\n    x = helper()\n    assert x == 1"),
    ("test_multiline_last_statement",
     "def test_multiline_last_statement():\n    assert helper() in (\n        1, 2,\n    )  # 注释在多行语句的最后一行"),
    ("test_async", "async def test_async():\n    assert True  # async too"),
    # a nested function loses the indentation of its first line on every line
    ("test_nested", "@pytest.mark.slow\ndef test_nested():\n    return 1  # nested"),
])
def test_localcheck_source_is_whole_lines(lc_file, name, want):
    assert build.extract_test_source(f"{lc_file}::{name}")[1] == want


def test_localcheck_source_errors_still_raise(lc_file):
    with pytest.raises(build.BuildError, match="test test_missing not found"):
        build.extract_test_source(f"{lc_file}::test_missing")
    with pytest.raises(build.BuildError, match="not found"):
        build.extract_test_source("verify/test_nope.py::test_plain")


def test_localcheck_renders_the_trailing_comment(lc_file):
    out = build.render_localcheck(f"{lc_file}::test_decorated", {"title": "t"})
    assert "uv run pytest verify/test_x.py::test_decorated" in out
    code = re.search(r'<pre class="code"><code>(.*)</code></pre></details>', out, re.S).group(1)
    assert text_of(code).rstrip("\n").endswith("assert n > 0  # 结尾的注释也要显示")
    assert text_of(code).startswith('@pytest.mark.parametrize("n", [')


def test_verify_localcheck_and_plain_code_are_not_numbered():
    out = render('```verify id=1.1-a title="t"\nx = 1\nassert x == 1\n```\n\n```python\na = 1\nb = 2\n```\n')
    assert 'class="line"' not in out and "numbered" not in out
    lc = build.render_localcheck("verify/test_build.py::test_hl_lines_handles_any_text", {})
    assert 'class="line"' not in lc and "numbered" not in lc


# ---------------------------------------------------------------- nav, TOC, hidden pages

def chapter(slug, kind="chapter", order=0, num="", **meta):
    return build.Chapter(path=Path(f"{slug}.md"), slug=slug, body="",
                         meta={"slug": slug, "kind": kind, "order": order, "chapter": num, "title": slug, **meta})


BOOK = [
    chapter("index", "extra", 0, bare=True, hidden=True),
    chapter("ch00", num=0),
    chapter("part3", "part", 75, "III"),
    chapter("ch09", order=100, num=9),
    chapter("appendix-a", "appendix", 110, "A"),
    chapter("appendix-b", "appendix", 120, "B"),
    chapter("playground", "extra", 160, "P"),
    chapter("errata", "extra", 200, "Z"),
    chapter("kitchen", "extra", 999, "K", hidden=True),
]


def test_nav_labels_the_appendix_and_extra_groups():
    nav = build.nav_html(BOOK, BOOK[3])
    items = re.findall(r'<li class="([^"]+)">(?:<a href="([^"]+)")?', nav)
    order = [href or cls for cls, href in items]
    assert order == ["ch00.html", "part3.html", "ch09.html", "nav-group", "appendix-a.html", "appendix-b.html",
                     "nav-group", "playground.html", "errata.html"]
    assert re.findall(r'<li class="nav-group"><span class="nav-group-label">([^<]+)</span></li>', nav) == \
        ["附录", "实践与参考"]
    assert "index.html" not in nav and "kitchen.html" not in nav


def test_toc_counts_cells_without_failing_on_another_pages_duplicate():
    a = chapter("ch01", num=1)
    a.body = "## 1.1 x\n\n```verify id=1.1-a\nx = 1\n```\n\n```verify id=1.1-a\ny = 2\n```\n"
    toc = build.toc_html([a])
    assert '<span class="toc-count">2 项验证</span>' in toc
    with pytest.raises(build.BuildError, match="duplicate verify id"):
        build.collect_cells(a.body)          # the gate itself is unchanged


def test_hidden_pages_are_noindex_but_the_cover_is_not():
    template = (build.WEB / "template.html").read_text(encoding="utf-8")
    pages = {ch.slug: build.render_page(template, BOOK, ch, "<p>x</p>") for ch in BOOK}
    meta = '<meta name="robots" content="noindex">'
    assert meta in pages["kitchen"]
    assert meta not in pages["index"] and meta not in pages["ch09"]
    for page in pages.values():
        assert "{{" not in page and "}}" not in page


# ---------------------------------------------------------------- the gates

def test_fence_gate_rejects_backticks_in_info_strings():
    with pytest.raises(build.BuildError, match="backtick in the info string"):
        build.check_fences("ch01", '```verify id=1.1-a title="`x`"\nx = 1\n```\n')


@pytest.mark.parametrize("fence,message", [
    ('```verify id=1.1-a title="t"\nx = 1\n```', "1 verify cells in the source but 0 rendered"),
    ("```widget kitchen-charts\n{}\n```", "1 widget fences but 0 rendered"),
])
def test_rendered_count_gate(tmp_path, monkeypatch, fence, message):
    # A ~~~ fence that is never closed swallows the rest of the page, the cell or widget below it included.
    content = tmp_path / "content"
    content.mkdir()
    (content / "x.md").write_text(f"---\nslug: x\n---\n\n~~~text\nnever closed\n\n{fence}\n", encoding="utf-8")
    monkeypatch.setattr(build, "CONTENT", content)
    monkeypatch.setattr(build, "copy_assets", lambda out: None)
    with pytest.raises(build.BuildError, match=message):
        build.build(None, run=False, out=tmp_path / "out")


def test_control_character_gate(tmp_path, monkeypatch):
    (tmp_path / "content").mkdir()
    (tmp_path / "web" / "widgets").mkdir(parents=True)
    (tmp_path / "web" / "js").mkdir()
    (tmp_path / "content" / "x.md").write_text("ok\nformula \x07lpha\n", encoding="utf-8")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "CONTENT", tmp_path / "content")
    monkeypatch.setattr(build, "WEB", tmp_path / "web")
    with pytest.raises(build.BuildError, match="U\\+0007"):
        build.check_control_chars()


def test_link_gate(tmp_path, monkeypatch):
    content = tmp_path / "content"
    content.mkdir()
    (content / "a.md").write_text("---\nslug: a\n---\n", encoding="utf-8")
    (content / "b.md").write_text("---\nslug: b\n---\n", encoding="utf-8")
    monkeypatch.setattr(build, "CONTENT", content)
    (tmp_path / "a.html").write_text('<h2 id="s1-1">x</h2><a href="b.html#v-1">x</a><a href="#s1-1">y</a>',
                                     encoding="utf-8")
    (tmp_path / "b.html").write_text('<div id="v-2"></div><a href="c.html">z</a>', encoding="utf-8")
    with pytest.raises(build.BuildError) as e:
        build.check_links(tmp_path, ["a", "b"], full=True)
    assert "b.html#v-1 (no such id on b.html)" in str(e.value)
    assert "c.html# (no such page)" in str(e.value)
    build.check_links(tmp_path, ["a"], full=False)      # partial builds only warn
