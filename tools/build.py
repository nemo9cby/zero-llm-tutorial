"""Build the static tutorial site: content/*.md -> site/*.html.

Markdown dialect (on top of CommonMark + tables + $math$ + footnotes):

  ```listing ch01/01_char_tokenizer.py#L5-L20      real DLFS6 code, read from third_party/dlfs6
  ```verify id=1.1-a title="..." [files=a.txt,b.bin]  NumPy-only check; executed at build time in the
                                                    Pyodide-mirror env, output embedded, re-runnable in browser
  ```localcheck verify/test_ch01.py::test_name     torch-level check; shows the test source + command
  ```widget tokentape caption="..."                JSON config body; hydrated by web/widgets/<name>.js
  ```shell / ```python / ...                        ordinary highlighted code

  ::: note | warn | column <title> | cs336 <title> | confusion <title> | gpu <title> | think <title> | answer
  :::

Usage:  uv run python tools/build.py [--no-cells] [--only ch01]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
WEB = ROOT / "web"
SITE = ROOT / "site"
BUILD = ROOT / "build"
DLFS6 = ROOT / "third_party" / "dlfs6"
DLFS6_URL = "https://github.com/oreilly-japan/deep-learning-from-scratch-6"

# The in-browser runtime is Pyodide 314.0.7 (Python 3.14, NumPy 2.4.6, regex 2026.3.32).
# Verify cells are executed at build time in an env that mirrors it, so embedded outputs match the browser.
PYODIDE_MIRROR = ["uv", "run", "--isolated", "--no-project", "--python", "3.14",
                  "--with", "numpy==2.4.6", "--with", "regex==2026.3.32", "python", "-I"]

CALLOUTS = {
    "note": "注",
    "warn": "注意",
    "column": "专栏",
    "cs336": "CS336 对照",
    "confusion": "你可能会困惑",
    "gpu": "GPU 实验",
    "think": "想一想",
    "answer": "答案",
}


class BuildError(Exception):
    pass


@dataclass
class Chapter:
    path: Path
    meta: dict
    body: str
    slug: str = ""
    sections: list = field(default_factory=list)
    cells: list = field(default_factory=list)


# ---------------------------------------------------------------- parsing helpers

def parse_info(info: str) -> tuple[str, str, dict]:
    """'verify id=1.1-a title="x y"' -> ('verify', '', {'id': '1.1-a', 'title': 'x y'}).
    The first bare word after the kind is returned as `arg` (e.g. a file path)."""
    parts = shlex.split(info, posix=True)
    kind = parts[0] if parts else ""
    arg, attrs = "", {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            attrs[k] = v
        elif not arg:
            arg = p
    return kind, arg, attrs


def read_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, text[m.end():]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def dlfs6_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(DLFS6), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


# ---------------------------------------------------------------- code rendering

_FORMATTER = HtmlFormatter(nowrap=True)


def hl(code: str, lang: str) -> str:
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except Exception:
        lexer = TextLexer()
    return highlight(code, lexer, _FORMATTER)


def render_listing(arg: str, attrs: dict, commit: str) -> str:
    path, _, frag = arg.partition("#")
    src = DLFS6 / path
    if not src.is_file():
        raise BuildError(f"listing: {path} not found in third_party/dlfs6")
    lines = src.read_text(encoding="utf-8").splitlines()
    start, end = 1, len(lines)
    if frag:
        m = re.fullmatch(r"L(\d+)(?:-L(\d+))?", frag)
        if not m:
            raise BuildError(f"listing: bad line range #{frag} for {path}")
        start = int(m.group(1))
        end = int(m.group(2) or start)
        if not (1 <= start <= end <= len(lines)):
            raise BuildError(f"listing: range {frag} outside 1..{len(lines)} for {path}")
    code = "\n".join(lines[start - 1:end])
    anchor = f"#L{start}-L{end}" if frag else ""
    url = f"{DLFS6_URL}/blob/{commit}/{path}{anchor}"
    title = attrs.get("title", "")
    rng = f"第 {start}–{end} 行" if frag else "全文"
    return (
        f'<figure class="listing" data-start="{start}">'
        f'<figcaption><a class="listing-path" href="{esc(url)}" target="_blank" rel="noopener">{esc(path)}</a>'
        f'<span class="listing-range">{rng}</span>'
        + (f'<span class="listing-title">{esc(title)}</span>' if title else "")
        + f'</figcaption><pre class="code"><code>{hl(code, "python")}</code></pre></figure>'
    )


def render_verify(code: str, attrs: dict, result: dict | None) -> str:
    cid = attrs.get("id")
    if not cid:
        raise BuildError("verify cell without id=")
    title = attrs.get("title", "")
    files = attrs.get("files", "")
    out = result.get("stdout", "") if result else ""
    data = {"id": cid, "golden": out, "files": [f for f in files.split(",") if f]}
    return (
        f'<div class="verify" id="v-{esc(cid)}" data-cell="{esc(json.dumps(data, ensure_ascii=False))}">'
        f'<div class="verify-head"><span class="verify-id">验证 {esc(cid)}</span>'
        f'<span class="verify-title">{esc(title)}</span>'
        f'<button class="verify-run" type="button">在浏览器中运行</button></div>'
        f'<pre class="code verify-code"><code>{hl(code.rstrip(), "python")}</code></pre>'
        f'<textarea class="verify-src" hidden>{esc(code)}</textarea>'
        f'<div class="verify-out"><div class="verify-out-label">构建时的输出（Python 3.14 + NumPy 2.4.6）</div>'
        f'<pre>{esc(out) if out else "（无输出，断言全部通过）"}</pre></div>'
        f'<div class="seal" aria-live="polite" data-state="idle" title="点击运行此验证"><span>验</span></div>'
        f'</div>'
    )


def extract_test_source(spec: str) -> tuple[str, str]:
    file, _, name = spec.partition("::")
    path = ROOT / file
    if not path.is_file():
        raise BuildError(f"localcheck: {file} not found")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return file, ast.get_source_segment(text, node) or ""
    raise BuildError(f"localcheck: test {name} not found in {file}")


def render_localcheck(arg: str, attrs: dict) -> str:
    file, src = extract_test_source(arg)
    cmd = f"uv run pytest {arg}"
    note = attrs.get("title", "")
    return (
        f'<div class="localcheck"><div class="localcheck-head">'
        f'<span class="localcheck-label">本地验证（需要 PyTorch）</span>'
        + (f'<span class="localcheck-title">{esc(note)}</span>' if note else "")
        + f'</div><pre class="cmd"><code>{esc(cmd)}</code></pre>'
        f'<details><summary>查看测试代码 {esc(file)}</summary>'
        f'<pre class="code"><code>{hl(src, "python")}</code></pre></details></div>'
    )


def render_widget(name: str, attrs: dict, body: str, fig: str) -> str:
    body = body.strip() or "{}"
    try:
        json.loads(body)
    except json.JSONDecodeError as e:
        raise BuildError(f"widget {name}: config is not valid JSON ({e})") from e
    caption = attrs.get("caption", "")
    return (
        f'<figure class="widget" id="fig-{fig}" data-widget="{esc(name)}" data-config="{esc(body)}">'
        f'<div class="widget-mount"><noscript>此交互图需要 JavaScript。</noscript></div>'
        f'<figcaption><span class="fignum">图 {fig}</span>{esc(caption)}</figcaption></figure>'
    )


# ---------------------------------------------------------------- markdown setup

def make_md(state: dict) -> MarkdownIt:
    md = (MarkdownIt("commonmark", {"html": True, "typographer": False})
          .enable(["table", "strikethrough"])
          .use(front_matter_plugin)
          .use(footnote_plugin)
          .use(dollarmath_plugin, allow_space=True, allow_digits=False, double_inline=True,
               renderer=lambda content, opts: (
                   f'<div class="math-block">{esc(content)}</div>' if opts.get("display_mode")
                   else f'<span class="math-inline">{esc(content)}</span>')))

    def callout_render(kind):
        def render(self, tokens, idx, options, env):
            tok = tokens[idx]
            if tok.nesting == 1:
                rest = tok.info.strip()[len(kind):].strip()
                label = CALLOUTS[kind]
                title = f"{label}：{rest}" if rest and kind not in ("think",) else (rest or label)
                if kind == "answer":
                    return f'<details class="callout callout-answer"><summary>{esc(label)}（先想一想再展开）</summary>\n'
                return (f'<aside class="callout callout-{kind}"><div class="callout-title">'
                        f'<span class="callout-kind">{esc(label)}</span>'
                        + (f'<span class="callout-text">{esc(rest)}</span>' if rest else "")
                        + '</div>\n')
            return "</details>\n" if kind == "answer" else "</aside>\n"
        return render

    for kind in CALLOUTS:
        md.use(container_plugin, name=kind, render=callout_render(kind))

    default_fence = md.renderer.rules.get("fence")

    def fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        kind, arg, attrs = parse_info(tok.info)
        if kind == "listing":
            return render_listing(arg, attrs, state["commit"])
        if kind == "verify":
            cid = attrs.get("id", "")
            return render_verify(tok.content, attrs, state["results"].get(cid))
        if kind == "localcheck":
            return render_localcheck(arg, attrs)
        if kind == "widget":
            state["fig"] += 1
            return render_widget(arg, attrs, tok.content, f"{state['chapter']}-{state['fig']}")
        lang = kind or ""
        return f'<pre class="code" data-lang="{esc(lang)}"><code>{hl(tok.content, lang)}</code></pre>\n'

    md.add_render_rule("fence", fence)

    def heading_ids(st):
        for i, tok in enumerate(st.tokens):
            if tok.type == "heading_open" and tok.tag in ("h2", "h3"):
                text = st.tokens[i + 1].content
                m = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", text)
                hid = "s" + m.group(1).replace(".", "-") if m else "h-" + hashlib.md5(text.encode()).hexdigest()[:8]
                tok.attrSet("id", hid)
                if tok.tag == "h2":
                    state["sections"].append((hid, text))
    md.core.ruler.push("heading_ids", heading_ids)
    return md


# ---------------------------------------------------------------- verify cells

VERIFY_RE = re.compile(r"^```verify[ \t]+(.*?)\n(.*?)^```[ \t]*$", re.S | re.M)


def collect_cells(body: str) -> list[dict]:
    cells, seen = [], set()
    for m in VERIFY_RE.finditer(body):
        _, _, attrs = parse_info("verify " + m.group(1))
        cid = attrs.get("id")
        if not cid:
            raise BuildError(f"verify cell without id: {m.group(1)!r}")
        if cid in seen:
            raise BuildError(f"duplicate verify id {cid}")
        seen.add(cid)
        cells.append({"id": cid, "code": m.group(2),
                      "files": [f for f in attrs.get("files", "").split(",") if f]})
    return cells


def run_cells(slug: str, cells: list[dict], out: Path) -> dict:
    """Run a chapter's cells in the Pyodide-mirror env; cache by content hash (code + data files)."""
    if not cells:
        return {}
    BUILD.mkdir(exist_ok=True)
    h = hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode() + (WEB / "cell_harness.py").read_bytes())
    for f in sorted({f for c in cells for f in c["files"]}):
        src = WEB / f
        if not src.is_file():
            raise BuildError(f"{slug}: verify cell declares files={f}, but web/{f} does not exist")
        h.update(f.encode() + src.read_bytes())
    key = h.hexdigest()[:16]
    cache = BUILD / f"cells-{slug}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached["results"]
    inp = BUILD / f"cells-{slug}.in.json"
    inp.write_text(json.dumps({"cells": cells, "data_dir": str(out / "data")}, ensure_ascii=False),
                   encoding="utf-8")
    proc = subprocess.run(PYODIDE_MIRROR + [str(ROOT / "tools" / "run_cells.py"), str(inp)],
                          capture_output=True, text=True, encoding="utf-8",
                          env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        raise BuildError(f"cell runner crashed for {slug}:\n{proc.stderr[-3000:]}")
    results = {r["id"]: r for r in json.loads(proc.stdout)}
    failed = [r for r in results.values() if r.get("error")]
    if failed:
        msg = "\n\n".join(f"[{r['id']}]\n{r['error']}" for r in failed)
        raise BuildError(f"{len(failed)} verify cell(s) failed in {slug}:\n{msg}")
    slow = [f"{r['id']} ({r['seconds']:.1f}s)" for r in results.values() if r["seconds"] > 3]
    if slow:
        print(f"  warning: slow cells in {slug}: {', '.join(slow)}", file=sys.stderr)
    cache.write_text(json.dumps({"key": key, "results": results}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return results


# ---------------------------------------------------------------- page assembly

def load_chapters(only: str | None) -> list[Chapter]:
    chapters = []
    for path in CONTENT.glob("*.md"):
        meta, body = read_front_matter(path.read_text(encoding="utf-8"))
        ch = Chapter(path=path, meta=meta, body=body, slug=meta.get("slug", path.stem))
        chapters.append(ch)
    chapters.sort(key=lambda c: (c.meta.get("order", 999), c.path.name))
    slugs = [c.slug for c in chapters]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        raise BuildError(f"duplicate slugs: {sorted(dupes)}")
    if only:
        missing = [s for s in only.split(",") if s not in {c.slug for c in chapters}]
        if missing:
            raise BuildError(f"--only: unknown slugs {missing}")
    return chapters


def nav_html(chapters: list[Chapter], current: Chapter) -> str:
    items = []
    for ch in chapters:
        if ch.meta.get("hidden"):
            continue
        kind = ch.meta.get("kind", "chapter")
        num = ch.meta.get("chapter", "")
        label = f'<span class="nav-num">{esc(str(num))}</span>' if num != "" and kind != "part" else ""
        classes = [f"kind-{kind}"] + (["current"] if ch is current else [])
        sub = ""
        if ch is current and ch.sections:
            sub = "<ol>" + "".join(f'<li><a href="#{hid}">{esc(t)}</a></li>' for hid, t in ch.sections) + "</ol>"
        items.append(f'<li class="{" ".join(classes)}"><a href="{ch.slug}.html">{label}'
                     f'{esc(ch.meta.get("nav_title", ch.meta.get("title", ch.slug)))}</a>{sub}</li>')
    return "<ol class=\"nav-chapters\">" + "".join(items) + "</ol>"


def neighbors(chapters: list[Chapter], ch: Chapter) -> str:
    visible = [c for c in chapters if not c.meta.get("hidden")]
    if ch not in visible:
        return ""
    i = visible.index(ch)
    prev = visible[i - 1] if i > 0 else None
    nxt = visible[i + 1] if i + 1 < len(visible) else None
    out = ['<nav class="pager">']
    if prev:
        out.append(f'<a class="prev" href="{prev.slug}.html"><span>上一章</span>{esc(prev.meta.get("title", ""))}</a>')
    if nxt:
        out.append(f'<a class="next" href="{nxt.slug}.html"><span>下一章</span>{esc(nxt.meta.get("title", ""))}</a>')
    out.append("</nav>")
    return "".join(out)


H2_RE = re.compile(r"^## +(\d+(?:\.\d+)+) +(.+?)\s*$", re.M)


def toc_html(chapters: list[Chapter]) -> str:
    """Book-style table of contents for the index page, parsed straight from the sources."""
    out = ['<ol class="book-toc">']
    for ch in chapters:
        if ch.meta.get("hidden"):
            continue
        n_cells = len(collect_cells(ch.body))
        secs = "".join(f'<li><a href="{ch.slug}.html#s{num.replace(".", "-")}"><span class="toc-num">{esc(num)}</span>'
                       f'{esc(title)}</a></li>' for num, title in H2_RE.findall(ch.body))
        badge = f'<span class="toc-count">{n_cells} 项验证</span>' if n_cells else ""
        out.append(f'<li class="toc-chapter"><a class="toc-title" href="{ch.slug}.html">'
                   f'<span class="toc-chnum">{esc(str(ch.meta.get("chapter", "")))}</span>'
                   f'{esc(ch.meta.get("title", ""))}</a>{badge}<ol>{secs}</ol></li>')
    out.append("</ol>")
    return "".join(out)


def build(only: str | None, run: bool, out: Path) -> None:
    chapters = load_chapters(only)
    commit = dlfs6_commit()
    template = (WEB / "template.html").read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    copy_assets(out)

    rendered = {}
    for ch in chapters:
        if only and ch.slug not in only.split(","):
            continue
        ch.cells = collect_cells(ch.body)
        results = run_cells(ch.slug, ch.cells, out) if run else {}
        state = {"commit": commit, "results": results, "fig": 0,
                 "chapter": ch.meta.get("chapter", ch.slug), "sections": ch.sections}
        html_body = make_md(state).render(ch.body)
        rendered[ch.slug] = html_body.replace("<!-- toc -->", toc_html(chapters))
        print(f"  {ch.slug}: {len(ch.cells)} verify cells, {state['fig']} figures")

    for ch in chapters:
        if ch.slug not in rendered:
            continue
        num = ch.meta.get("chapter", "")
        head = (f'<header class="chapter-head"><div class="chapter-num">{esc(str(num))}</div>'
                f'<h1>{esc(ch.meta.get("title", ""))}</h1>'
                + (f'<p class="chapter-lede">{esc(ch.meta["lede"])}</p>' if ch.meta.get("lede") else "")
                + "</header>") if not ch.meta.get("bare") else ""
        page = (template
                .replace("{{title}}", esc(ch.meta.get("title", "")))
                .replace("{{description}}", esc(ch.meta.get("lede", "")))
                .replace("{{nav}}", nav_html(chapters, ch))
                .replace("{{packages}}", esc(json.dumps(ch.meta.get("pyodide_packages", ["numpy"]))))
                .replace("{{body}}", head + rendered[ch.slug] + neighbors(chapters, ch))
                .replace("{{slug}}", ch.slug))
        (out / f"{ch.slug}.html").write_text(page, encoding="utf-8")


def copy_assets(out: Path) -> None:
    """Copy web/{css,js,widgets,data} into the output dir (before cells run: they read out/data)."""
    for sub in ("css", "js", "widgets", "data"):
        src = WEB / sub
        if src.exists():
            dst = out / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    shutil.copy2(WEB / "cell_harness.py", out / "js" / "cell_harness.py")
    (out / "css" / "pygments.css").write_text(pygments_css(), encoding="utf-8")


def pygments_css() -> str:
    # Token colors come from CSS variables in site.css so light/dark themes stay in sync.
    m = {"k": "kw", "kn": "kw", "kd": "kw", "kc": "kw", "ow": "kw", "nb": "builtin", "bp": "builtin",
         "s": "str", "s1": "str", "s2": "str", "sa": "str", "se": "str", "si": "str", "sd": "str",
         "c": "comment", "c1": "comment", "cm": "comment", "ch": "comment",
         "m": "num", "mi": "num", "mf": "num", "mh": "num", "nf": "fn", "nc": "fn", "fm": "fn", "nd": "fn"}
    return "\n".join(f".code .{k} {{ color: var(--syn-{v}); }}" for k, v in m.items()) + \
        "\n.code .c, .code .c1, .code .cm { font-style: italic; }\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated chapter slugs to rebuild")
    ap.add_argument("--no-cells", action="store_true", help="skip executing verify cells")
    ap.add_argument("--out", type=Path, default=SITE,
                    help="output directory (default site/; use build/preview-<slug> for private previews)")
    args = ap.parse_args()
    out = args.out.resolve()
    try:
        build(args.only, run=not args.no_cells, out=out)
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"site written to {out}")


if __name__ == "__main__":
    main()
