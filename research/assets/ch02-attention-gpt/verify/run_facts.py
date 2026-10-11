# Execute every snippet in facts.py (CPython), record stdout + time -> ../facts_cpython.json,
# and export the browser subset -> ../pyodide_test/facts_browser.json for run_facts.mjs.
# Usage (from anywhere):  uv run --project <dlfs6> --with transformers python run_facts.py [ID ...]
import contextlib, io, json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
REPO = r"C:\Users\chenf\Projects\_refs\dlfs6"
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(KIT)

from facts import FACTS  # noqa: E402

wanted = set(sys.argv[1:])
results = []
for f in FACTS:
    if wanted and f["id"] not in wanted:
        continue
    buf = io.StringIO()
    t = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(f["code"].strip() + "\n", f"<{f['id']}>", "exec"), {"__name__": "__fact__"})
        ok, err = True, ""
    except Exception:
        ok, err = False, traceback.format_exc(limit=2)
    dt = time.perf_counter() - t
    os.chdir(KIT)  # a snippet may chdir
    results.append(dict(id=f["id"], ok=ok, seconds=round(dt, 2), output=buf.getvalue().rstrip(), error=err))
    print(f"[{'ok' if ok else 'FAIL'}] {f['id']} ({dt:.2f}s)", flush=True)
    if not ok:
        print(err)

out = os.path.join(KIT, "facts_cpython.json")
prev = {}
if wanted and os.path.exists(out):
    prev = {r["id"]: r for r in json.load(open(out, encoding="utf-8"))["results"]}
prev.update({r["id"]: r for r in results})
order = [f["id"] for f in FACTS if f["id"] in prev]
meta = dict(python=sys.version.split()[0], platform=sys.platform)
try:
    import torch, numpy
    meta.update(torch=torch.__version__, numpy=numpy.__version__)
except ImportError:
    pass
json.dump(dict(meta=meta, results=[prev[i] for i in order]), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
browser = [dict(id=f["id"], code=f["code"]) for f in FACTS if f["browser"]]
json.dump(browser, open(os.path.join(KIT, "pyodide_test", "facts_browser.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(prev)} facts) and {len(browser)} browser facts")
