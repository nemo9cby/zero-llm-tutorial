// Run the browser-tagged facts (NumPy/math/json only) inside Pyodide and record outputs -> <kit>/facts_pyodide.json
// Usage: npm i (in this folder, installs pyodide 314.0.7)  then  node run_facts.mjs <kit-root>
import { loadPyodide } from "pyodide";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const KIT = process.argv[2];
const facts = JSON.parse(readFileSync(join(KIT, "pyodide_test", "facts_browser.json"), "utf8"));
const t0 = performance.now();
const py = await loadPyodide();
await py.loadPackage("numpy");
const boot = performance.now() - t0;

py.FS.writeFile("np_gpt.py", readFileSync(join(KIT, "verify", "np_gpt.py"), "utf8"));
py.FS.writeFile("ch02_attn_maps.json", readFileSync(join(KIT, "ch02_attn_maps.json"), "utf8"));
py.FS.mkdir("pyodide_test");
py.FS.writeFile("pyodide_test/tiny_ref.npz", readFileSync(join(KIT, "pyodide_test", "tiny_ref.npz")));
py.runPython(`import sys, io, contextlib, time, traceback\nsys.path.insert(0, ".")`);

const results = [];
for (const f of facts) {
  py.globals.set("SRC", f.code.trim() + "\n");
  const [ok, out, secs] = py.runPython(`
buf = io.StringIO(); t = time.perf_counter()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(SRC, "<fact>", "exec"), {"__name__": "__fact__"})
    ok = True
except Exception:
    ok = False; buf.write(traceback.format_exc(limit=2))
[ok, buf.getvalue().rstrip(), round(time.perf_counter() - t, 2)]
`).toJs();
  results.push({ id: f.id, ok, seconds: secs, output: out });
  console.log(`[${ok ? "ok" : "FAIL"}] ${f.id} (${secs}s)`);
}
const meta = { pyodide: py.version, numpy: py.runPython("import numpy; numpy.__version__"), python: py.runPython("sys.version.split()[0]"), boot_ms: Math.round(boot) };
writeFileSync(join(KIT, "facts_pyodide.json"), JSON.stringify({ meta, results }, null, 1));
console.log(JSON.stringify(meta));
