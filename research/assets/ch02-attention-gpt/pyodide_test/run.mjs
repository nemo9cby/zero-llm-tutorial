// Run the NumPy-only verification files inside Pyodide (WASM CPython), the same runtime a browser page would use.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const VERIFY = process.argv[2];
const t0 = performance.now();
const py = await loadPyodide();
const t1 = performance.now();
await py.loadPackage("numpy");
const t2 = performance.now();
console.log(`pyodide ${py.version} boot ${(t1 - t0).toFixed(0)} ms, numpy load ${(t2 - t1).toFixed(0)} ms`);

for (const f of ["np_gpt.py", "v_browser_suite.py", "v01_softdict_np.py", "v03_scaling_np.py"]) {
  py.FS.writeFile(f, readFileSync(join(VERIFY, f), "utf8"));
}
py.FS.writeFile("tiny_ref.npz", readFileSync("tiny_ref.npz"));

for (const f of ["v_browser_suite.py", "v01_softdict_np.py", "v03_scaling_np.py"]) {
  console.log(`\n######## ${f} (Pyodide) ########`);
  const t = performance.now();
  await py.runPythonAsync(`import sys; sys.path.insert(0, "."); exec(open("${f}").read(), {"__name__": "__main__"})`);
  console.log(`[${f}] ${(performance.now() - t).toFixed(0)} ms`);
}
