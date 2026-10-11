// Time the trained codebot GPT (fp16 weights -> float32 math) in Pyodide NumPy: is an in-browser demo feasible?
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const VERIFY = process.argv[2];
const py = await loadPyodide();
await py.loadPackage("numpy");
py.FS.writeFile("np_gpt.py", readFileSync(join(VERIFY, "np_gpt.py"), "utf8"));
py.FS.writeFile("codebot_fp16.npz", readFileSync("codebot_fp16.npz"));
await py.runPythonAsync(`
import sys, time; sys.path.insert(0, ".")
import numpy as np
from np_gpt import gpt_forward
Z = np.load("codebot_fp16.npz")
P = {k: Z[k].astype(np.float32) for k in Z.files if k not in ("ids", "ref_logits")}
P["embed.weight"] = P["unembed.weight"]
ids = [int(i) for i in Z["ids"]]
t = time.perf_counter(); out = gpt_forward(ids, P, 6, 6); dt = time.perf_counter() - t
print(f"trained codebot in Pyodide: C={len(ids)} fwd {dt*1000:.0f} ms, max|diff vs torch fp32| = {np.abs(out - Z['ref_logits']).max():.2e}, argmax agree: {(out.argmax(-1) == Z['ref_logits'].argmax(-1)).mean()*100:.0f}%")
for C in (64, 256):
    x = list(np.random.default_rng(0).integers(0, 1000, C))
    t = time.perf_counter(); gpt_forward(x, P, 6, 6); print(f"C={C}: {1000*(time.perf_counter()-t):.0f} ms")
`);
