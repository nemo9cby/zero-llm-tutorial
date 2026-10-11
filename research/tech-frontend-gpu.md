# Frontend + GPU execution feasibility (tech dossier)

Date 2026-10-11. Machine: i7-11700F (8C/16T), RTX 3060 Ti 8 GB (driver 610.60, CUDA UMD 13.3), Windows 11,
Chrome 154 (Playwright), Node 24.13.0 / V8 13.6, uv 0.9.13.

Evidence tags: **[M]** measured here (script path given) · **[S]** read in source code or vendor docs (URL) · **[E]** estimate, assumption stated.

> **Caveat on CPU numbers.** While I measured, other workflow agents held the CPU at 92–100 % (16 python processes).
> Every JS/Pyodide CPU timing below is therefore a pessimistic lower bound. Expect roughly 1.5–2× on an idle machine.
> GPU timings are not affected.

All prototypes are in `research/tech-kit/` (ready to reuse) and in the scratch dir listed in the appendix.

---

## 中文摘要

- **浏览器里的 Python**：Pyodide 314.0.7（2026-09-14 发布；Python 3.14.2；numpy 2.4.6、matplotlib 3.10.8、scipy 1.18.0；**没有 torch**）。
  从 `cdn.jsdelivr.net/npm/pyodide@314.0.7/` 加载时，必须显式传 `packageBaseUrl: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/"`，
  否则 `loadPackage("numpy")` 会 404。这一点已实测，源码里也能看到原因。core+numpy 走网络约 9.2 MB（brotli 压缩后），热缓存下加载约 4–5 s + 1–2 s。
- **交互单元格**：每页一个 Web Worker 内核，所有 cell 共享全局变量。stdout 通过 `setStdout({batched})` 回传。
  中断需要 SharedArrayBuffer，也就是 COOP/COEP 跨源隔离：实测有隔离时会抛 KeyboardInterrupt，变量都还在；没有隔离就只能结束 Worker 再重启，状态全部丢失。matplotlib 用 Agg 出 PNG，第一张图约 0.7 s。
- **claude.ai artifact 托管**：CDN 只允许加载脚本，fetch 只能访问同源。要用 Pyodide，就得把 wasm、stdlib 和 wheel 当作 artifact 附件自己托管；
  单个二进制文件上限 15 MB，pyodide.asm.wasm 是 9.6 MB，放得下。另外 CSP 必须允许 WebAssembly，公开报道说有 `'unsafe-eval'`。
  我用 `tech-kit/artifact-runtime-probe.html` 在本地模拟的 CSP 下测过；真实环境发布后打开一次就能确认。
- **浏览器推理**：手写 JS 前向（不依赖任何框架）的结果和 PyTorch 对得上：f32 下 logit 最大误差 1.1e-5，贪心解码 64 个 token 完全一致。
  纯 JS 约 20–50 tok/s；WASM SIMD 内核只有 359 字节，约 85–145 tok/s。int8 每行量化后权重 11.2 MB，贪心结果仍然一致。
  WebGPU 估计能到几百 tok/s，但实现复杂，放在可选项。
- **浏览器训练**：microgpt.js（4,192 个参数）在 Chrome 里跑 1000 步用了 14 s。参数量在 1 万到 10 万之间的字符级 transformer，用 Float32Array 或 WASM 在 1 分钟内把 loss 明显降下来是可行的。
- **GPU**：Modal 客户端 1.6.1。价格：H100 $3.95/h、L4 $0.80/h、B200 $6.25/h，每月有 $30 免费额度。
  GPU 函数默认可被抢占，而且关不掉，所以必须 checkpoint 加 Retries。续训模板已经在本机 GPU 上用 `.local()` 验证过。
  ch09 用了 torch.compile，镜像里要 `apt_install("build-essential")`。
- **成本**：
  - CodeBot 预训练：不到 $1。
  - StoryBot：约 $1–2。
  - WebBot 原样单卡 H100：约 9 h，约 $36。
  - WebBot 原样 8×H100：约 $290，处理的 token 数是单卡的 8 倍。
  - 推荐“预算版”：3B token，约 2 h，约 $8。
- **本地 3060 Ti**：pyproject 里配 `[[tool.uv.index]] explicit=true` 和 `[tool.uv.sources]` 的 marker，Windows 上就会装 torch 2.14.1+cu130（支持 sm_86，bf16 可用）。实测：
  - CodeBot：37 min。
  - StoryBot：训练 69 min，再加 eval 时间。
  - WebBot：不可行，约 12 天。
  - GRPO：约 1.7 h，瓶颈在 CPU。

---

## 0. Decisions at a glance

| Need | Recommendation | Why (evidence) |
|---|---|---|
| Instant interactive widgets such as softmax, masks, RoPE, BPE steps, bf16 rounding | **Plain JS on Float32Array**, no Python | Zero download, works under every CSP [M §1] |
| "Run this Python" cells that mirror the book code | **Pyodide 314.0.7 in a module Web Worker**, lazy-loaded on first click, numpy only by default | Verified kernel with shared state, stdout and interrupt [M §A.5]. PyTorch is unavailable in-browser, so cells use numpy translations of the torch code |
| End-to-end demo: "talk to CodeBot" | **Custom JS forward pass + WASM SIMD matvec**, int8 or f16 weights as same-origin files | Bit-for-bit greedy parity with PyTorch, 85–145 tok/s, no framework [M §B] |
| GPU training | **Modal** for one-command, pay-per-second runs (StoryBot ≈ $1–2, WebBot budget ≈ $8). **RunPod community** for the cheapest long WebBot runs | Prices [S §C.2, §D.1]; cost model [§D.5] |
| Local runs | uv project with Windows → cu130 index; CodeBot, StoryBot, SFT, DPO and GRPO run locally, WebBot does not | [M §E] |
| Hosting | **Plain static host** (GitHub Pages or Cloudflare Pages) unless the tutorial must live in a claude.ai artifact. Artifacts work but need self-hosted Pyodide and weights (≤ 15 MB per binary) | [S/M §1] |

---

## 1. Hosting targets and what their CSP allows

### 1.1 claude.ai artifact host [S]

- **Scripts.** The Artifact page contract allows external scripts only from `cdnjs.cloudflare.com`, `cdn.jsdelivr.net/npm/`, `unpkg.com`, `cdn.tailwindcss.com` and `code.jquery.com`.
  On those CDNs, "anything but a script — stylesheets, images, media, fetch/XHR/WebSocket, a library's runtime fetches" is blocked.
- **Fetch and workers.** "`fetch()` of files published alongside the page works with relative URLs." "Web Workers work from your own files or `blob:` URLs; service workers and WebRTC do not."
- **File limits.** Each binary file ≤ 15 MB, each text file ≤ 16 MB. One publish carries ≤ 64 MB / 255 files; a version holds ≤ 256 MB / 511 files.
- **Frame policy reported by an outside audit** (pluto.security, 2026-08-06): `script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; connect-src 'self'`, with each artifact served on its own `<uuid>.frame.claudeusercontent.com` origin.
  `'unsafe-eval'` also permits WebAssembly compilation. COOP/COEP headers are not reported, so assume **no SharedArrayBuffer**.
  Source: https://pluto.security/blog/inside-claude-artifacts-how-your-agent-publishes-to-the-web/

### 1.2 Emulation results [M]

My server set `default-src 'none'; script-src 'self' 'unsafe-inline' blob: <the 5 CDNs> [+X]; worker-src 'self' blob:; connect-src 'self' blob: data:`.
Files: `scratch/tech/web/server.mjs`, `csp.html`, `worker.html`.

| Probe | no eval keyword | `+'wasm-unsafe-eval'` | `+'unsafe-eval'` (reported artifact policy) |
|---|---|---|---|
| `WebAssembly.compile` | ❌ blocked | ✅ | ✅ |
| `eval` / `new Function` | ❌ | ❌ | ✅ |
| Worker from `blob:` (classic and module) | ✅ | ✅ | ✅ |
| `fetch` same-origin | ✅ | ✅ | ✅ |
| `fetch` cdn.jsdelivr.net | ❌ connect-src | ❌ | ❌ |
| `import()` from cdn.jsdelivr.net/npm | ✅ | ✅ | ✅ |
| Pyodide with its files fetched from the CDN | ❌ (inferred: same connect-src block) | ❌ connect-src violations on `pyodide-lock.json`, `python_stdlib.zip`, `pyodide.asm.wasm`; hangs, never rejects | ❌ (inferred: same connect-src block) |
| Pyodide **self-hosted** next to the page | ❌ hangs | ✅ core 4.7 s + numpy 0.9 s; worker 6.5 s + 1.2 s | ✅ core 4.1 s + numpy 1.8 s |

Pyodide never rejects its promise when WASM is blocked. **Guard every boot with a timeout and pre-check `WebAssembly.compile`** on an 8-byte module.

### 1.3 Probe page for the real host

`research/tech-kit/artifact-runtime-probe.html` follows the artifact page contract. It checks:
- wasm compile and eval;
- WASM SIMD speed, plain JS vs SIMD GFLOP/s;
- classic and module blob workers;
- same-origin fetch, CDN fetch and CDN `import()`;
- cross-origin isolation;
- the WebGPU adapter;
- `Float16Array`;
- IndexedDB;
- optionally a self-hosted Pyodide boot, when published together with `pyodide/*` files.

Locally, under the `'unsafe-eval'` emulation, every check returned the expected result, including "Python 3.14.2, numpy 2.4.6; core 4102 ms, numpy 1822 ms" [M].
**Publish it once and open it** to replace the reported policy with ground truth. A "Copy results as JSON" button is built in.

### 1.4 Plain static host (GitHub Pages, Cloudflare Pages, `python -m http.server`)

- Everything works.
- Pyodide can load straight from jsDelivr.
- Weights can come from Hugging Face (CORS `*`).
- A service worker can make the site work offline.
- SharedArrayBuffer needs COOP+COEP headers. GitHub Pages cannot set headers, so use the `coi-serviceworker` shim: https://github.com/gzuidhof/coi-serviceworker.
  Under `COEP: require-corp`, jsDelivr assets still load: dynamic `import()` and `fetch` are CORS requests and jsDelivr sends `access-control-allow-origin: *` [M].
  A plain `<script src>` needs the `crossorigin` attribute.

---

## A. In-browser Python: Pyodide

### A.1 Version facts

| Item | Value | Evidence |
|---|---|---|
| Latest stable | **314.0.7** (2026-09-14). Also `stable-0.29` = 0.29.5; `next` = 315.0.0-alpha.2 | `npm view pyodide dist-tags` [M] |
| Versioning | The major version now tracks CPython: 314.x = Python 3.14 | release list [M] |
| Runtime | Python 3.14.2, Emscripten 5.0.3, ABI `2026_0`, wheel tag `pyemscripten_2026_0_wasm32`, 357 packages | `pyodide-lock.json` `info` [M] |
| Useful packages | numpy 2.4.6, matplotlib 3.10.8, scipy 1.18.0, pandas 3.0.2, regex 2026.3.32, tiktoken 0.12.0 (pulls requests), safetensors 0.7.0, tqdm 4.67.3, scikit-learn 1.8.0, sympy 1.14.0 | lock file [M] |
| Not available | **torch**, tokenizers, onnxruntime, jax | lock file [M] |

### A.2 The two jsDelivr paths: why `loadPackage('numpy')` fails from `/npm/`

- **`/npm/pyodide@314.0.7/`** holds exactly 14 files: `pyodide.js`, `.mjs`, `.asm.mjs`, `.asm.wasm`, `python_stdlib.zip`, `pyodide-lock.json`, `d.ts` files, `console*.html`. **No wheels.** [M jsDelivr API]
- **`/pyodide/v314.0.7/full/`** is the full distribution with every wheel. It is served with `access-control-allow-origin: *` and `cache-control: public, max-age=31536000, immutable` [M].
- **Loader source** (`pyodide.mjs`, minified): `e.cdnUrl = k(e.packageBaseUrl ?? "https://cdn.jsdelivr.net/pyodide/v${U}/full/") … e.packageBaseUrl ??= V(lockFileURL)`.
  The runtime (`pyodide.asm.mjs`) uses `cdnURL` only when `IN_NODE`. In browsers, `installBaseUrl = config.packageBaseUrl`, which defaults to the lock file's directory, i.e. the npm path. [S]
- **Measured result** [M `cdn.html`]: with no option, the console shows `404 … cdn.jsdelivr.net/npm/pyodide@314.0.7/numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl`, and `import numpy` raises `ModuleNotFoundError`.
  With `packageBaseUrl` set, numpy 2.4.6 loads.
- Wheels are fetched with **SRI** (`integrity` sha256 from the lock). Self-hosted copies must be byte-identical; a mismatch is "blocked" [M console].

```js
// main thread or worker; works from either jsDelivr path
const V = "314.0.7";
const { loadPyodide } = await import(`https://cdn.jsdelivr.net/npm/pyodide@${V}/pyodide.mjs`);
const py = await loadPyodide({ packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${V}/full/` });
await py.loadPackage("numpy");
// Self-hosted (required on the artifact host): indexURL must be ABSOLUTE.
// const base = new URL("pyodide/", location.href).href;
// const py = await loadPyodide({ indexURL: base });   // packageBaseUrl defaults to the same dir
```

### A.3 Payload sizes [M]

| File | Raw | Over the wire (brotli) |
|---|---|---|
| pyodide.asm.wasm | 9,598,218 B | 3,438,516 B |
| pyodide.asm.mjs | 1,250,344 | 262,041 |
| python_stdlib.zip | 2,545,637 | 2,545,637 (already zipped) |
| pyodide-lock.json | 119,077 | 24,708 |
| pyodide.mjs / .js | 17,931 / 18,912 | ~5 K |
| **core total** | **13.5 MB** | **≈ 6.3 MB** |
| numpy-2.4.6 wheel | 2,960,568 | 2,929,885 |
| matplotlib closure (12 wheels: matplotlib 6.98 MB, numpy, fonttools 1.15, pillow 1.04, pytz 0.51, dateutil, pyparsing, contourpy, packaging, kiwisolver, six, cycler) | 13.27 MB | ≈ 13 MB |
| scipy | not measured separately; cold `loadPackage(['numpy','scipy','matplotlib'])` took 12.6 s | |

Core + numpy ≈ **9.2 MB transferred**. With matplotlib it is ≈ 22 MB, so load matplotlib lazily on the first plotting cell.

### A.4 Load timings [M] (Chrome 154, contended CPU)

| Scenario | Core boot | Packages |
|---|---|---|
| Main thread, cold network | 9.0 s | numpy 5.6 s (cold) |
| Main thread, HTTP cache warm (`transferSize` 0 for every file) | 4.1 s | numpy 1.9 s |
| Module worker, warm | 4.8–5.7 s | numpy 0.8–1.3 s |
| Worker, numpy + scipy + matplotlib, partly cold | 7.9 s | 12.6 s |
| Self-hosted under emulated artifact CSP, main thread | 4.1–4.7 s | numpy 0.9–1.8 s |
| First `import numpy` inside a cell after `loadPackage` | 1.4–1.7 s (one-time) | |

### A.5 Worker kernel: shared state, stdout and interrupt [M]

Files: `tech-kit/pyworker.js` (worker) and `tech-kit/pykernel.js` (main-thread client). Both were tested in `scratch/tech/web/kernel.html` and `worker.html`:

- **Shared state.** Cell 1 `W = np.arange(6.0).reshape(2,3)`; cell 2 `W @ np.ones(3)` printed `[3.0, 12.0]` and returned `"15"`.
- **With COOP/COEP** (`crossOriginIsolated=true`): an infinite loop was stopped after 1.5 s with **KeyboardInterrupt**, and the next cell still saw `W` (`True`).
- **Without isolation**: `interrupt()` terminates and reboots the worker in ≈ 5.7 s. The pending cell gets "kernel restarted: Python state was reset", and the next cell sees `False`.

```js
// pyworker.js (module worker). One interpreter per page => all cells share globals.
let ready = null;
self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    ready = (async () => {
      const t0 = performance.now();
      const { loadPyodide } = await import(msg.loaderUrl);
      const py = await loadPyodide({ indexURL: msg.indexURL, packageBaseUrl: msg.packageBaseUrl });
      const t1 = performance.now();
      if (msg.interruptBuffer) py.setInterruptBuffer(msg.interruptBuffer);
      await py.loadPackage(msg.packages, { messageCallback: () => {} });
      postMessage({ type: 'ready', loadMs: Math.round(t1 - t0), pkgMs: Math.round(performance.now() - t1), version: py.version });
      return py;
    })().catch((err) => { postMessage({ type: 'fatal', error: String(err && err.message || err) }); throw err; });
    return;
  }
  if (msg.type === 'run') {
    const { id, code } = msg;
    try {
      const py = await ready;                          // runs queue behind initialisation
      py.setStdout({ batched: (s) => postMessage({ type: 'stdout', id, s }) });
      py.setStderr({ batched: (s) => postMessage({ type: 'stderr', id, s }) });
      await py.loadPackagesFromImports(code, { messageCallback: () => {} });
      const t0 = performance.now();
      let r = await py.runPythonAsync(code);
      const ms = Math.round(performance.now() - t0);
      if (r && typeof r.toJs === 'function') { const p = r; r = p.toJs(); p.destroy(); }
      postMessage({ type: 'done', id, result: r === undefined ? null : String(r), ms });
    } catch (err) {
      postMessage({ type: 'error', id, error: String(err && err.message || err).trim().split('\n').slice(-1)[0] });
    }
  }
};
```

```js
// page side (full class in tech-kit/pykernel.js)
import { PyKernel } from './pykernel.js';
const V = '314.0.7';
const kernel = new PyKernel({
  workerUrl: 'pyworker.js',
  loaderUrl: `https://cdn.jsdelivr.net/npm/pyodide@${V}/pyodide.mjs`,
  indexURL: `https://cdn.jsdelivr.net/npm/pyodide@${V}/`,         // artifact host: new URL('pyodide/', location.href).href
  packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${V}/full/`, // artifact host: same as indexURL
});
await kernel.start();                       // call lazily, on the first "Run" click
const r = await kernel.run(cellSource, { onStdout: (s) => out.append(s + '\n') });
// r = {ok, result, ms} | {ok:false, error}; a Stop button calls kernel.interrupt()
```

Design rules for "run cells":
1. One kernel per page, shared by every cell.
2. Cells run strictly in order (the class queues them).
3. Each cell declares the cells it depends on. Clicking cell 5 first silently runs cells 1–4.
4. "Reset" = `kernel.restart()`.
5. For isolated demos on the same page, pass a fresh namespace: `py.runPythonAsync(code, { globals: py.toPy({}) })`.
   Verified [M `ns.html`]: the page-global `x` stayed `"page-global"`, the namespace got `x="isolated"` and `y=4.0`, and `__builtins__` was inserted automatically.
6. Run `import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)` once at boot.
   matplotlib 3.10.8 under Pyodide prints a `MatplotlibDeprecationWarning` about float x/y in `savefig` [M].

### A.6 matplotlib inside the worker [M]

The DOM is unavailable in a worker, so render with Agg and ship PNG bytes. The first figure took **661 ms**, giving a 13,472-byte PNG.

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, io, base64, numpy as np
fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(np.sin(np.linspace(0, 6, 100)))
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
"data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()   # cell result -> <img src=...>
```

### A.7 Numeric speed inside Pyodide [M]

numpy in Pyodide has no optimized BLAS:

| Op | Result |
|---|---|
| numpy `a @ a`, 384×384 fp32 | 136–156 ms → **0.73–0.83 GFLOP/s** |
| numpy 1024³ | 5.9 s → 0.36 GFLOP/s |
| numpy matvec 1536×384 | 1.2–1.4 ms → 0.8–1.0 GFLOP/s |
| `scipy.linalg.blas.sgemm` (OpenBLAS), 384³ | 49 ms → **2.3 GFLOP/s** |

Implication: in-browser numpy cells should stay at "toy" sizes (d ≤ 128, T ≤ 64, a few layers). One forward pass of a 2-layer d=64 numpy transformer takes milliseconds. A numpy training loop beyond about 1e8 FLOPs per step will feel slow.

### A.8 Caching and offline

- **Repeat visits.** jsDelivr's `immutable, max-age=1y` headers give a warm HTTP cache: every file showed transferSize 0 on reload [M].
- **Offline on a static host.** A service worker that caches `pyodide/*` plus the wheels you use, e.g. Workbox CacheFirst [E]. Not possible in artifacts (no service workers).
- **Node** has `loadPyodide({packageCacheDir})`; in Node only, missing wheels fall back to `cdnUrl` [S].

### A.9 Alternatives

| Option | Version | Verdict |
|---|---|---|
| **PyScript** | 2026.7.3 (`@pyscript/core` 0.7.31) | Wraps Pyodide or MicroPython with `<script type="py">`. Same CSP and self-hosting needs, and a heavier abstraction. It brings no capability a custom worker kernel lacks. |
| **JupyterLite** | 0.8.6 (pyodide-kernel 0.8.6) | A full notebook UI built as a static site with many files and a service-worker file system. Good as an optional "open this chapter as a notebook" link on GitHub Pages; **not artifact-compatible**. |
| **Plain JS cells** | n/a | Best for the 80 % of minimal checks that need no Python (§0). |

---

## B. In-browser inference and training without frameworks

### B.1 Weight export format [M]

`tech-kit/export_weights.py` reads `codebot/model_pretrain.pt` (HF `koki0702/zero-llm-data`, 44.5 MB) and writes:
- `codebot.<fmt>.bin` + `codebot.<fmt>.json`. The manifest holds the config plus `{name, shape, dtype, offset, nbytes[, scales_offset]}` per tensor, each 4-byte aligned for typed-array views.
- `tokenizer.json` with the 743 ordered BPE merges.
- `ref.json` with PyTorch logits and greedy tokens.
- The tied `unembed.weight` is dropped, and the forward pass reuses `embed.weight`.

| Format | CodeBot (11.12 M params) | Max \|Δlogit\| vs PyTorch | Greedy 64 tokens | Fits one artifact binary file (≤ 15 MB)? |
|---|---|---|---|---|
| f32 | 44.49 MB | **1.05e-5** | identical | no (3 shards) |
| f16 | 22.24 MB | 3.7e-3 | identical | no (2 shards) |
| int8, per output row, symmetric (+ f32 scales; 1-D params in f16) | **11.23 MB** | 0.13 | identical | **yes** |

Projected for the other bots [E: params × bytes]:
- StoryBot, 22.70 M: f16 45 MB, int8 ≈ 23 MB (2 shards).
- WebBot, 152.3 M: f16 305 MB, int8 ≈ 153 MB. Not a browser target; serve it only from a GPU or show recorded samples.

Manifest excerpt:

```json
{"config": {"vocab_size": 1000, "max_context_len": 256, "embed_dim": 384, "n_head": 6, "n_layer": 6, "ff_dim": 1536},
 "tied_unembed": true,
 "tensors": [{"name": "embed.weight", "shape": [1000, 384], "offset": 0, "scales_offset": 384000, "dtype": "q8", "nbytes": 388000}, "... 77 tensors"]}
```

### B.2 JS forward pass and parity [M]

`tech-kit/gpt.mjs` is dependency-free and runs in a browser, a worker or Node. It implements pre-LN, LayerNorm (eps 1e-5), exact-erf GELU (`nn.GELU()` default), learned positions, tied unembedding with bias, a KV cache, and a port of the BPE tokenizer.
Running `node tech-kit/bench_infer.mjs <export_dir>` gave:
- tokenizer ids identical: `"def add(a, b):\n" → [304,882,40,97,44,345,285,10]`;
- greedy continuation identical to PyTorch: `" return a + b\n\nresult = add(3, 4)\nprint(result)\n<|endoftext|>…"`.

Core of one decoding step (excerpt):

```js
export function step(sess, id) {             // feed one token, append K/V, return next-token logits
  const { cfg: { E, H, L, FF, V, T }, W } = sess.model, { x, h, q, a, o, f, att } = sess;
  const t = sess.pos, D = E / H, scale = 1 / Math.sqrt(D);
  if (t >= T) throw new Error('context full: re-prefill the last T tokens (learned positions cannot slide)');
  const wte = W['embed.weight'], wpe = W['pos_embed.weight'];
  for (let i = 0; i < E; i++) x[i] = wte[id * E + i] + wpe[t * E + i];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`, Kc = sess.K[l], Vc = sess.Vc[l];
    layerNorm(h, x, W[p + 'norm1.weight'], W[p + 'norm1.bias'], E);
    linear(q, W[p + 'attn.W_q.weight'], h, null, E, E);
    linear(Kc.subarray(t * E, (t + 1) * E), W[p + 'attn.W_k.weight'], h, null, E, E);
    linear(Vc.subarray(t * E, (t + 1) * E), W[p + 'attn.W_v.weight'], h, null, E, E);
    for (let hd = 0; hd < H; hd++) {           // causal mask for free: the cache only holds positions <= t
      const off = hd * D; let mx = -Infinity;
      for (let j = 0; j <= t; j++) { let s = 0; for (let d = 0; d < D; d++) s += q[off + d] * Kc[j * E + off + d]; att[j] = s * scale; if (att[j] > mx) mx = att[j]; }
      let z = 0; for (let j = 0; j <= t; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
      for (let d = 0; d < D; d++) a[off + d] = 0;
      for (let j = 0; j <= t; j++) { const w = att[j] / z; for (let d = 0; d < D; d++) a[off + d] += w * Vc[j * E + off + d]; }
    }
    linear(o, W[p + 'attn.W_o.weight'], a, null, E, E);
    for (let i = 0; i < E; i++) x[i] += o[i];
    layerNorm(h, x, W[p + 'norm2.weight'], W[p + 'norm2.bias'], E);
    linear(f, W[p + 'ffn.layers.0.weight'], h, W[p + 'ffn.layers.0.bias'], E, FF);
    for (let i = 0; i < FF; i++) f[i] = gelu(f[i]);
    linear(o, W[p + 'ffn.layers.2.weight'], f, W[p + 'ffn.layers.2.bias'], FF, E);
    for (let i = 0; i < E; i++) x[i] += o[i];
  }
  layerNorm(h, x, W['norm.weight'], W['norm.bias'], E);
  linear(sess.logits, wte, h, W['unembed.bias'], E, V);   // tied embedding
  sess.pos++;
  return sess.logits;
}
```

Notes:
- **Learned absolute positions** mean the KV cache cannot slide. Past 256 tokens, re-prefill the last 256 tokens per step, as the book's `generate()` does, or stop at 256.
- **f16 decode.** Use a 65,536-entry LUT loop: 69–81 ms for 11 M values. `Float32Array.from(u16, fn)` took 3.9 s [M]. Native `Float16Array` exists in Chrome 154 and Node 24 (115 ms copy) [M].
- **The book's CodeBot `generate()` has no KV cache.** In JS that would be about 256× more work per token at full context, so the KV cache is mandatory. It is also a nice ch05 tie-in.

### B.3 Throughput [M]

Single thread, contended CPU (`node --import ./hi.mjs` sets HIGH priority).

| Kernel | Result |
|---|---|
| Plain JS 4-accumulator matvec (384×384, 1536×384, 384×1536, 1000×384) | **0.5–1.3 GFLOP/s** (1.3 measured once the CPU load fell to ~60 %) |
| Plain JS L1-resident dot product | 0.88–0.95 GFLOP/s |
| Plain JS batched `X·Wᵀ`, 256×384×1536 | 0.57 GFLOP/s |
| WASM SIMD f32x4 matvec (359-byte module, `tech-kit/matvec_simd.wasm`, WAT in `simd_bench.mjs`) | **8.6–12.8 GFLOP/s**, ≈ 10–14× JS, max \|Δ\| 2e-6 |
| WebGPU, RTX 3060 Ti via Chrome (`navigator.gpu` adapter "nvidia / ampere"), naive one-thread-per-row WGSL matvec | 1536×384: 21 GFLOP/s (56 µs/dispatch); 384×1536: 6.9 (171 µs); 1000×384: 16.8 (46 µs) |
| WebGPU `mapAsync` readback of 4 KB of logits | median **1.5 ms**, p90 3.9 ms |

End-to-end CodeBot decode, KV cache, after an 8-token prompt:

| Backend | Weights | Decode tok/s | Prefill tok/s |
|---|---|---|---|
| plain JS (`gpt.mjs`) | f32 / f16 / int8 | **17–50** (varies with CPU contention; 38–50 at ~60 % load) | 18–50 |
| JS + WASM SIMD linears (`gpt_simd.mjs`) | f32 | **83–143** (falls toward 83–111 at positions 72–236 because attention is still plain JS) | 86–135 |
| WebGPU | f32 | **[E] 250–500**: about 37 matvec dispatches per token at 20–170 µs each with the naive kernel, plus 1.5 ms readback. A coalesced, workgroup-reduction kernel would cut the dispatch time several-fold | — |

StoryBot [E]: it costs about 35 MFLOP per token vs 22 for CodeBot, giving ≈ 50–90 tok/s with WASM SIMD. That is fine for "tell me a story" streaming.

The WASM kernel (all `nn.Linear` calls go through it; requires `nIn % 16 == 0`):

```wat
(module
  (memory (export "mem") 512)
  (func (export "matvec") (param $W i32) (param $x i32) (param $y i32) (param $nIn i32) (param $nOut i32)
    (local $o i32) (local $i i32) (local $row i32) (local $xi i32) (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (block $done_o (loop $lo
      (br_if $done_o (i32.ge_u (local.get $o) (local.get $nOut)))
      (local.set $row (i32.add (local.get $W) (i32.shl (i32.mul (local.get $o) (local.get $nIn)) (i32.const 2))))
      (local.set $a0 (v128.const f32x4 0 0 0 0)) (local.set $a1 (v128.const f32x4 0 0 0 0))
      (local.set $a2 (v128.const f32x4 0 0 0 0)) (local.set $a3 (v128.const f32x4 0 0 0 0))
      (local.set $i (i32.const 0))
      (block $done_i (loop $li
        (br_if $done_i (i32.ge_u (local.get $i) (local.get $nIn)))
        (local.set $xi (i32.add (local.get $x) (i32.shl (local.get $i) (i32.const 2))))
        (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (v128.load offset=0 (local.get $row)) (v128.load offset=0 (local.get $xi)))))
        (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (v128.load offset=16 (local.get $row)) (v128.load offset=16 (local.get $xi)))))
        (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (v128.load offset=32 (local.get $row)) (v128.load offset=32 (local.get $xi)))))
        (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (v128.load offset=48 (local.get $row)) (v128.load offset=48 (local.get $xi)))))
        (local.set $row (i32.add (local.get $row) (i32.const 64)))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $li)))
      (local.set $a0 (f32x4.add (f32x4.add (local.get $a0) (local.get $a1)) (f32x4.add (local.get $a2) (local.get $a3))))
      (f32.store (i32.add (local.get $y) (i32.shl (local.get $o) (i32.const 2)))
        (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a0)) (f32x4.extract_lane 1 (local.get $a0)))
                 (f32.add (f32x4.extract_lane 2 (local.get $a0)) (f32x4.extract_lane 3 (local.get $a0)))))
      (local.set $o (i32.add (local.get $o) (i32.const 1)))
      (br $lo)))))
```

Ship it as a base64 string inside the page (the probe page does exactly this). No toolchain is needed at runtime, and the WAT stays readable for a "how SIMD works" sidebar.

### B.4 Framework fallbacks [M versions / S docs]

| Library | Version | Notes |
|---|---|---|
| onnxruntime-web | 1.30.0 | `ort-wasm-simd-threaded.wasm` 14.24 MB (CPU); `…jsep.wasm` 28.31 MB (WebGPU). Fetches its wasm from jsDelivr by default, which the artifact CSP blocks: self-host and set `ort.env.wasm.wasmPaths`. The jsep build exceeds the 15 MB artifact file limit. Export path: `torch.onnx.export(model, ids, …, dynamic_axes=…)` |
| transformers.js (`@huggingface/transformers`) | 4.3.1 | Wraps ORT. Needs `env.allowRemoteModels=false`, `env.localModelPath`, and self-hosted ORT wasm on the artifact host. Overkill for a 1000-token-vocab custom GPT |
| web-llm (`@mlc-ai/web-llm`) | 0.2.86 | Needs MLC-compiled models; heavy |
| TensorFlow.js | 4.22.0 | Last release 2025-01; avoid |

Verdict: the custom JS/WASM path is smaller (≈ 10 KB of code vs 14–28 MB of runtime), has exact parity, and the tutorial controls it fully. Keep ORT only as a "compare against a real runtime" sidebar on a static host.

### B.5 In-browser training feasibility

- **Measured** [M `scratch/tech/web/microgpt.html`]. microgpt.js is @xenova's JS port of Karpathy's Feb-2026 *microgpt*: 1 layer, n_embd 16, 4 heads, block 16, 27-char vocabulary, **4,192 params**, scalar autograd in a Worker.
  It ran **1000 steps in 14.0 s** in Chrome on this contended CPU. Last-step loss went from 3.366 to 2.650, then it printed plausible names.
  - https://karpathy.github.io/2026/02/12/microgpt/ — 1000 steps ≈ 1 min in CPython on a MacBook
  - https://huggingface.co/spaces/webml-community/microgpt.js
- **Scaling [E].** Tensor-level ops on Float32Array at ≈ 0.7 GFLOP/s (plain JS) or ≈ 8–10 GFLOP/s (WASM SIMD matmuls):
  - a char-level model with ≈ 15 K params (1 layer, d=32, T=32, batch 8) costs ≈ 3e7 FLOPs per step, so 1000+ steps fit in a minute in plain JS;
  - a ≈ 100 K-param model (2 layers, d=64, T=64, batch 16) costs ≈ 7e8 FLOPs per step: about 1 step/s in plain JS, or ~5–7 steps/s with SIMD matmuls, which is enough to show Shakespeare loss falling from 4.17 to ~2.3 in a minute.
- **Other projects:**
  - TabGPT, ≈ 24 M params, full WebGPU pretraining in one tab: https://github.com/jbingen/tabgpt
  - trekhleb's homemade-gpt-js, TF.js char-level minGPT with in-browser training: https://trekhleb.medium.com/homemade-gpt-js-ead5feb475f0
  - 0hq/WebGPT, WebGPU inference: https://github.com/0hq/WebGPT
  - Karpathy's ConvNetJS/recurrentjs, char-RNN training in the browser (2015).
- **Recommendation.** One "train a tiny GPT in your browser" widget at microgpt scale or about 10× larger, in a Worker, with a live loss curve. It ties nicely to Saito's "build it from scratch" ethos and to CS336 A1's "overfit one minibatch" debugging advice.

---

## C. Modal

### C.1 Client and API [M introspection of `modal==1.6.1`, PyPI 2026-10-03]

The template below uses only these calls; their signatures were checked with `inspect.signature`:
- `modal.App`
- `@app.function(gpu=…, timeout=300 default, volumes=…, secrets=…, retries=…, single_use_containers=…, nonpreemptible=…, region=…, ephemeral_disk=…)`
- `modal.Image.debian_slim(python_version)`
- `.uv_pip_install(*pkgs, index_url, extra_index_url, gpu, …)`, `.pip_install(…)`, `.uv_sync(uv_project_dir, groups, extras, frozen=True)`
- `.add_local_dir(local, remote, copy=False, ignore=[…])`, `.add_local_file`, `.add_local_python_source(*modules)`
- `.apt_install`, `.env`, `.run_commands`
- `modal.Volume.from_name(name, create_if_missing, version)` with `.commit()` and `.reload()`
- `modal.Secret.from_name(name, required_keys)`
- `modal.Retries(max_retries, initial_delay, backoff_coefficient)`
- `Function.with_options(gpu=…, timeout=…, …)` and `Function.spawn(...).get()`

CLI: `modal run [-d/--detach] file.py::fn --arg v`. `--detach` means "Don't stop the app if the local process dies or disconnects."

Other facts:
- `add_local_*` files are attached at container start by default (`copy=False`), so code edits do not trigger an image rebuild.
- `remote_path` must be POSIX absolute. Passing a Windows path raised `InvalidError` [M].

### C.2 GPU strings and prices [S https://modal.com/pricing, https://modal.com/docs/guide/gpu]

| `gpu=` | $/s | **$/h** |
|---|---|---|
| `"T4"` | 0.000164 | **0.59** |
| `"L4"` | 0.000222 | **0.80** |
| `"A10"` | 0.000306 | **1.10** |
| `"L40S"` | 0.000542 | **1.95** |
| `"A100-40GB"` | 0.000583 | **2.10** |
| `"A100-80GB"` (`"A100"` may be upgraded to 80 GB at the 40 GB price) | 0.000694 | **2.50** |
| `"RTX-PRO-6000"` | 0.000842 | 3.03 |
| `"H100"` (may land on an H200 at the H100 price; `"H100!"` pins it) | 0.001097 | **3.95** |
| `"H200"` | 0.001261 | **4.54** |
| `"B200"` (`"B200+"` may run on a B300, billed as B200) | 0.001736 | **6.25** |
| `"B300"` (needs CUDA ≥ 13.1) | 0.001972 | 7.10 |

- **Multi-GPU.** `"H100:8"` gives up to 8 GPUs per container for B300/B200/H200/H100/A100/L4/T4/L40S, and up to 4 for A10. All GPUs sit on one machine.
- **Fallback lists.** `gpu=["H100", "A100-80GB"]`.
- **Multi-node** uses `@modal.clustered` and is private beta.
- **CPU and RAM are billed on top**: $0.0000131 per core-second (≈ $0.047/core-h) and $0.00000222 per GiB-second (≈ $0.008/GiB-h). Typically +$0.05–0.2/h for a training job.
- **Free credits.** Starter $30/month; Team $100/month.
- **Region pinning multiplier.** 1.15× (broad) or 1.75× (narrow).
- **Volumes.** $0.09/GiB-month, with 1 TiB/month free.

### C.3 Runtime facts that shape the templates

| Fact | Consequence | Source |
|---|---|---|
| Host driver **580.95.05, CUDA driver API 13.0** | The PyPI Linux `torch==2.14.1` wheel (CUDA 13.0 deps: `cuda-toolkit 13.0.3`, `nvidia-cudnn-cu13`) runs as-is | https://modal.com/docs/guide/cuda [S]; PyPI metadata [M] |
| **"All Modal Functions are subject to preemption by default."** `nonpreemptible=True` (3× CPU/RAM price) is **not supported for GPU functions** | Checkpoint to a Volume, use `retries=modal.Retries(initial_delay=0.0, …)` and `single_use_containers=True`, and resume on start | https://modal.com/docs/guide/preemption, `06_gpu_and_ml/long-training.py` [S] |
| Timeout 1 s – **24 h** per attempt (default 300 s) | WebBot on an A100 (≈ 22 h) is near the cap; prefer H100/B200 or a resumable loop | https://modal.com/docs/guide/timeouts [S] |
| Containers boot in about 1 s; imports and model loads dominate warm-up | Put data on a Volume, not in an `__init__` download | https://modal.com/docs/guide/cold-start [S] |
| Volumes v1: < 50 k files recommended, up to 2.5 GB/s; background commits every few seconds plus one at shutdown | A few large `.bin` files are ideal. Copy a 5.3 GB memmap to local disk before random-access training | https://modal.com/docs/guide/volumes [S] |
| Triton, which `torch.compile` uses in ch09, needs `gcc`/`clang` ("Failed to find C compiler…"). `debian_slim` is a slim Debian image without a compiler [E] | `.apt_install("build-essential")` | triton `python/triton/runtime/build.py` [S] |
| `wandb.init` in ch09 without credentials | Default `WANDB_MODE=disabled`. To log, run `modal secret create wandb-secret WANDB_API_KEY=…`, add `secrets=[modal.Secret.from_name("wandb-secret")]`, and set `WANDB_MODE=online` | Modal examples use `wandb-secret` [S] |

### C.4 Template 1: single GPU, data from a Volume, checkpoints back

`research/tech-kit/modal_app.py`, validated as follows:
1. `import modal_app` with modal 1.6.1 registers `book`, `ddp`, `prepare`, `storybot_resumable` and entrypoint `main`.
2. `storybot_resumable.local(max_iters=25, ckpt_every=10)` ran on the local RTX 3060 Ti against a fake 1 M-token memmap.
   A second call printed `resuming at step 25`, a third `resuming at step 40`.
3. The smoke test caught a real bug, now fixed: `torch.load(..., map_location="cuda")` moves the saved CPU RNG state to the GPU, and `torch.set_rng_state` then fails with "RNG state must be a torch.ByteTensor".
4. Not run on Modal itself, because that costs money. The book-script runner (`book`) was reviewed but not executed locally, since it would write into the read-only book checkout.

```python
"""Run the DLFS6 (ゼロから作るDeep Learning ❻) training scripts on Modal GPUs.

Layout on the Volume "zero-llm":  /vol/ws/<repo-relative path>  (datasets, tokenizers, checkpoints)
The book scripts read/write repo-relative paths, so each run links /vol/ws into the repo checkout
and copies new *.pt / *.png back afterwards. That also chains stages (pretrain -> SFT -> GRPO/DPO).

  modal run modal_app.py::prepare --bot storybot                      # HF -> Volume, CPU only
  modal run --detach modal_app.py --script ch06/05_pretrain.py --gpu L4
  modal run --detach modal_app.py::storybot_resumable                 # checkpoint/resume demo
  modal run --detach modal_app.py::ddp --script ch09/01_pretrain.py   # 8 x H100, torchrun
  modal volume get zero-llm ws/storybot/model_pretrain.pt .
"""
import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL_REPO = Path(os.environ.get("DLFS6_DIR", Path(__file__).parent / "dlfs6"))  # git clone of the book repo
REMOTE_REPO = "/root/dlfs6"
REPO = os.environ.get("ZLLM_REPO", REMOTE_REPO)  # env overrides only for local `.local()` smoke tests
WS = Path(os.environ.get("ZLLM_WS", "/vol/ws"))
HF = "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"
FILES = {  # what each bot needs that is NOT committed to the repo (codebot data ships with the repo)
    "codebot": ["model_pretrain.pt", "model_sft.pt"],
    "storybot": ["tiny_stories_train.bin", "tiny_stories_valid.bin", "merge_rules.pkl", "model_pretrain.pt"],
    "webbot": ["owt_train.bin", "owt_valid.bin", "merge_rules.pkl"],
}

vol = modal.Volume.from_name("zero-llm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # Triton (torch.compile in ch09) shells out to gcc
    .uv_pip_install("torch==2.14.1", "numpy<2", "matplotlib", "tqdm", "regex", "wandb")  # Linux wheel = CUDA 13.0
    .env({"PYTHONUNBUFFERED": "1", "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")})
    .add_local_dir(LOCAL_REPO, REMOTE_REPO, ignore=["**/.git/**", "**/.venv/**", "storybot/*.bin", "storybot/*.txt",
                                             "webbot/*.bin", "webbot/*.txt", "**/*.pt"])
)
app = modal.App("zero-llm", image=image)
RUN = dict(volumes={"/vol": vol}, timeout=24 * 3600, single_use_containers=True,
           retries=modal.Retries(initial_delay=0.0, max_retries=3))  # GPU containers are preemptible


def _commit() -> None:
    if not modal.is_local():  # inside a container: persist now (Volumes also auto-commit every few seconds)
        vol.commit()


def _link_workspace(copy_over_bytes: int = 1 << 30) -> set[Path]:
    """Expose /vol/ws/** inside the repo. Big memmapped corpora are copied to local disk for fast random reads."""
    ws, linked = WS, set()
    for src in ws.rglob("*"):
        if not src.is_file():
            continue
        dst = Path(REPO) / src.relative_to(ws)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        if src.stat().st_size > copy_over_bytes:
            shutil.copyfile(src, dst)
        else:
            dst.symlink_to(src)
        linked.add(dst)
    return linked


def _save_outputs(linked: set[Path]) -> None:
    for f in Path(REPO).rglob("*"):
        if f.suffix in {".pt", ".png"} and f.is_file() and not f.is_symlink() and f not in linked:
            dst = WS / f.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            print(f"saved {dst}")
    _commit()


@app.function(volumes={"/vol": vol}, timeout=3 * 3600)
def prepare(bot: str = "storybot", include_models: bool = False):
    """Download a bot's data from the book's Hugging Face dataset straight into the Volume."""
    import urllib.request
    out = WS / bot
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES[bot]:
        if name.endswith(".pt") and not include_models:
            continue
        dst = out / name
        if not dst.exists():
            print(f"downloading {bot}/{name}")
            urllib.request.urlretrieve(f"{HF}/{bot}/{name}", dst)
        print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB")
    _commit()


@app.function(gpu="L4", **RUN)
def book(script: str):
    """Run one book script unchanged, e.g. ch03/01_pretrain.py or ch06/05_pretrain.py."""
    linked = _link_workspace()
    subprocess.run(["nvidia-smi", "-L"], check=False)
    subprocess.run(["python", script], cwd=REPO, check=True)
    _save_outputs(linked)


@app.function(gpu="H100:8", **RUN)
def ddp(script: str = "ch09/01_pretrain.py", nproc: int = 8):
    """Single-node multi-GPU run with torchrun (the ch09 script initialises DDP from RANK/LOCAL_RANK)."""
    linked = _link_workspace()
    subprocess.run(["torchrun", "--standalone", f"--nproc-per-node={nproc}", script], cwd=REPO, check=True)
    _save_outputs(linked)


@app.function(gpu="L4", **RUN)
def storybot_resumable(max_iters: int = 40000, ckpt_every: int = 1000, batch_size: int = 32, max_lr: float = 1e-3):
    """ch06 StoryBot pretraining loop with atomic checkpoints, so a preempted container resumes, not restarts."""
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, REPO)
    from storybot.model import GPT

    ws = WS / "storybot"
    ckpt = ws / "resume.pt"
    data = np.memmap(ws / "tiny_stories_train.bin", dtype=np.uint16, mode="r")
    model = GPT(10000, 256, 512, 16, 4, 1344, 10000).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr)
    start = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)  # RNG state must stay a CPU ByteTensor
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); start = s["step"] + 1
        torch.set_rng_state(s["rng"]); print(f"resuming at step {start}")

    def lr_at(it, warmup=200):
        return max_lr * it / warmup if it < warmup else max_lr * (1 - (it - warmup) / (max_iters - warmup))

    for it in range(start, max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        ix = torch.randint(len(data) - 257, (batch_size,))
        xy = torch.stack([torch.from_numpy(data[i:i + 257].astype(np.int64)) for i in ix]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xy[:, :-1])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), xy[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 500 == 0:
            print(f"step {it} loss {loss.item():.4f}")
        if (it + 1) % ckpt_every == 0 or it == max_iters - 1:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": it, "rng": torch.get_rng_state()}, tmp)
            tmp.replace(ckpt)  # atomic: a preemption mid-save never leaves a torn checkpoint
            _commit()
    model.save(str(ws / "model_pretrain.pt"))
    _commit()


@app.local_entrypoint()
def main(script: str = "ch03/01_pretrain.py", gpu: str = "L4"):
    # .spawn().get() rather than .remote(): Modal's long-training example notes .remote() calls expire after 24 h.
    book.with_options(gpu=gpu).spawn(script).get()
```

Notes:
- `resume.pt` for StoryBot is 272 MB (model + AdamW m and v). Saving every 1000 steps is cheap.
- The book scripts have no resume logic. For runs over 1 h, prefer a resumable loop like `storybot_resumable` and stamp the commit or config into the checkpoint.
- **GPU choice per job:**
  - CodeBot ch03 runs in fp32, so a T4 is fine.
  - ch06 needs bf16, so use L4/A10/A100/H100. T4 (Turing) has no native bf16.
  - ch09 at micro-batch 32×1024 needs ≈ 30 GB, so A100-80GB, H100 or B200. A100-40GB is tight. On 24 GB cards use micro-batch 16 + accumulation 8 (§D.4).

### C.5 Template 2: 8-GPU torchrun DDP (single node)

`research/tech-kit/modal_ddp_min.py` imports cleanly. The string it patches, `max_iters = 100000`, exists at `ch09/01_pretrain.py:111` [M].

```python
"""Minimal 8-GPU DDP job on Modal: torchrun inside one container (single node, 8 x H100).
  modal run modal_app.py::prepare --bot webbot      # once: data -> Volume "zero-llm" (see modal_app.py)
  modal run --detach modal_ddp_min.py
"""
import os
import shutil
import subprocess
from pathlib import Path

import modal

LOCAL_REPO = Path(os.environ.get("DLFS6_DIR", Path(__file__).parent / "dlfs6"))
vol = modal.Volume.from_name("zero-llm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")                        # gcc for Triton / torch.compile
    .uv_pip_install("torch==2.14.1", "numpy<2", "matplotlib", "tqdm", "regex", "wandb")
    .env({"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})  # set "online" + add the wandb secret to log
    .add_local_dir(LOCAL_REPO, "/root/dlfs6", ignore=["**/.git/**", "**/.venv/**", "**/*.bin", "**/*.pt",
                                                      "storybot/*.txt", "webbot/*.txt"])
)
app = modal.App("zero-llm-ddp", image=image)


@app.function(gpu="H100:8", volumes={"/vol": vol}, timeout=24 * 3600,
              retries=modal.Retries(initial_delay=0.0, max_retries=2), single_use_containers=True)
              # secrets=[modal.Secret.from_name("wandb-secret")]  <- after `modal secret create wandb-secret WANDB_API_KEY=...`
def train(max_iters: int = 12_500):
    repo = Path("/root/dlfs6")
    for f in ("owt_train.bin", "owt_valid.bin"):        # 5.3 GB memmap: local NVMe beats network FS for random reads
        shutil.copyfile(f"/vol/ws/webbot/{f}", repo / "webbot" / f)
    script = repo / "ch09" / "01_pretrain.py"           # 8 ranks x 131k tokens/step: 12.5k steps = the 1-GPU 100k-step token budget
    src = script.read_text().replace("max_iters = 100000", f"max_iters = {max_iters}")
    (repo / "ch09" / "01_pretrain_modal.py").write_text(src)
    subprocess.run(["torchrun", "--standalone", "--nproc-per-node=8", "ch09/01_pretrain_modal.py"], cwd=repo, check=True)
    out = Path("/vol/ws/webbot"); out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / "webbot" / "model_pretrain.pt", out / "model_pretrain.pt")
    vol.commit()


@app.local_entrypoint()
def main(max_iters: int = 12_500):
    train.spawn(max_iters).get()
```

Why 12,500? In ch09 every rank draws its own micro-batches of 32×1024, 4 accumulation steps, for **131,072 tokens per rank per step**.
Unchanged `torchrun --nproc-per-node=8` therefore trains 8× the tokens of a 1-GPU run, 104.9 B instead of 13.1 B, for 8× the money (§D.5).

- The in-process alternative, from Modal's `14_clusters/simple_torch_cluster.py`, is `from torch.distributed.run import parse_args, run; run(parse_args([...]))`.
- Rank-0-only evaluation, 64.5 M tokens taking ≈ 1 min on an H100 [E], stays well under NCCL's default 10-minute process-group timeout.
  Source: `kProcessGroupNCCLDefaultTimeout = 10 * 60 * 1000` ms in `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` [S].
  If you enlarge the validation set, raise it with `init_process_group(timeout=timedelta(minutes=30))`.

### C.6 Command cheat-sheet (Windows PowerShell or Git Bash)

```bash
uv tool install modal               # or: uvx modal ...   (client 1.6.1)
modal setup                         # browser login -> ~/.modal.toml (already present on this machine)
modal volume create zero-llm        # optional; create_if_missing=True also works
modal run modal_app.py::prepare --bot storybot
modal run --detach modal_app.py --script ch06/05_pretrain.py --gpu L4
modal app list ; modal app logs <app-id> ; modal app stop <app-id>
modal volume ls zero-llm ws/storybot
modal volume get zero-llm ws/storybot/model_pretrain.pt ./model_pretrain.pt
modal secret create wandb-secret WANDB_API_KEY=xxxxxxxx      # only if logging to W&B
```

### C.7 Windows client caveats

- Python 3.14 + modal 1.6.1 prints `DeprecationWarning: 'asyncio.WindowsSelectorEventLoopPolicy' is deprecated` on every command [M]. It is harmless; use Python 3.12 for the client or set `PYTHONWARNINGS=ignore`.
- The console code page here is GBK. Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` before `modal run`, so rich progress output and the book's Japanese `print`s cannot hit `UnicodeEncodeError` [E, precaution].
- Pass local paths as `pathlib.Path`; Windows paths worked as the local side of `add_local_dir` [M]. Remote paths must be POSIX absolute [M].
- If the tutorial repo is checked out with `core.autocrlf=true`, `.sh` files reach Linux with CRLF (`$'\r': command not found`). Add `*.sh text eol=lf` to `.gitattributes` [E]. `tech-kit/setup_gpu_box.sh` is LF [M].
- `modal shell` and other interactive features were not tested on Windows.

---

## D. RunPod and Lambda

### D.1 On-demand prices (USD per GPU-hour)

**RunPod** [S https://www.runpod.io/pricing]. The page's JSON-LD gives Community/Secure tiers. The visible cards sometimes show a third number that matches neither ("card"), so confirm in the console before budgeting.

| GPU | Community | Secure | Card |
|---|---|---|---|
| H100 SXM 80 GB | **2.69** | 3.49 | 3.99 |
| H100 PCIe | 1.99 | 2.89 | 2.89 |
| H100 NVL | 2.59 | 3.19 | 3.19 |
| H200 | 3.59 | 4.59 | 5.29 |
| B200 | 5.98 | 6.79 | 7.99 |
| A100 SXM 80 GB | 1.39 | 1.59 | 1.79 |
| A100 PCIe 80 GB | 1.19 | 1.59 | 1.79 |
| L40S | 0.79 | 1.09 | 1.09 |
| RTX 4090 | **0.34** | 0.74 | 0.89 |
| RTX 5090 | 0.69 | 0.99 | 1.19 |
| L4 | 0.44 | 0.49 | 0.59 |
| A40 / RTX A6000 | 0.35 / 0.33 | 0.49 / 0.53 | 0.59 |

RunPod storage: container disk $0.10/GB-mo (wiped on stop). Volume disk `/workspace` $0.10 running / $0.20 stopped (deleted on terminate). Network volume `/workspace` $0.07/GB-mo (< 1 TB), attachable to several pods; attach it at creation.

**Lambda** [S https://lambda.ai/pricing, https://docs.lambda.ai/public-cloud/billing/] bills on-demand per minute; its pricing page lists no H200.

| Instance | $/GPU-h |
|---|---|
| 1× H100 SXM | **4.29** |
| 8× H100 SXM | **3.99** |
| 1× H100 PCIe | 3.29 |
| 1× A100 SXM 40 GB | 1.99 |
| 8× A100 SXM 80 GB | 2.79 |
| 1× B200 | 6.99 |
| 8× B200 | 6.69 |
| GH200 | 2.29 |
| A10 | 1.29 |
| A6000 | 1.09 |

Lambda filesystems: about $0.20/GiB-mo ("example rate"; the real price is shown at creation), mounted at `/lambda/nfs/<name>`. They must be in the same region and attached at launch.

### D.2 Launch and SSH

- **RunPod.**
  - Add your public key under Account → Credentials (or `SSH_PUBLIC_KEY` per pod).
  - CLI:
    ```bash
    bash <(curl -sL cli.runpod.io)
    runpodctl config --apiKey KEY
    runpodctl pod create --template-id runpod-torch-v21 --gpu-id "NVIDIA GeForce RTX 4090" --gpu-count 1 \
      --container-disk-in-gb 40 --network-volume-id <id> --cloud-type COMMUNITY
    ```
  - SSH: `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519` (exposed TCP, supports scp), or the proxied `ssh <pod>-<hash>@ssh.runpod.io -i …` (no scp).
  - Lifecycle: `runpodctl pod stop|delete <id>`.
- **Lambda.**
  - Launch from the dashboard, or via the API (one launch call per ~12 s):
    ```bash
    curl -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
      -d '{"region_name":"us-west-1","instance_type_name":"gpu_1x_h100_sxm5","ssh_key_names":["me"],"file_system_names":["zero-llm"],"quantity":1}' \
      https://cloud.lambda.ai/api/v1/instance-operations/launch
    ```
  - SSH: `ssh ubuntu@<ip>`.
  - The image is Ubuntu 22.04 with Lambda Stack (drivers, CUDA, PyTorch). Keep the project in a uv venv so Lambda Stack's system torch never leaks in.

### D.3 Reproducible setup script

`research/tech-kit/setup_gpu_box.sh`, validated:
- `bash -n` passes, and the file is LF-only.
- A dry run under Git Bash on Windows exercised `nvidia-smi`, the uv install check, the shallow clone, and `uv sync --frozen --python 3.12` against the book's `uv.lock`. That installed 24 packages including `torch==2.9.1` in 2 m 37 s.
- It then stopped, by design, at the CUDA assertion, because the book lock's Windows torch is the CPU build. On Linux, the same lock resolves the CUDA 12.8 build (`nvidia-*-cu12` deps present in `uv.lock`) [M].

```bash
#!/usr/bin/env bash
# Reproducible DLFS6 environment on a fresh Linux GPU box (RunPod pod or Lambda instance).
#   RunPod : WORK=/workspace   (volume disk or network volume; survives pod stop)
#   Lambda : WORK=/lambda/nfs/<filesystem-name>   (persistent filesystem attached at launch)
# Usage    : WORK=/workspace BOTS="storybot" bash setup_gpu_box.sh
set -euo pipefail

WORK="${WORK:-/workspace}"
BOTS="${BOTS:-storybot}"                      # any of: codebot storybot webbot
REPO_URL="https://github.com/oreilly-japan/deep-learning-from-scratch-6.git"
HF="https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main"

mkdir -p "$WORK"
export UV_CACHE_DIR="$WORK/.cache/uv"          # keep wheel downloads on the persistent disk
export UV_PYTHON_INSTALL_DIR="$WORK/.cache/python"
export UV_LINK_MODE=copy                       # cache and venv may sit on different filesystems

echo "== GPU / driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

cd "$WORK"
[ -d dlfs6/.git ] || git clone --depth 1 "$REPO_URL" dlfs6
cd dlfs6

# The book's uv.lock resolves torch 2.9.1 from PyPI; on Linux that wheel is the CUDA 12.8 build
# (CUDA 12.x minor-version compatibility needs driver >= 525; >= 570 recommended).
uv sync --frozen --python 3.12
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: compare `nvidia-smi` driver with torch.version.cuda"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {p.name} sm_{p.major}{p.minor} | "
      f"{p.total_memory / 2**30:.0f} GiB | bf16 {torch.cuda.is_bf16_supported()}")
PY

command -v gcc >/dev/null 2>&1 || echo "WARNING: no gcc; torch.compile (ch09) needs it: apt-get install -y build-essential"

fetch() {  # fetch <bot> <file>: skip files that already exist (re-runs are cheap)
  local dst="$1/$2"
  [ -s "$dst" ] && { echo "have $dst"; return; }
  curl -fL --retry 5 --retry-delay 3 -o "$dst.part" "$HF/$1/$2" && mv "$dst.part" "$dst"
}
for bot in $BOTS; do
  case "$bot" in
    codebot)  fetch codebot model_pretrain.pt ;;   # tiny_codes.* and merge_rules.pkl ship with the repo
    storybot) for f in tiny_stories_train.bin tiny_stories_valid.bin merge_rules.pkl; do fetch storybot "$f"; done ;;
    webbot)   for f in owt_train.bin owt_valid.bin merge_rules.pkl; do fetch webbot "$f"; done ;;
    *) echo "unknown bot: $bot" >&2; exit 1 ;;
  esac
done

cat <<EOF
Ready. Long runs should survive SSH disconnects:
  cd $WORK/dlfs6 && tmux new -s train
  uv run python ch06/05_pretrain.py                                    # single GPU
  WANDB_MODE=disabled uv run torchrun --standalone --nproc-per-node=8 ch09/01_pretrain.py
EOF
```

If you want torch 2.14 (CUDA 13) instead of the book's lock, the host driver must be ≥ 580. Run `nvidia-smi` first; RunPod community hosts vary.
For broad driver compatibility on Linux use the `cu126` or `cu128` index; `cu126` lacks Blackwell `sm_100` kernels.

### D.4 Data sizes and GPU memory

| Item | Size | Source |
|---|---|---|
| `storybot/tiny_stories_train.bin` | 1,082,458,446 B = **541.2 M tokens** | HF `x-linked-size` [M] |
| `storybot/tiny_stories_valid.bin` | 10.9 MB = 5.47 M tokens | [M] |
| `webbot/owt_train.bin` | 5,298,009,600 B = **2.649 B tokens** | [M] |
| `webbot/owt_valid.bin` | 129 MB = 64.5 M tokens | [M] |
| Raw text | TinyStories 2.23 GB, OWT 11.9 GB | [M] |
| `codebot/tiny_codes.bin` | 2.68 M tokens, in the repo. 20 k × 8192 tokens ≈ **61 epochs** (the book deliberately overfits) | [M] |
| Pretrained weights on HF | `codebot/model_{pretrain,sft,grpo}.pt` (44.5 MB each), `storybot/model_pretrain.pt` (90.8 MB). **No** `webbot/model_pretrain.pt` and no `storybot/model_dpo.pt` (15-byte "not found" bodies) | [M] |

WebBot peak memory, measured on the 3060 Ti with bf16 and no compile: micro-batch 1 → 2.97 GB, 2 → 3.89 GB, 4 → 5.86 GB, 8 → 9.71 GB.
That extrapolates to ≈ 30–34 GB at micro-batch 32, consistent with `repo-ch06-ch09-scale.md` §4.

### D.5 Cost table

Method:
- FLOPs = 6·N·D for training plus 2·N·D for evaluation forward passes, where N counts every matmul weight (non-embedding + LM head).
  Measured counts: CodeBot 11.02 M, StoryBot 17.58 M (≈ CS336 A1's "~17M"), WebBot 113.9 M.
- An attention term 6·L·T·d per token is included.
- Per-token training FLOPs: CodeBot 69.7 M, StoryBot 108.6 M, WebBot 740.1 M.
- **Eval overhead is real.** StoryBot runs 81 full validation passes over 5.47 M tokens (+45 % FLOPs). WebBot runs 101 passes over 64.5 M tokens (+17 %).
- MFU assumptions: WebBot 35 % on H100 (torch.compile + bf16 + flash SDPA; range 25–45 %), 45 % on A100, 25 % on B200.
  CodeBot and StoryBot are launch-overhead bound, so their GPU times are scaled from the measured 3060 Ti step times and CS336's statement that the StoryBot-identical A1 config runs in "roughly 20–30 minute runtime on 1 B200".
- Prices are from §C.2 and §D.1. Modal costs exclude its small CPU/RAM surcharge.

| Job | GPU | FLOPs | Hours | USD |
|---|---|---|---|---|
| CodeBot pretrain (ch03/01: 11.1 M params, 20 k × 32 × 256 = 163.8 M tok, fp32) | **RTX 3060 Ti local [M]** | 1.14e16 | **0.61** (109.7 ms/it) | 0 |
| same | Modal T4 [E] | 1.14e16 | 1.0–1.5 | 0.6–0.9 |
| same | Modal L4 [E] | 1.14e16 | 0.5–0.75 | 0.4–0.6 |
| same | Modal H100 [E] | 1.14e16 | 0.12–0.25 | 0.5–1.0 |
| same | RunPod RTX 4090 [E] | 1.14e16 | 0.2–0.3 | 0.07–0.22 |
| CodeBot SFT (ch03/03, 500 it) | any [E] | 2.9e14 | ≈ 0.02 | < 0.05 |
| CodeBot GRPO (ch03/09, 500 it × 256 sampled answers) | 3060 Ti local [M→E] | generation-bound | **≈ 1.8** (47 ms per sample → 12 s/it) | 0 (CPU/launch-bound: a bigger GPU barely helps) |
| StoryBot pretrain (ch06/05: 22.7 M params, 327.7 M tok + 81 evals) | **RTX 3060 Ti local [M+E]** | 3.56e16 + 1.60e16 eval | **≈ 1.7** (102.9 ms/it training ⇒ 1.14 h, eval ≈ 0.5 h) | 0 |
| same | Modal L4 [E] | 5.2e16 | 1.2–1.8 | 1.0–1.4 |
| same | Modal A100-40GB [E] | 5.2e16 | 0.5–0.8 | 1.0–1.7 |
| same | Modal H100 [E] | 5.2e16 | 0.3–0.5 | 1.2–2.0 |
| same | RunPod RTX 4090 [E] | 5.2e16 | 0.5–0.8 | 0.2–0.6 |
| StoryBot DPO (ch06/08, 1000 it × 8 pairs; policy fwd+bwd on chosen and rejected, plus reference forward) | any [E] | ≈ 6e14 | < 0.05 | < 0.1 |
| **WebBot as written, 1 GPU** (ch09: 152.3 M params, 100 k × 131 k = 13.1 B tok, + 101 evals) | Modal H100 [E, 35 % MFU] | 9.70e18 + 1.61e18 | **9.1** (range 7–12) | **36** (28–47) |
| same | Modal B200 [E, 25 %] | 1.13e19 | 5.6 | 35 |
| same | Modal A100-80GB [E, 45 %] | 1.13e19 | 22.4 (near the 24 h cap) | 56 |
| same | RunPod H100 SXM [E] | 1.13e19 | 9.1 | 24 (community) – 32 (secure) |
| same | Lambda 1× H100 SXM [E] | 1.13e19 | 9.1 | 39 |
| same | RTX 3060 Ti local [M→E] | 1.13e19 | ≈ 300 (12.5 k tok/s at micro-batch 4) | infeasible |
| **WebBot torchrun ×8 unchanged** (104.9 B tok) | Modal H100:8 [E] | 7.8e19 + 1.6e18 | 9.1 wall (72.6 GPU-h) | **287** |
| same | Lambda 8× H100 [E] | — | 9.1 wall | 290 |
| same | RunPod 8× H100 SXM [E] | — | 9.1 wall | 195–253 |
| **WebBot ×8 with max_iters 12,500** (same 13.1 B tok as 1 GPU; `modal_ddp_min.py`) | Modal H100:8 [E] | 9.70e18 + 2.2e17 | ≈ 1.2 wall (≈ 9.2 GPU-h) | ≈ 36–40 |
| **WebBot "budget"** (1 GPU, max_iters 23 k ⇒ 3.0 B tok ≈ 20 tok/param, eval every 1 k) | Modal H100 [E] | 2.23e18 + 3.8e17 | **2.1** | **≈ 8.3** |
| same | RunPod H100 SXM [E] | 2.6e18 | 2.1 | 5.6–7.3 |

How to verify any row before paying for all of it:
1. Run the first 200 iterations (≈ 1–3 min).
2. Read tqdm's `it/s`.
3. Hours = `max_iters / it/s / 3600`, plus evaluations. Time one evaluation pass separately.
4. On Modal, `book.with_options(gpu=..., timeout=600)` caps the cost of such a probe.

Cross-check from another dossier (`repo-ch06-ch09-scale.md`): WebBot 1×H100 "约 8–10 h" and StoryBot "H100 10–20 min" excluding eval, both consistent with this table.

---

## E. Local CUDA on Windows (RTX 3060 Ti, sm_86)

### E.1 pyproject: CUDA wheels on Windows, PyPI elsewhere [M]

Verified in `scratch/tech/localcuda`:
- `uv lock` resolves **`torch 2.14.1+cu130` for `sys_platform == 'win32'`** from download.pytorch.org and **`torch 2.14.1` from PyPI for everything else**.
  The PyPI Linux wheel already pulls CUDA 13.0 libraries; macOS gets the CPU/MPS wheel.
- `uv sync` downloaded the **1.99 GB** wheel in 21 min on this connection.
- On the GPU it reported `torch 2.14.1+cu130 | CUDA 13.0 | NVIDIA GeForce RTX 3060 Ti | cc 8.6 | bf16 True`, with arch list `sm_75 … sm_86 … sm_120`.

```toml
[project]
name = "zero-llm"
version = "0.1.0"
requires-python = ">=3.10,<3.13"        # the book pins numpy<2; numpy 1.26.4 has no cp313/cp314 wheels
dependencies = ["torch>=2.5.0", "numpy>=1.24,<2", "matplotlib>=3.7", "tqdm>=4.65", "regex>=2023.0", "wandb>=0.16"]

# PyTorch CUDA 13.0 wheels for Windows only. explicit = true => used ONLY for packages pinned to it below.
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]
```

Windows wheel sizes for torch 2.14.1 cp312 [M HEAD]: cu130 1.99 GB, cu132 1.99 GB, cu126 2.60 GB. cu128 stops at 2.11.0.
Index docs: https://docs.astral.sh/uv/guides/integration/pytorch/.

Variant when one machine type needs both, e.g. Linux CPU CI vs Linux GPU, verified with `uv lock` and `uv sync --extra cu130|cpu --dry-run` [M]:

```toml
[project.optional-dependencies]
cpu = ["torch>=2.5"]
cu130 = ["torch>=2.5"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu130" }]]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", extra = "cpu" }, { index = "pytorch-cu130", extra = "cu130" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
# uv sync --extra cu130   -> torch 2.14.1+cu130 ;  uv sync --extra cpu -> torch 2.14.1+cpu
```

Quick one-off without a project: `uv pip install torch --torch-backend=auto` picks the index from the detected driver; it works in the `uv pip` interface only [S].

### E.2 Measured training throughput on the 3060 Ti [M]

Script `scratch/tech/localcuda/bench_gpu.py`. Synthetic tokens; the loops mirror the book: AdamW, bf16 autocast and clipping where the book uses them.

| Model / setting | ms/iter | tokens/s | peak mem | Implied full run |
|---|---|---|---|---|
| CodeBot, fp32 as book, B=32×256 | **109.7** | 74.6 k | 2.09 GB | 20 k it = **36.6 min** |
| CodeBot, TF32 matmul | 96.8 | 84.6 k | 2.09 GB | 32 min |
| CodeBot, bf16 autocast | 90.6 | 90.4 k | 1.52 GB | 30 min |
| StoryBot, bf16 + clip as book, B=32×256 | **102.9** | 79.6 k | 2.73 GB | 40 k it = **68.6 min** + eval (≈ 30 min [E]) |
| WebBot, bf16, mb=8×1024 | 15,845 | 517 | **9.71 GB > 8 GB** | sysmem fallback, unusable |
| WebBot, bf16, mb=4×1024 | 327 | 12.5 k | 5.86 GB | 13.1 B tok ≈ 12 days |
| WebBot, mb=1 | 271 | 3.8 k | 2.97 GB | forward ≈ 135 ms of fixed launch overhead |

Microbenchmarks: bf16 4096³ matmul **34.3 TFLOP/s**; fp32 9.4 TFLOP/s; 1 GiB fp32→bf16 cast at 397 GB/s; a small 768² cast takes 43 µs, so WDDM launch overhead is visible.

### E.3 Windows gotchas

1. **Silent VRAM spill.** Since driver 536.40, Windows falls back to shared system memory instead of raising CUDA OOM. Here micro-batch 8 took 15.8 s/it, 48× slower per iteration than micro-batch 4 (24× per token).
   Set *NVIDIA Control Panel → Manage 3D settings → CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback* (driver ≥ 546.01) per python.exe, or keep peak memory below VRAM minus about 2 GB used by the desktop.
   Source: NVIDIA answer 5490, https://nvidia.custhelp.com/app/answers/detail/a_id/5490
2. **Launch overhead.** WDDM plus eager PyTorch makes small models launch-bound: WebBot at mb=1 spends ~250 ms/it outside the math. Linux cloud boxes are better.
3. **ch09 does not run on Windows** (other dossier, `repo-ch06-ch09-scale.md` §2.10): `torch.compile` needs `cl`/Triton, NCCL is absent, and torchrun fails with no libuv. Use Modal, RunPod or Lambda, or WSL2.
4. **The book's own `.venv` has `torch 2.9.1+cpu`** [M], because the book's `uv.lock` uses PyPI. Never "fix" the book repo; use the tutorial's own pyproject.
5. **numpy<2 vs Python 3.14.** uv picked CPython 3.14 by default here. Pin `.python-version` to 3.12, or the book's `numpy<2` has no wheels.
6. **GRPO (ch03/09) is CPU-bound.** Measured 47 ms per sampled answer, of which 10 ms is Python BPE `encode`. 500 iterations ≈ 1.8 h locally, and an H100 would not change that much.

---

## F. Recommendations for the tutorial build

1. **Two-tier interactivity.**
   - Tier 1: JS widgets for every "minimal verification" (instant, CSP-proof).
   - Tier 2: lazily loaded Pyodide cells (`pyworker.js` + `pykernel.js`) running *numpy translations* of the book's torch code.
   - Keep a local `uv run` command next to each cell for the original PyTorch version. Check the numpy and torch versions against each other offline with a parity script.
2. **End-to-end example: chat with the CodeBot you will train.**
   - Ship `codebot.q8.bin` (11.2 MB, one artifact-sized file), `tokenizer.json` and `gpt.mjs` + the SIMD kernel, running in a Worker with streaming tokens.
   - Include a toggle "Use my weights": the reader drops in a `model_pretrain.pt` they exported with `export_weights.py` from Modal or locally.
   - Show the PyTorch-parity badge, i.e. the greedy-identical test, as a "verify it yourself" button.
3. **Hosting decision.** If the tutorial is published as a claude.ai artifact:
   - self-host `pyodide/` (core 13.5 MB + numpy 3 MB) as artifact files, publishing `.mjs` with `contentType: text/javascript`;
   - expect no SharedArrayBuffer, so Stop = restart;
   - publish `artifact-runtime-probe.html` first to confirm `wasm-compile`, WebGPU and isolation.
   Otherwise, a GitHub Pages build with `coi-serviceworker` gives the best experience (real interrupts, CDN Pyodide, offline).
4. **GPU chapters.**
   - "Run it yourself" buttons map to `modal run …` commands from `modal_app.py`.
   - Default GPU per stage: CodeBot T4/L4, StoryBot L4 or H100, WebBot H100 or B200.
   - Every page shows an ETA/cost calculator fed by the §D.5 formulas and a "measure 200 steps first" recipe.
   - Make the WebBot budget run (≈ $8) the recommended path. Present the full 13 B-token run (≈ $36) and the 8×H100 variants as optional.
5. **CS336 tie-ins with hard numbers** (handout v26.0.3, Spring 2026):
   - StoryBot's config is **identical** to A1's TinyStories run (vocab 10 k, ctx 256, d 512, d_ff 1344, 4 layers, 16 heads, θ 10 k, 327,680,000 tokens).
   - A1's target is val loss **≤ 1.45**, in "20–30 minutes on 1 B200".
   - Its CPU/MPS fallback, 40.96 M tokens on an M4 Max, reaches val loss 1.80 in 1 h 22 m on CPU or 36 min on MPS. That is a ready-made local alternative for readers without a GPU.
   - A1's leaderboard caps OWT training at 45 min on a B200, a natural "beat it with WebBot" challenge.
6. **Before shipping any GPU number in the tutorial,** replace the [E] H100/B200 rows by a 200-step probe on Modal (≈ $0.1–0.5 each, including container start and the first image build).
7. **Tutorial repo hookup.** The root `zero-llm-tutorial/pyproject.toml`, written by another agent, has a plain `torch>=2.5.0`, so `uv sync` on this Windows box installs the **CPU** wheel.
   Add the E.1 `[[tool.uv.index]]` and `[tool.uv.sources]` block; its `requires-python` of 3.11–3.13 and `.python-version` 3.12 already fit.
   Its `gpu` dependency group (`modal`, `wandb`) matches the Modal templates. Its `.gitattributes` (`* text=auto eol=lf`) already prevents the CRLF problem in §C.7.

---

## Appendix: reproduce

Scratch root: `C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\tech\`

| What | Command |
|---|---|
| Param counts | `cd dlfs6 && PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I <scratch>/calc/params.py <scratch>/localcuda` |
| FLOPs/time table | `uv run --no-project python -I <scratch>/calc/cost_table.py` |
| GPU throughput | `cd <scratch>/localcuda && uv run python -I bench_gpu.py <scratch>/localcuda` (WebBot small batches: `bench_webbot.py`) |
| GRPO generation timing | `uv run --with regex python -I <scratch>/calc/grpo_gen_timing2.py <dlfs6> <scratch>/models/codebot/model_sft.pt` |
| Weight export | `cd <scratch>/localcuda && uv run --with regex python -I <scratch>/jsinfer/export_weights.py <dlfs6> <ckpt.pt> <out>` |
| JS parity/speed | `node <scratch>/jsinfer/bench_infer.mjs <out>`; `node <scratch>/wasmsimd/bench_simd_infer.mjs <out>`; `node <scratch>/jsinfer/matmul_bench.mjs` |
| Browser tests | `node <scratch>/web/server.mjs <scratch>/web 18765`, then open `http://127.0.0.1:18765/` + `cdn.html[?pbu=1]`, `worker.html[?coi=1&pkgs=numpy,scipy,matplotlib]`, `csp.html?csp=strict|wasm|eval&py=self|cdn`, `kernel.html[?coi=1]`, `webgpu.html`, `microgpt.html`, `probe.html?csp=eval` |
| Modal template smoke test | Write a fake `<ws>/storybot/tiny_stories_train.bin` (1 M random uint16 < 10000), then `cd research/tech-kit && PYTHONDONTWRITEBYTECODE=1 DLFS6_DIR=<dlfs6> ZLLM_REPO=<dlfs6> ZLLM_WS=<ws> uv run --project <scratch>/localcuda --with modal python -c "import modal_app as m; m.storybot_resumable.local(max_iters=25, ckpt_every=10)"`. Run it twice to see `resuming at step 25`. The CUDA venv was deleted to save 4 GB; uv rebuilds it from its cache on first use |

Primary sources:
- Pyodide on npm and jsDelivr: https://www.npmjs.com/package/pyodide, https://cdn.jsdelivr.net/pyodide/v314.0.7/full/
- Modal: https://modal.com/pricing · https://modal.com/docs/guide/gpu · https://modal.com/docs/guide/volumes · https://modal.com/docs/guide/preemption · https://modal.com/docs/guide/timeouts · https://modal.com/docs/guide/cuda · https://modal.com/docs/guide/cold-start · https://modal.com/docs/guide/secrets · https://github.com/modal-labs/modal-examples (`06_gpu_and_ml/long-training.py`, `14_clusters/simple_torch_cluster.py`)
- RunPod: https://www.runpod.io/pricing · https://docs.runpod.io/pods/configuration/use-ssh · https://docs.runpod.io/pods/storage/types · https://docs.runpod.io/runpodctl/reference/runpodctl-pod
- Lambda: https://lambda.ai/pricing · https://docs.lambda.ai/public-cloud/billing/ · https://docs.lambda.ai/public-cloud/filesystems/
- uv + PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/ · PyTorch wheel indexes https://download.pytorch.org/whl/cu130/torch/
- CS336 A1 handout (v26.0.3, Spring 2026): https://github.com/stanford-cs336/assignment1-basics
- WebGPU status: https://web.dev/blog/webgpu-supported-major-browsers · https://caniuse.com/webgpu
- Artifact host CSP audit: https://pluto.security/blog/inside-claude-artifacts-how-your-agent-publishes-to-the-web/
