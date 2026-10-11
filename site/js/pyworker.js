/* Pyodide module worker: runs verify cells off the main thread.
   Protocol (each message carries a numeric `id` echoed in the reply):
     {type: 'init', indexURL, packages, harnessURL} -> {ok, version} | {error}
     {type: 'run', cellId, code, files}               -> {stdout, error, seconds}
   All cells of one page share a single namespace, mirroring tools/run_cells.py.
   (A module worker importing pyodide.mjs; classic importScripts of pyodide.js fails in current Chromium.) */

let pyodide = null;
let runCell = null;
let ns = null;

async function init({ indexURL, packages, harnessURL }) {
  const { loadPyodide } = await import(indexURL + 'pyodide.mjs');
  pyodide = await loadPyodide({ indexURL, packageBaseUrl: indexURL });
  if (packages && packages.length) await pyodide.loadPackage(packages);
  const harness = await (await fetch(harnessURL)).text();
  pyodide.FS.writeFile('/home/pyodide/cell_harness.py', harness);
  pyodide.runPython('import sys; sys.path.insert(0, "/home/pyodide")\nfrom cell_harness import run_cell');
  runCell = pyodide.globals.get('run_cell');
  ns = pyodide.globals.get('dict')();
  ns.set('__name__', '__main__');
  return { ok: true, version: pyodide.version };
}

async function stageFiles(files) {
  for (const f of files || []) {
    const path = '/home/pyodide/' + f;
    if (pyodide.FS.analyzePath(path).exists) continue;
    const resp = await fetch(new URL('../' + f, self.location.href));
    if (!resp.ok) throw new Error(`无法下载数据文件 ${f}（HTTP ${resp.status}）`);
    const dir = path.slice(0, path.lastIndexOf('/'));
    pyodide.FS.mkdirTree(dir);
    pyodide.FS.writeFile(path, new Uint8Array(await resp.arrayBuffer()));
  }
}

async function run({ cellId, code, files }) {
  await stageFiles(files);
  const res = runCell(code, ns, cellId);
  const out = res.toJs({ dict_converter: Object.fromEntries });
  res.destroy();
  return out;
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    const reply = type === 'init' ? await init(e.data) : await run(e.data);
    self.postMessage({ id, ...reply });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err), fatal: type === 'init' });
  }
};
