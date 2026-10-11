// Pyodide "kernel" in a module Web Worker: one interpreter per page, so all cells share globals.
// Protocol: {type:'init', loaderUrl, indexURL, packageBaseUrl, packages, interruptBuffer?} -> {type:'ready'|'fatal'}
//           {type:'run', id, code} -> {type:'stdout'|'stderr', id, s}* then {type:'done'|'error', id, ...}
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
      const t2 = performance.now();
      postMessage({ type: 'ready', loadMs: Math.round(t1 - t0), pkgMs: Math.round(t2 - t1), version: py.version });
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
