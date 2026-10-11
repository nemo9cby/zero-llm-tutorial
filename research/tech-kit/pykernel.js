// Main-thread client for pyworker.js: one Pyodide interpreter per page, so every cell shares globals.
// Cells run strictly in order. interrupt() raises KeyboardInterrupt when the page is cross-origin isolated
// (SharedArrayBuffer available); otherwise it restarts the worker and the page's Python state is lost.
export class PyKernel {
  constructor({ workerUrl, loaderUrl, indexURL, packageBaseUrl, packages = ['numpy'] }) {
    Object.assign(this, { workerUrl, loaderUrl, indexURL, packageBaseUrl, packages });
    this.sab = self.crossOriginIsolated ? new Int32Array(new SharedArrayBuffer(4)) : null;
    this.queue = Promise.resolve();
    this.pending = new Map();
    this.seq = 0;
  }

  start() {
    this.worker = new Worker(this.workerUrl, { type: 'module' });
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = ({ data: m }) => {
        if (m.type === 'ready') return resolve(m);
        if (m.type === 'fatal') return reject(new Error(m.error));
        const p = this.pending.get(m.id);
        if (!p) return;
        if (m.type === 'stdout') return p.onStdout(m.s);
        if (m.type === 'stderr') return p.onStderr(m.s);
        this.pending.delete(m.id);
        p.resolve(m.type === 'done' ? { ok: true, result: m.result, ms: m.ms } : { ok: false, error: m.error });
      };
      this.worker.onerror = (e) => reject(new Error(e.message || 'worker failed to start'));
    });
    const { loaderUrl, indexURL, packageBaseUrl, packages } = this;
    this.worker.postMessage({ type: 'init', loaderUrl, indexURL, packageBaseUrl, packages, interruptBuffer: this.sab });
    return this.ready;
  }

  run(code, { onStdout = () => {}, onStderr = () => {} } = {}) {
    const task = async () => {
      await this.ready;
      if (this.sab) this.sab[0] = 0;
      const id = ++this.seq;
      return new Promise((resolve) => {
        this.pending.set(id, { resolve, onStdout, onStderr });
        this.worker.postMessage({ type: 'run', id, code });
      });
    };
    const result = this.queue.then(task, task);
    this.queue = result.catch(() => {});
    return result;
  }

  interrupt() {
    if (this.sab) { this.sab[0] = 2; return 'interrupted'; }           // SIGINT -> KeyboardInterrupt
    this.restart();
    return 'restarted';
  }

  restart() {
    this.worker?.terminate();
    for (const p of this.pending.values()) p.resolve({ ok: false, error: 'kernel restarted: Python state was reset' });
    this.pending.clear();
    this.queue = Promise.resolve();
    return this.start();
  }
}
