/* Page runtime: theme + nav, KaTeX, widget hydration, and in-browser verify cells. */

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';

// ---------------------------------------------------------------- chrome
function setupChrome() {
  const root = document.documentElement;
  document.querySelector('.theme-toggle')?.addEventListener('click', () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) { /* storage may be blocked */ }
  });
  const nav = document.getElementById('nav');
  const toggle = document.querySelector('.nav-toggle');
  toggle?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  nav?.addEventListener('click', (e) => {
    if (e.target.closest('a') && nav.classList.contains('open')) toggle.click();
  });
}

// ---------------------------------------------------------------- math
function renderMath() {
  if (!window.katex) return;
  for (const el of document.querySelectorAll('.math-inline, .math-block')) {
    try {
      window.katex.render(el.textContent, el, {
        displayMode: el.classList.contains('math-block'), throwOnError: false, strict: 'ignore',
      });
    } catch (e) { el.classList.add('math-error'); }
  }
}

// ---------------------------------------------------------------- tables
// The <wbr> break points in table code are off (site.css) unless the table has .wbr-on, which a table gets only when it
// overflows without them (each table is its own scroll box: scrollWidth > clientWidth): identifiers stay whole wherever
// the table fits. Measured again when web fonts finish loading and when a table's width changes (a resized window, a
// <details> that opens). Switching break points changes a table's height, never its width, so it cannot set off another
// measurement; the observer only marks tables, and they are measured in the next frame, outside the observer's loop.
function fitTables() {
  const tables = [...document.querySelectorAll('.chapter table')].filter((t) => t.querySelector('wbr'));
  if (!tables.length) return;
  const widths = new WeakMap();                  // the width each table was last measured at
  const measure = (list) => {
    for (const t of list) t.classList.remove('wbr-on');      // all writes, then all reads: one layout per batch
    const wide = list.filter((t) => t.scrollWidth > t.clientWidth);
    for (const t of list) widths.set(t, t.clientWidth);
    for (const t of wide) t.classList.add('wbr-on');
  };
  measure(tables);
  if (document.fonts) {                          // after measure(): the fonts it needs have started loading by now
    document.fonts.ready.then(() => measure(tables));
    document.fonts.addEventListener?.('loadingdone', () => measure(tables));    // e.g. inside a <details> opened later
  }
  if (typeof ResizeObserver !== 'function') return;
  const changed = new Set();
  let frame = 0;
  const observer = new ResizeObserver((entries) => {
    for (const e of entries) if (e.target.clientWidth !== widths.get(e.target)) changed.add(e.target);
    if (!changed.size || frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const list = [...changed];
      changed.clear();
      measure(list);
    });
  });
  for (const t of tables) observer.observe(t);
}

// ---------------------------------------------------------------- widgets
async function hydrateWidgets() {
  for (const fig of document.querySelectorAll('figure.widget[data-widget]')) {
    const name = fig.dataset.widget;
    const mountEl = fig.querySelector('.widget-mount');
    try {
      const mod = await import(`../widgets/${name}.js`);
      mod.mount(mountEl, JSON.parse(fig.dataset.config || '{}'));
    } catch (err) {
      mountEl.textContent = `交互图 ${name} 加载失败：${err.message}`;
      console.error(err);
    }
  }
}

// ---------------------------------------------------------------- verify cells
class PyRunner {
  constructor(packages) { this.packages = packages; this.seq = 0; this.pending = new Map(); this.ready = null; }
  call(msg) {
    return new Promise((resolve) => {
      const id = ++this.seq;
      this.pending.set(id, resolve);
      this.worker.postMessage({ ...msg, id });
    });
  }
  init() {
    if (this.ready) return this.ready;
    this.worker = new Worker(new URL('./pyworker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const { id, ...rest } = e.data;
      const resolve = this.pending.get(id);
      if (resolve) { this.pending.delete(id); resolve(rest); }
    };
    this.ready = this.call({
      type: 'init', indexURL: PYODIDE_URL, packages: this.packages,
      harnessURL: new URL('./cell_harness.py', import.meta.url).href,
    }).then((r) => { if (r.error) { this.ready = null; throw new Error(r.error); } return r; });
    return this.ready;
  }
  async run(cell) {
    await this.init();
    return this.call({ type: 'run', cellId: cell.id, code: cell.code, files: cell.files });
  }
}

const norm = (s) => (s || '').replace(/[ \t]+$/gm, '').replace(/\s+$/, '');

function setupVerify() {
  const els = [...document.querySelectorAll('.verify')];
  if (!els.length) return;
  const packages = JSON.parse(document.body.dataset.packages || '["numpy"]');
  const runner = new PyRunner(packages);
  const counter = document.querySelector('.verify-counter');
  const runAllBtn = document.querySelector('.run-all');
  const cells = els.map((el) => {
    const data = JSON.parse(el.dataset.cell);
    return {
      el, id: data.id, golden: data.golden, files: data.files,
      code: el.querySelector('.verify-src').value,
      seal: el.querySelector('.seal'), out: el.querySelector('.verify-out'), state: 'idle',
    };
  });

  const updateCounter = () => {
    const passed = cells.filter((c) => c.state === 'pass').length;
    counter.hidden = false;
    counter.textContent = `本页验证 ${passed} / ${cells.length}`;
  };
  const setState = (c, state, label) => {
    c.state = state;
    c.seal.dataset.state = state;
    c.seal.dataset.label = label;
    if (state === 'pass') {
      c.seal.classList.remove('stamped');
      void c.seal.offsetWidth; // restart the stamp animation
      c.seal.classList.add('stamped');
    }
    updateCounter();
  };
  const showOutput = (c, html) => { c.out.innerHTML = html; };
  const esc = (s) => s.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));

  let queue = Promise.resolve();
  const runOne = async (c) => {
    setState(c, 'running', '运行中');
    if (!runner.ready) showOutput(c, '<div class="verify-out-label">正在浏览器里启动 Python（Pyodide，首次约需十几秒）…</div>');
    let r;
    try {
      r = await Promise.race([
        runner.run(c),
        new Promise((_, reject) => setTimeout(() => reject(new Error('等待超过 90 秒：网络可能拦截了 cdn.jsdelivr.net，或浏览器禁用了 WebAssembly。')), 90000)),
      ]);
    } catch (err) {
      setState(c, 'fail', '未运行');
      showOutput(c, `<div class="verify-out-label">无法启动浏览器内的 Python。可以复制代码，用本地 Python 运行。</div><pre class="err">${esc(err.message)}</pre>`);
      return 'fatal';
    }
    const label = '<div class="verify-out-label">你的浏览器刚刚运行的输出</div>';
    if (r.error) {
      setState(c, 'fail', '未通过');
      showOutput(c, `${label}<pre>${esc(r.stdout || '')}</pre><pre class="err">${esc(r.error)}</pre>`);
      return false;
    }
    if (norm(r.stdout) !== norm(c.golden)) {
      setState(c, 'mismatch', '输出不同');
      showOutput(c, `${label}<pre>${esc(r.stdout)}</pre><div class="verify-out-label diff">与构建时的输出不一致，构建时为：</div><pre class="diff">${esc(c.golden)}</pre>`);
      return true;
    }
    setState(c, 'pass', '已验证');
    showOutput(c, `${label}<pre>${esc(r.stdout || '（无输出，断言全部通过）')}</pre>`);
    return true;
  };
  // Running cell k first runs every earlier cell that has not run yet (they share one namespace).
  const runUpTo = (k) => {
    queue = queue.then(async () => {
      for (let i = 0; i <= k; i++) {
        const c = cells[i];
        if (i < k && (c.state === 'pass' || c.state === 'mismatch')) continue;
        const ok = await runOne(c);
        if (ok === true) continue;
        // A failed prerequisite (or a runtime that cannot start) means later cells cannot run either.
        for (let j = i + 1; j <= k; j++) {
          if (cells[j].state !== 'pass') {
            setState(cells[j], 'fail', '未运行');
            showOutput(cells[j], `<div class="verify-out-label">前面的验证 ${cells[i].id} 没有通过，这一段没有运行。</div>`);
          }
        }
        break;
      }
    });
    return queue;
  };

  cells.forEach((c, k) => {
    c.seal.dataset.label = '未运行';
    c.el.querySelector('.verify-run').addEventListener('click', () => runUpTo(k));
    c.seal.addEventListener('click', () => runUpTo(k));
    c.seal.setAttribute('role', 'button');
    c.seal.tabIndex = 0;
    c.seal.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); runUpTo(k); } });
  });
  runAllBtn.hidden = false;
  runAllBtn.addEventListener('click', () => runUpTo(cells.length - 1));
  updateCounter();
}

setupChrome();
renderMath();
fitTables();                                     // after the math: formulas in cells change the widths
setupVerify();
hydrateWidgets();
