// Headless browser check for built pages.
//
//   node tools/check_pages.mjs <site-dir> [slug ...] [--no-cells] [--shots <dir> [--widget-shots]]
//
// For each page: load it, fail on console errors / page errors / failed requests,
// wait for every widget to mount, then click "运行本页全部验证" and wait until every
// verify cell's seal reaches a terminal state. Exit code 1 if any page fails.
// With --shots, also saves desktop (1280px) and phone (375px) screenshots.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith('--')));
const shotsIdx = args.indexOf('--shots');
const shotsDir = shotsIdx >= 0 ? args[shotsIdx + 1] : null;
const positional = args.filter((a, i) => !a.startsWith('--') && !(shotsIdx >= 0 && i === shotsIdx + 1));
const siteDir = path.resolve(positional[0] || 'site');
let slugs = positional.slice(1);
if (!slugs.length) {
  slugs = fs.readdirSync(siteDir).filter((f) => f.endsWith('.html')).map((f) => f.slice(0, -5));
}

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.py': 'text/plain; charset=utf-8', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.bin': 'application/octet-stream', '.wasm': 'application/wasm', '.txt': 'text/plain; charset=utf-8' };

function serve(root) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = path.join(root, url === '/' ? 'index.html' : url);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// --widget-shots: one screenshot per figure.widget, in light and dark, named <slug>-fig<N>-<width>-<theme>.png
async function widgetShots(page, slug, width) {
  const figs = await page.$$('figure.widget');
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);
    await page.waitForTimeout(250); // charts re-render on theme change
    for (let i = 0; i < figs.length; i++) {
      await figs[i].scrollIntoViewIfNeeded();
      await figs[i].screenshot({ path: path.join(shotsDir, `${slug}-fig${i + 1}-${width}-${theme}.png`) });
    }
  }
  await page.evaluate(() => { delete document.documentElement.dataset.theme; });
}

async function checkPage(browser, base, slug) {
  const problems = [];
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('requestfailed', (r) => {
    const u = r.url();
    if (!u.includes('fonts.g')) problems.push(`requestfailed: ${u} (${r.failure()?.errorText})`);
  });
  page.on('response', (r) => { if (r.status() >= 400) problems.push(`HTTP ${r.status()}: ${r.url()}`); });

  await page.goto(`${base}/${slug}.html`, { waitUntil: 'load' });
  await page.waitForTimeout(800); // let widget modules import and mount

  const widgets = await page.$$eval('figure.widget', (figs) => figs.map((f) => ({
    name: f.dataset.widget,
    mounted: f.querySelector('.widget-mount').children.length > 0
      && !/加载失败/.test(f.querySelector('.widget-mount').textContent),
  })));
  for (const w of widgets) if (!w.mounted) problems.push(`widget ${w.name} did not mount`);

  // KaTeX renders a formula it cannot parse as red source text (throwOnError: false): count those.
  const badMath = await page.$$eval('.katex-error, .math-error', (els) =>
    els.map((e) => `${(e.getAttribute('title') || 'parse error').slice(0, 120)} :: ${e.textContent.slice(0, 60)}`));
  for (const m of badMath) problems.push(`math: ${m}`);

  const overflow = async (label) => {
    const wide = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (wide > 1) problems.push(`horizontal overflow ${wide}px at ${label}`);
    // Elements sticking out of their figure (a few px of axis label slip past the page-level check),
    // unless an ancestor inside the figure scrolls or clips them on purpose.
    const sticking = await page.evaluate(() => {
      const out = [];
      document.querySelectorAll('figure.widget').forEach((fig, i) => {
        const box = fig.getBoundingClientRect();
        for (const el of fig.querySelectorAll('*')) {
          const r = el.getBoundingClientRect();
          if (!r.width || (r.right <= box.right + 2 && r.left >= box.left - 2)) continue;
          let clipped = false;
          for (let a = el.parentElement; a && a !== fig; a = a.parentElement) {
            const ox = getComputedStyle(a).overflowX;
            if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') { clipped = true; break; }
          }
          if (!clipped) {
            out.push(`figure ${i + 1} <${el.tagName.toLowerCase()}> sticks out by ${Math.round(Math.max(r.right - box.right, box.left - r.left))}px`);
            break;
          }
        }
      });
      return out;
    });
    for (const s of sticking) problems.push(`${s} at ${label}`);
  };
  await overflow('1280px');

  let cells = { total: 0, pass: 0, other: [] };
  const nCells = await page.$$eval('.verify', (els) => els.length);
  if (nCells && !flags.has('--no-cells')) {
    await page.click('.run-all');
    try {
      await page.waitForFunction(() => [...document.querySelectorAll('.seal')]
        .every((s) => ['pass', 'mismatch', 'fail'].includes(s.dataset.state)), null, { timeout: 240000, polling: 500 });
    } catch (e) { problems.push('timeout waiting for verify cells'); }
    cells = await page.$$eval('.verify', (els) => {
      const out = { total: els.length, pass: 0, other: [] };
      for (const el of els) {
        const st = el.querySelector('.seal').dataset.state;
        if (st === 'pass') out.pass += 1;
        else out.other.push(`${el.id}: ${st} — ${el.querySelector('.verify-out').innerText.slice(0, 400)}`);
      }
      return out;
    });
    for (const o of cells.other) problems.push(`cell ${o}`);
  }

  if (shotsDir) {
    fs.mkdirSync(shotsDir, { recursive: true });
    await page.screenshot({ path: path.join(shotsDir, `${slug}-desktop.png`), fullPage: false });
    if (flags.has('--widget-shots')) await widgetShots(page, slug, 'desktop');
  }
  await page.setViewportSize({ width: 375, height: 812 });
  await page.waitForTimeout(300);
  await overflow('375px');
  if (shotsDir) {
    await page.screenshot({ path: path.join(shotsDir, `${slug}-phone.png`), fullPage: false });
    if (flags.has('--widget-shots')) await widgetShots(page, slug, 'phone');
  }
  await page.close();
  return { slug, widgets: widgets.length, cells, problems };
}

const server = await serve(siteDir);
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();
let failed = 0;
try {
  for (const slug of slugs) {
    const r = await checkPage(browser, base, slug);
    const status = r.problems.length ? 'FAIL' : 'ok';
    if (r.problems.length) failed += 1;
    console.log(`${status.padEnd(4)} ${slug}: ${r.widgets} widgets, cells ${r.cells.pass}/${r.cells.total} stamped`);
    for (const p of r.problems) console.log(`     - ${p}`);
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
