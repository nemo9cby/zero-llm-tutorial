/* P.9 Parity with PyTorch: re-run the prompts of ref.json (written by tools/export_codebot.py with the book's own
   codebot.model.GPT) in this browser and compare logits, greedy tokens, attention and the logit lens.
   "int8" = PyTorch with exactly the weights this page reconstructs (checks the implementation);
   "float32" = PyTorch with the checkpoint's original weights (adds the effect of the int8 rounding).
   config: { checkpoint, tolerance } */
import { h } from './_lib.js';
import { injectStyle, getIndex, engine, ensureModel, checkpointSwitch, statusLine, DATA } from './playground-common.js';

const NAMES = { code: '续写代码', chat: '对话模板', math: '加法', long: '长 prompt（越过 256）' };

/* The reference greedy sequences keep going past <|endoftext|> (so that every checkpoint compares as many tokens),
   while the book's generate() stops there. A fork from the float32 reference after its first <|endoftext|> (eosAt)
   therefore changes nothing in the answer the book's code would print. */
const forksAfterEnd = (r) => r.greedyF32.firstMismatch >= 0 && r.eosAt >= 0 && r.greedyF32.firstMismatch > r.eosAt;

/** Rows of compareToReference -> how many prompts fork from the float32 reference before / only after <|endoftext|>. */
export function float32Forks(rows) {
  const forks = rows.filter((r) => r.greedyF32.firstMismatch >= 0);
  const late = forks.filter(forksAfterEnd).length;
  return { early: forks.length - late, late };
}

/** The sentence of the verdict about the float32 weights (the effect of int8 rounding, not of the implementation). */
export function float32Verdict({ early, late }) {
  if (!early && !late) return '换成原始的 float32 权重，贪心生成的序列也全部相同。';
  return '换成原始的 float32 权重时，'
    + (early ? `有 ${early} 个 prompt 贪心生成的序列在中途分岔：这是 int8 量化带来的差别，不是实现的差别。` : '')
    + (late ? `${early ? '另有' : '有'} ${late} 个 prompt 只在 <|endoftext|> 之后分岔：参考序列遇到它也不停，`
      + '原书的 generate() 却在那里就停下了，所以两边的回答相同。' : '');
}

export async function mount(root, config = {}) {
  injectStyle();
  const cfg = { checkpoint: 'grpo', tolerance: 1e-4, ...config };
  const box = h('div', { class: 'w-pg' }, h('p', { class: 'w-pg-note' }, '正在准备……'));
  root.append(box);
  let entries;
  try {
    entries = await getIndex();
  } catch (err) {
    box.replaceChildren(h('p', { class: 'w-pg-error' }, `这张图没能启动：${err.message}`));
    return;
  }
  let name = cfg.checkpoint, busy = false;
  const sw = checkpointSwitch(entries, name, (n) => { name = n; });
  const backend = h('select', { 'aria-label': '计算后端' }, h('option', { value: 'auto' }, 'WASM SIMD（默认）'), h('option', { value: 'js' }, '纯 JavaScript'));
  const long = h('input', { type: 'checkbox', checked: true });
  const runBtn = h('button', { type: 'button', class: 'w-pg-btn primary' }, '与 PyTorch 对拍');
  const status = statusLine();
  const table = h('div', { class: 'w-pg-scroll' });
  const verdict = h('div', { class: 'w-label', 'aria-live': 'polite' });
  box.replaceChildren(h('div', { class: 'w-row' }, sw.el),
    h('div', { class: 'w-row' }, h('label', { class: 'w-pg-field' }, '后端', backend),
      h('label', { class: 'w-pg-field' }, long, '包括 251 个 token 的长 prompt（WASM 约 10 秒，纯 JS 约 1 分钟）')),
    h('div', { class: 'w-row' }, runBtn), status.el, table, verdict);

  const e = (x) => (x == null ? '—' : x.toExponential(1));
  const agree = (a, late = '') => `${a.same}/${a.total}${a.firstMismatch >= 0 ? `（第 ${a.firstMismatch + 1} 个起不同${late}）` : ''}`;
  const agreeF32 = (r) => agree(r.greedyF32, forksAfterEnd(r) ? `，已在第 ${r.eosAt + 1} 个的 <|endoftext|> 之后` : '');

  const COLS = ['prompt', 'token 数', 'max|Δlogit|（int8 权重）', '贪心解码一致（int8）', 'max|Δlogit|（float32 权重）',
    '贪心解码一致（float32）', '注意力 max|Δ|', 'logit lens', '用时'];
  function render(rows) {
    table.replaceChildren(h('table', { class: 'w-pg-table w-pg-rtable' },
      h('thead', {}, h('tr', {}, COLS.map((t) => h('th', {}, t)))),
      h('tbody', {}, rows.map((r) => h('tr', {}, [
        NAMES[r.name] || r.name, `${r.n} + ${r.newTokens}`, e(r.maxDiffQ8), agree(r.greedyQ8), r.maxDiffF32.toFixed(3),
        agreeF32(r), e(r.attentionMaxDiff), r.lensTop1 ? `第一名全同，概率差 ≤ ${e(r.lensMaxDiff)}` : '第一名不同',
        `${(r.ms / 1000).toFixed(1)} s`,
      ].map((v, j) => h('td', { 'data-label': COLS[j] }, v)))))));
  }

  runBtn.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    runBtn.disabled = true;
    sw.buttons.forEach((b) => { b.disabled = true; });
    verdict.textContent = '';
    try {
      await ensureModel(name, status);
      const entry = entries.find((c) => c.name === name);
      status.say(`${entry.label}：逐个 prompt 重算并与 PyTorch 的记录比较……`);
      const rows = [];
      const handle = await engine().call('parity', name, {
        refURL: new URL(entry.ref, DATA).href, backend: backend.value, skip: long.checked ? [] : ['long'],
      }, (ev) => { rows.push(ev.result); render(rows); });
      const res = await handle.promise;
      const ok = rows.every((r) => r.maxDiffQ8 < cfg.tolerance && r.greedyQ8.firstMismatch === -1 && r.lensTop1);
      const maxQ8 = Math.max(...rows.map((r) => r.maxDiffQ8));
      verdict.textContent = (ok
        ? `实现一致：与使用同一份 int8 权重的 PyTorch 相比，logits 最多差 ${maxQ8.toExponential(1)}（容差 ${cfg.tolerance}），贪心生成的 token 全部相同。`
        : `出现了超出容差的差异（最大 ${maxQ8.toExponential(1)}），说明实现有问题。`)
        + float32Verdict(float32Forks(rows))
        + `（后端：${res.backend === 'wasm-simd' ? 'WASM SIMD' : '纯 JavaScript'}）`;
      status.say('完成。');
    } catch (err) {
      status.error(err);
    } finally {
      busy = false;
      runBtn.disabled = false;
      sw.buttons.forEach((b) => { b.disabled = false; });
    }
  });
  status.say('点按钮开始；参考数据是 PyTorch 在导出权重时记录下来的（ref.json）。');
}
