/* 图：BPE 单步训练器（1.3.3 节）。
   每一步数一遍相邻 pair，合并出现次数最多的一对（平票规则可选）；对照表同时跑三种平票规则。
   训练由 _bpe.js 的 BPETrainer 完成，它与原书 ch01/03–06 和 codebot 的训练函数逐条相同
   （三种规则都对拍过，tests/js/lib-bpe.test.mjs）。 */
import { h, fmt } from './_lib.js';
import { BPETrainer, ENDOFTEXT, pretokenize, splitSpecial } from './_bpe.js';
import { injectStyle, tokenLabel, bytesRepr, presetButtons } from './ch01-util.js';

export const RULES = [
  { value: 'first', label: '先出现者胜（ch01/03–06）' },
  { value: 'ids', label: '比 id（codebot）' },
  { value: 'bytes', label: '比字节串（CS336）' },
];
const SHORT = { first: '先出现', ids: '比 id', bytes: '比字节串' };

const Z = ENDOFTEXT;
export const PRESETS = [
  { label: '原书 03 的例句', text: 'Hello world! This is BPE training.', pretokenize: false },
  { label: 'zz 与 aa（平票）', text: ['zzq', 'zzq', 'zzq', 'zz', 'zz', 'zz', 'aaq', 'aaq', 'aaq', 'aa', 'aa'].join(Z), pretokenize: false },
  { label: '重叠：aaaaaaa', text: 'aaaaaaa', pretokenize: false },
  { label: '原书 06 的例句', text: `Say hello! Why hello? Just hello.${Z}Good morning!`, pretokenize: true },
  { label: 'Python 代码', text: 'for i in range(10):\n    if i % 2 == 0:\n        print(i)\n    else:\n        print(-i)\n', pretokenize: true },
];

/** Three trainers (one per tie-break) that are stepped together. Pure: tested without a DOM. */
export class Race {
  constructor(text, pre) {
    this.text = text;
    this.pre = pre;
    this.trainers = Object.fromEntries(RULES.map((r) => [r.value, new BPETrainer(text, { tieBreak: r.value, pretokenize: pre })]));
    this.history = Object.fromEntries(RULES.map((r) => [r.value, []]));
    this.start = this.trainers.ids.tokens;
  }

  get steps() { return this.history.ids.length; }

  /** One merge for every rule; each history entry gets the number of places actually replaced. */
  step() {
    let any = false;
    for (const r of RULES) {
      const t = this.trainers[r.value];
      const before = t.tokens;
      const rec = t.step();
      if (!rec) continue;
      any = true;
      this.history[r.value].push({ ...rec, replaced: before - rec.tokens, bytes: t.vocab[rec.newId] });
    }
    return any;
  }

  /** The pair each tie-break would merge next, judged on the state of `rule`'s trainer. */
  nextPicks(rule) {
    const t = this.trainers[rule];
    const keep = t.tieBreak;
    const picks = {};
    for (const r of RULES) {
      t.tieBreak = r.value;
      picks[r.value] = t.best();
    }
    t.tieBreak = keep;
    return picks;
  }

  /** First step (0-based) where the three rules learned different tokens, or -1. */
  firstDivergence() {
    for (let k = 0; k < this.steps; k++) {
      const toks = RULES.map((r) => this.history[r.value][k]?.bytes).map((b) => (b ? [...b].join(',') : ''));
      if (new Set(toks).size > 1) return k;
    }
    return -1;
  }

  /** The whole text in the current tokens of `rule`: [{ids, special}] per pre-token or special token. */
  segmentation(rule) {
    const t = this.trainers[rule];
    const wordIds = new Map(t.words.map((w) => [w.text, w.ids]));
    const out = [];
    splitSpecial(this.text, [ENDOFTEXT]).forEach((part, i) => {
      if (i % 2) { out.push({ special: true, ids: [] }); return; }
      const pieces = this.pre ? pretokenize(part) : (part ? [part] : []);
      for (const p of pieces) out.push({ special: false, ids: wordIds.get(p) });
    });
    return out;
  }
}

export function mount(root, config = {}) {
  injectStyle();
  let rule = config.rule || 'first';
  const preset = PRESETS[config.preset ?? 0];
  const input = h('textarea', { class: 'w-input', rows: 2, spellcheck: 'false', 'aria-label': '训练用的文字，可以编辑' });
  input.value = config.text ?? preset.text;
  const pre = h('input', { type: 'checkbox', checked: config.pretokenize ?? preset.pretokenize, 'aria-label': '先做预分词（1.6 节）' });
  const select = h('select', { 'aria-label': '平票规则' }, RULES.map((r) => h('option', { value: r.value, selected: r.value === rule }, r.label)));
  const stepBtn = h('button', { type: 'button' }, '下一步');
  const tenBtn = h('button', { type: 'button' }, '走 10 步');
  const resetBtn = h('button', { type: 'button' }, '重置');
  const status = h('div', { class: 'w-ch01-stats', 'aria-live': 'polite' });
  const chips = h('div', { class: 'tokens w-ch01-tokens' });
  const candTitle = h('div', { class: 'w-label' });
  const candidates = h('div', { class: 'w-ch01-wrap' });
  const picks = h('div', { class: 'w-ch01-note' });
  const history = h('div', { class: 'w-ch01-history' });
  const compare = h('div', { class: 'w-ch01-wrap' });

  root.append(
    presetButtons(PRESETS, (p) => { input.value = p.text; pre.checked = p.pretokenize; reset(); }, '预设的训练文字'),
    input,
    h('div', { class: 'w-ch01-bar' }, h('label', {}, '平票规则', select), h('label', {}, pre, '预分词'), stepBtn, tenBtn, resetBtn),
    status, chips, candTitle, candidates, picks,
    h('div', { class: 'w-label' }, '已经学到的规则（当前平票规则）'), history,
    h('div', { class: 'w-label' }, '三种平票规则并排：每一步各自学到的 token'), compare,
  );

  let race;
  function reset() {
    race = new Race(input.value, pre.checked);
    render();
  }

  function render() {
    const t = race.trainers[rule];
    const hist = race.history[rule];
    const exhausted = !t.best();
    status.replaceChildren(
      '学到 ', h('b', {}, hist.length), ' 条规则；文字现在是 ', h('b', {}, fmt(t.tokens)), ' 个 token（开始时 ',
      fmt(race.start), ' 个，', fmt(t.textBytes), ' 个字节）', exhausted ? '；已经没有可合并的 pair 了' : '',
    );
    stepBtn.disabled = tenBtn.disabled = exhausted && !RULES.some((r) => race.trainers[r.value].best());

    // the text in its current tokens
    let n = 0;
    chips.replaceChildren(...race.segmentation(rule).flatMap((seg) => {
      if (seg.special) return [h('span', { class: 'tok w-ch01-special', title: '特殊 token：训练时先在这里切开' }, h('b', {}, ENDOFTEXT), h('i', {}, '特殊'))];
      return seg.ids.map((id) => {
        const lab = tokenLabel(t.vocab[id]);
        return h('span', { class: `tok c${n++ % 6}`, title: `id ${id}` }, h('b', {}, lab.text), h('i', {}, id));
      });
    }));

    // candidates for the next merge
    const top = t.top(6);
    const best = t.best();
    candTitle.textContent = best ? `下一步的候选（出现次数最多的 ${top.length} 个 pair）` : '';
    candidates.replaceChildren(best ? h('table', { class: 'w-ch01-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'pair'), h('th', {}, '合并成'), h('th', { class: 'w-ch01-num' }, '出现次数'), h('th', {}, ''))),
      h('tbody', {}, top.map(({ pair, count }) => {
        const chosen = pair[0] === best.pair[0] && pair[1] === best.pair[1];
        const merged = new Uint8Array([...t.vocab[pair[0]], ...t.vocab[pair[1]]]);
        return h('tr', { class: chosen ? 'pick' : '' },
          h('td', {}, h('code', {}, `(${pair[0]}, ${pair[1]})`)),
          h('td', {}, h('code', {}, bytesRepr(merged))),
          h('td', { class: 'w-ch01-num' }, count),
          h('td', {}, chosen ? '← 本规则选它' : count === best.count ? '并列' : ''));
      }))) : '');
    if (best && best.tied > 1) {
      const p = race.nextPicks(rule);
      picks.replaceChildren(`${best.tied} 个 pair 并列第一。三种规则会选：`,
        ...RULES.flatMap((r, i) => [i ? '；' : '', `${SHORT[r.value]} → `, h('code', {}, bytesRepr(new Uint8Array([...t.vocab[p[r.value].pair[0]], ...t.vocab[p[r.value].pair[1]]])))]));
    } else {
      picks.replaceChildren(best ? '没有并列，三种规则会选同一个 pair。' : '');
    }

    // history of the selected rule
    history.replaceChildren(...hist.map((rec) => h('div', {},
      `第 ${rec.step + 1} 步：`, h('code', {}, bytesRepr(t.vocab[rec.pair[0]])), ' + ', h('code', {}, bytesRepr(t.vocab[rec.pair[1]])),
      ' → ', h('code', {}, bytesRepr(rec.bytes)), `（id ${rec.newId}）出现 ${rec.count} 次`,
      rec.replaced < rec.count ? h('span', { class: 'w-ch01-bad' }, `，只替换了 ${rec.replaced} 处（重叠）`) : `，替换 ${rec.replaced} 处`,
      rec.tied > 1 ? `；${rec.tied} 个并列` : '')));
    if (!hist.length) history.replaceChildren(h('div', { class: 'w-ch01-note' }, '还没有开始。点“下一步”。'));

    // the three rules side by side
    const div = race.firstDivergence();
    compare.replaceChildren(race.steps ? h('table', { class: 'w-ch01-table' },
      h('thead', {}, h('tr', {}, h('th', {}, '步'), RULES.map((r) => h('th', {}, SHORT[r.value])))),
      h('tbody', {}, Array.from({ length: race.steps }, (_, k) => {
        const cells = RULES.map((r) => race.history[r.value][k]);
        const keys = cells.map((c) => (c ? bytesRepr(c.bytes) : '—'));
        const differ = new Set(keys).size > 1;
        return h('tr', { class: k === div ? 'first-diff' : '' }, h('td', { class: 'w-ch01-num' }, k + 1),
          keys.map((key) => h('td', { class: differ ? 'diff' : '' }, h('code', {}, key))));
      }))) : '',
    race.steps ? h('div', { class: 'w-ch01-note' }, div < 0 ? '到目前为止，三种规则学到的 token 完全相同。'
      : `第 ${div + 1} 步起三种规则开始分叉（橙色）。`) : '');
  }

  stepBtn.addEventListener('click', () => { race.step(); render(); });
  tenBtn.addEventListener('click', () => { for (let i = 0; i < 10 && race.step(); i++); render(); });
  resetBtn.addEventListener('click', reset);
  select.addEventListener('change', () => { rule = select.value; render(); });
  pre.addEventListener('change', reset);
  let timer = 0;
  input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(reset, 250); });
  reset();
}
