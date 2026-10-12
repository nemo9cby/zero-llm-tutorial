/* 图：软字典（2.1 节）。七部电影的权重随查询、温度、相似度的定义变化；可把相似度放大 10 倍，
   观察不减最大值的 softmax 何时溢出成 NaN。数值与验证 2.1-b/c/d/e 相同（JS 测试对拍）。 */
import { h } from './_lib.js';
import { softmax, injectStyle } from './ch02-util.js';

export const MOVIES = [
  { key: [8, 2, 3], value: 85, name: '偏动作' },
  { key: [3, 9, 1], value: 70, name: '偏剧情' },
  { key: [1, 2, 9], value: 60, name: '偏喜剧' },
  { key: [5, 5, 5], value: 75, name: '均衡型' },
  { key: [7, 6, 2], value: 80, name: '动作剧情' },
  { key: [2, 7, 6], value: 65, name: '喜剧剧情' },
  { key: [9, 1, 1], value: 90, name: '纯动作' },
];

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));

/** Similarity of query q and key k: 'dot' (the book), 'cos', or 'dist' (negative Euclidean distance). */
export function similarity(kind, q, k) {
  if (kind === 'cos') return dot(q, k) / (norm(q) * norm(k) || 1);
  if (kind === 'dist') return -Math.hypot(...q.map((v, i) => v - k[i]));
  return dot(q, k);
}

/** The soft dictionary: scores -> softmax(scores * scale / T) -> weighted sum of the ratings. */
export function softDict(q, { T = 1, kind = 'dot', scale = 1, naive = false } = {}) {
  const scores = MOVIES.map((m) => similarity(kind, q, m.key) * scale);
  const weights = softmax(scores.map((s) => s / T), naive);
  const result = weights.reduce((acc, w, i) => acc + w * MOVIES[i].value, 0);
  return { scores, weights, result };
}

const fmtScore = (s) => (Number.isInteger(s) ? String(s) : s.toFixed(2));

export function mount(root) {
  injectStyle();
  const q = [6, 4, 5];
  const state = { logT: 0, kind: 'dot', scale: 1, naive: false };

  const sliders = ['动作', '剧情', '喜剧'].map((name, i) => {
    const out = h('span', { class: 'w-ch02-num' }, q[i]);
    const input = h('input', { type: 'range', min: 0, max: 10, step: 1, value: q[i], 'aria-label': `新电影的${name}性（0–10）` });
    input.addEventListener('input', () => { q[i] = +input.value; out.textContent = input.value; render(); });
    return h('label', {}, name, input, out);
  });
  const tOut = h('span', { class: 'w-ch02-num' });
  const tInput = h('input', { type: 'range', min: -2, max: 2, step: 0.05, value: 0, 'aria-label': '温度 T（对数刻度，0.01 到 100）' });
  tInput.addEventListener('input', () => { state.logT = +tInput.value; render(); });
  const kindSel = h('select', { 'aria-label': '相似度的定义' },
    h('option', { value: 'dot' }, '点积（原书）'), h('option', { value: 'cos' }, '余弦相似度'),
    h('option', { value: 'dist' }, '负欧氏距离'));
  kindSel.addEventListener('change', () => { state.kind = kindSel.value; render(); });
  const scaleBox = h('input', { type: 'checkbox', 'aria-label': '把相似度放大 10 倍' });
  scaleBox.addEventListener('change', () => { state.scale = scaleBox.checked ? 10 : 1; render(); });
  const naiveBox = h('input', { type: 'checkbox', 'aria-label': '使用不减最大值的朴素 softmax' });
  naiveBox.addEventListener('change', () => { state.naive = naiveBox.checked; render(); });
  const reset = h('button', { type: 'button' }, '复位');
  reset.addEventListener('click', () => {
    [6, 4, 5].forEach((v, i) => { q[i] = v; sliders[i].querySelector('input').value = v; sliders[i].querySelector('span').textContent = v; });
    Object.assign(state, { logT: 0, kind: 'dot', scale: 1, naive: false });
    tInput.value = 0; kindSel.value = 'dot'; scaleBox.checked = false; naiveBox.checked = false;
    render();
  });

  const bars = h('div', { class: 'w-ch02-bars', role: 'list', 'aria-label': '七部电影的权重' });
  const summary = h('div', { class: 'w-row' });
  const hint = h('p', { class: 'w-ch02-note', 'aria-live': 'polite' });
  root.append(
    h('div', { class: 'w-ch02-controls' }, h('span', { class: 'w-label' }, '新电影'), ...sliders),
    h('div', { class: 'w-ch02-controls' }, h('label', {}, '温度 T', tInput, tOut), kindSel, reset),
    h('div', { class: 'w-ch02-controls' },
      h('label', {}, scaleBox, '相似度 ×10'), h('label', {}, naiveBox, '朴素 softmax（不减最大值）')),
    bars, summary, hint,
  );

  function render() {
    const T = 10 ** state.logT;
    tOut.textContent = T >= 10 ? T.toFixed(0) : T.toFixed(2);
    const { scores, weights, result } = softDict(q, { T, kind: state.kind, scale: state.scale, naive: state.naive });
    const finite = weights.filter(Number.isFinite);
    const top = finite.length ? Math.max(...finite) : NaN;
    bars.replaceChildren(...MOVIES.flatMap((m, i) => {
      const w = weights[i];
      const isNaN_ = !Number.isFinite(w);
      return [
        h('span', { class: 'lab', role: 'listitem', title: `${m.name}，评分 ${m.value}，相似度 ${fmtScore(scores[i])}` },
          h('b', {}, `(${m.key.join(',')})`), ` ${m.name} · ${m.value}分`),
        h('span', { class: 'track', 'aria-hidden': 'true' },
          h('span', { class: `fill${w === top ? ' top' : ''}`, style: { width: `${isNaN_ ? 0 : (w * 100).toFixed(2)}%`, display: 'block' } })),
        h('span', { class: `pct${isNaN_ ? ' w-ch02-bad' : ''}` }, isNaN_ ? 'NaN' : `${(w * 100).toFixed(2)}%`),
      ];
    }));
    const bad = !Number.isFinite(result);
    summary.replaceChildren(
      h('span', { class: 'w-label' }, '预测评分'),
      h('span', { class: `w-ch02-big${bad ? ' w-ch02-bad' : ''}` }, bad ? 'NaN' : result.toFixed(2)),
      h('span', { class: 'w-label' }, `相似度：${scores.map(fmtScore).join('、')}`),
    );
    const maxScore = Math.max(...scores.map((s) => s / T));
    if (bad) hint.textContent = `出现了 NaN：最大的相似度 / T = ${maxScore.toFixed(0)}，超过了 709.8，exp 溢出成 inf，inf / inf = NaN。取消“朴素 softmax”，先减去最大值就没事了。`;
    else if (state.kind === 'cos' && T > 0.2) hint.textContent = '余弦相似度只在 −1 到 1 之间，T = 1 时七个权重几乎一样大；把 T 调到 0.05 左右再比较。';
    else if (T <= 0.02) hint.textContent = 'T 很小：几乎只剩相似度最大的那一部。查询与某个键完全一样时，遇到并列（比如 (5,5,5) 与 (7,6,2)、(2,7,6)），权重会平分。';
    else if (T >= 50) hint.textContent = `T 很大：权重接近平均，预测接近七个评分的平均 75。`;
    else hint.textContent = '点积不只看方向，还看长度：(7,6,2) 比 (5,5,5) 长，所以拿到了最大的权重。';
  }
  render();
}
