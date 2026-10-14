/* 配置翻译器：选一个预设，或者粘贴 Hugging Face 的 config.json，看同一组数在 DLFS6、CS336、nanoGPT 里叫什么。
   config: { data: "data/glossary/sources.json", preset: "olmo2" }
   键名对照来自 verify/_glossary_sources.py 记录的一手来源（HF 的 config.json 与 GPT2Config.attribute_map、
   CS336 A1 的 adapters.py、nanoGPT 的 GPTConfig、DLFS6 的 GPT.__init__）。 */
import { h, fmt } from './_lib.js';

export const ROSETTA = [
  { key: 'vocab', zh: '词表大小', hf: ['vocab_size'], dlfs6: 'vocab_size', cs336: 'vocab_size', nanogpt: 'vocab_size' },
  { key: 'ctx', zh: '上下文长度', hf: ['max_position_embeddings', 'n_positions', 'n_ctx'], dlfs6: 'max_context_len',
    cs336: 'context_length', nanogpt: 'block_size' },
  { key: 'width', zh: '残差流宽度', hf: ['hidden_size', 'n_embd', 'd_model'], dlfs6: 'embed_dim', cs336: 'd_model',
    nanogpt: 'n_embd' },
  { key: 'heads', zh: '注意力头数', hf: ['num_attention_heads', 'n_head'], dlfs6: 'n_head', cs336: 'num_heads',
    nanogpt: 'n_head' },
  { key: 'kv', zh: 'KV 头数', hf: ['num_key_value_heads'], dlfs6: 'n_kv_head', cs336: '（A1 没有）', nanogpt: '（没有）' },
  { key: 'headDim', zh: '每头维数', hf: ['head_dim'], dlfs6: '（embed_dim // n_head）', cs336: 'd_k',
    nanogpt: '（n_embd // n_head）' },
  { key: 'layers', zh: '层数', hf: ['num_hidden_layers', 'n_layer'], dlfs6: 'n_layer', cs336: 'num_layers',
    nanogpt: 'n_layer' },
  { key: 'ff', zh: 'FFN 宽度', hf: ['intermediate_size', 'n_inner'], dlfs6: 'ff_dim', cs336: 'd_ff',
    nanogpt: '固定为 4 × n_embd' },
  { key: 'theta', zh: 'RoPE θ', hf: ['rope_theta'], dlfs6: 'theta', cs336: 'rope_theta', nanogpt: '（没有 RoPE）' },
];

/** Resolve each quantity from a HF-style config; fills the documented defaults (n_inner → 4·n_embd, KV → heads,
 *  head_dim → hidden_size / heads when the config does not state it). */
export function translate(cfg) {
  const rows = ROSETTA.map((r) => {
    const hfKey = r.hf.find((k) => cfg[k] !== undefined && cfg[k] !== null);
    return { ...r, hfKey: hfKey || null, value: hfKey ? cfg[hfKey] : null, note: '' };
  });
  const get = (k) => rows.find((r) => r.key === k);
  const width = get('width').value, heads = get('heads').value;
  const ff = get('ff');
  if (ff.value == null && width != null && ('n_embd' in cfg || cfg.model_type === 'gpt2')) {
    ff.value = 4 * width;
    ff.note = 'n_inner 缺省，GPT-2 取 4 × n_embd';
  }
  const kv = get('kv');
  if (kv.value == null && heads != null) {
    kv.value = heads;
    kv.note = '缺省：与头数相同（普通的多头 Attention）';
  }
  const hd = get('headDim');
  if (hd.value == null && width != null && heads) {
    hd.value = width / heads;
    hd.note = '配置里没有，按宽度 ÷ 头数算出';
  }
  const derived = {};
  if (hd.value != null) derived.headDim = hd.value;
  if (hd.hfKey && width != null && heads && hd.value * heads !== width) derived.headDimMismatch = width / heads;
  if (heads && kv.value) derived.groupSize = heads / kv.value;
  if (ff.value != null && width) derived.ffRatio = ff.value / width;
  const L = get('layers').value, V = get('vocab').value;
  if (L != null && width != null) derived.roughNonEmbedding = 12 * L * width * width;
  if (L != null && width != null && heads && ff.value != null) {
    derived.blocks = L * blockParams(cfg, width, heads, kv.value, ff.value);
    derived.shares = blockShares(cfg, width, heads, kv.value, ff.value);
  }
  if (V != null && width != null) derived.embedding = V * width;
  return { rows, derived };
}

const isGated = (cfg) => (cfg.hidden_act ? /silu|swish/i.test(cfg.hidden_act) : 'intermediate_size' in cfg);

/** Parameters of one Transformer block from its matrix shapes (two norms, no biases): Q is d × (H·D) and O is
 *  (H·D) × d, K and V are d × (n_kv·D), the FFN has 3 matrices when gated (SwiGLU) and 2 otherwise.
 *  D is the config's head_dim when it states one, else d / H. */
export function blockParams(cfg, width, heads, kv, ff) {
  const D = cfg.head_dim ?? width / heads;
  return 2 * width * heads * D + 2 * width * (kv ?? heads) * D + (isGated(cfg) ? 3 : 2) * width * ff + 2 * width;
}

/** The same block in units of d²: attention and FFN separately (the rough 12·d² assumes 4 + 8). */
export function blockShares(cfg, width, heads, kv, ff) {
  const D = cfg.head_dim ?? width / heads;
  const nFF = isGated(cfg) ? 3 : 2;
  return { attn: (2 * heads * D + 2 * (kv ?? heads) * D) / width, ffn: nFF * ff / width, ffMatrices: nFF };
}

const num = (x) => String(Number(x.toFixed(2)));

/** Why the exact block count and the rough 12·L·d² differ (null when they agree within 5 %). */
export function roughNote(derived) {
  const { blocks, roughNonEmbedding: rough, shares } = derived;
  if (blocks == null || rough == null || !shares || Math.abs(blocks / rough - 1) <= 0.05) return null;
  const pct = Math.round(Math.abs(1 - rough / blocks) * 100);
  return `粗估 12·L·d² 假设每层 Attention 4d²、FFN 8d²（两个 d × 4d 的矩阵）。这个模型每层约 ${num(shares.attn + shares.ffn)}d²：`
    + `Attention ${num(shares.attn)}d²，FFN ${num(shares.ffn)}d²（${shares.ffMatrices} 个矩阵），`
    + `所以粗估${rough < blocks ? '低' : '高'}了约 ${pct}%。`;
}

/** The DLFS6 constructor call for this config, or a comment when DLFS6's classes cannot express it. */
export function dlfs6Call(rows, cfg = {}) {
  const v = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const gqa = v.kv != null && v.heads != null && v.kv !== v.heads;
  const mod = gqa ? 'webbot' : (v.theta != null ? 'storybot' : 'codebot');
  if (cfg.head_dim != null && v.width != null && v.heads && cfg.head_dim * v.heads !== v.width) {
    return `# ${mod}/model.py 写不出这个模型：它用 embed_dim // n_head 计算每头维数，\n`
      + `# 这里会得到 ${Math.floor(v.width / v.heads)}，而配置写的是 head_dim = ${cfg.head_dim}`;
  }
  const args = [['vocab_size', v.vocab], ['max_context_len', v.ctx], ['embed_dim', v.width], ['n_head', v.heads],
    ...(gqa ? [['n_kv_head', v.kv]] : []), ['n_layer', v.layers], ['ff_dim', v.ff],
    ...(v.theta != null ? [['theta', v.theta]] : []),
    ...(mod === 'codebot' ? [['dropout_rate', cfg.resid_pdrop ?? 0.1]] : [])];   // codebot's GPT has no default
  return `# ${mod}/model.py 的写法\nGPT(${args.map(([k, x]) => `${k}=${x ?? '?'}`).join(', ')})`;
}

export const PRESETS = (src) => [
  { id: 'gpt2', label: 'GPT-2（HF config.json）', cfg: src.hf.gpt2.config },
  { id: 'olmo2', label: 'OLMo-2-0425-1B（CS336 2026 A5）', cfg: src.hf.olmo2_0425_1b.config },
  { id: 'qwen', label: 'Qwen2.5-Math-1.5B（CS336 2025 A5）', cfg: src.hf.qwen25_math_1p5b.config },
  ...(src.hf.qwen3_0p6b ? [{ id: 'qwen3', label: 'Qwen3-0.6B（单列 head_dim）', cfg: src.hf.qwen3_0p6b.config }] : []),
  { id: 'codebot', label: 'CodeBot（换成 GPT-2 的键名）', cfg: { model_type: 'gpt2', vocab_size: 1000, n_positions: 256,
    n_embd: 384, n_head: 6, n_layer: 6, n_inner: 1536 } },
  { id: 'storybot', label: 'StoryBot（换成 HF 的键名）', cfg: { vocab_size: 10000, max_position_embeddings: 256,
    hidden_size: 512, num_attention_heads: 16, num_hidden_layers: 4, intermediate_size: 1344, rope_theta: 10000 } },
  { id: 'webbot', label: 'WebBot（换成 HF 的键名）', cfg: { vocab_size: 50000, max_position_embeddings: 1024,
    hidden_size: 768, num_attention_heads: 12, num_key_value_heads: 4, num_hidden_layers: 12, intermediate_size: 2048,
    rope_theta: 10000 } },
];

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h('style', {}, `
.w-gcfg textarea { width: 100%; min-height: 7.5rem; font: 12.5px/1.5 var(--mono); color: var(--ink); background: var(--paper);
  border: 1px solid var(--rule); border-radius: 4px; padding: .5rem .6rem; resize: vertical; }
.w-gcfg select { font: inherit; font-size: .9rem; color: var(--ink); background: var(--paper); border: 1px solid var(--rule);
  border-radius: 4px; padding: .25rem .4rem; max-width: 100%; }
.w-gcfg table { font-size: .85rem; }
.w-gcfg td code, .w-gcfg th code { white-space: nowrap; }
.w-gcfg td:first-child, .w-gcfg th:first-child { white-space: nowrap; }
.w-gcfg .err { color: var(--kincha); font-size: .85rem; }
.w-gcfg pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.w-gcfg-narrow { display: none; list-style: none; padding: 0; margin: .4rem 0; font-size: .85rem; }
.w-gcfg-narrow li { padding: .3rem 0; border-bottom: 1px solid var(--rule); overflow-wrap: anywhere; }
.w-gcfg-narrow b { font-weight: 600; }
@media (max-width: 560px) {
  .w-gcfg table.w-gcfg-wide { display: none; }
  .w-gcfg-narrow { display: block; }
}
`));
}

export async function mount(root, config) {
  injectStyle();
  let src;
  try {
    const res = await fetch(config.data || 'data/glossary/sources.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    src = await res.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `数据加载失败：${err.message}`));
    return;
  }
  const presets = PRESETS(src);
  const box = h('div', { class: 'w-gcfg' });
  const select = h('select', { 'aria-label': '选择一个预设配置' },
    ...presets.map((p) => h('option', { value: p.id }, p.label)));
  const area = h('textarea', { 'aria-label': 'config.json 的内容（可以修改或粘贴自己的）', spellcheck: 'false' });
  const out = h('div', { 'aria-live': 'polite' });
  box.append(h('div', { class: 'w-row' }, h('label', { class: 'w-label' }, '预设：'), select), area, out);
  root.append(box);

  const code = (text) => (text == null || text === '' ? '—' : h('code', {}, String(text)));
  const cell = (text) => h('td', {}, code(text));
  const render = () => {
    let cfg;
    try {
      cfg = JSON.parse(area.value);
      if (typeof cfg !== 'object' || cfg === null || Array.isArray(cfg)) throw new Error('需要一个 JSON 对象');
    } catch (err) {
      out.replaceChildren(h('p', { class: 'err' }, `这不是合法的 JSON：${err.message}`));
      return;
    }
    const { rows, derived } = translate(cfg);
    const shown = (r) => (r.value == null ? '—' : fmt(r.value));
    const table = h('table', { class: 'w-gcfg-wide' },
      h('thead', {}, h('tr', {}, ...['含义', '这份 config', '值', 'DLFS6', 'CS336 A1', 'nanoGPT'].map((t) => h('th', {}, t)))),
      h('tbody', {}, ...rows.map((r) => h('tr', {},
        h('td', {}, r.zh), cell(r.hfKey), h('td', {}, shown(r), r.note ? h('div', { class: 'w-label' }, r.note) : null),
        cell(r.dlfs6), cell(r.cs336), cell(r.nanogpt)))));
    // narrow screens: one quantity per line instead of a six-column table
    const narrow = h('ul', { class: 'w-gcfg-narrow' }, ...rows.map((r) => h('li', {},
      h('b', {}, `${r.zh}：`), ...(r.hfKey ? [code(r.hfKey), ' = '] : []), `${shown(r)} → DLFS6 `, code(r.dlfs6),
      ' · CS336 ', code(r.cs336), ' · nanoGPT ', code(r.nanogpt), r.note ? h('div', { class: 'w-label' }, r.note) : null)));
    const facts = [];
    if (derived.headDimMismatch != null) {
      facts.push(`head_dim（${fmt(derived.headDim)}）≠ 残差流宽度 ÷ 头数（${fmt(derived.headDimMismatch)}）：`
        + 'DLFS6 的类用 embed_dim // n_head 计算每头维数，写不出这个模型');
    } else if (derived.headDim != null) {
      facts.push(`每头维数 D = 残差流宽度 ÷ 头数 = ${fmt(derived.headDim)}`);
    }
    if (derived.groupSize != null && derived.groupSize !== 1) facts.push(`GQA：每 ${fmt(derived.groupSize)} 个查询头共用一组 K、V`);
    if (derived.ffRatio != null) facts.push(`FFN 宽度是残差流宽度的 ${derived.ffRatio.toFixed(2)} 倍`);
    if (derived.blocks != null) facts.push(`全部 Transformer 块的参数（按矩阵形状算，每层含两个 norm，不含 bias）：${fmt(derived.blocks)}`
      + `；常用的粗估 12 · L · d² = ${fmt(derived.roughNonEmbedding)}`);
    const why = roughNote(derived);
    if (why) facts.push(why);
    if (derived.embedding != null) facts.push(`一张 embedding 表 V · d = ${fmt(derived.embedding)}`);
    out.replaceChildren(table, narrow, h('ul', {}, ...facts.map((f) => h('li', {}, f))),
      h('pre', { class: 'cmd' }, h('code', {}, dlfs6Call(rows, cfg))));
  };
  const load = () => {
    const p = presets.find((x) => x.id === select.value) || presets[0];
    area.value = JSON.stringify(p.cfg, null, 2);
    render();
  };
  select.addEventListener('change', load);
  area.addEventListener('input', render);
  select.value = config.preset || 'gpt2';
  load();
}
