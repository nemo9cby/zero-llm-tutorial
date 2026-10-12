/* 配置翻译器：选一个预设，或者粘贴 Hugging Face 的 config.json，看同一组数在 DLFS6、CS336、nanoGPT 里叫什么。
   config: { data: "data/glossary/sources.json" }
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
  { key: 'layers', zh: '层数', hf: ['num_hidden_layers', 'n_layer'], dlfs6: 'n_layer', cs336: 'num_layers',
    nanogpt: 'n_layer' },
  { key: 'ff', zh: 'FFN 宽度', hf: ['intermediate_size', 'n_inner'], dlfs6: 'ff_dim', cs336: 'd_ff',
    nanogpt: '固定为 4 × n_embd' },
  { key: 'theta', zh: 'RoPE θ', hf: ['rope_theta'], dlfs6: 'theta', cs336: 'rope_theta', nanogpt: '（没有 RoPE）' },
];

/** Resolve each quantity from a HF-style config; fills the documented defaults (n_inner → 4·n_embd, KV → heads). */
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
    kv.note = '缺省：与头数相同（普通多头注意力）';
  }
  const derived = {};
  if (width != null && heads) derived.headDim = width / heads;
  if (heads && kv.value) derived.groupSize = heads / kv.value;
  if (ff.value != null && width) derived.ffRatio = ff.value / width;
  const L = get('layers').value, V = get('vocab').value;
  if (L != null && width != null) derived.roughNonEmbedding = 12 * L * width * width;
  if (L != null && width != null && heads && ff.value != null) derived.blocks = L * blockParams(cfg, width, heads, kv.value, ff.value);
  if (V != null && width != null) derived.embedding = V * width;
  return { rows, derived };
}

/** Parameters of one Transformer block from its matrix shapes (two norms, no biases):
 *  Q and O are d×d, K and V are d×(n_kv·D), the FFN has 3 matrices when gated (SwiGLU) and 2 otherwise. */
export function blockParams(cfg, width, heads, kv, ff) {
  const kvDim = (kv ?? heads) * (width / heads);
  const gated = cfg.hidden_act ? /silu|swish/i.test(cfg.hidden_act) : 'intermediate_size' in cfg;
  return 2 * width * width + 2 * width * kvDim + (gated ? 3 : 2) * width * ff + 2 * width;
}

export function dlfs6Call(rows) {
  const v = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const gqa = v.kv != null && v.heads != null && v.kv !== v.heads;
  const args = [['vocab_size', v.vocab], ['max_context_len', v.ctx], ['embed_dim', v.width], ['n_head', v.heads],
    ...(gqa ? [['n_kv_head', v.kv]] : []), ['n_layer', v.layers], ['ff_dim', v.ff],
    ...(v.theta != null ? [['theta', v.theta]] : [])];
  const mod = gqa ? 'webbot' : (v.theta != null ? 'storybot' : 'codebot');
  return `# ${mod}/model.py 的写法\nGPT(${args.map(([k, x]) => `${k}=${x ?? '?'}`).join(', ')})`;
}

const PRESETS = (src) => [
  { id: 'gpt2', label: 'GPT-2（HF config.json）', cfg: src.hf.gpt2.config },
  { id: 'olmo2', label: 'OLMo-2-0425-1B（CS336 2026 A5）', cfg: src.hf.olmo2_0425_1b.config },
  { id: 'qwen', label: 'Qwen2.5-Math-1.5B（CS336 2025 A5）', cfg: src.hf.qwen25_math_1p5b.config },
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
.w-gcfg .err { color: var(--shu); font-size: .85rem; }
.w-gcfg pre { white-space: pre-wrap; overflow-wrap: anywhere; }
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

  const cell = (text) => h('td', {}, text == null || text === '' ? '—' : h('code', {}, String(text)));
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
    const table = h('table', {},
      h('thead', {}, h('tr', {}, ...['含义', '这份 config', '值', 'DLFS6', 'CS336 A1', 'nanoGPT'].map((t) => h('th', {}, t)))),
      h('tbody', {}, ...rows.map((r) => h('tr', {},
        h('td', {}, r.zh), cell(r.hfKey), h('td', {}, r.value == null ? '—' : fmt(r.value), r.note ? h('div', { class: 'w-label' }, r.note) : null),
        cell(r.dlfs6), cell(r.cs336), cell(r.nanogpt)))));
    const facts = [];
    if (derived.headDim != null) facts.push(`每头维数 D = 残差流宽度 ÷ 头数 = ${fmt(derived.headDim)}`);
    if (derived.groupSize != null && derived.groupSize !== 1) facts.push(`GQA：每 ${fmt(derived.groupSize)} 个查询头共用一组 K、V`);
    if (derived.ffRatio != null) facts.push(`FFN 宽度是残差流宽度的 ${derived.ffRatio.toFixed(2)} 倍`);
    if (derived.blocks != null) facts.push(`全部 Transformer 块的参数（按矩阵形状算，每层含两个 norm，不含 bias）：${fmt(derived.blocks)}`
      + `；常用的粗估 12 · L · d² = ${fmt(derived.roughNonEmbedding)}`);
    if (derived.embedding != null) facts.push(`一张嵌入表 V · d = ${fmt(derived.embedding)}`);
    out.replaceChildren(table, h('ul', {}, ...facts.map((f) => h('li', {}, f))), h('pre', { class: 'cmd' }, h('code', {}, dlfs6Call(rows))));
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
