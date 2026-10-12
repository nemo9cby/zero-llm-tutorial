/* 图：训练好的 CodeBot 的位置向量（2.5.4 节）。data/ch02/pos_embed_cos.json 由 model_pretrain.pt 导出，
   verify/test_ch02.py 从 checkpoint 重算核对；正文的数字由验证 2.5-e 断言。 */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';

export async function mount(root) {
  let data;
  try {
    const r = await fetch('data/ch02/pos_embed_cos.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    root.append(h('p', { class: 'w-label' }, `位置向量数据加载失败（${err.message}）。`));
    return;
  }
  const pts = (arr) => arr.map((v, i) => [i + 1, v]);
  const box = h('div');
  const low = data.trained.reduce((b, v, i) => (v < data.trained[b] ? i : b), 0);
  root.append(box, h('p', { class: 'w-label' },
    `间隔 1、8、64 的平均余弦相似度：训练后 ${[0, 7, 63].map((i) => data.trained[i].toFixed(3)).join('、')}，`
    + `最低点在间隔 ${low + 1}（${data.trained[low].toFixed(2)}）；`
    + `随机初始化 ${[0, 7, 63].map((i) => data.fresh[i].toFixed(3)).join('、')}。`));
  lineChart(box, {
    series: [{ name: '训练后', points: pts(data.trained) }, { name: '随机初始化', points: pts(data.fresh) }],
    xLabel: '两个位置的间隔 k', yLabel: '平均余弦相似度', height: 240,
    refLines: [{ y: 0, label: '' }],
  });
}
