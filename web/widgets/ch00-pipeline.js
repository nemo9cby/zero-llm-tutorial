/* 图：LLM 的流水线地图。选一台机器人，再点一道工序，看它用什么数据、优化什么目标、产出什么。
   Numbers come from data/ch00/bots.json, which verify/test_ch00.py checks against the book's scripts.
   config: { bot: "codebot", stage: "pretrain" } */
import { h, fmt } from './_lib.js';

const STAGES = [
  { id: 'pretrain', name: '预训练', en: 'pretraining' },
  { id: 'SFT', name: 'SFT', en: 'supervised fine-tuning' },
  { id: 'GRPO', name: '强化学习', en: 'GRPO' },
  { id: 'DPO', name: '偏好学习', en: 'DPO' },
];

const OBJECTIVE = {
  pretrain: ['预测下一个 token：在每个位置上，让正确的下一个 token 概率尽量大。',
    'loss = 所有位置上 −log p(下一个 token | 前文) 的平均（交叉熵）'],
  SFT: ['同一个交叉熵，但只在“回答”部分计算；指令部分的标签设为 −100，不计入 loss。',
    'loss = 回答部分 −log p(下一个 token | 前文) 的平均'],
  GRPO: ['模型自己作答，按奖励给回答打分；比同组平均分高的回答，概率被推高，低的被压低。',
    '梯度方向 = Σ advantage × ∇log p(回答)，advantage = 得分 − 组内平均分'],
  DPO: ['给成对的回答（喜欢的 / 不喜欢的），让模型相对参考模型更偏向喜欢的那个。',
    'loss = −log σ(β × [喜欢的回答的 Δ − 不喜欢的回答的 Δ])，Δ = log π − log π_ref'],
};

const LINKS = {
  codebot: { pretrain: 'ch03.html#s3-1', SFT: 'ch03.html#s3-3', GRPO: 'ch03.html#s3-9' },
  storybot: { pretrain: 'ch06.html#s6-5', DPO: 'ch06.html#s6-8' },
  webbot: { pretrain: 'ch09.html#s9-7' },
};

const CSS = `
.w-ch00-pipe .bots { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .8rem; }
.w-ch00-pipe .bots button[aria-pressed="true"] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch00-pipe .flow { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5rem; }
.w-ch00-pipe .stage { position: relative; text-align: left; padding: .5rem .6rem; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--paper); line-height: 1.35; min-height: 4.2rem; }
.w-ch00-pipe .stage b { display: block; font-size: .95rem; }
.w-ch00-pipe .stage i { display: block; font-style: normal; font-size: .75rem; color: var(--ink-3); }
.w-ch00-pipe .stage small { display: block; font-size: .75rem; color: var(--ink-2); margin-top: .2rem; }
.w-ch00-pipe .stage.on { border-color: var(--ai); box-shadow: inset 0 0 0 1px var(--ai); }
.w-ch00-pipe .stage.off { opacity: .45; border-style: dashed; }
.w-ch00-pipe .stage:not(:last-child)::after { content: '→'; position: absolute; right: -.5rem; top: 1.2rem;
  transform: translateX(50%); color: var(--ink-3); font-size: .8rem; }
.w-ch00-pipe .detail { margin-top: .9rem; padding: .7rem .9rem; background: var(--paper-2); border-radius: 6px; font-size: .9rem; }
.w-ch00-pipe .detail dl { display: grid; grid-template-columns: 4.5em minmax(0, 1fr); gap: .25rem .8rem; margin: 0; }
.w-ch00-pipe .detail dt { color: var(--ink-3); }
.w-ch00-pipe .detail dd { margin: 0; overflow-wrap: anywhere; }
.w-ch00-pipe .detail code { font-size: .82em; }
.w-ch00-pipe .formula { font-family: var(--mono); font-size: .8rem; color: var(--ink-2); display: block; margin-top: .15rem; }
@media (max-width: 560px) {
  .w-ch00-pipe .flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .w-ch00-pipe .stage:nth-child(2)::after { content: ''; }
}`;

function stageFacts(bot, stageId) {
  if (stageId === 'pretrain') {
    const p = bot.pretrain, c = bot.corpus;
    return {
      data: `${bot.data}：${fmt(c.text_bytes)} 字节的原始文本，编码后 ${fmt(c.tokens)} 个 token`,
      scale: `${fmt(p.max_iters)} 步 × ${p.batch_size}${p.accumulation_steps > 1 ? ` × ${p.accumulation_steps}` : ''} 条 × ${fmt(p.context_len)} token = ${fmt(p.tokens)} 个 token`,
      output: `${bot.id}/model_pretrain.pt：会“续写”的基础模型`,
      script: p.script,
    };
  }
  const s = bot.post.find((x) => x.stage === stageId);
  if (!s) return null;
  const files = { SFT: 'codebot/model_sft.pt：会按模板回答的助手',
    GRPO: 'codebot/model_grpo.pt：算术更准的助手',
    DPO: 'storybot/model_dpo.pt：更爱写快乐结局（要自己训练，Hugging Face 上没有）' };
  const data = { SFT: `${fmt(s.examples)} 条“指令 → 回答”（${s.data}）`,
    GRPO: `${s.data}，外加奖励函数：回答里最后一个整数等于正确答案得 1 分`,
    DPO: `${fmt(s.examples)} 对（同一个开头，快乐结局 / 悲伤结局）（${s.data}）` };
  const scale = { SFT: `${fmt(s.max_iters)} 步 × ${s.batch_size} 条`,
    GRPO: `${fmt(s.max_iters)} 步 × ${s.batch_size} 道题 × 每题 ${s.group_size} 个回答`,
    DPO: `${fmt(s.max_iters)} 步 × ${s.batch_size} 对，β = ${s.beta}` };
  return { data: data[stageId], scale: scale[stageId], output: files[stageId], script: s.script };
}

export async function mount(root, config) {
  root.append(h('style', {}, CSS));
  const box = h('div', { class: 'w-ch00-pipe' });
  root.append(box);
  let bots;
  try {
    const r = await fetch('data/ch00/bots.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    bots = (await r.json()).bots;
  } catch (err) {
    box.append(h('p', { class: 'w-label' }, `无法读取 data/ch00/bots.json（${err.message}）。`));
    return;
  }
  let botId = config.bot || 'codebot';
  let stageId = config.stage || 'pretrain';
  const botRow = h('div', { class: 'bots', role: 'group', 'aria-label': '选择机器人' });
  const flow = h('div', { class: 'flow', role: 'group', 'aria-label': '训练工序' });
  const detail = h('div', { class: 'detail', 'aria-live': 'polite' });
  box.append(botRow, flow, detail);

  const render = () => {
    const bot = bots.find((b) => b.id === botId);
    const active = new Set(['pretrain', ...bot.post.map((p) => p.stage)]);
    if (!active.has(stageId)) stageId = 'pretrain';
    botRow.replaceChildren(...bots.map((b) => h('button', {
      type: 'button', 'aria-pressed': String(b.id === botId),
      onclick: () => { botId = b.id; render(); },
    }, `${b.name}（第 ${b.part} 部）`)));
    flow.replaceChildren(...STAGES.map((s, i) => {
      const on = active.has(s.id);
      return h('button', {
        type: 'button', class: `stage${s.id === stageId ? ' on' : ''}${on ? '' : ' off'}`,
        'aria-pressed': String(s.id === stageId), disabled: !on,
        'aria-label': `${i + 1}. ${s.name}${on ? '' : `（${bot.name} 不做这一步）`}`,
        onclick: () => { stageId = s.id; render(); },
      }, h('b', {}, `${i + 1}. ${s.name}`), h('i', {}, s.en), h('small', {}, on ? '点击查看' : `${bot.name} 不做`));
    }));
    const f = stageFacts(bot, stageId);
    const [say, formula] = OBJECTIVE[stageId];
    const link = LINKS[bot.id][stageId];
    detail.replaceChildren(
      h('dl', {},
        h('dt', {}, '数据'), h('dd', {}, f.data),
        h('dt', {}, '目标'), h('dd', {}, say, h('span', { class: 'formula' }, formula)),
        h('dt', {}, '规模'), h('dd', {}, f.scale),
        h('dt', {}, '产物'), h('dd', {}, f.output),
        h('dt', {}, '代码'), h('dd', {}, h('code', {}, f.script), ' · ', h('a', { href: link }, '正文讲解'))),
      bot.post.length ? '' : h('p', { class: 'w-label', style: 'margin:.6rem 0 0' },
        `${bot.name} 只做预训练：原书第 III 部到预训练和生成为止，没有后训练。`),
    );
  };
  render();
}
