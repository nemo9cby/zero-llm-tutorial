/* 图：从 REINFORCE 到 GRPO 的“家谱”，以及配置行解码器。
   所有算法都写成 ∇J ≈ E[ Σ_t Φ_t ∇log π(a_t|s_t) ]。前五代只改 Φ；后两代改“用谁的样本”（重要性比值与裁剪）和归一化。
   前四行沿用《深度学习入门4：强化学习》9.6 节的编号。config: {} */
import { h } from './_lib.js';
import { injectStyle, segmented } from './ch03-common.js';

export const LINEAGE = [
  { tag: '(1)', name: '最简单的策略梯度', phi: 'Φ_t = G(τ)　整个回合的总回报',
    where: '《深度学习入门4》9.1 节（式 9.1、9.2）；本章 3.6 节',
    needs: '只需要采样', note: '每个动作都按整条轨迹的回报加权，方差很大。' },
  { tag: '(2)', name: 'REINFORCE', phi: 'Φ_t = G_t　只看动作之后的回报',
    where: '《深度学习入门4》9.2 节（式 9.3）；本章 3.5.4 节',
    needs: '只需要采样', note: '语言模型只在最后给奖励、γ = 1 时，G_t 处处等于 G(τ)，与 (1) 完全相同。' },
  { tag: '(3)', name: '带基线的 REINFORCE', phi: 'Φ_t = G_t − b(s_t)',
    where: '《深度学习入门4》9.3 节（式 9.4）；本章 3.7.2 节；附录 C',
    needs: '一个不依赖动作的基线', note: '期望不变，方差下降（验证 3.7-b：1.589 → 0.531）。' },
  { tag: '(4)', name: 'Actor-Critic', phi: 'Φ_t = r_t + γV(s′) − V(s_t)　TD 误差（s′ 是下一个状态）',
    where: '《深度学习入门4》9.4 节（式 9.6）', needs: '再训练一个价值网络 V（critic）',
    note: 'PPO 通常用 critic 的估计计算 GAE，在偏差与方差之间折中。对 LLM 来说，critic 往往又是一个 Transformer。' },
  { tag: '(5)', name: 'GRPO 的组内优势', phi: 'Φ_t = A_i = r_i − mean(r_1 … r_G)',
    where: '本章 3.7.3 节；09_grpo.py 第 88–89 行', needs: '每个 prompt 采 G 个回答；不需要 critic',
    note: '组均值是 V(prompt) 的蒙特卡洛估计。期望梯度是真梯度的 (G−1)/G 倍（验证 3.7-f）。' },
  { tag: '(6)', name: '重要性比值 + 裁剪（PPO-clip）', phi: '目标 min(ρ_t·A, clip(ρ_t, 1−ε, 1+ε)·A)，ρ_t = π_θ/π_old',
    where: '本章 3.8 节；09_grpo.py 第 110–129 行', needs: '旧策略的概率 old_probs',
    note: '让同一批样本可以更新多次。ρ = 1 时梯度与 (5) 完全相同（验证 3.8-e）。' },
  { tag: '(7)', name: 'Dr. GRPO 式归一化', phi: '不除以 std；loss = Σ_tokens / (B·G)，不除以 |o_i|',
    where: '本章 3.7 节专栏；09_grpo.py 第 89、128–129 行', needs: '——',
    note: '去掉 DeepSeekMath 原版里的两处偏差；09_grpo.py 也没有 KL 项。' },
];

export const PRESETS = [
  { name: '09_grpo.py', terms: [
    ['GRPO', '用组内平均作基线的策略梯度，没有 critic（3.7.3 节）。', '09_grpo.py 第 75–96 行 generate_group'],
    ['group_size = 8', '每道题采 8 个回答，组内互相比较（3.7.3 节）。', '09_grpo.py 第 144 行'],
    ['32 prompts/step', '每轮 32 道题，× 8 = 256 条回答；81 道题分成 32、32、17 三批（验证 3.9-a）。', '09_grpo.py 第 145、156 行'],
    ['temperature = 1.0', '采样温度。1.0 表示按模型原本的分布采样（3.2.2 节）。', '09_grpo.py 第 84 行'],
    ['max_new_tokens = 1000', '每个回答最多生成的 token 数；generate() 的默认值（3.2.4 节）。', 'codebot/utils.py 第 6 行'],
    ['ε = 0.2', '裁剪范围：比值被限制在 [0.8, 1.2]（3.8.4 节）。', '09_grpo.py 第 143 行'],
    ['2 updates / rollout batch', '同一批回答更新 2 次，第 2 次是离策略的（3.8.1 节）。', '09_grpo.py 第 141、181 行'],
    ['lr = 7e-6', 'AdamW 的学习率，比 SFT 小 40 多倍（3.9.1 节）。', '09_grpo.py 第 139 行'],
    ['no KL', '没有参考模型，也没有 KL 惩罚，只靠裁剪和小学习率控制步幅。', '09_grpo.py 全文'],
  ] },
  { name: '典型的 PPO 式 RLHF', terms: [
    ['PPO', '用重要性比值与裁剪做多次更新的策略梯度（3.8.4 节）。', 'DLFS6 没有实现；对应 3.8 节'],
    ['GAE', '用 critic 估计的价值计算优势，在偏差与方差之间折中（3.7.2 节）。', 'DLFS6 没有实现'],
    ['critic / value model', '估计 V(s) 的价值网络，需要与策略一起训练（《深度学习入门4》9.4 节）。', 'GRPO 用组均值代替它'],
    ['KL penalty', '惩罚策略偏离参考模型（通常是 SFT 模型）太远。', '09_grpo.py 没有这一项'],
    ['reward model', '从人类偏好数据学出来的打分模型（第 6 章 6.7 节）。', 'CodeBot 用可验证奖励代替'],
    ['trajectories / step', '每个优化步使用的回答条数，= 题数 × 每题回答数。', '09_grpo.py：32 × 8 = 256'],
  ] },
];

export function mount(root) {
  injectStyle();
  root.classList.add('w-ch03');
  root.append(h('style', {}, `
.w-ch03-lin details { border-bottom: 1px solid var(--rule); padding: .35rem 0; }
.w-ch03-lin summary { cursor: pointer; display: flex; flex-wrap: wrap; gap: .2rem .7rem; align-items: baseline; }
.w-ch03-lin summary .tag { font-family: var(--mono); color: var(--ai); min-width: 2.2em; }
.w-ch03-lin summary .phi { font-family: var(--mono); font-size: .82rem; color: var(--ink-2); overflow-wrap: anywhere; }
.w-ch03-lin dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .15rem .8rem; margin: .4rem 0 .2rem 2.9em; font-size: .85rem; }
.w-ch03-lin dt { color: var(--ink-3); }
.w-ch03-lin dd { margin: 0; overflow-wrap: anywhere; }
.w-ch03-lin .terms { display: flex; flex-wrap: wrap; gap: 4px; margin: .5rem 0; }
.w-ch03-lin .terms button { font: 12.5px/1.4 var(--mono); padding: .15rem .45rem; }
.w-ch03-lin .terms button[aria-pressed=true] { background: var(--ai); color: var(--paper); border-color: var(--ai); }
.w-ch03-lin .explain { min-height: 3.2rem; padding: .5rem .7rem; background: var(--paper-2); border-radius: 4px; font-size: .88rem; }
@media (max-width: 520px) { .w-ch03-lin dl { grid-template-columns: minmax(0, 1fr); margin-left: 0; } }
`));
  root.classList.add('w-ch03-lin');

  root.append(h('div', { class: 'w-label' }, '统一写法（《深度学习入门4》9.6 节）：∇J ≈ E[ Σ_t Φ_t ∇log π(a_t|s_t) ]。前四行沿用该书的编号，点开每一行查看细节。'));
  for (const row of LINEAGE) {
    root.append(h('details', {},
      h('summary', {}, h('span', { class: 'tag' }, row.tag), h('b', {}, row.name), h('span', { class: 'phi' }, row.phi)),
      h('dl', {},
        h('dt', {}, '出处'), h('dd', {}, row.where),
        h('dt', {}, '需要什么'), h('dd', {}, row.needs),
        h('dt', {}, '说明'), h('dd', {}, row.note))));
  }

  root.append(h('div', { class: 'w-label', style: 'margin-top:1.2rem' }, '配置行解码器：点一个词，看它是什么、在 DLFS6 代码的哪里。'));
  const terms = h('div', { class: 'terms', role: 'group', 'aria-label': '配置中的术语' });
  const explain = h('div', { class: 'explain', 'aria-live': 'polite' });
  let preset = 0;
  const show = (t) => explain.replaceChildren(h('b', {}, t[0]), '：', t[1], h('div', { class: 'muted' }, t[2]));
  const renderTerms = () => {
    const list = PRESETS[preset].terms;
    const btns = list.map((t) => h('button', { type: 'button', 'aria-pressed': 'false' }, t[0]));
    btns.forEach((b, i) => {
      const act = () => { btns.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j))); show(list[i]); };
      b.addEventListener('click', act);
      b.addEventListener('focus', act);
    });
    terms.replaceChildren(...btns);
    explain.replaceChildren(h('span', { class: 'muted' }, '（点上面的任意一个词）'));
  };
  root.append(segmented(PRESETS.map((p) => p.name), 0, (i) => { preset = i; renderTerms(); }, '选择一行配置'), terms, explain);
  renderTerms();
}
