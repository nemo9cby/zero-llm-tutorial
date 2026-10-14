/* 图：“loss 不是进度条”。一个 softmax 策略的多臂老虎机，按 09_grpo.py 的方式训练：
   每步采 G 个动作，A = r − mean(r)，比值 = 1 时 loss = −Σ A·L / G（L 是回答的 token 数）。
   梯度只经过第一个 token（后面的 token 概率为 1），所以“回答长度不等”只改变 loss 的数值，不改变学习。
   这是玩具的假设：在 09_grpo.py 里，长回答的每个 token 都带梯度。长度不等时 loss = 2 × 组内奖励的方差（L = 3 − 2r）。
   config: { arms: 4, good: 2, G: 8, lr: 1.0, steps: 30 } */
import { h } from './_lib.js';
import { lineChart } from './_plot.js';
import { injectStyle, softmax, sample, rng, toggle, advantages } from './ch03-common.js';

export function mount(root, config) {
  injectStyle();
  const K = config.arms || 4, good = config.good ?? 2, G = config.G || 8, lr = config.lr || 1.0;
  const maxSteps = config.steps || 30;
  let theta, rand, losses, rewards, unequal = false, timer = null, seed = 0;

  root.classList.add('w-ch03');
  const lossBox = h('div'), rewBox = h('div');
  const stepBtn = h('button', { type: 'button' }, '更新一步');
  const runBtn = h('button', { type: 'button' }, `再自动运行 ${maxSteps} 步`);
  const resetBtn = h('button', { type: 'button' }, '换个种子重来');
  const lenToggle = toggle('回答长度不等（答错的回答更长）', false, (v) => { unequal = v; reset(false); });
  const note = h('div', { class: 'stat', 'aria-live': 'polite' });
  root.append(h('div', { class: 'ctl' }, stepBtn, runBtn, resetBtn), h('div', { class: 'ctl' }, lenToggle),
    h('div', { class: 'two' }, lossBox, rewBox), note);

  // 动作 good 的回答长 1 个 token；其余动作在“长度不等”时长 3 个 token
  const lengthOf = (a) => (unequal && a !== good ? 3 : 1);
  const lossChart = lineChart(lossBox, { series: [{ name: 'loss', points: [[0, 0]] }], xLabel: '更新步数', yLabel: 'loss（比值 = 1 时）', height: 200, table: false, yMin: -0.2, yMax: 0.6 });
  const rewChart = lineChart(rewBox, { series: [{ name: '期望奖励', points: [[0, 1 / K]] }], xLabel: '更新步数', yLabel: '期望奖励 P(选中正确的臂)', height: 200, table: false, yMin: 0, yMax: 1 });

  function reset(newSeed = true) {
    if (newSeed) seed += 1;
    stop();
    theta = Array(K).fill(0);
    rand = rng(1000 + seed);
    losses = [];
    rewards = [[0, softmax(theta)[good]]];
    draw();
  }

  function step() {
    if (losses.length >= 200) return;
    const pi = softmax(theta);
    const acts = Array.from({ length: G }, () => sample(pi, rand));
    const r = acts.map((a) => (a === good ? 1 : 0));
    const A = advantages(r);
    const loss = -A.reduce((s, a, i) => s + a * lengthOf(acts[i]), 0) / G;   // −Σ A·L / N
    // d loss / d θ = −(1/G) Σ A_i (onehot(a_i) − π)
    const grad = Array(K).fill(0);
    acts.forEach((a, i) => { for (let j = 0; j < K; j++) grad[j] -= (A[i] * ((j === a ? 1 : 0) - pi[j])) / G; });
    theta = theta.map((t, j) => t - lr * grad[j]);
    losses.push([losses.length + 1, loss]);
    rewards.push([rewards.length, softmax(theta)[good]]);
    draw();
  }

  function draw() {
    const n = losses.length;
    lossChart.update({ series: [{ name: 'loss', points: n ? [[0, 0], ...losses] : [[0, 0]] }] });
    rewChart.update({ series: [{ name: '期望奖励', points: rewards }] });
    const last = rewards[rewards.length - 1][1];
    const lastLoss = n ? losses[n - 1][1] : 0;
    note.textContent = n
      ? `第 ${n} 步：loss ${lastLoss >= 0 ? '+' : ''}${lastLoss.toFixed(3)}，期望奖励 ${last.toFixed(3)}（起点 ${(1 / K).toFixed(2)}）。`
        + (unequal
          ? ' 长度不等时，loss = −Σ A·L / N = 2 × 组内奖励的方差：组里有对有错时为正，全对或全错时回到 0。'
            + '它后来变小，是因为组里没了差异、信号没了，不是因为学会了。'
            + '在这个玩具里，答错的回答多出来的 token 概率恒为 1，不产生梯度，所以学习的方向与长度相等时完全一样。'
          : ' 每一步的 loss 都恰好是 0：组内优势之和为 0，各回答又一样长。')
      : '点“更新一步”开始。每一步都重新采样 8 个动作、重新减去组均值。';
  }

  function stop() { if (timer) { clearInterval(timer); timer = null; runBtn.textContent = `再自动运行 ${maxSteps} 步`; } }
  stepBtn.addEventListener('click', () => { stop(); step(); });
  resetBtn.addEventListener('click', () => reset(true));
  runBtn.addEventListener('click', () => {
    if (timer) { stop(); return; }
    const target = Math.min(200, losses.length + maxSteps);
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { while (losses.length < target) step(); return; }
    runBtn.textContent = '暂停';
    timer = setInterval(() => { if (losses.length >= target) stop(); else step(); }, 120);
  });
  reset(false);
  while (losses.length < maxSteps) step();   // 先给结果：打开页面时已经跑完 30 步，可以重置后一步步看
}
