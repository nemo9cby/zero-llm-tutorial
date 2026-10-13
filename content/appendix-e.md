---
chapter: E
slug: appendix-e
order: 150
kind: appendix
title: DPO 最优解的推导
lede: 6.7 节说，带 KL 正则的 RLHF 目标有一个闭式的最优解，DPO 正是靠它绕开了奖励模型和强化学习。本附录从目标函数出发，逐行推导最优策略、隐式奖励和 DPO 损失，并在一个只有 4 个回答的小策略上把每一步都数值验证一遍。
pyodide_packages: [numpy]
---

[6.7 节](ch06.html#s6-7)比较了两条对齐的路线。RLHF 先用偏好数据训练一个奖励模型，再用强化学习最大化奖励，同时用 KL 散度把策略拴在参考模型附近。DPO 则直接拿偏好数据训练策略，它的损失函数是这样写的：

```listing ch06/08_dpo.py#L98-L113
```

真正的计算只有最后两步：算出 `logits`，再取 `-F.logsigmoid`。这短短几行里藏着几个问题：$\beta$ 为什么出现在 sigmoid 里面？“当前模型减参考模型”的对数概率差是什么意思？RLHF 里那个需要对所有可能回答求和、根本算不出来的配分函数，又是怎么消失的？

本附录一步一步回答这些问题。我们先写出带 KL 正则的目标，推导出它的最优策略 $\pi^*$；然后把最优策略反过来，把奖励写成策略的函数，这就是“隐式奖励”；接着代入 Bradley-Terry 偏好模型，看配分函数怎样抵消；最后得到 DPO 的损失，并在一个只有 4 个回答的小例子上确认：用 DPO 训练到最优，得到的正是 RLHF 的最优策略。推导沿用 DPO 原论文第 4 节与附录 A.1 的思路（[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)，2026-10 查证）。

## E.1 问题设定：带 KL 正则的目标

先固定一个问题 $x$，它可能的回答 $y$ 构成一个有限集合（回答长度有上限）。为了让记号简洁，下面省略 $x$。RLHF 想要的策略 $\pi$ 要同时满足两件事：期望奖励高，并且离参考策略 $\pi_{\text{ref}}$（预训练或 SFT 之后的模型）不远。写成一个目标函数：

$$
\begin{gathered}
\mathcal J(\pi) = \mathbb E_{y\sim\pi}\big[r(y)\big] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})\\
\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}}) = \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}
\end{gathered}
$$

这里，$r(y)$ 是回答 $y$ 的奖励，$\beta > 0$ 控制“离参考策略多远”的代价：$\beta$ 越大，越不愿意偏离。KL 散度有一条重要的性质：$\mathrm{KL}(p\,\|\,q) \ge 0$，当且仅当 $p = q$ 时等于 0（吉布斯不等式）。证明很短。$\log$ 是凹函数，由**琴生不等式**（Jensen's inequality），按概率加权的“对数的平均”不超过“平均的对数”：

$$
\begin{aligned}
-\mathrm{KL}(p\,\|\,q) &= \sum_{p(y)>0} p(y)\log\frac{q(y)}{p(y)}\\
&\le \log\sum_{p(y)>0} p(y)\,\frac{q(y)}{p(y)}\\
&= \log\sum_{p(y)>0} q(y)\\
&\le \log 1 = 0
\end{aligned}
$$

求和只取 $p(y) > 0$ 的 $y$（$p(y) = 0$ 的项按惯例记为 0）。$\log$ 是严格凹的，第一个“$\le$”只在 $q(y)/p(y)$ 处处相等时取等号；最后一个“$\le$”取等号，要求 $q$ 在 $p$ 为 0 的地方也为 0。两者合起来，就是 $p = q$。

先手算一个 KL：$p = (0.5, 0.5)$、$q = (0.9, 0.1)$ 时，$\mathrm{KL}(p\,\|\,q) = 0.5\ln\frac{0.5}{0.9} + 0.5\ln\frac{0.5}{0.1} = -0.2939 + 0.8047 = 0.5108$。这里用的是自然对数，单位叫 **nats**；换成以 2 为底就是 bits，$0.5108 / \ln 2 = 0.737$ bits。

接下来用一个小例子贯穿全附录：一共只有 4 个回答，参考策略是 $\pi_{\text{ref}} = (0.4, 0.3, 0.2, 0.1)$，奖励是 $r = (0, 1, 2, 0.5)$，$\beta = 0.5$。

```verify id=E-a title="KL 散度：手算、非负、nats 与 bits"
import numpy as np

def kl(p, q):
    """KL(p ‖ q)，自然对数，单位 nats。"""
    return float(np.sum(p * np.log(p / q)))

hand = kl(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
print(f"KL = {hand:.4f} nats = {hand / np.log(2):.4f} bits")
assert abs(hand - 0.5108) < 1e-4
assert abs(1 / np.log(2) - 1.4427) < 1e-4                # 1 nat = 1.4427 bits

rng = np.random.default_rng(0)
pairs = rng.dirichlet(np.ones(4), size=(2000, 2))      # 2000 对随机分布
kls = [kl(p, q) for p, q in pairs]
print(f"2000 对随机分布：KL 的最小值 {min(kls):.4f}")
assert min(kls) > 0 and kl(pairs[0, 0], pairs[0, 0]) == 0   # 吉布斯不等式

pi_ref = np.array([0.4, 0.3, 0.2, 0.1])                # 贯穿全附录的小例子
r = np.array([0.0, 1.0, 2.0, 0.5])
beta = 0.5
```

::: confusion 每一步的 KL 加起来，是不是就是“策略总共移动了多少”？
训练日志里常常每一步都记录一个 KL，把它们加起来，感觉就像“走过的路程”。但上面目标函数里的 KL 衡量的是**最终的策略离参考策略有多远**，与“每步 KL 之和”是两回事，而且谁大谁小都有可能。来回摆动时，和可以远大于它：A→B→A→B→A，每步 KL 之和是 1.76 nats，最后却回到了 A，相对起点的 KL 是 0。沿直线小步前进时，和又可以远小于它：从 A 到 B 分 100 等步走，每步 KL 之和只有 0.0044 nats，而 $\mathrm{KL}(B\,\|\,A) = 0.368$。严格地讲，KL 不是距离：它不对称，也不满足三角不等式；相邻的两个分布之间，它近似于距离的平方，所以同一段路切成 $n$ 步，每步之和大约缩小为 $1/n$。可以带走的结论：**想知道离参考策略多远，就直接算 $\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$，不要把每步的 KL 加起来。**
:::

```verify id=E-b title="每步 KL 之和与相对起点的 KL：来回摆动时和大得多，小步直走时和小得多"
A, B = np.array([0.5, 0.5]), np.array([0.9, 0.1])

def odometer(path):                      # 把每一步的 KL 加起来
    return sum(kl(path[k + 1], path[k]) for k in range(len(path) - 1))

swing = [A, B, A, B, A]                  # 来回摆动两次，回到起点
print(f"来回摆动：每步 KL 之和 {odometer(swing):.2f} nats，"
      f"相对起点的 KL {kl(swing[-1], swing[0]):.1f}")
assert abs(odometer(swing) - 1.7578) < 1e-4 and kl(swing[-1], swing[0]) == 0

sums = {}
for n in (2, 10, 100):                   # 从 A 沿直线走到 B，分 n 等步
    straight = [(1 - t) * A + t * B for t in np.linspace(0, 1, n + 1)]
    sums[n] = odometer(straight)
    print(f"直线分 {n:3d} 步：每步 KL 之和 {sums[n]:.4f} nats，n × 和 = {n * sums[n]:.3f}")
print(f"终点相对起点：KL(B‖A) = {kl(B, A):.3f}；反过来 KL(A‖B) = {kl(A, B):.3f}")
assert sums[100] < 0.01 < kl(B, A)       # 小步直走：和远小于终点的 KL（违反三角不等式）
assert abs(100 * sums[100] / (10 * sums[10]) - 1) < 0.05   # 和大约缩小为 1/n
assert abs(kl(A, B) - kl(B, A)) > 0.1                   # KL 不对称
```

## E.2 最优策略：π* ∝ π_ref · exp(r/β)

现在求使 $\mathcal J(\pi)$ 最大的 $\pi$。思路是把 $\mathcal J$ 改写成“一个 KL 加一个常数”，这样最大值一眼就能看出来。每一行下面用小字写着理由：

$$
\begin{aligned}
\mathcal J(\pi) &= \sum_y \pi(y)\,r(y)\\
&\qquad - \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\\[-3pt]
&\quad{\footnotesize\text{（写成求和）}}\\[2pt]
&= -\beta\sum_y \pi(y)\Big[\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\\
&\qquad\qquad - \frac{r(y)}{\beta}\Big]\\[-3pt]
&\quad{\footnotesize\text{（提出 }-\beta\text{）}}\\[2pt]
&= -\beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)\,e^{r(y)/\beta}}\\[-3pt]
&\quad{\footnotesize\text{（合并对数）}}\\[2pt]
&= -\beta\sum_y \pi(y)\log\frac{\pi(y)}{Z\,\pi^*(y)}\\[-3pt]
&\quad{\footnotesize\text{（代入 }Z\text{ 与 }\pi^*\text{）}}\\[2pt]
&= -\beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi^*(y)}\\
&\qquad + \beta\log Z\sum_y \pi(y)\\[-3pt]
&\quad{\footnotesize\text{（拆开对数）}}\\[2pt]
&= -\beta\,\mathrm{KL}(\pi\,\|\,\pi^*) + \beta\log Z\\[-3pt]
&\quad{\footnotesize\text{（}\textstyle\sum_y\pi(y) = 1\text{）}}
\end{aligned}
$$

“代入 $Z$ 与 $\pi^*$”那一步用到的两个定义是：

$$
\begin{aligned}
Z &= \sum_y \pi_{\text{ref}}(y)\,e^{r(y)/\beta}\\
\pi^*(y) &= \frac{1}{Z}\,\pi_{\text{ref}}(y)\,e^{r(y)/\beta}
\end{aligned}
$$

$Z$ 叫作**配分函数**（partition function），它的作用是让 $\pi^*$ 的总和为 1，所以 $\pi^*$ 是一个合法的概率分布。最后一行里，$\beta\log Z$ 与 $\pi$ 无关；由吉布斯不等式，$\mathrm{KL}(\pi\,\|\,\pi^*) \ge 0$，当且仅当 $\pi = \pi^*$ 时为 0。所以 $\mathcal J$ 的最大值是 $\beta\log Z$，在 $\pi = \pi^*$ 时取到。这就是 DPO 原论文的式 (4)。

$\pi^*$ 的意思很直观：把参考策略的每个回答按 $e^{r/\beta}$ 重新加权，奖励高的回答被放大。手算小例子：$r/\beta = (0, 2, 4, 1)$，所以 $\pi_{\text{ref}}\,e^{r/\beta} = (0.4,\ 0.3e^2,\ 0.2e^4,\ 0.1e) = (0.400, 2.217, 10.920, 0.272)$，求和得 $Z = 13.808$，于是 $\pi^* = (0.029, 0.161, 0.791, 0.020)$。奖励最高的回答在参考策略下只有 0.2 的概率，在 $\pi^*$ 下涨到了 0.79。

```verify id=E-c title="π* 与 Z：逐行验证改写，π* 打败 2000 个随机策略"
def objective(p):
    return float(p @ r) - beta * kl(p, pi_ref)

def fmt(v, digits=3):
    return "(" + ", ".join(f"{x:.{digits}f}" for x in v) + ")"

w = pi_ref * np.exp(r / beta)
Z = w.sum()
pi_star = w / Z
print(f"π_ref·e^(r/β) = {fmt(w)}，Z = {Z:.3f}")
print(f"π* = {fmt(pi_star)}")
assert abs(Z - 13.808) < 1e-3 and np.isclose(pi_star.sum(), 1)

candidates = rng.dirichlet(np.ones(4), size=2000)
for p in candidates:                # 最后一行的恒等式对任何 π 都成立
    assert np.isclose(objective(p), -beta * kl(p, pi_star) + beta * np.log(Z))
best_random = max(objective(p) for p in candidates)
print(f"J(π*) = {objective(pi_star):.4f} = β·log Z = {beta * np.log(Z):.4f}，"
      f"随机策略里最好的 J = {best_random:.4f}")
assert np.isclose(objective(pi_star), beta * np.log(Z))
assert best_random < objective(pi_star)
```

$\beta$ 是一个“旋钮”。$\beta$ 很大时，$e^{r/\beta} \approx 1$，$\pi^*$ 几乎等于 $\pi_{\text{ref}}$；$\beta$ 很小时，$e^{r/\beta}$ 把奖励最高的回答放大到压倒一切，$\pi^*$ 趋向只选它。下面的验证扫一遍 $\beta$，同时看 $\pi^*$ 离参考策略有多远、期望奖励有多高：

```verify id=E-d title="β 从大到小：π* 从 π_ref 走向只选奖励最高的回答"
def optimal_policy(beta_):
    logits = np.log(pi_ref) + r / beta_
    e = np.exp(logits - logits.max())                # 减去最大值，避免 β 很小时溢出
    return e / e.sum()

for beta_ in (100, 10, 1, 0.5, 0.1, 0.01):
    p = optimal_policy(beta_)
    kl_p = kl(np.clip(p, 1e-300, 1), pi_ref)          # 下溢的 0 不影响求和
    print(f"β = {beta_:6}：π* = {fmt(p)}，"
          f"KL(π*‖π_ref) = {kl_p:.3f}，E[r] = {p @ r:.3f}")
assert np.allclose(optimal_policy(100), pi_ref, atol=0.01)
assert optimal_policy(0.01)[2] > 0.999                     # 只选 r = 2 的回答
assert np.allclose(optimal_policy(beta), pi_star)
```

## E.3 隐式奖励：把奖励写成策略的函数

E.2 节从奖励得到了策略。反过来，也可以从策略读出奖励。对 $\pi^*$ 的定义取对数，再移项：

$$
\begin{aligned}
\log\pi^*(y) &= \log\pi_{\text{ref}}(y) + \frac{r(y)}{\beta} - \log Z\\[-3pt]
&\quad{\footnotesize\text{（对 }\pi^*\text{ 的定义取对数）}}\\[2pt]
r(y) &= \beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)} + \beta\log Z\\[-3pt]
&\quad{\footnotesize\text{（移项，两边乘以 }\beta\text{）}}
\end{aligned}
$$

这就是 DPO 原论文的式 (5)。它说：最优策略相对参考策略的对数概率比，乘以 $\beta$，就是奖励，只差一个与回答无关的常数 $\beta\log Z$。我们把 $\hat r(y) = \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}$ 叫作策略 $\pi$ 的**隐式奖励**（implicit reward）：任何一个策略，配上参考策略，都“隐含”着一个奖励模型。DPO 论文的副标题“你的语言模型其实是一个奖励模型”说的就是这件事。

```verify id=E-e title="隐式奖励加上 β·log Z，恰好还原 r"
r_hat = beta * np.log(pi_star / pi_ref)
print(f"隐式奖励 β·log(π*/π_ref) = {fmt(r_hat, 4)}")
print(f"再加上 β·log Z 之后     = {fmt(r_hat + beta * np.log(Z), 4)}")
assert np.allclose(r_hat + beta * np.log(Z), r)
# 两两之差完全相同
assert np.allclose(r_hat[:, None] - r_hat[None, :], r[:, None] - r[None, :])
```

## E.4 Bradley-Terry 模型与 Z 的消去

[6.7.2 节](ch06.html#s6-7-2)介绍了偏好数据的 **Bradley-Terry 模型**（下面简称 BT 模型）：给定同一个问题的两个回答 $y_w$ 和 $y_l$，人们更喜欢 $y_w$ 的概率是

$$
\begin{aligned}
P(y_w \succ y_l) &= \frac{e^{r(y_w)}}{e^{r(y_w)} + e^{r(y_l)}}\\
&= \sigma\big(r(y_w) - r(y_l)\big)
\end{aligned}
$$

其中 $\sigma(z) = 1/(1+e^{-z})$ 是 sigmoid 函数。第二个等号只是分子分母同除以 $e^{r(y_w)}$。关键在于，偏好概率只取决于两个奖励的**差**。记 $\hat r^*(y) = \beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}$ 为最优策略的隐式奖励，由 E.3 节，$r(y) = \hat r^*(y) + \beta\log Z$。代进去：

$$
\begin{aligned}
P(y_w \succ y_l) &= \sigma\big(r(y_w) - r(y_l)\big)\\[-3pt]
&\quad{\footnotesize\text{（BT 模型）}}\\[2pt]
&= \sigma\Big(\hat r^*(y_w) + \beta\log Z\\
&\qquad\quad - \hat r^*(y_l) - \beta\log Z\Big)\\[-3pt]
&\quad{\footnotesize\text{（E.3 节）}}\\[2pt]
&= \sigma\big(\hat r^*(y_w) - \hat r^*(y_l)\big)\\[-3pt]
&\quad{\footnotesize\text{（}\beta\log Z\text{ 抵消）}}
\end{aligned}
$$

把 $\hat r^*$ 展开写回来，就是

$$
\begin{aligned}
P(y_w \succ y_l) = \sigma\Big(&\beta\log\frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)}\\
&- \beta\log\frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}\Big)
\end{aligned}
$$

$Z$ 要对所有可能的回答求和，对语言模型来说根本算不出来；而它恰好在差值里抵消了。于是偏好概率只用到四个可以计算的量：两个回答在当前策略和参考策略下的对数概率。这就是 DPO 原论文的式 (6)。同样的道理，给所有回答的奖励加上同一个常数，偏好概率不变，所以奖励本来就只能确定到“差一个常数”为止。

```verify id=E-f title="偏好概率只看奖励差；用隐式奖励算，Z 消失"
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

pref = sigmoid(r[:, None] - r[None, :])       # pref[w, l] = P(y_w ≻ y_l)
er = np.exp(r)
assert np.allclose(pref, er[:, None] / (er[:, None] + er[None, :]))
via_policy = sigmoid(r_hat[:, None] - r_hat[None, :])  # 只用 π* 与 π_ref
assert np.allclose(pref, via_policy)
# 所有奖励加上同一个常数，偏好概率不变
assert np.allclose(pref, sigmoid((r + 7.0)[:, None] - (r + 7.0)[None, :]))
print(f"P(y2 ≻ y0) = σ(2 − 0) = {pref[2, 0]:.4f}，"
      f"用隐式奖励算也是 {via_policy[2, 0]:.4f}")
```

## E.5 DPO 的损失

最后一步：我们手里没有 $\pi^*$，只有一个待训练的策略 $\pi_\theta$ 和一堆偏好数据 $(y_w, y_l)$。把式 (6) 里的 $\pi^*$ 换成 $\pi_\theta$，就得到了“$\pi_\theta$ 认为人们更喜欢 $y_w$ 的概率”。用最大似然估计来训练，也就是最小化这个概率的负对数：

$$
\begin{gathered}
\mathcal L_{\text{DPO}}(\theta) = -\,\mathbb E_{(y_w, y_l)}\big[\log\sigma(z)\big]\\
z = \beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}
\end{gathered}
$$

这里，$z$ 是 sigmoid 里面的量，也就是两个回答的隐式奖励之差。这正是本附录开头那段代码：`logits` 就是 $z$，只是把四个对数概率换了一种分组方式，写成 $\beta\big[(\log\pi_\theta(y_w) - \log\pi_\theta(y_l)) - (\log\pi_{\text{ref}}(y_w) - \log\pi_{\text{ref}}(y_l))\big]$，再用 `-F.logsigmoid(...).mean()` 取负对数并对 batch 求平均。

开头的前两个问题也有了答案。“当前模型减参考模型”的对数概率差，乘以 $\beta$ 就是隐式奖励（E.3 节）；sigmoid 里面装的是两个回答的隐式奖励之差，所以 $\beta$ 出现在 sigmoid 里面。

有三件事可以马上手算出来：

1. **刚开始训练时，损失是 $\ln 2$。** DLFS6 用同一个预训练模型初始化 `model` 和 `ref_model`（[6.8 节](ch06.html#s6-8)），这时 $z = 0$，损失是 $-\log\sigma(0) = \log 2 \approx 0.6931$。真实的运行也是如此：在 H100 上原样运行 `08_dpo.py`，第 0 步记录的损失正是 0.6931（[验证 E-j](#v-E-j)）。
2. **一个具体的数。** 设 $\beta = 0.1$，当前策略下两个回答的对数概率是 $-10$ 和 $-12$，参考策略下都是 $-11$。于是 $z = 0.1\times\big((-10+12) - (-11+11)\big) = 0.2$，损失是 $\log(1+e^{-0.2}) = 0.5981$。
3. **梯度的权重。** 损失 $-\log\sigma(z)$ 对 $\log\pi_\theta(y_w)$ 求导得 $-\beta\,\sigma(-z)$，对 $\log\pi_\theta(y_l)$ 求导得 $+\beta\,\sigma(-z)$。上面的例子里是 $\mp0.1\times\sigma(-0.2) = \mp0.0450$。$\sigma(-z)$ 在隐式奖励排错了顺序（$z < 0$）时接近 1，排对了时接近 0，所以模型越是“看错”的样本，得到的推力越大。DPO 原论文第 4 节对这个权重有同样的解释。

```verify id=E-g title="DPO 损失：初始为 ln 2，手算 0.5981，梯度权重 β·σ(−z)"
def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)          # log σ(z) = −log(1 + e^(−z))

def dpo_loss(lp_chosen, lp_rejected, ref_chosen, ref_rejected, beta):
    """与 08_dpo.py 的 compute_dpo_loss 相同；输入是每条回答的对数概率之和。"""
    z = beta * ((lp_chosen - lp_rejected) - (ref_chosen - ref_rejected))
    return -np.mean(log_sigmoid(z))

same = rng.normal(-50, 10, size=(8, 2))                 # 当前策略 = 参考策略
init = dpo_loss(same[:, 0], same[:, 1], same[:, 0], same[:, 1], 0.1)

def example(lp_w):                     # 手算的例子，只改 log π(y_w)
    one = lambda v: np.array([v])
    return dpo_loss(one(lp_w), one(-12.0), one(-11.0), one(-11.0), 0.1)

hand = example(-10.0)
h = 1e-6                                                # 中心差分求导
grad_w = (example(-10.0 + h) - example(-10.0 - h)) / (2 * h)
print(f"初始损失 {init:.6f}（ln 2 = {np.log(2):.6f}）")
print(f"手算的例子 {hand:.4f}，∂L/∂log π(y_w) = {grad_w:.5f}")
assert abs(init - np.log(2)) < 1e-12 and abs(hand - 0.5981) < 1e-4
assert abs(grad_w - (-0.1 * sigmoid(-0.2))) < 1e-8
```

真实的 StoryBot 上是不是也一样？下面的本地测试从 `ch06/08_dpo.py` 里取出 `DPODataset`、`get_sequence_logprobs` 和 `compute_dpo_loss` 这三个部件，用 Hugging Face 上发布的两个 StoryBot checkpoint 来算。策略和参考必须不同，否则损失恒为 $\ln 2$，所以我们拿预训练第 5000 步的中间 checkpoint 当策略，预训练完成的模型当参考。在前 4 对真实的偏好数据上，原书的 PyTorch 损失与本页的 NumPy `dpo_loss` 一致。测试还确认了三件事。第一，`08_dpo.py` 的 `model` 和 `ref_model` 都从预训练完成的同一个 checkpoint 加载，所以训练开始时策略等于参考。第二，策略等于参考时，损失恰为 $\ln 2$。第三，把问题部分也算进对数概率，损失不变：同一个问题在两条回答里贡献的对数概率完全相同，在差值里抵消了。

```localcheck verify/test_appendix.py::test_e_dpo_loss needs="PyTorch" title="真实 StoryBot：原书的 compute_dpo_loss = 本页的 dpo_loss；初始为 ln 2"
```

::: confusion DPO 的损失里有 π_θ(y_w)，训练时是不是要让 π_θ 先生成回答？
GRPO 训练时确实要让模型自己生成回答（3.9 节），所以很容易以为 DPO 也一样。其实不用：$y_w$ 和 $y_l$ 是事先准备好的文字。DLFS6 的 `tiny_stories_dpo.json` 里有 144 对，每对共用一个故事开头，`chosen`（$y_w$）是快乐的结局，`rejected`（$y_l$）是悲伤的结局。比如第一对里，找到玩具的 Tim “was so happy”，没找到的 Tim “was very sad”。$\pi_\theta$ 和 $\pi_{\text{ref}}$ 只负责给这两段固定的文字“打分”：把文字喂进去，算出每个 token 的对数概率再求和。这和 SFT 计算交叉熵一样，一次前向传播就够了，没有任何采样。上面的验证 E-g 只用了四个对数概率，也说明了这一点；本地测试还检查了 `08_dpo.py` 里确实没有调用生成函数，而 `09_grpo.py` 调用了。可以带走的对照：**SFT 用数据里的回答、以交叉熵打分；GRPO 让策略自己生成、用奖励函数打分；DPO 用事先配好对的回答、以隐式奖励的差打分。**
:::

## E.6 偏好服从 BT 模型时，DPO 训练到最后就是 π*

推导告诉我们，DPO 与 RLHF 瞄准的是同一个最优策略。用小例子来确认：假设偏好数据无穷多，每一对回答都按 Bradley-Terry 模型和真实奖励 $r$ 给出偏好。4 个回答两两配对，一共 6 对；同一对的两种标注都会出现，$y_w$ 赢的比例是 $P_{wl} = P(y_w \succ y_l)$。这时 DPO 的期望损失（对 6 对取平均）是

$$
\begin{aligned}
\mathcal L(\theta) &= -\frac16\sum_{w<l}\Big[P_{wl}\log\sigma(z_{wl})\\
&\qquad + (1 - P_{wl})\log\sigma(-z_{wl})\Big]
\end{aligned}
$$

其中 $z_{wl}$ 就是 E.5 节的 $z$：

$$
z_{wl} = \beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}
$$

策略用 softmax 参数化，$\pi_\theta = \operatorname{softmax}(\boldsymbol\theta)$。这里还有一个小小的巧合：$\log\pi_\theta(y_w) - \log\pi_\theta(y_l) = \theta_w - \theta_l$，softmax 的分母也在差值里抵消了，和 $Z$ 一样。从 $\boldsymbol\theta = \log\pi_{\text{ref}}$（即 $\pi_\theta = \pi_{\text{ref}}$）出发做梯度下降：

```verify id=E-h title="在小例子上用 DPO 训练到收敛：结果就是 RLHF 的最优策略 π*"
w_idx, l_idx = np.triu_indices(4, k=1)                  # 全部 6 对回答
p_pref = sigmoid(r[w_idx] - r[l_idx])                   # 真实的偏好概率
pair_sign = np.eye(4)[w_idx] - np.eye(4)[l_idx]         # 每一对：y_w 处 +1，y_l 处 −1

def expected_dpo(theta, p=p_pref, beta_=beta):
    """期望 DPO 损失及其梯度；log π_θ(w) − log π_θ(l) = θ_w − θ_l。p 是 y_w 赢的比例。"""
    ref_gap = np.log(pi_ref[w_idx] / pi_ref[l_idx])
    z = beta_ * (pair_sign @ theta - ref_gap)           # pair_sign @ θ 就是每一对的 θ_w − θ_l
    loss = -np.mean(p * log_sigmoid(z) + (1 - p) * log_sigmoid(-z))
    dz = -(p * sigmoid(-z) - (1 - p) * sigmoid(z)) / len(z)
    return loss, beta_ * (dz @ pair_sign)

theta = np.log(pi_ref)                                  # 从 π_θ = π_ref 出发
print(f"初始损失 {expected_dpo(theta)[0]:.4f}（每一对都是 ln 2）")
for step in range(3000):
    loss, grad = expected_dpo(theta)
    theta -= 2.0 * grad
pi_dpo = np.exp(theta - theta.max())
pi_dpo /= pi_dpo.sum()
print(f"DPO 训练的结果 = {fmt(pi_dpo, 4)}")
print(f"RLHF 的最优 π* = {fmt(pi_star, 4)}")
assert np.abs(pi_dpo - pi_star).max() < 1e-6

# 收敛时的损失不是 0：每一对最低只能降到偏好概率的二元熵
entropy = -(p_pref * np.log(p_pref) + (1 - p_pref) * np.log(1 - p_pref))
final = expected_dpo(theta)[0]
print(f"收敛时的损失 {final:.4f}，6 对偏好概率的二元熵的平均 {entropy.mean():.4f}")
assert abs(final - entropy.mean()) < 1e-9
```

DPO 只看了偏好数据，从来没有见过奖励 $r$，也没有估计过 $Z$，却收敛到了 RLHF 的最优策略。这就是“直接偏好优化”这个名字的含义。注意收敛时的损失是 0.5551，而不是 0。同一对回答的两种标注都会出现，所以每一对的损失最低只能降到 $-P\ln P - (1-P)\ln(1-P)$，也就是偏好概率 $P$ 的**二元熵**（binary entropy），6 对的平均正是 0.5551。

当然，这依赖几个理想化的条件：偏好服从 BT 模型，数据足够多，覆盖了需要比较的回答；策略的表达能力也足够。DLFS6 的偏好数据却不满足第一个条件。`tiny_stories_dpo.json` 的 144 对里，每个故事开头只出现一次，标注也只有一种：`chosen` 赢。从数据看，$P_{wl} = 1$。按 BT 模型，$\sigma(r_w - r_l) = 1$ 意味着奖励差是无穷大，$\pi^*$ 里 $y_l$ 的概率是 0，而且不管 $\beta$ 取多少都是如此。DPO 的损失也就没有有限的最小点：每一对只剩 $-\log\sigma(z)$ 这一项，它随 $z$ 单调下降，训练会一直把 $z$ 往上推。在小例子上把标注换成这种确定性的试一试，奖励高的那个总是赢：

```verify id=E-i title="标注是确定性的：损失没有最小点，π_θ 越过 π*，走向只选一个回答"
p_hard = (r[w_idx] > r[l_idx]).astype(float)           # 奖励高的那个总是赢：P 只取 0 或 1
print("确定性的标注：" + "，".join(f"P(y{w} ≻ y{l}) = {p:.0f}" for w, l, p in zip(w_idx, l_idx, p_hard)))

hard = {}
for beta_ in (0.5, 2.0):
    theta = np.log(pi_ref)
    losses = []
    for step in range(3000):                            # 与验证 E-h 相同的梯度下降
        loss, grad = expected_dpo(theta, p_hard, beta_)
        losses.append(loss)
        theta -= 2.0 * grad
    pi_hard = np.exp(theta - theta.max())
    hard[beta_] = (np.array(losses), pi_hard / pi_hard.sum())
    print(f"β = {beta_}：损失 {losses[0]:.4f} → {losses[-1]:.5f}，π_θ = {fmt(hard[beta_][1])}，"
          f"π* = {fmt(optimal_policy(beta_))}")

for beta_, (losses, pi_hard) in hard.items():
    assert np.all(np.diff(losses) < 0)                  # 每一步都在下降：没有停下来的点
    assert losses[-1] < 0.01 and pi_hard[2] > 0.97 > optimal_policy(beta_)[2]
assert hard[0.5][1][2] > 0.999
```

两个 $\beta$ 下，损失每一步都在下降，$\pi_\theta$ 都越过了各自的 $\pi^*$，一路走向只选 $y_2$；$\beta = 2$ 的 $\pi^*$ 本来相当分散。KL 正则在这里拴不住策略。

真实的运行也是这样。我们在 H100 上原样运行了 `08_dpo.py`（运行 storybot-h100）。第 0 步的损失是 0.6931，恰好是 $\ln 2$；之后每 100 步的平均损失都比前 100 步低，最后一步（第 1000 步）降到了 0.0045：

```verify id=E-j title="H100 上原样运行 08_dpo.py：损失从 ln 2 一路降到 0.0045，没有停下来" files=data/runs/storybot-h100/08_dpo.metrics.jsonl
import json

with open("data/runs/storybot-h100/08_dpo.metrics.jsonl", encoding="utf-8") as f:
    dpo_log = {row["it"]: row["loss"] for row in map(json.loads, f)}
by_100 = [np.mean([v for it, v in dpo_log.items() if a <= it < a + 100]) for a in range(0, 1000, 100)]
first, last = dpo_log[0], dpo_log[1000]
z_min = -np.log(np.expm1(last))          # 解 −log σ(z) = log(1 + e^(−z)) = last
print(f"第 0 步 {first:.4f}（ln 2 = {np.log(2):.4f}），第 1000 步 {last:.4f}")
print("每 100 步的平均损失：" + "，".join(f"{v:.4f}" for v in by_100))
print(f"−log σ(z) = {last} 时 z = {z_min:.2f}；β = 0.1，z/β = {z_min / 0.1:.0f} nats")
assert abs(first - np.log(2)) < 1e-4 and last < 0.01
assert all(a > b for a, b in zip(by_100, by_100[1:]))  # 每 100 步都比前 100 步低
assert abs(-log_sigmoid(z_min) - last) < 1e-12 and abs(z_min - 5.40) < 0.01
```

日志里的损失是一个 batch 里 8 对的平均。$-\log\sigma(z)$ 是单调递减的凸函数，由琴生不等式（E.1 节），8 对的平均损失不小于“平均 $z$ 处的损失”，所以这 8 对的平均 $z$ 至少是 5.4。`08_dpo.py` 的 $\beta = 0.1$，所以与参考模型相比，两段回答的对数概率之差平均被拉开了 54 nats 以上。[6.8 节](ch06.html#s6-8)在 CPU 上也跑了这 1000 步：chosen 的对数概率平均上升 13.1 nats，rejected 平均下降 39.4 nats（[验证 6.8-c](ch06.html#v-6.8-c)），拉开的距离主要来自把 $y_l$ 压下去。1000 步、batch 8，相当于把同一批数据看了 55 遍以上（$1000\times8/144 \approx 55.6$），每看一遍，$z$ 就再被推高一些。

这个问题有专门的研究。IPO 的论文第 4.2 节指出：偏好越接近确定，KL 正则就越弱，作者称之为过拟合；数据少时，即使真实的偏好概率是 0.8，样本里也很可能每次都是同一个结果。论文也提到，提前停止（early stopping）这类常见做法可以充当额外的正则；第 5 节则为此提出了新的损失 IPO（[Azar et al., 2023](https://arxiv.org/abs/2310.12036)，2026-10 查证）。

下面的交互图可以调 $\beta$ 和每个回答的奖励，看最优策略、隐式奖励和偏好概率怎样随之变化。

```widget appendix-dpo caption="π* 把参考策略按 e^(r/β) 重新加权：β 越小，越集中在奖励最高的回答上；隐式奖励加上 β·log Z 总是等于 r。π_θ = π* 时，一对回答的期望损失降到最小，正是偏好概率的二元熵。"
{"pi_ref": [0.4, 0.3, 0.2, 0.1], "r": [0, 1, 2, 0.5], "beta": 0.5}
```

::: cs336 A5 补充作业：DPO
CS336 作业 5 的补充部分（安全与 RLHF）要求实现逐样本的 DPO 损失。两边用的是同一个损失，设置对比如下：

| | CS336 A5 补充 | DLFS6 的 6.8 节 |
|---|---|---|
| 模型 | Llama 3.1 8B | StoryBot（22.7M 参数） |
| 数据 | Anthropic HH | 自制的“快乐结局 / 悲伤结局”，144 对 |
| 参考模型 | SFT 之后的模型 | 预训练模型 |
| 优化器 | RMSprop，学习率 $10^{-6}$ | AdamW，学习率 $5\times10^{-6}$ |
| batch | 等效 64 | 8 |
| $\beta$ | 0.1 | 0.1 |
| 监控 | 验证集上的隐式奖励准确率 | 训练损失 |

隐式奖励准确率是 $\hat r(y_w) > \hat r(y_l)$ 的比例。理论部分对应第 15 讲（SFT、RLHF 与 DPO）（[A5 仓库](https://github.com/stanford-cs336/assignment5-alignment)，2026-10 查证）。
:::

## 小结

本附录从带 KL 正则的目标 $\mathbb E[r] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$ 出发，把它改写成 $-\beta\,\mathrm{KL}(\pi\,\|\,\pi^*) + \beta\log Z$，于是最优策略一眼可见：$\pi^* \propto \pi_{\text{ref}}\,e^{r/\beta}$。对它取对数，奖励就可以写成策略的函数 $\beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z$；代入只看奖励差的 Bradley-Terry 模型，算不出来的 $\beta\log Z$ 正好抵消。把 $\pi^*$ 换成待训练的 $\pi_\theta$，做最大似然，就是 DPO 的损失。

DPO 训练不需要采样，只需要给固定的回答对算对数概率；初始损失是 $\ln 2$；梯度对“看错”的样本加权更大。在偏好服从 BT 模型、数据足够的理想情况下，它收敛到的正是 RLHF 的最优策略。DLFS6 的标注却是确定性的，这时损失没有有限的最小点，训练只会把隐式奖励之差越推越大，$\beta$ 也拴不住策略。

### 本附录所学的内容

- KL 散度非负，单位是 nats（乘以 1.4427 得 bits）；正则项约束的是最终离参考策略多远，不是每步 KL 之和（[验证 E-a](#v-E-a)、[验证 E-b](#v-E-b)）。
- 带 KL 正则的目标等于 $-\beta\,\mathrm{KL}(\pi\,\|\,\pi^*) + \beta\log Z$，最优策略是 $\pi^* \propto \pi_{\text{ref}}\,e^{r/\beta}$，最大值是 $\beta\log Z$（[验证 E-c](#v-E-c)）。
- $\beta$ 越大 $\pi^*$ 越接近 $\pi_{\text{ref}}$，越小越集中在奖励最高的回答上（[验证 E-d](#v-E-d)）。
- 隐式奖励 $\beta\log(\pi^*/\pi_{\text{ref}})$ 与真实奖励只差常数 $\beta\log Z$（[验证 E-e](#v-E-e)）。
- Bradley-Terry 偏好概率只看奖励差，所以 $Z$ 在 DPO 里抵消（[验证 E-f](#v-E-f)）。
- DPO 损失初始为 $\ln 2$；梯度权重是 $\beta\,\sigma(-z)$（[验证 E-g](#v-E-g)）。
- 偏好服从 BT 模型时，DPO 训练的结果就是 RLHF 的最优策略；标注是确定性的（DLFS6 正是这样）时，损失没有有限的最小点，真实运行的损失从 $\ln 2$ 一路降到 0.0045（[验证 E-h](#v-E-h)、[验证 E-i](#v-E-i)、[验证 E-j](#v-E-j)）。

五个附录到此结束。它们各自把正文里一句“可以证明”变成了能一步步核对的推导：UTF-8 的位模式、$\sqrt d$ 的来历、基线的无偏性、Adam 的偏差校正，以及 DPO 的最优解。回到正文时，如果某个公式又让你停下来，不妨像这样，把每一步都交给一个验证单元。接下来的 [CodeBot 工作台](playground.html)则反过来，把全书的零件重新装回一台能在浏览器里运行的机器人。
