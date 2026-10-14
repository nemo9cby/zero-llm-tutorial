---
chapter: C
slug: appendix-c
order: 130
kind: appendix
title: 带基线策略梯度的无偏性证明
lede: 3.7 节说：把奖励 r 换成 r − b，策略梯度的期望不变、方差变小。本附录逐行证明这句话，弄清楚基线为什么只能依赖状态，再看 GRPO 的组均值为什么让梯度整体缩小为 (G−1)/G。
pyodide_packages: [numpy]
---

[3.7 节](ch03.html#s3-7)给策略梯度加上了**基线**（baseline）：把奖励 $r$ 换成 $r - b$，梯度的期望不变，方差却可以变小。GRPO 更进一步，用同一个问题的一组回答的平均奖励来当基线。这里有三个问题需要回答：期望为什么不变？方差在什么条件下才会变小？组均值里包含了回答自己的奖励，它还满足证明的前提吗？

本附录把这三件事一一讲清楚。我们先回顾策略梯度的推导，在一个能手算的 4 臂老虎机上逐行验证；然后证明基线不改变期望，并看清楚它为什么只能依赖状态、不能依赖动作；接着计算方差，求出最好的基线；最后分析 GRPO 的组均值：它让梯度的期望整体缩小为原来的 $(G-1)/G$，方向完全不变。如果你读过《深度学习入门4：强化学习》的附录 D，前两步的写法会很眼熟。

## C.1 准备：策略梯度与对数导数技巧

先固定一个问题（prompt）$x$。语言模型对它生成回答 $y$ 的概率是 $\pi_\theta(y)$，回答得到的奖励是 $r(y)$，目标是让期望奖励 $J(\theta) = \mathbb E_{y\sim\pi_\theta}[r(y)]$ 尽量大。回答的长度有上限（`max_new_tokens`），所以所有可能的回答是有限多个，期望可以写成求和，求和与求导也可以交换顺序。推导如下：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_y \pi_\theta(y)\, r(y)\\[-3pt]
&\quad{\footnotesize\text{（期望写成求和）}}\\[2pt]
&= \sum_y \nabla_\theta \pi_\theta(y)\; r(y)\\[-3pt]
&\quad{\footnotesize\text{（}r\text{ 与 }\theta\text{ 无关；有限和与求导交换）}}\\[2pt]
&= \sum_y \pi_\theta(y)\,\nabla_\theta \log \pi_\theta(y)\; r(y)\\[-3pt]
&\quad{\footnotesize\text{（对数导数技巧）}}\\[2pt]
&= \mathbb E_{y\sim\pi_\theta}\big[\,r(y)\,\nabla_\theta \log\pi_\theta(y)\,\big]\\[-3pt]
&\quad{\footnotesize\text{（写回期望）}}
\end{aligned}
$$

第三行用到的**对数导数技巧**（log-derivative trick）只是链式法则：$\nabla\log\pi = \nabla\pi / \pi$，两边乘以 $\pi$ 就得到 $\nabla\pi = \pi\,\nabla\log\pi$。它的好处是把“对概率求导”变成了“在采样出来的回答上求 $\log\pi$ 的导数”，后者正是反向传播能算的东西。$\nabla_\theta\log\pi_\theta(y)$ 经常出现，我们把它简记为 $\mathbf s(y)$。对语言模型来说，$\log\pi_\theta(y) = \sum_t \log\pi_\theta(y_t \mid x, y_{<t})$，所以 $\mathbf s(y)$ 就是回答里每个 token 的对数概率梯度之和。

为了能手算，我们用最简单的“语言模型”：回答只有一个 token，从 4 个候选里选，这就是 **4 臂老虎机**（《深度学习入门4：强化学习》第 1 章的老虎机问题）。策略是 $\pi = \operatorname{softmax}(\boldsymbol\theta)$，这时

$$
\mathbf s(a) = \nabla_\theta\log\pi_a = \mathbf e_a - \boldsymbol\pi
$$

其中 $\mathbf e_a$ 是第 $a$ 个分量为 1 的 one-hot 向量。这个公式来自 $\log\pi_a = \theta_a - \log\sum_j e^{\theta_j}$，对 $\theta_j$ 求导得 $\delta_{aj} - \pi_j$。取 $\boldsymbol\theta = \mathbf 0$，于是 $\boldsymbol\pi = (\frac14, \frac14, \frac14, \frac14)$；奖励 $\mathbf r = (0, 0, 1, 0)$，也就是只有第 2 号臂（从 0 数起）是“答对”。手算：

$$
\begin{aligned}
J &= \textstyle\sum_a \pi_a r_a = \tfrac14\\
\nabla J &= \textstyle\sum_a \pi_a r_a (\mathbf e_a - \boldsymbol\pi) = \tfrac14(\mathbf e_2 - \boldsymbol\pi)\\
&= \big(-\tfrac1{16},\, -\tfrac1{16},\, \tfrac3{16},\, -\tfrac1{16}\big)
\end{aligned}
$$

梯度的意思很直观：提高答对那一臂的 logit，压低其余三臂。下面逐行数值验证推导的每一步：

```verify id=C-a title="4 臂老虎机：逐行验证策略梯度的推导"
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

theta = np.zeros(4)
r = np.array([0.0, 0.0, 1.0, 0.0])
pi = softmax(theta)
S = np.eye(4) - pi                      # 第 a 行是 s(a) = e_a − π

def J(th):
    return softmax(th) @ r

h = 1e-6                                # 中心差分的步长
grad_fd = np.array([(J(theta + h * e) - J(theta - h * e)) / (2 * h)
                    for e in np.eye(4)])                        # 第 1 行
dpi = np.diag(pi) - np.outer(pi, pi)    # dpi[a, j] = ∂π_a/∂θ_j
grad_sum = r @ dpi                                              # 第 2 行
grad_score = (pi * r) @ S                                       # 第 3、4 行
logpi_fd = np.array([(np.log(softmax(theta + h * e)) -
                      np.log(softmax(theta - h * e))) / (2 * h)
                     for e in np.eye(4)]).T                     # ∇log π_a
print("∇J =", np.array2string(grad_score, precision=4))
assert np.allclose(grad_fd, grad_sum, atol=1e-9)
assert np.allclose(grad_sum, grad_score)
assert np.allclose(logpi_fd, S, atol=1e-8)                      # s(a) = e_a − π
assert np.allclose(grad_score, [-1/16, -1/16, 3/16, -1/16])     # 手算的结果
```

## C.2 基线不改变期望

关键是一条引理：**$\mathbf s(y)$ 的期望是零向量**。推导如下：

$$
\begin{aligned}
\mathbb E_{y\sim\pi_\theta}\big[\mathbf s(y)\big] &= \sum_y \pi_\theta(y)\,\nabla_\theta\log\pi_\theta(y)\\[-3pt]
&\quad{\footnotesize\text{（期望的定义）}}\\[2pt]
&= \sum_y \nabla_\theta\pi_\theta(y)\\[-3pt]
&\quad{\footnotesize\text{（对数导数技巧，反过来用）}}\\[2pt]
&= \nabla_\theta \sum_y \pi_\theta(y)\\[-3pt]
&\quad{\footnotesize\text{（有限和与求导交换）}}\\[2pt]
&= \nabla_\theta\, 1 = \mathbf 0\\[-3pt]
&\quad{\footnotesize\text{（概率之和恒为 1）}}
\end{aligned}
$$

最后一行是整个证明的核心：概率之和永远是 1，不管 $\theta$ 怎么变，所以它对 $\theta$ 的导数是 0。直观地说，提高一部分回答的概率，就必然压低另一部分，按概率加权平均下来，“推力”正好抵消。

有了引理，带基线的情形一步就出来了。只要 $b$ 不依赖这个回答 $y$：

$$
\begin{aligned}
\mathbb E\big[(r - b)\,\mathbf s\big] &= \mathbb E[r\,\mathbf s] - b\,\mathbb E[\mathbf s]\\[-3pt]
&\quad{\footnotesize\text{（期望的线性）}}\\[2pt]
&= \nabla_\theta J(\theta) - b\cdot\mathbf 0\\[-3pt]
&\quad{\footnotesize\text{（C.1 节的结论；引理）}}\\[2pt]
&= \nabla_\theta J(\theta)
\end{aligned}
$$

这里把 $r(y)$、$\mathbf s(y)$ 简写成了 $r$、$\mathbf s$。第一行能把 $b$ 提到期望外面，靠的正是“$b$ 不随 $y$ 变化”。这就是“无偏”：不管 $b$ 取什么值，估计量的期望都是真正的梯度。注意前提“$b$ 不随 $y$ 变化”：$b$ 可以依赖问题 $x$，也可以依赖与 $y$ 独立的其他东西，但不能依赖 $y$ 本身。在 4 臂老虎机上，期望可以精确地按 4 个臂加权求和：

```verify id=C-b title="E[s] = 0，所以无论 b 取多少，期望梯度都不变"
print("E[s] =", np.array2string(pi @ S, precision=4))
assert np.allclose(pi @ S, 0)                                   # 引理
for b in (0.0, 0.25, 1.0, -3.0, 10.0):
    expected = (pi * (r - b)) @ S          # Σ_a π_a (r_a − b) s(a)
    vec = np.array2string(expected, precision=4)
    print(f"b = {b:5.2f}：E[(r − b)s] = {vec}")
    assert np.allclose(expected, grad_score)
```

## C.3 基线改变的是方差

期望不变，基线又图什么呢？图的是方差。如果你读过《深度学习入门4：强化学习》9.3 节，一定记得那个考试的例子：三个人的分数是 90、40、50，方差是 466.67；减去各自往常的平均分（82、46、49）之后，剩下 8、−6、1，方差降到 32.67。基线做的是同一件事：先减去“预期能拿到的奖励”，剩下的才是“这次比预期好多少”。

我们来精确地算一下。估计量 $\mathbf g_b = (r - b)\,\mathbf s$ 是一个向量，用它各个分量的方差之和（总方差）来衡量波动：

$$
\begin{aligned}
V(b) &= \mathbb E\big[\|\mathbf g_b\|^2\big] - \big\|\mathbb E[\mathbf g_b]\big\|^2\\[-3pt]
&\quad{\footnotesize\text{（方差的定义）}}\\[2pt]
&= \mathbb E[r^2\|\mathbf s\|^2] - 2b\,\mathbb E[r\|\mathbf s\|^2]\\
&\qquad + b^2\,\mathbb E[\|\mathbf s\|^2] - \|\nabla J\|^2\\[-3pt]
&\quad{\footnotesize\text{（展开）}}\\[2pt]
&= V(b^*) + \mathbb E[\|\mathbf s\|^2]\,(b - b^*)^2\\[-3pt]
&\quad{\footnotesize\text{（对 }b\text{ 配方）}}
\end{aligned}
$$

第一行是“方差 = 二阶矩 − 均值的平方”，用在向量的每个分量上再求和。展开之后的最后一项 $\|\nabla J\|^2$ 用到了 C.2 节：均值 $\mathbb E[\mathbf g_b] = \nabla J$，与 $b$ 无关。最后一行的 $b^*$ 是

$$
b^* = \frac{\mathbb E\big[r\,\|\mathbf s\|^2\big]}{\mathbb E\big[\|\mathbf s\|^2\big]}
$$

$V(b)$ 是一条开口向上的抛物线，最低点在 $b^*$。$b^*$ 是奖励的加权平均，权重是 $\|\mathbf s\|^2$；如果每个回答的 $\|\mathbf s\|^2$ 都一样，$b^*$ 就是平均奖励 $\mathbb E[r]$，这也是实践中最常用的基线。

回到 4 臂老虎机。每一臂的 $\|\mathbf s\|^2 = (\frac34)^2 + 3\times(\frac14)^2 = \frac34$，都一样，所以 $b^* = \mathbb E[r] = \frac14$。再由 C.1 节，$\|\nabla J\|^2 = 3\times(\frac1{16})^2 + (\frac3{16})^2 = \frac3{64}$。手算两个值：$V(0) = \frac34\,\mathbb E[r^2] - \|\nabla J\|^2 = \frac3{16} - \frac3{64} = \frac9{64} \approx 0.141$；$V(\frac14) = \frac34\operatorname{Var}[r] - \frac3{64} = \frac9{64} - \frac3{64} = \frac6{64} \approx 0.094$。方差降低了三分之一。反过来，基线选得太离谱，方差反而会变大：$b = 1$ 时 $V = \frac{33}{64} \approx 0.516$。基线不是越大越好，也不是“有就比没有强”。

```verify id=C-c title="方差是关于 b 的抛物线，最低点在 b*；蒙特卡洛也看得到"
# 先核对《深度学习入门4》9.3 节的考试例子
exam = np.array([90.0, 40.0, 50.0])
usual = np.array([82.0, 46.0, 49.0])                  # 各自往常的平均分
print(f"考试分数的方差 {exam.var():.2f}，减去往常的平均分之后 {(exam - usual).var():.2f}")
assert round(exam.var(), 2) == 466.67 and round((exam - usual).var(), 2) == 32.67

def total_var(pi, S, r, b):
    """单个样本的估计量 (r − b) s 的总方差（精确，按各臂概率加权）。"""
    g = (r - b)[:, None] * S
    mean = pi @ g
    return pi @ (g ** 2).sum(axis=1) - mean @ mean

def best_baseline(pi, S, r):
    s2 = (S ** 2).sum(axis=1)                     # 每一臂的 ‖s‖²
    return (pi * r * s2).sum() / (pi * s2).sum()

b_star = best_baseline(pi, S, r)
for b in (0.0, 0.25, 1.0):
    print(f"b = {b:4.2f}：V(b) = {total_var(pi, S, r, b):.6f}")
assert abs(b_star - 0.25) < 1e-12
assert np.isclose(grad_score @ grad_score, 3 / 64)
assert np.isclose(total_var(pi, S, r, 0.0), 9 / 64)
assert np.isclose(total_var(pi, S, r, 0.25), 6 / 64)
assert np.isclose(total_var(pi, S, r, 1.0), 33 / 64)

# 换一个不均匀的策略：‖s‖² 各不相同，b* 不再等于 E[r]
pi2 = softmax(np.array([1.0, 0.0, 0.0, -1.0]))
S2, r2 = np.eye(4) - pi2, np.array([0.0, 1.0, 0.0, 1.0])
b2 = best_baseline(pi2, S2, r2)
s2_mean = pi2 @ (S2 ** 2).sum(axis=1)
print(f"不均匀的策略：E[r] = {pi2 @ r2:.4f}，b* = {b2:.4f}")
for b in (0.0, pi2 @ r2, b2, 2.0):
    v = total_var(pi2, S2, r2, b)
    assert np.isclose(v, total_var(pi2, S2, r2, b2) + s2_mean * (b - b2) ** 2)
    print(f"  b = {b:.4f}：V(b) = {v:.4f}")
v_none, v_er, v_best = (total_var(pi2, S2, r2, b) for b in (0.0, pi2 @ r2, b2))
assert v_er < 1.2 * v_best < v_none     # 用 E[r] 当基线，方差已接近最小值；不加基线时大得多

# 蒙特卡洛：用 8 个样本估计一次梯度，重复 4000 次
rng = np.random.default_rng(0)
arms = rng.choice(4, size=(4000, 8), p=pi)
for b in (0.0, 0.25):
    est = ((r[arms] - b)[..., None] * S[arms]).mean(axis=1)   # (4000, 4)
    spread = est.var(axis=0).sum()
    mean = np.array2string(est.mean(0), precision=3)
    theory = total_var(pi, S, r, b) / 8
    print(f"b = {b:.2f}：4000 次估计的平均 {mean}，"
          f"总方差 {spread:.5f}（理论 V(b)/8 = {theory:.5f}）")
    assert np.allclose(est.mean(0), grad_score, atol=0.01)     # 无偏：每一维的标准误不超过 0.002
    assert abs(spread / theory - 1) < 0.1                       # 方差的相对标准误约 2%

# 每一维也有自己的方差抛物线，最低点是这一维的最优基线 b*_k = E[r s_k²] / E[s_k²]
e_k = pi @ S ** 2                                     # E[s_k²]
b_k = (pi * r) @ S ** 2 / e_k
print(f"各维的最优基线 b*_k = {np.array2string(b_k, precision=4)}；"
      f"按 E[s_k²] 加权平均 = {b_k @ e_k / e_k.sum():.4f}")
assert np.allclose(b_k, [1 / 12, 1 / 12, 3 / 4, 1 / 12])
assert np.isclose(b_k @ e_k / e_k.sum(), b_star)        # b* 是各维之间的折中
assert np.allclose((r - 0.75) * S[:, 2], 3 / 16)        # b = 0.75：第 2 维的估计恒为 3/16
assert np.isclose(total_var(pi, S, r, 0.75), 3 * total_var(pi, S, r, b_star))   # 四维之和却是最小值的 3 倍
```

两种基线的估计平均值都落在 $\nabla J$ 附近（无偏），但 $b = \frac14$ 时，估计值的总方差小了三分之一。不均匀的那个策略说明了另一件事：$b^*$ 和 $\mathbb E[r]$ 并不相等（0.4435 对 0.2689），但用 $\mathbb E[r]$ 当基线，方差已经离最小值不远了（0.1283 对 0.1090，而不加基线是 0.2333）。

还有一个细节，下面的交互图会用到。梯度有 4 维，$V(b)$ 是四个分量的方差之和。每一维的方差 $V_k(b)$ 单独看也是一条抛物线，最低点是这一维自己的最优基线 $b^*_k = \mathbb E[r\,s_k^2]/\mathbb E[s_k^2]$，其中 $s_k$ 是 $\mathbf s$ 的第 $k$ 个分量。$b^*$ 则是各个 $b^*_k$ 按 $\mathbb E[s_k^2]$ 加权的平均，是四维之间的折中。在均匀的 4 臂老虎机上，第 2 号臂那一维的 $b^*_2 = \frac34$，其余三维都是 $\frac1{12}$，加权平均正好是 $b^* = \frac14$。$b = \frac34$ 时，每一臂给出的第 2 维估计都恰好是 $\frac3{16}$，这一维的方差是 0；可这时其余三维的方差很大，四维之和是最小值的 3 倍（验证 C-c 的最后几行）。

下面的交互图可以拖动基线 $b$。上图是 400 次“用 $N$ 个样本估计梯度”的结果分布，竖线是真正的梯度。上图只能画一维，默认是奖励最高那一臂的那一维，可以切换。下图是单个样本的方差：实线是四维之和 $V(b)$，虚线是上图那一维的 $V_k(b)$。

```widget appendix-baseline caption="拖动 b：估计值的分布始终以真梯度为中心（无偏），宽窄却在变。上图这一维在它自己的最优基线（下图虚线的最低点）处最窄；四维的方差之和（下图实线）在 b* 处最小。"
{"preset": "uniform", "n": 8}
```

## C.4 逐个 token 的基线：只能看前缀

到目前为止，基线是对整个回答而言的。语言模型的回答其实是一串 token，每个 token 都是一次动作，状态是“问题加上已经生成的前缀”。基线也可以逐个 token 地设：第 $t$ 个 token 的基线 $b_t$ 可以依赖前缀 $y_{<t}$，但不能依赖 $y_t$ 本身以及它之后的 token。记 $\mathbf s_t = \nabla\log\pi(y_t \mid y_{<t})$，证明的关键是先固定前缀，再对下一个 token 求期望：

$$
\begin{aligned}
\mathbb E\big[b_t\,\mathbf s_t\big] &= \mathbb E_{y_{<t}}\Big[\,b_t\;\mathbb E\big[\mathbf s_t \,\big|\, y_{<t}\big]\Big]\\[-3pt]
&\quad{\footnotesize\text{（先固定前缀）}}\\[2pt]
&= \mathbb E_{y_{<t}}\big[\,b_t\cdot\mathbf 0\,\big] = \mathbf 0\\[-3pt]
&\quad{\footnotesize\text{（C.2 节的引理）}}
\end{aligned}
$$

第一行能把 $b_t$ 提到内层期望的外面，是因为前缀固定之后它就是一个常数。第二行的内层期望正是 C.2 节的引理，只不过概率分布换成了“给定前缀时下一个 token 的分布”。

::: warn 基线绝不能依赖这个动作本身
如果 $b$ 依赖 $y_t$，就不能从内层期望里提出来，引理也就用不上了。《深度学习入门4：强化学习》附录 D 末尾的提示框说的正是这件事：$\mathbb E[G_t\,\nabla\log\pi(A_t\mid S_t)] \ne 0$，因为回报 $G_t$ 会随动作 $A_t$ 而变。下面的验证构造了一个依赖动作的“基线”，期望梯度立刻就偏了。
:::

用一个只有 2 个 token、词表只有 2 个词的小语言模型来验证。全部 4 种回答可以一一枚举，所以期望是精确的：

```verify id=C-d title="两个 token 的小语言模型：依赖前缀的基线无偏，依赖动作的不行"
rng = np.random.default_rng(3)
th1 = rng.normal(size=2)              # 第 1 个 token 的 logits
th2 = rng.normal(size=(2, 2))         # 第 2 个 token 的 logits（按 y1 分开）
R = rng.normal(size=(2, 2)) + 1.0     # 每种回答 (y1, y2) 的奖励

def J2(p):                            # p = 把 th1、th2 拉平成 6 个参数
    p1 = softmax(p[:2])
    return sum(p1[y1] * softmax(p[2 + 2 * y1: 4 + 2 * y1])[y2] * R[y1, y2]
               for y1 in range(2) for y2 in range(2))

params = np.concatenate([th1, th2.ravel()])
true_grad = np.array([(J2(params + 1e-6 * e) - J2(params - 1e-6 * e)) / 2e-6
                      for e in np.eye(6)])

def expected_estimate(b1, b2):
    """精确期望：Σ 概率 × [(R − b1)·s1 + (R − b2(y1, y2))·s2]"""
    p1, g = softmax(th1), np.zeros(6)
    for y1 in range(2):
        p2 = softmax(th2[y1])
        for y2 in range(2):
            s1, s2 = np.zeros(6), np.zeros(6)
            s1[:2] = np.eye(2)[y1] - p1                     # ∇log π(y1)
            s2[2 + 2 * y1: 4 + 2 * y1] = np.eye(2)[y2] - p2  # ∇log π(y2 | y1)
            g += p1[y1] * p2[y2] * ((R[y1, y2] - b1) * s1
                                    + (R[y1, y2] - b2(y1, y2)) * s2)
    return g

no_base = expected_estimate(0.0, lambda y1, y2: 0.0)
prefix = expected_estimate(0.7, lambda y1, y2: [0.3, -1.2][y1])  # 只看前缀
action = expected_estimate(0.7, lambda y1, y2: R[y1, y2])        # 看了 y2 本身
for name, g in (("不加基线", no_base), ("依赖前缀", prefix), ("依赖动作", action)):
    print(f"{name}：与真梯度的最大偏差 {np.abs(g - true_grad).max():.2e}")
assert np.abs(no_base - true_grad).max() < 1e-8
assert np.abs(prefix - true_grad).max() < 1e-8
assert np.abs(action - true_grad).max() > 0.1
```

DLFS6 的 GRPO 走的是最简单的一种：同一个回答里的每个 token 都用同一个优势 $A = r - b$，$b$ 是同一个问题下一组回答的平均奖励。在奖励只在末尾给出、不打折扣的情况下，这正是带基线的 REINFORCE（原书的实现在几处细节上与它不同，C.5 节末尾再说）。不过，这个平均里也包含了回答自己的奖励，下一节就来处理这个问题。

## C.5 GRPO 的组均值：梯度缩小为 (G−1)/G

GRPO 对同一个问题采样 $G$ 个回答 $y_1, \dots, y_G$，得到奖励 $r_1, \dots, r_G$，用组内平均当基线。DLFS6 的写法只有两行：

```listing ch03/09_grpo.py#L88-L89
```

麻烦在于，组均值里包含了 $r_i$ 自己，它依赖 $y_i$，不满足 C.2 节的前提。那么期望变成了什么呢？把第 $i$ 项单独拿出来算：

$$
\begin{aligned}
A_i &= r_i - \frac1G\sum_{j=1}^G r_j\\[-3pt]
&\quad{\footnotesize\text{（组均值当基线）}}\\[2pt]
&= \Big(1 - \frac1G\Big)\,r_i - \frac1G\sum_{j\ne i} r_j\\[-3pt]
&\quad{\footnotesize\text{（把 }r_i\text{ 单独拿出来）}}\\[2pt]
\mathbb E[A_i\,\mathbf s_i] &= \Big(1 - \frac1G\Big)\,\mathbb E[r_i\,\mathbf s_i]\\
&\qquad - \frac1G\sum_{j\ne i}\mathbb E[r_j\,\mathbf s_i]\\[-3pt]
&\quad{\footnotesize\text{（期望的线性）}}\\[2pt]
&= \frac{G-1}{G}\,\nabla J\\
&\qquad - \frac1G\sum_{j\ne i}\mathbb E[r_j]\;\mathbb E[\mathbf s_i]\\[-3pt]
&\quad{\footnotesize\text{（C.1 节；独立）}}\\[2pt]
&= \frac{G-1}{G}\,\nabla J\\[-3pt]
&\quad{\footnotesize\text{（C.2 节的引理）}}
\end{aligned}
$$

这里 $\mathbf s_i = \mathbf s(y_i)$。标着“C.1 节；独立”的那一步有两处依据：第一项用到 C.1 节的 $\mathbb E[r\,\mathbf s] = \nabla J$；第二项里，$y_j$ 与 $y_i$ 是独立采样的，所以 $r_j$ 与 $\mathbf s_i$ 独立，乘积的期望可以拆成期望的乘积。最后一步再用 C.2 节的引理 $\mathbb E[\mathbf s_i] = \mathbf 0$。

所以组均值基线并不是严格无偏的，但偏差只是一个常数倍：方向完全正确，长度缩小为 $(G-1)/G$。$G = 8$ 时是 $7/8$。再配合 Adam 这类对梯度尺度不敏感的优化器（[附录 D](appendix-d.html)），这个常数几乎没有影响。DLFS6 还把梯度裁剪到范数 1，这时 $7/8$ 的全部效果，相当于把裁剪的阈值放宽到 $8/7$（[D.4 节](appendix-d.html#sD-4)的提示框）。

如果想要严格无偏，可以把自己排除在外，用其余 $G - 1$ 个回答的平均当基线，这叫 **RLOO**（REINFORCE Leave-One-Out）。上面推导的第二行其实已经说明了两者的关系：

$$
\begin{aligned}
A_i^{\text{GRPO}} &= \frac{G-1}{G}\Big(r_i - \frac{1}{G-1}\sum_{j\ne i} r_j\Big)\\
&= \frac{G-1}{G}\,A_i^{\text{RLOO}}
\end{aligned}
$$

RLOO 的基线只依赖其他回答，与 $y_i$ 独立，满足 C.2 节的前提，所以严格无偏（[Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740) 第 2.3 节；该估计量最早由 Kool 等人在 2019 年提出；2026-10 查证）。下面在 4 臂老虎机上枚举一组 $G$ 个回答的全部 $4^G$ 种可能，精确计算两种估计量的期望：

```verify id=C-e title="枚举全部 4^G 种分组：GRPO 的期望是真梯度的 (G−1)/G，RLOO 恰好等于真梯度"
from itertools import product

def expected_group_grad(G, leave_one_out):
    total = np.zeros(4)
    for group in product(range(4), repeat=G):       # 一组 G 个回答的全部可能
        g = list(group)
        prob, rew = np.prod(pi[g]), r[g]
        if leave_one_out:
            A = (np.array([rew[i] - np.delete(rew, i).mean() for i in range(G)])
                 if G > 1 else np.zeros(1))
        else:
            A = rew - rew.mean()                     # 09_grpo.py 的第 89 行
        total += prob * (A[:, None] * S[g]).mean(axis=0)
    return total

def times_true(g):                                  # g 是真梯度的几倍
    return g @ grad_score / (grad_score @ grad_score)

for G in (1, 2, 3, 4):
    grpo = expected_group_grad(G, leave_one_out=False)
    line = f"G = {G}：GRPO 的期望 = {times_true(grpo):.4f} × ∇J"
    assert np.allclose(grpo, (G - 1) / G * grad_score)       # 恰好 (G−1)/G 倍
    if G > 1:                                         # RLOO 至少要 2 个回答
        rloo = expected_group_grad(G, leave_one_out=True)
        line += f"，RLOO = {times_true(rloo):.4f} × ∇J"
        assert np.allclose(rloo, grad_score)
    print(line)

# 恒等式：A_GRPO = (G−1)/G · A_RLOO
rew = rng.random(5)
a_rloo = np.array([rew[i] - np.delete(rew, i).mean() for i in range(5)])
assert np.allclose(rew - rew.mean(), 4 / 5 * a_rloo)
```

::: confusion 组里只有一个回答时，为什么完全学不动？
直觉上，一个回答也有奖励，总该能提供一点信号。但看看优势的定义：$G = 1$ 时组均值就是 $r_1$ 自己，$A_1 = r_1 - r_1 = 0$，不管答对答错都是 0。上面的验证也显示，$G = 1$ 时期望梯度是 $0 \times \nabla J$。同样的道理，一组回答全对或全错时，所有优势都是 0，这一组对梯度毫无贡献。可以带走的结论：**GRPO 只从同一组回答之间的差异里学习**，没有差异就没有信号。
:::

::: confusion G = 2 的 GRPO 有意义吗？
两个回答的平均“好像太粗糙了”，这个直觉很自然。不过代入公式：$A_1 = r_1 - \frac{r_1 + r_2}{2} = \frac{r_1 - r_2}{2}$，正好是 RLOO 优势 $r_1 - r_2$ 的一半。所以 $G = 2$ 的期望梯度是 $\frac12\nabla J$：方向完全正确，只是幅度减半，而 Adam 对梯度的整体缩放几乎不敏感（[附录 D](appendix-d.html)）。它真正的代价是信号太少：奖励只有对错两种、答对概率为 $p$ 时，一组产生信号的概率只有 $2p(1-p)$，$p = 0.5$ 时是 50%，而 $G = 8$ 时是 $1 - p^8 - (1-p)^8 \approx 99\%$。下面的验证给出具体数字。可以带走的结论：**$G = 2$ 不是错的，只是浪费：$p = 0.5$ 时，一半的组白白生成了回答，却没有梯度。**
:::

```verify id=C-f title="G 越大，越多的组有信号；G = 2 是一半的 RLOO"
for p in (0.1, 0.5, 0.9):
    # 组内奖励不全相同（有信号）的概率
    sig = {G: 1 - p**G - (1 - p)**G for G in (1, 2, 4, 8)}
    cells = "，".join(f"G = {G} 有信号 {v:.1%}" for G, v in sig.items())
    print(f"p = {p}：{cells}")
    assert np.isclose(sig[1], 0) and np.isclose(sig[2], 2 * p * (1 - p))

r1, r2 = 1.0, 0.0                       # G = 2：A = (r1 − r2) / 2
A = np.array([r1, r2]) - np.mean([r1, r2])
assert np.allclose(A, 0.5 * np.array([r1 - r2, r2 - r1]))   # 一半的 RLOO 优势
```

最后，把这一节的结论接回 DLFS6 的代码。3.9 节的 `grpo_loss` 带有 3.8 节讲的比率和裁剪。理想情况下，每次采样之后的第一次更新，新旧策略完全相同，比率恒为 1，这时 `grpo_loss` 的梯度恰好是本附录的估计量 $-\frac1N\sum_i A_i\,\mathbf s(y_i)$，其中 $N$ 是样本数。可是 DLFS6 原样运行时并不满足这个前提：更新时 `model` 处于 train 模式，`old_model` 却处于 eval 模式，dropout 让第一次更新的比率就偏离了 1（3.9 节测得过半 token 的比率落在 $[0.8, 1.2]$ 之外，[验证 3.9-g](ch03.html#v-3.9-g)）。所以下面的本地测试从 `ch03/09_grpo.py` 里取出 `grpo_loss`，在一个小 CodeBot 上分两步核对：先关掉 dropout，确认梯度恰好是本附录的估计量；再打开 dropout，确认比率和梯度都偏了。

```localcheck verify/test_appendix.py::test_c_grpo_grad needs="PyTorch" title="比率为 1 时，grpo_loss 的梯度就是带组均值基线的策略梯度；打开 dropout 后比率与梯度都偏了"
```

这个偏差的代价不小。本书在 RTX 3060 Ti 上把 `09_grpo.py` 各跑了一次：原样运行时，第 51 轮贪心解码的准确率是 60.5%；从同一次运行的 SFT checkpoint 出发，只在更新前加一行 `model.eval()`，第 51 轮就到了 98.8%。

```verify id=C-g title="运行记录：更新前加一行 model.eval()，第 51 轮的准确率从 60.5% 变成 98.8%" files=data/runs/codebot-3060ti/09_grpo.metrics.jsonl,data/runs/codebot-grpo-evalfix-3060ti/09_grpo.metrics.jsonl,data/runs/codebot-grpo-evalfix-3060ti/report.json
import json

def grpo_acc(run_id):                   # 第 it 轮贪心解码的准确率（%）
    with open(f"data/runs/{run_id}/09_grpo.metrics.jsonl", encoding="utf-8") as f:
        return {row["it"]: row["acc"] for row in map(json.loads, f)}

with open("data/runs/codebot-grpo-evalfix-3060ti/report.json", encoding="utf-8") as f:
    fix = json.load(f)
patches = fix["steps"][0]["patches"]
as_is, fixed = grpo_acc("codebot-3060ti"), grpo_acc("codebot-grpo-evalfix-3060ti")
print(f"唯一的改动：{patches[0]['new'].strip().splitlines()[0]}")
print(f"出发点：{fix['seeded'][0]['path']}，来自 {fix['seeded'][0]['from']}")
for it in (51, 500):
    print(f"第 {it:3d} 轮：原样 {as_is[it]:5.1f}%，加 model.eval() 之后 {fixed[it]:5.1f}%")
assert len(patches) == 1 and patches[0]["new"].lstrip().startswith("model.eval()")
assert fix["seeded"][0]["from"] == "tutorial run codebot-3060ti"
assert (as_is[51], as_is[500]) == (60.5, 86.4) and (fixed[51], fixed[500]) == (98.8, 100.0)
```

严格地说，原书的实现与本附录的估计量还有两处差别（[3.9 节](ch03.html#s3-9)）。一是回答先被解码成字符串，到了 `get_batch` 里再重新分词，训练用的 token 不一定是采样出来的那一串。二是 EOS 不在掩码里（[验证 3.9-b](ch03.html#v-3.9-b)），“什么时候停下”这一步不参与梯度。少了这一步，C.2 节的引理就不成立了：引理的求和要遍历所有可能的下一个 token，正是 EOS 那一项把总和补成了 0。去掉它，剩下的期望是 $-\pi(\text{EOS})\,\mathbf s(\text{EOS})$，基线乘上它也就不再是 0。下面在一个只有 a、b、EOS 三个候选的位置上验证：

```verify id=C-h title="EOS 不在掩码里：剩下的 s 的期望不是 0，基线不再无偏"
pi3 = np.array([0.5, 0.3, 0.2])          # 下一个 token 是 a、b 或 EOS
S3 = np.eye(3) - pi3                     # s(v) = e_v − π
full = pi3 @ S3                          # 三个候选都进梯度：C.2 节的引理
masked = pi3[:2] @ S3[:2]                # 采到 EOS 时，这一步不进梯度
def tidy(v):                             # 去掉 1e-17 量级的舍入误差再打印
    return np.array2string(np.round(v, 12) + 0.0, precision=4)

print(f"E[s]，全部候选：{tidy(full)}；EOS 不在掩码里：{tidy(masked)}")
assert np.allclose(full, 0)
assert np.allclose(masked, -pi3[2] * S3[2]) and np.abs(masked).max() > 0.1
```

所以本附录的无偏性和 $(G-1)/G$，描述的是理想化的估计量；上面的本地测试核对的，是给定 token ids 时 `grpo_loss` 的梯度。

::: cs336 A5：基线、方差与组归一化
CS336 作业 5（2026 版）的理论部分从 REINFORCE 讲起，有一道关于基线与方差的练习，还要求分析“除以组内标准差”怎样改变了优化的目标。它的变体表里，DLFS6 的做法（只减组均值、不除以标准差、损失按常数归一化）对应 Dr_GRPO 那一行（[A5 仓库](https://github.com/stanford-cs336/assignment5-alignment)，2026-10 查证）。
:::

## 小结

本附录证明了基线的两条性质。第一，**无偏**：由于概率之和恒为 1，$\mathbb E[\nabla\log\pi] = \mathbf 0$，所以减去任何不依赖当前回答的 $b$，期望梯度都不变；逐个 token 设基线时，$b_t$ 可以依赖前缀，不能依赖这个 token 本身。第二，**方差**是关于 $b$ 的抛物线，最低点 $b^*$ 是按 $\|\nabla\log\pi\|^2$ 加权的平均奖励，一般不等于 $\mathbb E[r]$；不过在例子里，用 $\mathbb E[r]$ 当基线，方差已经接近最小值（0.128 对 0.109，不加基线是 0.233）。基线选得离谱，方差反而会变大。

GRPO 的组均值包含了回答自己的奖励，严格说不满足前提，但代价只是让期望梯度整体乘以 $(G-1)/G$，方向不变；它与 RLOO 只差这个常数。$G = 1$ 时梯度恒为 0，$G = 2$ 时是一半的 RLOO，有效但浪费。回到 DLFS6 的代码，这些结论描述的是理想化的估计量：原书更新时开着 dropout，比率从第一次更新起就不是 1；EOS 不在掩码里，“何时停下”也不参与梯度。

### 本附录所学的内容

- 策略梯度 $\nabla J = \mathbb E[r\,\nabla\log\pi]$ 来自对数导数技巧；softmax 策略的 $\nabla\log\pi_a = \mathbf e_a - \boldsymbol\pi$（[验证 C-a](#v-C-a)）。
- $\mathbb E[\nabla\log\pi] = \mathbf 0$，所以不依赖回答的基线不改变期望梯度（[验证 C-b](#v-C-b)）。
- 估计量的方差是 $b$ 的抛物线，最低点 $b^* = \mathbb E[r\|\mathbf s\|^2]/\mathbb E[\|\mathbf s\|^2]$，是各维最优基线的加权平均；均匀的 4 臂老虎机上方差从 $9/64$ 降到 $6/64$，$b = 1$ 时反而升到 $33/64$（[验证 C-c](#v-C-c)）。
- 逐 token 的基线可以依赖前缀，依赖动作本身就会带偏（[验证 C-d](#v-C-d)）。
- GRPO 组均值的期望梯度是 $\frac{G-1}{G}\nabla J$，RLOO 恰好是 $\nabla J$，两者的优势只差常数倍（[验证 C-e](#v-C-e)）。
- $G = 1$ 没有梯度；$G = 2$ 有信号的概率是 $2p(1-p)$（[验证 C-f](#v-C-f)）。
- 原书的 `grpo_loss` 要在比率为 1、EOS 也进梯度时才等于这个估计量；更新时开着 dropout，比率就不是 1，加一行 `model.eval()`，第 51 轮的准确率从 60.5% 变成 98.8%（[验证 C-g](#v-C-g)、[验证 C-h](#v-C-h)）。

基线解决了“往哪个方向走”的噪声问题，走多远则交给优化器。[附录 D](appendix-d.html) 看 Adam 的偏差校正：为什么要除以 $1-\beta^t$，不除又会怎样。
