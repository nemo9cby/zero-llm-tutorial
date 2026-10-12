---
chapter: B
slug: appendix-b
order: 120
kind: appendix
title: Attention 缩放效果的证明
lede: 2.3 节用一万次采样看到：q·k 的方差约等于 d，除以 √d 之后约等于 1。本附录把这个经验变成证明，弄清楚它依赖哪些假设，再看看 √d 怎样让 softmax 不再随维度饱和。
pyodide_packages: [numpy]
---

[2.3 节](ch02.html#s2-3)做了一个实验：随机生成两个 10 维向量 $\mathbf q$ 和 $\mathbf k$，各分量都取自标准正态分布，计算点积 $\mathbf q\cdot\mathbf k$，重复一万次。点积的方差约为 10，除以 $\sqrt{10}$ 之后约为 1：

```listing ch02/03_attn_scaling.py#L24-L38
```

于是 Attention 的公式里多了一个 $\sqrt d$：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V
$$

CodeBot 的模型里也有对应的两行，其中 `D` 是每个头的维度：

```listing codebot/model.py#L33-L34
```

不过，实验只能说明“这一次大约如此”。本附录给出证明：只要 $\mathbf q$、$\mathbf k$ 的各个分量相互独立、均值为 0、方差为 1，点积的方差就**恰好**是 $d$。这也是 Transformer 原论文给出的理由，写在第 3.2.1 节的脚注 4 里（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)）。我们先准备期望和方差的几条性质，再逐行推导，每一步都配一个数值验证；然后看看假设不成立时会发生什么；最后回答缩放究竟为 softmax 解决了什么问题。

## B.1 准备：期望与方差的四条性质

推导之前，先把要用到的工具准备好。设 $X$、$Y$ 是随机变量，$c$ 是常数，$\mathbb E$ 表示期望，$\operatorname{Var}$ 表示方差：

- **性质 1**（期望的线性）：$\mathbb E[X + Y] = \mathbb E[X] + \mathbb E[Y]$，$\mathbb E[cX] = c\,\mathbb E[X]$。
- **性质 2**（独立时乘积可以拆开）：若 $X$、$Y$ 独立，则 $\mathbb E[XY] = \mathbb E[X]\,\mathbb E[Y]$。
- **性质 3**（方差的展开）：$\operatorname{Var}[X] = \mathbb E[X^2] - \mathbb E[X]^2$。
- **性质 4**（方差的缩放与相加）：$\operatorname{Var}[cX] = c^2\operatorname{Var}[X]$；若 $X$、$Y$ 独立，则 $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$。

性质 1、性质 3 和 $\operatorname{Var}[cX] = c^2\operatorname{Var}[X]$ 对任何随机变量都成立；性质 2 和“方差相加”要求独立。为了能手算，我们用一个只取三个值的随机变量来检验这些性质：$X$ 以 $1/8$ 的概率取 $-2$，以 $3/4$ 的概率取 $0$，以 $1/8$ 的概率取 $2$。它的均值是 $(-2)\cdot\frac18 + 0\cdot\frac34 + 2\cdot\frac18 = 0$，平方的均值是 $4\cdot\frac18 + 4\cdot\frac18 = 1$，所以方差是 $1 - 0^2 = 1$，正好符合“均值 0、方差 1”。下面的验证枚举所有取值组合，用分数精确计算，没有任何舍入误差；同时也演示一下，$Y = X$（不独立）时性质 2 和性质 4 会失效：

```verify id=B-a title="三值随机变量：四条性质，以及不独立时的反例"
from fractions import Fraction
from itertools import product

X = {-2: Fraction(1, 8), 0: Fraction(3, 4), 2: Fraction(1, 8)}  # 取值 → 概率

def E(f, n=1):
    """n 个相互独立、都服从 X 的变量，函数 f 的期望（枚举 3**n 种组合）。"""
    total = Fraction(0)
    for xs in product(X, repeat=n):
        p = Fraction(1)
        for x in xs:
            p *= X[x]
        total += p * f(*xs)
    return total

def Var(f, n=1):
    return E(lambda *xs: f(*xs) ** 2, n) - E(f, n) ** 2       # 性质 3

print("E[X] =", E(lambda x: x), "  Var[X] =", Var(lambda x: x))
print("独立：  E[XY] =", E(lambda x, y: x * y, 2),
      "  Var[X+Y] =", Var(lambda x, y: x + y, 2))
print("Y = X：E[XX] =", E(lambda x: x * x),
      "  Var[X+X] =", Var(lambda x: x + x))
assert E(lambda x: x) == 0 and Var(lambda x: x) == 1
assert E(lambda x, y: x * y, 2) == E(lambda x: x) ** 2        # 性质 2
assert Var(lambda x, y: x + y, 2) == 2                        # 性质 4
assert Var(lambda x: 3 * x) == 9 * Var(lambda x: x)     # Var[cX] = c²Var[X]
assert E(lambda x: x * x) != 0 and Var(lambda x: x + x) == 4  # 不独立就失效
```

## B.2 证明：q·k 的方差是 d

现在正式推导。设 $\mathbf q = (q_1, \dots, q_d)$、$\mathbf k = (k_1, \dots, k_d)$，这 $2d$ 个分量相互独立，均值都是 0，方差都是 1。点积是 $\mathbf q\cdot\mathbf k = \sum_{i=1}^d q_i k_i$。推导分三步，每一行右边写着理由。

**第 1 步：均值是 0。**

$$
\begin{aligned}
\mathbb E[\mathbf q\cdot\mathbf k] &= \mathbb E\Big[\sum_{i=1}^d q_i k_i\Big] && \text{（点积的定义）}\\
&= \sum_{i=1}^d \mathbb E[q_i k_i] && \text{（性质 1）}\\
&= \sum_{i=1}^d \mathbb E[q_i]\,\mathbb E[k_i] && \text{（性质 2：}q_i\text{ 与 }k_i\text{ 独立）}\\
&= 0 && \text{（均值都是 0）}
\end{aligned}
$$

**第 2 步：每一项 $q_i k_i$ 的方差是 1。**

$$
\begin{aligned}
\operatorname{Var}[q_i k_i] &= \mathbb E[q_i^2 k_i^2] - \mathbb E[q_i k_i]^2 && \text{（性质 3）}\\
&= \mathbb E[q_i^2]\,\mathbb E[k_i^2] - 0 && \text{（性质 2；第 1 步）}\\
&= (1 + 0^2)(1 + 0^2) = 1 && \text{（性质 3 反过来用）}
\end{aligned}
$$

第二行把性质 2 用在 $q_i^2$ 和 $k_i^2$ 上：$q_i$ 与 $k_i$ 独立，它们的平方也独立。第三行把性质 3 反过来写成 $\mathbb E[X^2] = \operatorname{Var}[X] + \mathbb E[X]^2$，再代入方差 1、均值 0。

**第 3 步：把 $d$ 项加起来，再除以 $\sqrt d$。**

$$
\begin{aligned}
\operatorname{Var}[\mathbf q\cdot\mathbf k] &= \sum_{i=1}^d \operatorname{Var}[q_i k_i] && \text{（性质 4：各项相互独立）}\\
&= d && \text{（第 2 步：每项都是 1）}\\
\operatorname{Var}\Big[\frac{\mathbf q\cdot\mathbf k}{\sqrt d}\Big] &= \frac1d\operatorname{Var}[\mathbf q\cdot\mathbf k] = 1 && \text{（性质 4：}\operatorname{Var}[cX] = c^2\operatorname{Var}[X]\text{）}
\end{aligned}
$$

第 3 步的第一行用到“各项相互独立”：$q_i k_i$ 只由第 $i$ 对分量决定，而不同的 $i$ 对应的分量相互独立，所以各项也相互独立。

证明完了。用 B.1 节的三值随机变量把每一步都精确地算一遍。以第 2 步为例，可以先心算：$q_i k_i$ 只有在两者都不为 0 时才不为 0，概率是 $\frac14\times\frac14 = \frac1{16}$，这时 $(q_i k_i)^2 = 16$，所以 $\mathbb E[q_i^2 k_i^2] = 16\times\frac1{16} = 1$。

```verify id=B-b title="数值验证每一步：d = 2、3 时精确等于 d"
term_mean = E(lambda q, k: q * k, 2)                  # 第 1 步的一项
term_var = Var(lambda q, k: q * k, 2)                 # 第 2 步
print(f"一项：E[q_i k_i] = {term_mean}，Var[q_i k_i] = {term_var}")
assert term_mean == 0 and term_var == 1
assert E(lambda q, k: (q * k) ** 2, 2) == E(lambda q: q * q) ** 2

for d in (2, 3):
    def dot(*xs):                                     # 前 d 个是 q，后 d 个是 k
        return sum(xs[i] * xs[d + i] for i in range(d))
    mean, var = E(dot, 2 * d), Var(dot, 2 * d)        # 第 1 步、第 3 步
    print(f"d = {d}：E[q·k] = {mean}，Var[q·k] = {var}，"
          f"Var[q·k/√d] = {var / d}")
    assert mean == 0 and var == d and var / d == 1
```

三值随机变量不是正态分布，结论照样精确成立。这说明证明只用到了“独立、均值 0、方差 1”，与具体的分布无关。再回到 2.3 节的设定，用正态分布做一次蒙特卡洛：

```verify id=B-c title="正态分布、一万个样本：Var[q·k] ≈ d"
import numpy as np

rng = np.random.default_rng(0)
N = 10_000
for d in (10, 64, 256):
    q = rng.standard_normal((N, d))
    k = rng.standard_normal((N, d))
    dots = (q * k).sum(axis=1)
    v, v_scaled = dots.var(), (dots / np.sqrt(d)).var()
    print(f"d = {d:3d}：Var[q·k] = {v:7.2f}，Var[q·k/√d] = {v_scaled:.4f}")
    # 一万个样本估计方差，相对标准误约 1.5%，这里允许 5%（约 3 倍标准误）
    assert abs(v / d - 1) < 0.05
```

顺带一提，2.3 节脚本打印的两个方差，比值总是恰好等于 $d$。这只是性质 4 的 $\operatorname{Var}[cX] = c^2\operatorname{Var}[X]$，与采样无关；真正需要采样来确认的，是 $\operatorname{Var}[\mathbf q\cdot\mathbf k] \approx d$ 本身。

## B.3 假设不成立时

证明有三个前提：均值为 0、方差为 1、相互独立。真实模型里的 $\mathbf q$、$\mathbf k$ 并不保证满足它们，那么结论会怎样变化呢？

先看方差。如果各分量的方差是 $\sigma_q^2$ 和 $\sigma_k^2$，第 2 步的结果就变成 $\sigma_q^2\sigma_k^2$，第 3 步变成：

$$
\operatorname{Var}[\mathbf q\cdot\mathbf k] = d\,\sigma_q^2\sigma_k^2,\qquad
\operatorname{Var}\Big[\frac{\mathbf q\cdot\mathbf k}{\sqrt d}\Big] = \sigma_q^2\sigma_k^2
$$

也就是说，除以 $\sqrt d$ 消掉的只是对维度的依赖，分数的尺度仍由 $\sigma_q$、$\sigma_k$ 决定。在真实模型里，它们由初始化决定，而且可以预先算出来。CodeBot 初始化权重和 embedding 时用的标准差都是 0.02，embedding 是 384 维，每个头 64 维（[2.8 节](ch02.html#s2-8)）。以刚初始化的第 0 层为例，一步一步来：

1. 输入是 token embedding 加位置 embedding，两者独立，所以每个分量的方差是 $0.02^2 + 0.02^2 = 0.0008$（性质 4）。
2. LayerNorm 把每个向量除以 $\sqrt{0.0008 + 10^{-5}}$，其中 $10^{-5}$ 是 `nn.LayerNorm` 默认的 eps，所以输出的方差是 $0.0008 / (0.0008 + 10^{-5}) \approx 0.9877$，略小于 1。
3. $q_i = \sum_{j=1}^{384} W_{ij}x_j$ 是 384 个独立项之和，每项的方差是 $0.02^2 \times 0.9877$，由性质 4 得 $\sigma_q^2 = 384 \times 0.02^2 \times 0.9877 \approx 0.1517$，$\sigma_k^2$ 也一样。
4. 每个头除以 $\sqrt{64} = 8$（注意不是 $\sqrt{384}$），分数的方差是 $\sigma_q^2\sigma_k^2$，标准差是 $\sigma_q\sigma_k \approx 0.1517$。

预测是 0.1517，比 1 小得多，所以刚初始化时注意力几乎是均匀的。用 NumPy 照着这个配方搭一个第 0 层来检验，顺便看看一行注意力里最大的权重有多大：

```verify id=B-d title="照着 CodeBot 的初始化：分数的标准差 ≈ 0.1517"
E_dim, H, D, eps, C = 384, 6, 64, 1e-5, 128
rng = np.random.default_rng(1)
x = rng.normal(0, 0.02, (C, E_dim)) + rng.normal(0, 0.02, (C, E_dim))
x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)
Wq = rng.normal(0, 0.02, (E_dim, H * D))
Wk = rng.normal(0, 0.02, (E_dim, H * D))
Q = (x @ Wq).reshape(C, H, D).transpose(1, 0, 2)     # (H, C, D)
K = (x @ Wk).reshape(C, H, D).transpose(1, 0, 2)
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D)       # (H, C, C)

pred = E_dim * 0.02**2 * 0.0008 / (0.0008 + eps)
print(f"预测 {pred:.4f}，实测 {scores.std():.4f}")
assert abs(scores.std() / pred - 1) < 0.03           # 随机种子不同时在 ±1% 内波动

last = scores[:, -1, :]                              # 最后一个位置看得到全部 C 个键
w = np.exp(last - last.max(1, keepdims=True))
w /= w.sum(1, keepdims=True)
print(f"最后一行注意力的最大权重：{w.max():.4f}（均匀分布是 1/{C} = {1 / C:.4f}）")
assert w.max() < 2 / C                               # 几乎均匀
```

真实的 CodeBot 是不是也这样，用 PyTorch 检查一下：取 `codebot/model.py` 里刚初始化的模型，确认它的注意力等于“除以 $\sqrt{64}$”的手算结果，并测量第 0 层分数的标准差。换三个随机种子，实测分别是 0.1511、0.1514、0.1515，与预测相差不到 1%：

```localcheck verify/test_appendix.py::test_b_codebot_scores title="真实 CodeBot：缩放因子是 √64；初始化时分数的标准差 ≈ 0.1517"
```

再看独立性。如果 $\mathbf q$ 和 $\mathbf k$ 相关，第 1 步就不成立了，点积的均值不再是 0。极端情况是 $\mathbf q = \mathbf k$：

::: think 如果 q 和 k 完全相同（各分量独立、标准正态），q·k 的均值和方差各是多少？
:::
::: answer
这时 $\mathbf q\cdot\mathbf k = \sum_i q_i^2$。均值是 $\sum_i \mathbb E[q_i^2] = d$，不再是 0。方差是 $\sum_i \operatorname{Var}[q_i^2] = d\,(\mathbb E[q_i^4] - 1) = d\,(3 - 1) = 2d$，其中 $\mathbb E[q_i^4] = 3$ 是标准正态分布的四阶矩。下面的验证单元给出了数值结果。训练好的模型里，$\mathbf q$ 和 $\mathbf k$ 正是靠这种“相关”让相关的 token 得到更高的分数；独立的假设描述的是刚初始化、什么都还没学到的状态。
:::

```verify id=B-e title="q = k 时：均值 d、方差 2d"
q = rng.standard_normal((20_000, 64))
self_dots = (q * q).sum(axis=1)
print(f"均值 {self_dots.mean():.2f}（d = 64），"
      f"方差 {self_dots.var():.1f}（2d = 128）")
assert abs(self_dots.mean() / 64 - 1) < 0.02
assert abs(self_dots.var() / 128 - 1) < 0.05
```

## B.4 方差对 softmax 意味着什么

方差是 $d$ 又怎么样呢？问题出在接下来的 softmax 上。设一个查询要和 $n$ 个键打分，分数是 $s_j = \mathbf q\cdot\mathbf k_j$。不缩放时，分数的标准差是 $\sqrt d$：$d = 64$ 时是 8，$d = 1024$ 时是 32。一行 32 个这样的分数里，第一名平均要比第二名高出 14 左右（下面的验证 B-g 会算出这个数）。softmax 把分数差变成指数，$e^{14}$ 约等于 120 万，于是几乎全部权重都落在第一名身上，注意力退化成“只看一个 token”。这就是 2.3 节说的饱和。

换个角度看，缩放就是温度。除以 $\sqrt d$ 相当于把 softmax 的温度设成 $T = \sqrt d$（[3.2 节](ch03.html#s3-2)）。反过来说，不缩放等于用温度 $1/\sqrt d$ 去读一组“正常尺度”的分数：维度越大，温度越低，softmax 越接近 argmax。

接下来用实验看这件事，我们需要大量的分数。直接生成 $n$ 个 $d$ 维的键太费时间，好在 B.1 节的性质可以帮忙。给定 $\mathbf q$ 时，$s_j = \sum_i q_i k_{ji}$ 是 $d$ 个独立正态变量的加权和，由性质 4，它的方差是 $\sum_i q_i^2 = \|\mathbf q\|^2$；而独立正态变量的加权和仍然服从正态分布，所以 $s_j \sim \mathcal N(0, \|\mathbf q\|^2)$。于是只要采样 $\mathbf q$，再用 $\|\mathbf q\|$ 乘以标准正态数，就得到了与真实采样分布相同的分数。先检验这个捷径：

```verify id=B-f title="给定 q 时，分数的方差等于 |q|²"
d = 64
q = rng.standard_normal(d)
keys = rng.standard_normal((20_000, d))
s = keys @ q
print(f"Var[s | q] = {s.var():.2f}，|q|² = {q @ q:.2f}")
assert abs(s.var() / (q @ q) - 1) < 0.05            # 2 万个样本，相对标准误约 1%
```

还要用到 softmax 的导数。softmax 的输出 $p_i$ 对输入 $s_j$ 的偏导数是 $\partial p_i/\partial s_j = p_i(\delta_{ij} - p_j)$，其中 $\delta_{ij}$ 在 $i = j$ 时为 1，否则为 0。写成矩阵，就是雅可比矩阵 $J = \operatorname{diag}(\mathbf p) - \mathbf p\mathbf p^\top$。梯度要穿过 softmax 往回传，必须乘上这个矩阵；它的大小（这里用 Frobenius 范数 $\|J\|_F$）可以看作 softmax 的“导电能力”。下面先用有限差分确认这个公式，再对 $d$ 从 1 到 4096，比较缩放前后的三个量：一行注意力的最大权重、最大权重超过 0.99 的行所占的比例，以及 $\|J\|_F$。每一行有 32 个键，每个 $d$ 重复 300 次：

```verify id=B-g title="不缩放时 softmax 随 d 饱和；缩放后与 d 无关"
def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def jacobian(p):                         # J[i, j] = p_i (δ_ij − p_j)，按行批量
    return p[:, :, None] * (np.eye(p.shape[-1]) - p[:, None, :])

s0 = np.array([[0.3, -1.0, 2.0, 0.5]])   # 用有限差分核对雅可比公式
num = np.stack([(softmax(s0 + 1e-6 * e) - softmax(s0 - 1e-6 * e))[0] / 2e-6
                for e in np.eye(4)], axis=1)
assert np.abs(num - jacobian(softmax(s0))[0]).max() < 1e-8

n, trials = 32, 300
dims = (1, 4, 16, 64, 256, 1024, 4096)
raw, scaled = [], []
for d in dims:
    qn = np.linalg.norm(rng.standard_normal((trials, d)), axis=1)
    s = qn[:, None] * rng.standard_normal((trials, n))   # s_j ~ N(0, |q|²)
    for rows, z in ((raw, s), (scaled, s / np.sqrt(d))):
        p = softmax(z)
        jf = np.sqrt((jacobian(p) ** 2).sum(axis=(1, 2)))
        rows.append((p.max(1).mean(), (p.max(1) > 0.99).mean(), jf.mean()))
    if d == 1024:                        # 不缩放时，第一名比第二名高出多少
        top2 = np.sort(s, axis=1)[:, -2:]
        gap = (top2[:, 1] - top2[:, 0]).mean()

print("max p：一行里最大的权重；sat：max p > 0.99 的比例；|J|F：雅可比范数")
print("      d |  不缩放 max p    sat   |J|F |  ÷√d max p   |J|F")
for d, (m1, s1, j1), (m2, _, j2) in zip(dims, raw, scaled):
    print(f"{d:7d} | {m1:12.3f} {s1:6.2f} {j1:6.3f} | {m2:10.3f} {j2:6.3f}")
print(f"d = 1024、不缩放时，第一名平均比第二名高 {gap:.1f}，"
      f"e^{gap:.1f} ≈ {np.exp(gap):,.0f}")
assert 10 < gap < 17

maxp_raw = [r[0] for r in raw]
maxp_scaled = [r[0] for r in scaled]
assert all(a < b for a, b in zip(maxp_raw, maxp_raw[1:]))   # 越来越尖
assert maxp_raw[-1] > 0.95 and raw[-1][1] > 0.75            # 大多已经饱和
assert max(maxp_scaled) / min(maxp_scaled) < 1.2            # 与 d 无关
assert raw[1][2] > raw[0][2]                                # ‖J‖ 先升……
assert raw[-1][2] < raw[1][2] / 5                           # ……后降
assert raw[-1][2] < scaled[-1][2] / 4                       # 大 d 时梯度很小
```

先看结论。不缩放时，最大权重随 $d$ 单调上升，$d = 4096$ 时平均超过 0.95，八成以上的行已经饱和；除以 $\sqrt d$ 之后，最大权重在各个 $d$ 下几乎相同。也就是说，缩放让注意力的“尖锐程度”不再依赖维度。

雅可比矩阵的范数值得多看一眼。缩放之后它在各个 $d$ 下保持不变；不缩放时，它在 $d$ 很大时确实趋近于 0，这正是原论文说的“把 softmax 推到梯度极小的区域”。但它并不是单调下降的：从 $d = 1$ 到 $d = 4$ 反而变大，因为适度尖锐的分布梯度最大，只有接近 one-hot 时梯度才消失。所以更准确的说法是：不缩放时，softmax 的有效温度随 $\sqrt d$ 变化，维度大到一定程度，梯度就几乎传不回去了。

下面的交互图可以拖动 $d$，切换是否缩放，看一行注意力怎样随之变化。

```widget appendix-scaling caption="不缩放时，d 越大，注意力越集中在一个键上；除以 √d 之后，注意力的形状与 d 无关。"
{"d": 64, "n": 16}
```

::: cs336 Lecture 3：让 attention 的 logits 保持在合理范围
CS336 作业 1 的 `scaled_dot_product_attention` 同样要求除以 $\sqrt{d_k}$。第 3 讲（架构与超参数）把它放进了一个更大的话题：怎样防止 attention 的 logits 变得过大。除了 $\sqrt{d_k}$，现代模型还会用 QK-norm（点积之前先把 $\mathbf q$、$\mathbf k$ 归一化）、logit soft-capping 等办法；作业 3 的训练 API 所用的模型就带有 QK-norm（[CS336 课程主页](https://cs336.stanford.edu/)）。
:::

最后补一个实现上的细节。WebBot 的模型直接调用 PyTorch 的 `F.scaled_dot_product_attention`（[8.2 节](ch08.html#s8-2)），没有显式地写除法。它默认的缩放因子是 $1/\sqrt{\text{查询的最后一维}}$，也就是每个头的维度，与 CodeBot 手写的 `D ** 0.5` 一致：

```localcheck verify/test_appendix.py::test_b_sdpa_default_scale title="SDPA 的默认缩放就是除以 √(每个头的维度)"
```

## 小结

本附录证明了 2.3 节的实验结论：只要 $\mathbf q$、$\mathbf k$ 的分量相互独立、均值为 0、方差为 1，$\operatorname{Var}[\mathbf q\cdot\mathbf k] = d$，除以 $\sqrt d$ 之后方差为 1。证明只用了期望的线性、独立时乘积的期望可以拆开、方差的展开式、独立时方差相加这四条性质，与分布的具体形状无关，我们用一个三值分布做了精确验证。

假设不成立时，结论也跟着变：方差为 $\sigma^2$ 时缩放后的方差是 $\sigma_q^2\sigma_k^2$，据此可以预先算出刚初始化的 CodeBot 的分数尺度（约 0.15）；$\mathbf q$ 与 $\mathbf k$ 相关时均值不再为 0。缩放真正解决的问题是：让 softmax 的“温度”不随维度变化。不缩放时，维度越大，注意力越接近 one-hot，梯度也就越难传回去。

### 本附录所学的内容

- 期望的线性、独立时 $\mathbb E[XY] = \mathbb E[X]\mathbb E[Y]$、$\operatorname{Var}[X] = \mathbb E[X^2] - \mathbb E[X]^2$、独立时方差相加；不独立时后两条会失效（[验证 B-a](#v-B-a)）。
- 分量独立、均值 0、方差 1 时，$\mathbb E[\mathbf q\cdot\mathbf k] = 0$、$\operatorname{Var}[\mathbf q\cdot\mathbf k] = d$，与分布无关（[验证 B-b](#v-B-b)、[验证 B-c](#v-B-c)）。
- 方差为 $\sigma_q^2$、$\sigma_k^2$ 时，缩放后的方差是 $\sigma_q^2\sigma_k^2$；刚初始化的 CodeBot 第 0 层，分数的标准差约为 0.1517（[验证 B-d](#v-B-d)）。
- $\mathbf q = \mathbf k$ 时，点积的均值是 $d$、方差是 $2d$，独立性是结论成立的前提（[验证 B-e](#v-B-e)）。
- 给定 $\mathbf q$ 时，分数服从 $\mathcal N(0, \|\mathbf q\|^2)$（[验证 B-f](#v-B-f)）。
- 不缩放时 softmax 随 $d$ 饱和，雅可比范数先升后降；除以 $\sqrt d$ 之后两者都与 $d$ 无关（[验证 B-g](#v-B-g)）。

缩放让 softmax 在任何维度下都保持同样的“温度”。[附录 C](appendix-c.html) 换到强化学习，证明 GRPO 的“减去组内平均”为什么不会把梯度带偏，以及它为什么能降低方差。
