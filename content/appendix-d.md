---
chapter: D
slug: appendix-d
order: 140
kind: appendix
title: Adam 的偏差校正
lede: Adam 的两个移动平均都从 0 起步，早期的估计偏向 0。本附录推导 E[m_t] = (1 − βᵗ)·E[g]，说明除以 1 − βᵗ 为什么能把偏差消掉；再看看不校正时更新量会怎样，以及校正修不好的东西。
pyodide_packages: [numpy]
---

[6.1 节](ch06.html#s6-1)介绍了 Adam：它维护梯度的移动平均 $m_t$（一阶矩）和梯度平方的移动平均 $v_t$（二阶矩），用 $m_t/\sqrt{v_t}$ 来决定每个参数走多远。DLFS6 在 `02_adamw.py` 里亲手实现了它，其中有两行“偏差校正”（バイアス補正）：

```listing ch06/02_adamw.py#L45-L60
```

`m_hat = m / (1 - beta1**t)`：为什么要除以 $1-\beta_1^t$？不除会怎样？本附录回答这两个问题。我们先看移动平均从 0 起步会发生什么，然后推导 $\mathbb E[m_t] = (1-\beta^t)\,\mathbb E[g]$，并换一个角度理解校正：它把移动平均的权重重新归一化。接着计算不校正时的更新量，结果多半与直觉相反。最后看看校正修不好的东西，并用 NumPy 重现 `02_adamw.py` 的输出。

## D.1 从 0 起步的移动平均

先把 Adam 的两个量写成同一种形式。设第 $t$ 步的梯度是 $g_t$，**指数移动平均**（exponential moving average）的定义是：

$$
m_t = \beta\, m_{t-1} + (1-\beta)\, g_t,\qquad m_0 = 0
$$

这里 $\beta$ 是衰减率，Adam 的默认值是 $\beta_1 = 0.9$（用于 $m$）和 $\beta_2 = 0.999$（用于 $v$，把 $g_t$ 换成 $g_t^2$）。如果你读过《深度学习入门4：强化学习》1.5 节，这正是那里用来应对非稳态问题的“固定步长”更新，步长 $\alpha$ 就是这里的 $1-\beta$；那一节的提示框也提到，初始值 $Q_0$ 会让估计带上偏差。Adam 的初始值是 0，偏差的方向也就是“偏向 0”。

用最简单的情形手算一下：梯度恒为 $g_t = 1$，$\beta = 0.9$。于是 $m_1 = 0.1$，$m_2 = 0.9\times0.1 + 0.1 = 0.19$，$m_3 = 0.9\times0.19 + 0.1 = 0.271$，以此类推。

真实的平均梯度是 1，$m_t$ 却从 0.1 开始慢慢爬升。仔细看这几个数：$0.1 = 1 - 0.9$，$0.19 = 1 - 0.9^2$，$0.271 = 1 - 0.9^3$。规律是 $m_t = 1 - 0.9^t$，所以除以 $1 - 0.9^t$ 恰好得到 1：

```verify id=D-a title="梯度恒为 1：m_t = 1 − 0.9ᵗ，校正之后恰好是 1"
beta = 0.9
m = 0.0
for t in range(1, 6):
    m = beta * m + (1 - beta) * 1.0
    m_hat = m / (1 - beta**t)
    print(f"t = {t}：m_t = {m:.5f}，1 − 0.9^t = {1 - beta**t:.5f}，"
          f"m̂_t = {m_hat:.5f}")
    assert abs(m - (1 - beta**t)) < 1e-12 and abs(m_hat - 1) < 1e-12
```

## D.2 推导：E[m_t] = (1 − βᵗ) E[g]

梯度恒定只是特例。真实的梯度带着噪声，我们要的是期望意义下的结论。这里先假设每一步梯度的期望都相同，记作 $\mathbb E[g]$（梯度的分布不随时间变化）。推导如下，每一行右边写着理由：

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t && \text{（定义）}\\
&= (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i}\, g_i && \text{（反复代入，}m_0 = 0\text{）}\\
\mathbb E[m_t] &= (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i}\,\mathbb E[g_i] && \text{（期望的线性）}\\
&= \mathbb E[g]\cdot(1-\beta)\sum_{k=0}^{t-1}\beta^{k} && \text{（}\mathbb E[g_i] = \mathbb E[g]\text{，令 }k = t - i\text{）}\\
&= \mathbb E[g]\cdot(1-\beta^t) && \text{（等比数列求和）}
\end{aligned}
$$

第二行可以手动展开前两步来确认：$m_1 = (1-\beta)g_1$，$m_2 = \beta(1-\beta)g_1 + (1-\beta)g_2$，越早的梯度乘上越多个 $\beta$。最后一行的等比数列求和是 $(1-\beta)(1 + \beta + \dots + \beta^{t-1}) = 1 - \beta^t$，把左边乘开，中间的项两两抵消就能看到。

于是 $\hat m_t = m_t/(1-\beta^t)$ 的期望正好是 $\mathbb E[g]$，偏差消失了。$v_t$ 的推导一模一样，只是把 $g$ 换成 $g^2$、把 $\beta$ 换成 $\beta_2$：$\mathbb E[v_t] = (1-\beta_2^t)\,\mathbb E[g^2]$。Adam 的原论文在第 3 节对 $v_t$ 做的正是这个推导，并指出：如果 $\mathbb E[g_i]$ 随时间缓慢变化，结论只是近似成立（[Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)）。

下面逐步数值验证：第二行（展开式）与递推式完全相等，最后一行的等比数列求和成立，再用蒙特卡洛确认期望的结论。梯度取自均值 $\mu = 0.5$、标准差 $\sigma = 1$ 的正态分布，重复 2 万次：

```verify id=D-b title="展开式、等比数列、期望：逐行验证"
import numpy as np

rng = np.random.default_rng(0)
beta1, beta2 = 0.9, 0.999

g = rng.normal(size=30)                        # 第 2 行：递推 = 展开式
m = 0.0
for gi in g:
    m = beta1 * m + (1 - beta1) * gi
unrolled = (1 - beta1) * sum(beta1**(30 - i) * g[i - 1] for i in range(1, 31))
assert abs(m - unrolled) < 1e-12

for t in (1, 7, 100):                          # 第 5 行：等比数列求和
    geometric = (1 - beta1) * sum(beta1**k for k in range(t))
    assert abs(geometric - (1 - beta1**t)) < 1e-12

mu, sigma, runs, T = 0.5, 1.0, 20_000, 50      # 第 3–5 行：蒙特卡洛
grads = rng.normal(mu, sigma, size=(runs, T))
m, v = np.zeros(runs), np.zeros(runs)
for t in range(1, T + 1):
    m = beta1 * m + (1 - beta1) * grads[:, t - 1]
    v = beta2 * v + (1 - beta2) * grads[:, t - 1] ** 2
    if t in (1, 2, 5, 10, 50):
        m_hat_mean = m.mean() / (1 - beta1**t)
        v_hat_mean = v.mean() / (1 - beta2**t)
        print(f"t = {t:2d}：E[m_t] ≈ {m.mean():.4f}"
              f"（(1 − β₁ᵗ)μ = {(1 - beta1**t) * mu:.4f}），"
              f"E[m̂_t] ≈ {m_hat_mean:.4f}，E[v̂_t] ≈ {v_hat_mean:.4f}")
        # m_t 的标准误不超过 σ·√((1 − β)/(1 + β))/√runs
        se_m = sigma * np.sqrt((1 - beta1) / (1 + beta1)) / np.sqrt(runs)
        assert abs(m.mean() - (1 - beta1**t) * mu) < 5 * se_m
        assert abs(v.mean() / (1 - beta2**t) / (mu**2 + sigma**2) - 1) < 0.05
```

最后一个断言的容差来自 $\hat v_t$ 的噪声：$t$ 很小时，$\hat v_t$ 几乎只由一两个梯度决定，波动很大（D.5 节会精确地算出它的大小）。

## D.3 校正之后：权重之和为 1

换一个角度看第二行的展开式。$m_t$ 是 $g_1, \dots, g_t$ 的加权和，权重是 $(1-\beta)\beta^{t-i}$，它们的和是多少呢？正是上面算过的 $1-\beta^t$，不到 1。差的那 $\beta^t$ 去哪儿了？它分给了初始值 $m_0 = 0$。$t$ 很小时，0 在这个“平均”里占了很大的份额，估计自然偏向 0。

校正做的事情很朴素：

$$
\hat m_t = \frac{m_t}{1-\beta^t} = \sum_{i=1}^t w_i\, g_i,\qquad w_i = \frac{(1-\beta)\,\beta^{t-i}}{1-\beta^t},\qquad \sum_{i=1}^t w_i = 1
$$

也就是说，把分给 0 的那份权重拿掉，再把剩下的权重放大，让它们加起来等于 1。这样 $\hat m_t$ 就是一个真正的加权平均：最近的梯度权重最大，越早的越小。$t = 1$ 时只有一个梯度，$\hat m_1 = g_1$。这个角度也解释了原论文的附加说明：梯度的期望随时间变化时，$\mathbb E[\hat m_t] = \sum_i w_i\,\mathbb E[g_i]$ 是近期期望的加权平均，不再等于某一个固定值。

```verify id=D-c title="校正 = 把权重重新归一化；梯度期望在变化时，得到近期期望的加权平均"
for t in (1, 3, 10, 50):
    w_raw = (1 - beta1) * beta1 ** (t - np.arange(1, t + 1))
    w = w_raw / (1 - beta1**t)
    print(f"t = {t:2d}：原始权重和 {w_raw.sum():.4f} + 分给 m₀ 的 {beta1**t:.4f}，"
          f"校正后 {w.sum():.4f}")
    assert abs(w_raw.sum() + beta1**t - 1) < 1e-12 and abs(w.sum() - 1) < 1e-12

# 梯度的期望随时间线性增长：E[g_i] = 0.1·i
t = 20
means = 0.1 * np.arange(1, t + 1)
w = (1 - beta1) * beta1 ** (t - np.arange(1, t + 1)) / (1 - beta1**t)
grads = rng.normal(means, 1.0, size=(20_000, t))
m = np.zeros(20_000)
for i in range(t):
    m = beta1 * m + (1 - beta1) * grads[:, i]
m_hat_mean = m.mean() / (1 - beta1**t)
print(f"E[m̂_20] ≈ {m_hat_mean:.4f}，Σ w_i·E[g_i] = {w @ means:.4f}，"
      f"最新的 E[g_20] = {means[-1]:.1f}")
assert abs(m_hat_mean - w @ means) < 0.01
```

## D.4 不校正会怎样：步子偏大，而不是偏小

::: think 如果去掉偏差校正，Adam 最初几步的更新量会偏大还是偏小？
:::
::: answer
直觉上会说“偏小”，因为 $m_t$ 偏向 0。但别忘了 $v_t$ 也偏向 0，而且偏得更厉害（$\beta_2$ 比 $\beta_1$ 更接近 1），它又在分母里。即使开了平方根，分母缩小得也比分子多，结果是**偏大**。下面就来算出偏大多少。
:::

Adam 的更新量是 $\mathrm{lr}\cdot\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$。忽略很小的 $\varepsilon$，不做校正时用的是 $m_t/\sqrt{v_t}$，两者之比是：

$$
\begin{aligned}
\frac{m_t}{\sqrt{v_t}} &= \frac{(1-\beta_1^t)\,\hat m_t}{\sqrt{(1-\beta_2^t)\,\hat v_t}} && \text{（代入 }m_t = (1-\beta_1^t)\hat m_t,\ v_t = (1-\beta_2^t)\hat v_t\text{）}\\
&= \underbrace{\frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}}_{f(t)}\cdot\frac{\hat m_t}{\sqrt{\hat v_t}} && \text{（把与梯度无关的部分提出来）}
\end{aligned}
$$

$f(t)$ 只与步数和两个衰减率有关。用默认值手算第 1 步：$f(1) = \frac{1-0.9}{\sqrt{1-0.999}} = \frac{0.1}{0.0316} \approx 3.16$。也就是说，不校正的第一步是校正后的 3 倍多。更糟的是，$1-\beta_1^t$ 涨得快，$\sqrt{1-\beta_2^t}$ 涨得慢，前十几步 $f(t)$ 还会继续变大：

```verify id=D-d title="不校正的更新量是校正后的 f(t) 倍：第 1 步 3.16，第 12 步达到 6.57"
t = np.arange(1, 100_001)
for b2 in (0.99, 0.999, 0.9999):
    f = (1 - beta1**t) / np.sqrt(1 - b2**t)
    k = int(f.argmax())
    print(f"β₂ = {b2}：f(1) = {f[0]:5.2f}，最大值 {f[k]:5.2f}（第 {t[k]} 步），"
          f"f(1000) = {f[999]:.3f}")
f = (1 - beta1**t) / np.sqrt(1 - beta2**t)
assert abs(f[0] - 0.1 / np.sqrt(0.001)) < 1e-12              # 手算的 3.16
assert int(f.argmax()) + 1 == 12 and abs(f.max() - 6.5685) < 1e-4
assert f[999] > 1.25 and f[-1] < 1.0001                       # 一千步后仍偏大 25%
```

默认的 $\beta_2 = 0.999$ 下，不校正的更新量在第 12 步是校正后的 6.6 倍，第 1000 步时仍然大 25%。$\beta_2$ 越接近 1，这个问题越严重。原论文第 3 节也指出了这一点：为了可靠地估计二阶矩需要平均很多个梯度（即 $\beta_2$ 接近 1），而恰恰在这种情况下，不做校正会让最初几步大得多。论文原文写的是“small value of β2”，结合上下文，指的是 $1-\beta_2$ 很小。

下面的交互图把 D.1 到 D.4 节的内容画在一起，可以调节两个衰减率，看看它们怎样影响偏差和 $f(t)$：

```widget appendix-adam caption="上图：未校正的 m_t 从 0 起步，系统性地偏低；校正后的 m̂_t 没有这种偏差，500 次的平均几乎贴着 μ，但单次的早期值只平均了很少几个梯度，波动很大。下图：不校正时的更新量是校正后的 f(t) 倍，β₂ 越接近 1，峰值越高。"
{"beta1": 0.9, "beta2": 0.999, "mu": 0.5, "sigma": 1.0}
```

校正之后，第一步又是什么样子呢？$t = 1$ 时 $\hat m_1 = g_1$、$\hat v_1 = g_1^2$，更新量是 $\mathrm{lr}\cdot g_1/(|g_1| + \varepsilon) \approx \mathrm{lr}\cdot\operatorname{sign}(g_1)$：不管梯度是 $10^{-6}$ 还是 50，每个参数都移动约 lr。

```verify id=D-e title="校正后的第一步：每个参数都移动约 lr，与梯度大小无关"
eps = 1e-8
g1 = np.array([1e-6, -3.0, 50.0])
m1, v1 = (1 - beta1) * g1, (1 - beta2) * g1**2          # 第 1 步的两个移动平均
m_hat, v_hat = m1 / (1 - beta1), v1 / (1 - beta2)       # 校正后恰好是 g 和 g²
step = m_hat / (np.sqrt(v_hat) + eps)                   # 更新量 ÷ lr
print("第一步的更新量 / lr =", ", ".join(f"{x:.6f}" for x in step))
assert np.allclose(step, np.sign(g1), atol=0.01)        # 梯度极小时 ε 才显出影响
assert abs(step[0] - 1e-6 / (1e-6 + 1e-8)) < 1e-12     # 0.990099…
```

这给了学习率一个具体的含义：在 Adam 里，lr 的单位是“参数每一步大约移动多少”，而不是“梯度乘以多少”。原论文第 2.1 节把它总结为“有效步长大致不超过 lr”。CodeBot 的权重初始化标准差是 0.02，预训练的学习率是 $3\times10^{-4}$：

```listing ch03/01_pretrain.py#L25-L25
```

所以在最初几步，每个权重大约移动它自身典型大小的 1.5%（$3\times10^{-4}/0.02$）。

::: confusion 把 loss（或 GRPO 的优势）整体乘以 10，Adam 走的步子会变大吗？
对 SGD 来说会，梯度乘以 10，步子就大 10 倍，所以很自然地以为 Adam 也一样。但 Adam 的更新量是 $\hat m_t/\sqrt{\hat v_t}$：梯度乘以 $c$，$\hat m_t$ 乘以 $c$，$\hat v_t$ 乘以 $c^2$，开方之后也是 $c$，两者抵消。手算一下：梯度恒为 2 时，$\hat m = 2$、$\sqrt{\hat v} = 2$，比值是 1；梯度恒为 20 时是 $20/20$，还是 1。所以 [附录 C](appendix-c.html) 里 GRPO 组均值带来的 $(G-1)/G$ 倍缩放，在 Adam 下几乎没有影响。不过这只在两种情况之外成立：一是梯度小到与 $\varepsilon$（$10^{-8}$）可比；二是用了梯度裁剪。DLFS6 的 GRPO 和 StoryBot 预训练都做了 `clip_grad_norm_(..., 1.0)`，裁剪的比例每一步都不同，Adam 就不再能把它整体抵消。可以带走的结论：**纯 Adam 对梯度的整体缩放不敏感，加上 ε 和裁剪之后只是近似如此。**下面的验证分别检查这三种情况。
:::

```verify id=D-f title="Adam 对梯度的整体缩放不敏感；ε 和梯度裁剪会打破这一点"
def adam_path(grads, clip=None, eps=1e-8, lr=1e-3):
    """按 Adam（无权重衰减）更新，返回每一步之后的参数。"""
    theta = np.zeros(grads.shape[1])
    m, v, path = np.zeros_like(theta), np.zeros_like(theta), []
    for t, g in enumerate(grads, start=1):
        if clip is not None and np.linalg.norm(g) > clip:  # 同 clip_grad_norm_
            g = g * clip / np.linalg.norm(g)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        step = (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
        theta = theta - lr * step
        path.append(theta.copy())
    return np.array(path)

# 50 步梯度，范数逐步变大
grads = rng.normal(size=(50, 5)) * np.linspace(0.2, 2.0, 50)[:, None]
size = np.abs(adam_path(grads)).max()
cases = {
    "梯度 ×10": (adam_path(grads), adam_path(10 * grads)),
    "梯度 ×10，裁剪到 1": (adam_path(grads, clip=1.0),
                           adam_path(10 * grads, clip=1.0)),
    "梯度 ×1e-6 与 ×1e-7": (adam_path(1e-6 * grads),
                            adam_path(1e-7 * grads)),
}
for name, (a, b) in cases.items():
    ratio = np.abs(a - b).max() / size
    print(f"{name}：参数轨迹的最大差别是参数大小的 {ratio:.1e} 倍")
diffs = [np.abs(a - b).max() / size for a, b in cases.values()]
assert diffs[0] < 1e-6 and diffs[1] > 0.05 and diffs[2] > 0.05
```

## D.5 校正修不好的：早期 v̂ 的噪声

偏差校正修正的是期望，修不好波动。D.3 节说 $\hat v_t$ 是最近若干个 $g^2$ 的加权平均；$t$ 很小时，参与平均的梯度很少，估计就很吵。衡量“相当于平均了几个样本”可以用**有效样本数** $n_{\text{eff}} = 1/\sum_i w_i^2$（权重都相等时，它就是样本个数）。把 D.3 节的 $w_i$ 代进去，再做一次等比数列求和：

$$
n_{\text{eff}}(t) = \frac{1}{\sum_{i=1}^t w_i^2} = \frac{(1-\beta^t)(1+\beta)}{(1-\beta)(1+\beta^t)}
$$

$\beta_2 = 0.999$ 时，$n_{\text{eff}}(1) = 1$，$n_{\text{eff}}(10) \approx 10$，$n_{\text{eff}}(1000) \approx 924$，最终趋近于 $\frac{1+\beta_2}{1-\beta_2} = 1999$。如果梯度服从均值为 0 的正态分布，$g^2$ 的相对标准差是 $\sqrt2$，于是 $\hat v_t$ 的相对标准差约为 $\sqrt{2/n_{\text{eff}}}$：第 1 步是 141%，第 100 步还有 14%。

```verify id=D-g title="有效样本数与 v̂ 的噪声：第 1 步相对标准差 141%"
def n_eff(beta, t):
    return (1 - beta**t) * (1 + beta) / ((1 - beta) * (1 + beta**t))

runs = 8000
grads = rng.standard_normal((runs, 100))
v = np.zeros(runs)
for t in range(1, 101):
    v = beta2 * v + (1 - beta2) * grads[:, t - 1] ** 2
    if t in (1, 10, 100):
        w = (1 - beta2) * beta2 ** (t - np.arange(1, t + 1)) / (1 - beta2**t)
        assert abs(1 / (w @ w) - n_eff(beta2, t)) < 1e-6 * t   # 公式 = 定义
        v_hat = v / (1 - beta2**t)
        rel = v_hat.std() / v_hat.mean()
        theory = np.sqrt(2 / n_eff(beta2, t))
        print(f"t = {t:3d}：n_eff = {n_eff(beta2, t):6.1f}，"
              f"v̂ 的相对标准差 {rel:.3f}（理论 √(2/n_eff) = {theory:.3f}）")
        assert abs(rel / theory - 1) < 0.1
print(f"t → ∞：n_eff → {(1 + beta2) / (1 - beta2):.0f}")
```

分母 $\sqrt{\hat v_t}$ 这么吵，早期的步长就忽大忽小。这是学习率 **warmup**（[6.3 节](ch06.html#s6-3)）的一个常见解释：先用小学习率走几百步，等 $\hat v_t$ 积累了足够多的样本再全速前进。RAdam 的论文正是从“自适应学习率在训练早期方差过大”出发，分析了 warmup 为什么有效，并提出了另一种修正（[Liu et al., 2020](https://arxiv.org/abs/1908.03265)）。

## D.6 回到 02_adamw.py

最后，用 NumPy 把 `02_adamw.py` 里的 `AdamW.step` 照抄一遍，重现脚本的输出。脚本用 `torch.manual_seed(0)` 初始化了一个 `Linear(2, 1)`，输入是 $(1, 2)$，目标是 3，学习率 0.1，训练 5 步，每步打印 loss。下面把初始权重写死（它们正是 `torch.manual_seed(0)` 给出的值，本地测试会确认这一点）：

```verify id=D-h title="NumPy 版 AdamW 重现 02_adamw.py 的五个 loss"
def adamw_step(p, grad, state, lr, betas=(0.9, 0.999), eps=1e-8,
               weight_decay=0.01):
    """照抄 02_adamw.py 的 AdamW.step；p 是 NumPy 数组，state 记录 t、m、v。"""
    b1, b2 = betas
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * grad           # 一阶矩
    state["v"] = b2 * state["v"] + (1 - b2) * grad**2        # 二阶矩
    m_hat = state["m"] / (1 - b1**t)                         # 偏差校正
    v_hat = state["v"] / (1 - b2**t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * p

# torch.manual_seed(0) 之后 Linear(2, 1) 的初始权重
w = np.array([-0.0052939653396606445, 0.37932294607162476])
b = np.array([-0.5819807648658752])
x, y = np.array([1.0, 2.0]), 3.0
states = [{"t": 0, "m": np.zeros(2), "v": np.zeros(2)},
          {"t": 0, "m": np.zeros(1), "v": np.zeros(1)}]
losses = []
for step in range(5):
    out = w @ x + b[0]
    loss = (out - y) ** 2
    dout = 2 * (out - y)                                     # d loss / d out
    w = adamw_step(w, dout * x, states[0], lr=0.1)
    b = adamw_step(b, np.array([dout]), states[1], lr=0.1)
    losses.append(f"{loss:.4f}")
    print(f"Step {step}: loss = {loss:.4f}")
# 与原书脚本打印的五个 loss 相同
assert losses == ["8.0011", "5.8991", "4.1294", "2.6933", "1.5855"]
```

脚本里实际训练用的是 `torch.optim.AdamW`，自己写的 `AdamW` 那一行被注释掉了。三者是否真的一致？下面的本地测试让本页的 `adamw_step`、`02_adamw.py` 的 `AdamW` 类和 `torch.optim.AdamW` 在同一个问题上各跑 200 步，比较参数：

```localcheck verify/test_appendix.py::test_d_adamw title="本页的 NumPy AdamW = 原书的 AdamW 类 = torch.optim.AdamW"
```

::: cs336 A1：自己实现 AdamW
CS336 作业 1 同样要求继承 `torch.optim.Optimizer` 自己实现 AdamW，并在 `Linear(3, 2)` 上训练 1000 步后与 `torch.optim.AdamW` 对拍，容差 $10^{-4}$。上面的本地测试用的正是 `Linear(3, 2)`：在 float64 下跑 200 步，DLFS6 的手写版本与 `torch.optim.AdamW` 的参数差别小于 $10^{-10}$。2026 版讲义的更新记录里还有一条“修正 AdamW 中权重衰减与梯度更新的先后顺序”；DLFS6 的写法用更新前的参数计算衰减项，与 PyTorch“先衰减、再更新”的写法在代数上相同（[A1 仓库](https://github.com/stanford-cs336/assignment1-basics)）。
:::

另外，原论文第 2 节还给出了一种更省计算的等价写法：不单独计算 $\hat m_t$ 和 $\hat v_t$，而是把校正并进学习率，$\mathrm{lr}_t = \mathrm{lr}\cdot\sqrt{1-\beta_2^t}/(1-\beta_1^t)$，再用 $\mathrm{lr}_t\cdot m_t/\sqrt{v_t}$ 更新。$\varepsilon = 0$ 时两者完全相同；$\varepsilon \ne 0$ 时，$\varepsilon$ 的含义略有不同：

```verify id=D-i title="把校正并进学习率：ε = 0 时与原写法完全相同"
grads = rng.normal(size=(30, 4))
m, v = np.zeros(4), np.zeros(4)
theta_a, theta_b, lr = np.zeros(4), np.zeros(4), 1e-3
for t, g in enumerate(grads, start=1):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    theta_a -= lr * (m / (1 - beta1**t)) / np.sqrt(v / (1 - beta2**t))  # 原写法
    lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)      # 并进学习率
    theta_b -= lr_t * m / np.sqrt(v)
print(f"30 步之后两种写法的最大差别：{np.abs(theta_a - theta_b).max():.1e}")
assert np.allclose(theta_a, theta_b, rtol=1e-12, atol=1e-15)
```

## 小结

Adam 的移动平均从 0 起步，展开之后是 $m_t = (1-\beta)\sum_i\beta^{t-i}g_i$，权重之和只有 $1-\beta^t$，剩下的 $\beta^t$ 分给了初始值 0，所以 $\mathbb E[m_t] = (1-\beta^t)\,\mathbb E[g]$。除以 $1-\beta^t$ 等于把权重重新归一化，得到一个真正的加权平均，期望就对了。

不校正的后果与直觉相反：$v_t$ 在分母里，偏差更大，所以早期的更新量**偏大**，默认参数下第 12 步达到 6.6 倍。校正之后，第一步每个参数都移动约 lr，这也是 Adam 的学习率“以参数为单位”的原因。校正修不好的是早期 $\hat v_t$ 的噪声，这是 warmup 的一个动机。

### 本附录所学的内容

- 从 0 起步的移动平均偏向 0：梯度恒为 1 时 $m_t = 1 - 0.9^t$（[验证 D-a](#v-D-a)）。
- $\mathbb E[m_t] = (1-\beta^t)\,\mathbb E[g]$，$v_t$ 同理；除以 $1-\beta^t$ 后期望正确（[验证 D-b](#v-D-b)）。
- 校正等于把移动平均的权重重新归一化到和为 1；梯度期望变化时得到近期期望的加权平均（[验证 D-c](#v-D-c)）。
- 不校正时更新量是校正后的 $f(t) = (1-\beta_1^t)/\sqrt{1-\beta_2^t}$ 倍：第 1 步 3.16，第 12 步 6.57（[验证 D-d](#v-D-d)）。
- 校正后的第一步约为 lr·sign(g)；Adam 对梯度的整体缩放不敏感，但 ε 和梯度裁剪会打破这一点（[验证 D-e](#v-D-e)、[验证 D-f](#v-D-f)）。
- $\hat v_t$ 的有效样本数是 $(1-\beta^t)(1+\beta)/((1-\beta)(1+\beta^t))$，早期噪声很大（[验证 D-g](#v-D-g)）。
- NumPy 版 AdamW 逐位重现了 `02_adamw.py` 的输出（[验证 D-h](#v-D-h)）。

优化器的故事到这里告一段落。[附录 E](appendix-e.html) 回到对齐：从带 KL 正则的目标出发，推导 DPO 的最优解，看看它的损失函数是怎样一步步来的。
