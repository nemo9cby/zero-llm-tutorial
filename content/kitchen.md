---
chapter: K
slug: kitchen
order: 999
kind: extra
hidden: true
title: 组件样板间
lede: 给作者看的活样板：每种组件与提示框的标准用法。不出现在目录里。
pyodide_packages: [numpy]
---

## K.1 共享图表

折线图：两条曲线、参考线、十字准线与数据表。

```widget kitchen-charts caption="共享图表库 _plot.js 的三种图：折线、柱状、热力图。"
{}
```

## K.2 验证单元

```verify id=K.1-a title="softmax 每行之和为 1，且对平移不变"
import numpy as np

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(0)
s = rng.normal(size=(3, 5))
p = softmax(s)
print(np.array2string(p.sum(axis=-1), precision=6))
assert np.allclose(p.sum(axis=-1), 1)
assert np.allclose(softmax(s + 100.0), p)
```

## K.3 提示框

::: note
补充说明，比如术语的由来。
:::

::: warn 不要在 transpose 之后直接 view
注意事项。
:::

::: column 专栏标题
专栏正文可以比较长，适合放与主线解耦的话题。
:::

::: cs336 A1 §3.5.4 scaled_dot_product_attention
CS336 对照的写法。
:::

::: confusion pg_loss 接近 0 是不是说明没在学？
先承认这个直觉为什么自然，再用一个能手算的例子把它拆开。
:::

::: gpu 在本机 RTX 3060 Ti 上运行
```shell
uv run python third_party/dlfs6/ch03/01_pretrain.py
```
:::

::: think 把温度设成 0.01 会发生什么？
:::
::: answer
logits 除以一个很小的数，softmax 退化成 argmax……
:::
