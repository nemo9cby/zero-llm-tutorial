# Render the "minimal verifiable facts" section (Markdown, Chinese) from facts.py + recorded results.
# Usage: python render_facts.py <out.md>
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from facts import FACTS  # noqa: E402

ZH = {
 "F01": "软字典：新电影 (6,4,5) 的预测评分为 78.66；72.62% 的权重落在 (7,6,2)，26.71% 落在 (5,5,5)",
 "F02": "点积相似度对向量长度敏感：点积最大的是 (7,6,2)，但余弦最相似、欧氏距离最近的都是 (5,5,5)",
 "F03": "“硬查询是软字典在 T→0 时的极限”只在没有并列时成立：用每个键去查，(8,2,3) 得 87.5（应为 85），(5,5,5) 得 73.33（应为 75）",
 "F04": "01 里的朴素 softmax（不减最大值）在分数约 710 时溢出成 NaN；减去最大值的稳定写法不会",
 "F05": "softmax 对平移不变，对缩放敏感",
 "F06": "矩阵形式 softmax(QKᵀ)V 复现 02 的三个预测 78.66 / 69.76 / 61.20，且每行权重之和为 1",
 "F07": "softmax([100,200,300]) 饱和（float32 下为 [0, 3.8e-44, 1]）；先除以 100 则得 [0.090, 0.245, 0.665]",
 "F08": "q、k ~ N(0, I_d) 时 Var(q·k) ≈ d，除以 √d 后 ≈ 1（两者之比恰为 d）",
 "F09": "不缩放时 d 越大，256 个 key 上的注意力越尖锐（饱和行占比上升）；除以 √d 后尖锐程度与 d 无关",
 "F10": "初始化（std 0.02）时，真实模型第 0 层的分数很小（std≈0.15），注意力几乎均匀，此时缩放与否影响不大",
 "F11": "没有 mask 也没有位置信息时，attention 对输入顺序是置换等变的；因果 mask 打破了这一点",
 "F12": "正弦位置编码：PE(p)·PE(p+k) 只取决于 k；|PE(p)|² = d/2",
 "F13": "训练后模型的可学习位置嵌入是平滑的：间隔 1/8/64 的平均余弦为 0.949 / 0.562 / 0.037（随机初始化约为 0）",
 "F14": "可学习的绝对位置嵌入限定了上下文长度：向 max_context_len=256 的模型输入 257 个 token 会触发 IndexError",
 "F15": "因果 softmax：上三角恰为 0，每行之和为 1，第 0 行为 [1,0,...]",
 "F16": "对书中 06 的 Attention 做因果性测试：改动第 3、4 个 token 后，位置 0–2 的输出逐位不变",
 "F17": "06 的 Attention 与 F.scaled_dot_product_attention(is_causal=True) 一致（value 维度 256 ≠ key 维度 64 也可以）",
 "F18": "一整行都被 mask 时 softmax 输出 NaN（tril 不会出现，padding mask 可能出现）",
 "F19": "softmax 做在错误的轴上时，即使被遮住的元素为 0，也会泄露未来信息",
 "F20": "为什么需要 mask（2.6.1–2.6.2）：在 i.i.d. 随机 token 上，因果模型只能停在 ln V，去掉 mask 的模型能“作弊”降到 ln(V)/C",
 "F21": "SDPA 的 is_causal=True 采用左上角对齐：1 个 query 对 5 个缓存 key 时只能看到 key 0，所以 KV-cache 解码必须用 is_causal=False",
 "F22": "07 等价于把 06 的 E×E value 矩阵换成 W_o·W_v（秩 64），参数从 98,304 降到 65,536",
 "F23": "位置 0 只能看到自己，因此它的输出恰好是自己的 value 向量",
 "F24": "08 的形状变化：(2,4,24) → view → (2,4,3,8) → transpose → (2,3,4,8)，此时张量已不连续（non-contiguous）",
 "F25": "transpose 之后直接 .view 会报 RuntimeError；.contiguous().view 与 .reshape 结果相同",
 "F26": "批量多头（reshape+transpose）与按权重行切片逐头计算的结果完全一致",
 "F27": "拆头时用 reshape(H,C,D) 而不先 transpose，会悄悄破坏因果性",
 "F28": "codebot 的 MultiHeadAttention、SDPA(is_causal=True) 路径、拷贝权重后的 torch.nn.MultiheadAttention 三者一致",
 "F29": "2.8.3 的“两个等价公式”：concat(heads)·W_oᵀ = Σ_h head_h·W_o[:, hD:(h+1)D]ᵀ",
 "F30": "模块创建后默认处于 train 模式：dropout 让两次前向结果不同，被 dropout 的注意力行之和也不再是 1",
 "F31": "MHA 参数量 = 4·E·(H·D) = 1,048,576（E=512, H=8, D=64，无 bias）",
 "F32": "书中的 LayerNorm 与 nn.LayerNorm 一致（eps 1e-5，有偏方差）",
 "F33": "若误用无偏方差，LN 输出会整体缩放 √((E−1)/E) = 0.998697（E=384）",
 "F34": "LN（γ=1, β=0）之后，每个 token 向量的均值为 0、方差为 1（扣除 eps 的影响）",
 "F35": "LayerNorm 按 token 归一化，样本的输出与 batch 里其他样本无关（BatchNorm 则有关）",
 "F36": "tanh 近似 GELU 与精确（erf）GELU 最多相差 4.73e-4；GELU 最小值为 −0.16997，位于 x = −0.752",
 "F37": "书中的 GELU = nn.GELU(approximate='tanh')；而 codebot 的 FFN 用的是 nn.GELU()，也就是 erf 版本",
 "F38": "09 的 Block（gamma/beta）与 codebot 的 Block（weight/bias）的 state_dict 不改名就不能互相加载",
 "F39": "残差流：把 W_o 和 FFN 输出层置零后，Block 严格等于恒等映射",
 "F40": "一个 Block（E=384, H=6, ff=4E）有 1,772,928 个参数 = 2 个 LN + 4E² + FFN（约等于 12E² = 1,769,472）",
 "F41": "ch02/ch03 所用 GPT 的参数量闭式解：11,121,640（不共享权重则为 11,505,640）",
 "F42": "对 parameters() 求 numel 之和得 11,121,640（共享矩阵只算一次）；对 state_dict 求和会重复计算；embed.weight 与 unembed.weight 是同一个对象",
 "F43": "保存的 checkpoint 约 44.5 MB ≈ 4 字节 × 11.12M 参数，说明共享矩阵只存了一份（官方 model_pretrain.pt 为 44,516,353 字节，HTTP HEAD 得到）",
 "F44": "初始化：Linear/Embedding ~ N(0, 0.02)，bias 为 0，LayerNorm 为 1/0；对共享矩阵而言，Linear 的初始化最后执行，因此最终生效",
 "F45": "初始化时的 loss 约为 6.98 = ln V + σ²/2（σ 是 logits 的标准差，约 0.39），略高于 ln(1000) = 6.9078",
 "F46": "整模型因果性：改动第 100 个 token，位置 0–99 的 logits 逐位不变",
 "F47": "eval() 下模型输出确定且与 batch 无关；train()（有 dropout）下不确定",
 "F48": "HF GPT-2 small 有 124,439,808 个参数；同样超参的 codebot GPT 有 124,453,201 个（多 50,257 个 lm_head bias，少 36,864 个 qkv/proj bias）",
 "F49": "把 HF GPT-2 的权重拷进 codebot GPT（qkv/proj bias 置零）后，logits 相差 3.5e-4（erf GELU）；换成 tanh GELU 后相差 1.9e-6",
 "F50": "训练好的 codebot：代码片段的精确 token id；在 32 个训练窗口上 CE 为 0.40（新初始化约 6.94–6.98）；从 'def' 开始的贪心生成",
 "F51": "训练后的注意力头（ch02_attn_maps.json）：L4H3 是 previous-token head（' b'→' +' 权重 0.99）；第 1 层约一半权重落在第一个 token 上（attention sink）",
 "F52": "纯 NumPy 的 GPT 前向（np_gpt.py）能复现小型 codebot GPT 的 torch logits，误差约 1e-7",
 "F53": "端到端自检：11.1M 参数的 GPT 能在 CPU 上过拟合一个固定 batch 的真实 TinyCodes token（50 步内从 7.0 降到约 0.02）",
}
NEEDS = {"": "仅 NumPy", "torch": "torch", "ckpt": "torch + 训练好的 checkpoint", "transformers": "torch + transformers"}

def read_json(name):
    with open(os.path.join(KIT, name), encoding="utf-8") as fh:
        return json.load(fh)


cp, py = read_json("facts_cpython.json"), read_json("facts_pyodide.json")
C = {r["id"]: r for r in cp["results"]}
P = {r["id"]: r for r in py["results"]}
assert set(ZH) == {f["id"] for f in FACTS}, "every fact needs a Chinese claim"

lines = []
m, pm = cp["meta"], py["meta"]
lines.append(f"> 下面每条事实的代码都由 `verify/run_facts.py` 原样执行（CPython {m['python']}，torch {m['torch']}，numpy {m['numpy']}，Windows，i7-11700F）。"
             f"标为“浏览器可跑”的条目还由 `pyodide_test/run_facts.mjs` 在 Pyodide {pm['pyodide']}（Python {pm['python']}，numpy {pm['numpy']}）里实际跑过，"
             f"输出与 CPython 一致（只有 F27/F29/F52 末位浮点数不同）。用时只计 snippet 本身；新开进程还要加上 `import torch` 约 2–3 s，Pyodide 冷启动约 {pm['boot_ms']/1000:.1f} s（含从 CDN 下载 numpy）。"
             "快照来自运行时的 JSON（`facts_cpython.json` / `facts_pyodide.json`）。")
lines.append("")
lines.append("| ID | 节 | 断言 | 依赖 | 浏览器 | 用时 |")
lines.append("|---|---|---|---|---|---|")
for f in FACTS:
    r = C[f["id"]]
    br = "可（Pyodide 实测）" if f["browser"] else "否"
    cell = ZH[f['id']].replace('|', r'\|')   # keep table columns intact
    lines.append(f"| {f['id']} | {f['sec']} | {cell} | {NEEDS[f['needs']]} | {br} | {r['seconds']} s |")
lines.append("")
for f in FACTS:
    r = C[f["id"]]
    assert r["ok"], f["id"]
    lines.append(f"#### {f['id']}（§{f['sec']}）{ZH[f['id']]}")
    lines.append("")
    tag = "浏览器：可（Pyodide 实测通过）" if f["browser"] else "浏览器：否"
    lines.append(f"依赖：{NEEDS[f['needs']]}；{tag}；CPython 用时 {r['seconds']} s")
    lines.append("")
    lines.append("```python")
    lines.append(f["code"].strip())
    lines.append("```")
    lines.append("")
    lines.append("实测输出：")
    lines.append("")
    lines.append("```text")
    lines.append(r["output"])
    lines.append("```")
    if f["browser"]:
        pr = P[f["id"]]
        assert pr["ok"], f["id"]
        if pr["output"] != r["output"]:
            lines.append("")
            lines.append(f"Pyodide 输出（{pr['seconds']} s，仅末位浮点差异）：")
            lines.append("")
            lines.append("```text")
            lines.append(pr["output"])
            lines.append("```")
        else:
            lines.append("")
            lines.append(f"Pyodide：输出逐字相同（{pr['seconds']} s）。")
    lines.append("")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", sys.argv[1], len(lines), "lines")
