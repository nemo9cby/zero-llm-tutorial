"""第 3 章 训练【基础篇】：本地验证（B 层）。

全部在 CPU 上运行，直接导入 third_party/dlfs6 的原书代码与官方 checkpoint（不运行会写文件的训练脚本）：

    CUDA_VISIBLE_DEVICES=-1 uv run pytest verify/test_ch03.py

页面上引用的每个“本地验证”都对应这里的一个测试；web/data/ch03/ 下的数据文件也在这里逐项复核。
"""
import ast
import inspect
import json
import math
from itertools import cycle

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from _ch03_dlfs6 import (ALPACA, CODEBOT, DLFS6, ROOT, RUN_CHECKPOINTS, arithmetic_prompts, load_defs, load_model,
                         load_run_model, load_tokenizer, run_checkpoint, verify_cell)

DATA = ROOT / "web" / "data" / "ch03"


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tok(dlfs6_path):
    return load_tokenizer()


@pytest.fixture(scope="module")
def models(dlfs6_path):
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    ms = {k: load_model(k) for k in ("pretrain", "sft", "grpo")}
    for m in ms.values():
        m.eval()
    return ms


@pytest.fixture(scope="module")
def grpo_fns(dlfs6_path):
    return load_defs("ch03/09_grpo.py", ["GRPODataset", "calculate_reward", "generate_group",
                                         "compute_probs", "grpo_loss"])


# ---------------------------------------------------------------- 3.1 预训练

def test_corpus_windows_and_epochs(dlfs6_path):
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16)
    assert len(ids) == 2_677_398
    assert (ids == 999).sum() == 21_835          # <|endoftext|>：夹在相邻两篇之间的分隔符
    assert ids[0] != 999 and ids[-1] != 999      # 开头和结尾都没有分隔符，所以一共 21,836 篇
    docs = [d for d in np.split(ids, np.where(ids == 999)[0]) if len(d[d != 999])]
    assert len(docs) == 21_835 + 1
    assert ids.min() == 9 and ids.max() == 999 and (ids == 0).sum() == 0
    TokenDataset = load_defs("ch03/01_pretrain.py", ["TokenDataset"])["TokenDataset"]
    ds = TokenDataset(ids, 256)
    assert len(ds) == 2_677_398 - 256
    x, y = ds[123]
    assert torch.equal(y[:-1], x[1:]) and y[-1].item() == int(ids[123 + 256])
    x, y = ds[len(ds) - 1]                        # 最后一个窗口恰好用到最后一个 token
    assert y[-1].item() == int(ids[-1])
    epochs = 20000 * 32 * 256 / len(ids)
    print(f"预训练看过的 token 位置 = 语料的 {epochs:.2f} 倍")
    assert abs(epochs - 61.19) < 0.01
    head = np.fromfile(DATA / "tiny_codes_head.bin", dtype=np.uint16)
    assert np.array_equal(head, ids[:4096])


def test_param_count_and_initial_loss(dlfs6_path):
    from codebot.model import GPT
    torch.manual_seed(0)
    m = GPT(1000, 256, 384, 6, 6, 4 * 384, 0.1)
    assert sum(p.numel() for p in m.parameters()) == 11_121_640
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16)
    starts = np.random.default_rng(0).integers(0, len(ids) - 257, 16)
    xb = torch.tensor(np.stack([ids[s:s + 256] for s in starts]).astype(np.int64))
    yb = torch.tensor(np.stack([ids[s + 1:s + 257] for s in starts]).astype(np.int64))
    m.eval()
    with torch.no_grad():
        logits = m(xb)
        loss = F.cross_entropy(logits.view(-1, 1000), yb.view(-1)).item()
    sigma = logits.std().item()
    print(f"初始 loss {loss:.4f}，ln 1000 = {math.log(1000):.4f}，logit 标准差 {sigma:.4f}")
    assert 6.9 < loss < 7.1
    assert abs(sigma - 0.02 * math.sqrt(384)) < 0.02           # 0.3919
    # ln V + σ²/2 把同一位置的 1000 个 logit 当作独立的正态变量，只是近似；在真实语料的 16 个窗口上
    # 实测相差约 0.027（第 2 章在随机 token 上做同样的检验，相差不到 0.01），所以这里的容差取 0.03
    assert abs(loss - (math.log(1000) + sigma ** 2 / 2)) < 0.03


def test_pretrained_checkpoint_fits_its_training_text(dlfs6_path, models):
    """没有验证集：官方 model_pretrain.pt 在训练语料上的 loss 很低（接近背诵）。"""
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16)
    starts = np.random.default_rng(1).integers(0, len(ids) - 257, 16)
    xb = torch.tensor(np.stack([ids[s:s + 256] for s in starts]).astype(np.int64))
    yb = torch.tensor(np.stack([ids[s + 1:s + 257] for s in starts]).astype(np.int64))
    with torch.no_grad():
        loss = F.cross_entropy(models["pretrain"](xb).view(-1, 1000), yb.view(-1)).item()
    print(f"训练语料上的 loss {loss:.4f}，困惑度 {math.exp(loss):.2f}")
    assert loss < 0.6 and math.exp(loss) < 2


def test_low_training_loss_is_not_verbatim_recall(dlfs6_path, models):
    """loss 很低 ≠ 逐字背诵：给出语料里的 96 个 token，让官方预训练模型贪心续写 32 个，
    与语料原文逐 token 比较。teacher forcing 下 top-1 准确率很高，自由续写却很少整段对上。"""
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16).astype(np.int64)
    starts = np.random.default_rng(123).integers(0, len(ids) - 300, 64)
    P, K = 96, 32                                            # 前缀 96 个 token，续写 32 个
    full = torch.tensor(np.stack([ids[s:s + P + K] for s in starts]))
    m = models["pretrain"]
    with torch.no_grad():
        lg = m(full[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 1000), full[:, 1:].reshape(-1)).item()
        top1 = (lg.argmax(-1) == full[:, 1:]).float().mean().item()
        cur = full[:, :P]
        for _ in range(K):                                   # 贪心续写，不看原文
            cur = torch.cat([cur, m(cur)[:, -1].argmax(-1, keepdim=True)], 1)
    match = (cur[:, P:] == full[:, P:]).int()
    verbatim = int((match.sum(1) == K).sum())                # 32 个 token 全部对上的窗口
    prefix = match.cumprod(1).sum(1)                         # 从头开始连续对上的 token 数
    print(f"teacher forcing：loss {loss:.3f}，top-1 准确率 {top1:.1%}；自由续写 32 个 token：逐 token 一致 "
          f"{match.float().mean().item():.1%}，整段一致的窗口 {verbatim}/64，连续一致的长度中位数 {int(prefix.median())}")
    assert loss < 0.6 and 0.835 <= top1 < 0.845              # 在语料上拟合得很紧（正文：84%）……
    assert verbatim == 5 and int(prefix.median()) < 16       # ……但整段复述的窗口只有 5 个（正文的数字）


def test_overfit_one_batch_first(dlfs6_path):
    """训练前的体检：同一个小 batch 反复训练，loss 必须能压到接近 0；做不到就先别跑大实验。"""
    from codebot.model import GPT
    ids = np.fromfile(CODEBOT / "tiny_codes.bin", dtype=np.uint16)
    torch.manual_seed(0)
    x = torch.tensor(ids[:4 * 33].astype(np.int64)).view(4, 33)
    xb, yb = x[:, :-1], x[:, 1:]
    m = GPT(1000, 32, 64, 4, 2, 256, 0.1)       # 缩小版 CodeBot（同一份 model.py，dropout 照旧 0.1）
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    losses = []
    for _ in range(120):
        loss = F.cross_entropy(m(xb).view(-1, 1000), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"同一个 batch 训练 120 步：loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[0] > 6.5 and losses[-1] < 0.3


# ---------------------------------------------------------------- 3.2 文本生成

def test_generate_is_the_same_function_everywhere(dlfs6_path):
    from codebot import utils
    src02 = (DLFS6 / "ch03/02_generate.py").read_text(encoding="utf-8")
    f02 = next(n for n in ast.parse(src02).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    futils = next(n for n in ast.parse(inspect.getsource(utils)).body
                  if isinstance(n, ast.FunctionDef) and n.name == "generate")
    assert ast.dump(f02) == ast.dump(futils)
    # 区别只在注释：去掉 ❶❷❸ 标号之后，还有两行注释的措辞不同
    def comments(src, fn):
        lines = src.splitlines()[fn.lineno - 1:fn.end_lineno]
        return [ln.split("#", 1)[1].strip().lstrip("❶❷❸ ") for ln in lines if "#" in ln]
    c02 = comments(src02, f02)
    cut = comments(inspect.getsource(utils), futils)
    assert len(c02) == len(cut) and sum(a != b for a, b in zip(c02, cut)) == 2
    # 只有温度这一个旋钮：没有 top_k / top_p 参数
    assert list(inspect.signature(utils.generate).parameters) == [
        "model", "tokenizer", "prompt", "max_new_tokens", "temperature"]


def test_temperature_zero_is_greedy_and_tiny_temperature_breaks(dlfs6_path, tok, models):
    from codebot.utils import generate
    m = models["sft"]
    p = ALPACA.format("3+4=")
    outs = {generate(m, tok, p, max_new_tokens=5, temperature=0) for _ in range(3)}
    assert len(outs) == 1                          # T=0 走 argmax 分支，每次都一样
    with torch.no_grad():
        logits = m(torch.tensor([tok.encode(p)]))[:, -1, :]
    probs = F.softmax(logits / 1e-45, dim=-1)      # float32 溢出成 inf，inf - inf = nan
    assert torch.isnan(probs).any()
    with pytest.raises(RuntimeError):              # 原书的 generate() 本身：torch.multinomial 拒绝 NaN 概率
        generate(m, tok, p, max_new_tokens=1, temperature=1e-45)
    ok = F.softmax(logits / 0.01, dim=-1)          # 0.01 不会溢出，但也不等于贪心解码
    assert not torch.isnan(ok).any()
    top2 = ok.topk(2).values[0].tolist()
    print(f"T=0.01 时前两名的概率：{top2[0]:.3f} / {top2[1]:.3f}")
    assert 0.85 < top2[0] < 0.95 and top2[1] > 0.05    # '8' 与 '6' 的 logit 只差 0.02：仍有一成机会采到第二名
    assert ok.argmax().item() == logits.argmax().item()


def test_generate_left_crops_context_and_switches_to_eval(dlfs6_path, tok, models):
    from codebot.model import GPT
    from codebot.utils import generate
    torch.manual_seed(0)
    small = GPT(1000, 8, 16, 2, 1, 32, 0.1)          # 只有 8 个位置
    prompt = "def add(a, b):\n    return a + b\n"
    assert len(tok.encode(prompt)) > 8
    small.train()
    out = generate(small, tok, prompt, max_new_tokens=5, temperature=1.0)
    assert out.startswith(prompt)                    # 返回值包含完整 prompt，但模型每步只看最后 8 个 token
    assert small.training is False                   # 副作用：generate() 把模型切到 eval 且不还原
    with pytest.raises(IndexError):
        small(torch.zeros(1, 9, dtype=torch.long))   # 不裁剪就越界：位置 embedding 只有 8 行
    with pytest.raises(IndexError):
        models["pretrain"](torch.zeros(1, 257, dtype=torch.long))   # CodeBot 只有 256 个位置


def test_samples_export(dlfs6_path, tok, models):
    import _ch03_export as ex
    exp = read_json("samples.json")
    for row in exp["chat"]:
        for name in ("sft", "grpo"):
            assert ex.chat_greedy(tok, models[name], row["instruction"]) == row[name], (name, row["instruction"])
    first = exp["pretrain_def"][0]
    assert ex.pretrain_sample(tok, models["pretrain"], first["seed"]) == first["text"]
    # 上限 60 与上限 1000 得到同一段文字：这一次没写满 60 个 token 就采到了 EOS
    assert ex.pretrain_sample(tok, models["pretrain"], first["seed"], max_new_tokens=1000) == first["text"]


def test_next_token_export_matches_checkpoints(dlfs6_path, tok, models):
    data = read_json("next_token.json")
    for c in data["contexts"]:
        with torch.no_grad():
            lg = models[c["model"]](torch.tensor([tok.encode(c["prompt"])]))[0, -1].numpy()
        assert np.abs(lg - np.array(c["logits"])).max() < 2e-4, c["label"]
    vocab = read_json("vocab.json")
    assert vocab["tokens"] == [tok.id_to_bytes[i].decode("utf-8", errors="replace") for i in range(1000)]


# ---------------------------------------------------------------- 3.3 SFT

def test_alpaca_ids_and_the_stale_comment(dlfs6_path, tok):
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    item = data[0]
    assert item == {"instruction": "Hello", "response": "Hello. What can I help you with?"}
    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
    ids = tok.encode(text)
    assert ids == [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10,
                   846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 932, 657, 63, 999]
    # 原书注释写的是 930；930 其实是 'alize'，932 才是 ' you'
    assert tok.id_to_bytes[932] == b" you" and tok.id_to_bytes[930] == b"alize"
    assert len(tok.encode(ALPACA.format("Hello"))) == 19


def test_sft_dataset_labels_padding_and_export(dlfs6_path, tok):
    """原书 SFTDataset 与页面数据 sft_tokens.json 逐位一致（全部 1092 条样本）；
    全量统计（23,635 个带 label 的位置等）由页面上的验证单元在这份数据上计算。"""
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    exp = read_json("sft_tokens.json")
    assert len(data) == len(exp["prompt_ids"]) == 1092
    assert exp["instructions"] == [d["instruction"] for d in data]
    k34 = exp["instructions"].index("3+4=")
    SFTDataset = load_defs("ch03/03_sft.py", ["SFTDataset"])["SFTDataset"]
    ds = SFTDataset(str(CODEBOT / "tiny_codes_sft.json"), tok, 256)
    assert len(ds.samples) == 1092
    for k, (ids, labels) in enumerate(ds.samples):
        p, r = exp["prompt_ids"][k], exp["response_ids"][k]
        full = p + r
        ref_ids = (full[:-1] + [0] * 256)[:256]
        ref_labels = (([-100] * len(p) + r)[1:] + [-100] * 256)[:256]
        assert ids == ref_ids and labels == ref_labels, k   # 导出的 token 能逐位重建原书的每一条样本
    ids, labels = ds.samples[k34]
    assert labels[:23] == [-100] * 21 + [55, 999] and set(labels[23:]) == {-100}
    assert ids[22] == 55 and set(ids[23:]) == {0}           # 输入里有 '7'，没有 EOS；之后全是 padding id 0
    lens = [len(p) + len(r) for p, r in zip(exp["prompt_ids"], exp["response_ids"])]
    assert max(lens) == 138 and sum(len(r) for r in exp["response_ids"]) == 23_635


def test_trailing_padding_does_not_change_earlier_logits(dlfs6_path, models):
    """因果掩码：补在尾部的 padding 不影响前面任何位置。补到 256 与补到 137，结果相同。"""
    exp = read_json("sft_tokens.json")
    for k in (exp["instructions"].index("3+4="), exp["instructions"].index("merge two sorted lists")):
        seq = (exp["prompt_ids"][k] + exp["response_ids"][k])[:-1]
        x256 = torch.tensor([seq + [0] * (256 - len(seq))])
        x137 = torch.tensor([seq + [0] * (137 - len(seq))])
        with torch.no_grad():
            a = models["sft"](x256)[0, :len(seq)]
            b = models["sft"](x137)[0, :len(seq)]
        diff = (a - b).abs().max().item()
        print(f"{len(seq)} 个真实 token：两种补齐方式的 logits 最大差 {diff:.2e}")
        assert diff < 1e-4                                   # 只剩 float32 的舍入误差
        # 训练时开着 dropout：同一个种子，两种形状抽到的 dropout 掩码不同，结果只是统计上等价
        import copy
        m = copy.deepcopy(models["sft"]).train()
        with torch.no_grad():
            torch.manual_seed(0)
            a = m(x256)[0, :len(seq)]
            torch.manual_seed(0)
            b = m(x137)[0, :len(seq)]
        assert (a - b).abs().max().item() > 1e-3


def test_cross_entropy_ignore_index_is_a_token_mean(dlfs6_path):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 6, 11, generator=g)
    labels = torch.randint(0, 11, (3, 6), generator=g)
    labels[0, :5] = -100          # 第 0 条只有 1 个有效 token
    labels[1, :2] = -100          # 第 1 条有 4 个
    loss = F.cross_entropy(logits.view(-1, 11), labels.view(-1), ignore_index=-100).item()
    lp = F.log_softmax(logits, -1)
    valid = labels != -100
    nll = -lp.gather(-1, labels.clamp(min=0)[..., None]).squeeze(-1)
    token_mean = nll[valid].mean().item()
    sample_mean = torch.stack([nll[i][valid[i]].mean() for i in range(3)]).mean().item()
    assert abs(loss - token_mean) < 1e-6 and abs(loss - sample_mean) > 1e-3


def test_prompt_masking_effect_on_losses(dlfs6_path, tok, models):
    exp = read_json("sft_tokens.json")
    pick = list(range(0, 1092, 8))                       # 137 条，固定子集
    seqs = [exp["prompt_ids"][k] + exp["response_ids"][k] for k in pick]
    L = max(map(len, seqs))
    x = torch.tensor([s + [0] * (L - len(s)) for s in seqs])
    out = {}
    for name in ("pretrain", "sft"):
        with torch.no_grad():
            lp = F.log_softmax(models[name](x[:, :-1]), -1)
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
        is_resp = torch.zeros_like(nll, dtype=torch.bool)
        is_prompt = torch.zeros_like(nll, dtype=torch.bool)
        for row, k in enumerate(pick):
            lp_, lr_ = len(exp["prompt_ids"][k]), len(exp["response_ids"][k])
            is_prompt[row, :lp_ - 1] = True              # 预测 prompt 自己的下一个 token
            is_resp[row, lp_ - 1:lp_ - 1 + lr_] = True   # 预测回答（含 EOS）
        out[name] = (nll[is_prompt].mean().item(), nll[is_resp].mean().item())
    print("prompt / response 上的 loss：", {k: tuple(round(v, 3) for v in t) for k, t in out.items()})
    assert out["pretrain"][1] > 3.0 and out["sft"][1] < 0.1        # 回答：学会了
    assert out["sft"][0] > out["pretrain"][0] + 2.0                # prompt：反而“忘了”怎么续写


# ---------------------------------------------------------------- 3.4 对话

def test_greedy_probe_before_and_after_rl(dlfs6_path, tok, models, grpo_fns):
    from codebot.utils import generate
    reward = grpo_fns["calculate_reward"]
    right = {}
    for name in ("sft", "grpo"):
        right[name] = [reward(gt, generate(models[name], tok, p, temperature=0)[len(p):]) == 1.0
                       for p, gt in arithmetic_prompts()]
    acc = {name: sum(r) for name, r in right.items()}
    fixed = sum(not a and b for a, b in zip(right["sft"], right["grpo"]))     # SFT 错、GRPO 对
    broken = sum(a and not b for a, b in zip(right["sft"], right["grpo"]))    # SFT 对、GRPO 错
    print("81 道加法，贪心解码答对的题数：", acc, f"；GRPO 修好 {fixed} 道，弄坏 {broken} 道")
    assert acc == {"sft": 44, "grpo": 67}
    assert (fixed, broken) == (29, 6) and 44 + fixed - broken == 67


def test_truncated_answer_scores_zero(dlfs6_path, tok, models, grpo_fns):
    from codebot.utils import generate
    reward = grpo_fns["calculate_reward"]
    p = ALPACA.format("2+9=")
    full = generate(models["sft"], tok, p, temperature=0)[len(p):]
    cut = generate(models["sft"], tok, p, max_new_tokens=1, temperature=0)[len(p):]
    assert (full, reward(11, full)) == ("11", 1.0)
    assert (cut, reward(11, cut)) == ("1", 0.0)      # 上限截断了答案：奖励为 0


def test_think_tag_is_just_ordinary_tokens(dlfs6_path, tok):
    assert tok.encode("<|endoftext|>") == [999] and tok.end_token_id == 999 == 256 + 743
    think = tok.encode("<think>")
    assert len(think) == 5 and 999 not in think          # CodeBot 没有专门的 <think> token
    assert b"".join(tok.id_to_bytes[i] for i in think) == b"<think>"


def test_free_running_length_export(dlfs6_path, tok, models):
    import _ch03_export as ex
    exp = read_json("sft_free_running.json")
    assert exp["indices"] == ex.free_running_indices()
    pick = exp["indices"][::16]                              # 每个 prompt 有自己的种子，可以单独抽查
    r = ex.sft_free_running_lengths(tok, models["sft"], pick)
    assert r["generated"] == exp["generated"][::16] and r["target"] == exp["target"][::16]
    # 改写过的 48 条指令（句末的标点去掉或加上）：都不在 SFT 数据里，模型却大多一字不差地背出原题的回答
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    rw = exp["reworded"]
    assert rw["instructions"] == [ex.reworded(data[i]["instruction"]) for i in exp["indices"]]
    assert not {d["instruction"] for d in data} & set(rw["instructions"])
    assert rw["ids"] == [tok.encode(t) for t in rw["texts"]]
    sft = read_json("sft_tokens.json")
    verbatim = [t == data[i]["response"] for t, i in zip(rw["texts"], exp["indices"])]
    # 页面上的验证单元比较的是 token id：编码是确定的、可逆的，所以 id 相同等价于文字相同
    assert verbatim == [g == sft["response_ids"][i][:-1] for g, i in zip(rw["ids"], exp["indices"])]
    print(f"改写过的 48 条指令：与训练回答一字不差的 {sum(verbatim)} 条，平均长度 {np.mean([len(g) for g in rw['ids']]):.1f}")
    assert sum(verbatim) == 42
    sub = sorted({0, 16, 32, rw["instructions"].index("output welcome.")})     # 抽查几条，含页面引用的那条跑偏的例子
    texts = ex.free_running_texts(tok, models["sft"], [rw["instructions"][j] for j in sub], [exp["indices"][j] for j in sub])
    assert texts == [rw["texts"][j] for j in sub]


def test_answer_distribution_export(dlfs6_path, tok, models):
    """answer_dists.json 由逐条前向算出；这里用另一条路径（19 个答案拼成一个 batch）抽查 9 道题。"""
    data = read_json("answer_dists.json")
    eos = tok.end_token_id
    answers = [tok.encode(str(a)) + [eos] for a in range(19)]
    for k, (p, gt) in enumerate(arithmetic_prompts()):
        if k % 10:
            continue
        pid = tok.encode(p)
        x = torch.tensor([pid + a + [0] * (3 - len(a)) for a in answers])   # 尾部 padding 不影响前面的位置
        for name in ("sft", "grpo"):
            with torch.no_grad():
                lp = F.log_softmax(models[name](x)[:, len(pid) - 1:-1], -1)
            for a, ans in enumerate(answers):
                prob = lp[a, torch.arange(len(ans)), torch.tensor(ans)].sum().exp().item()
                assert abs(prob - data[name][k][a]) < 2e-5, (name, p, a)
    # 只算标准切分：'10' 是一个 token（id 743）；“先 '1' 后 '0'”这条非标准路径的概率在 1e-5 量级
    assert tok.encode("10") == [743] and tok.encode("1") + tok.encode("0") == [49, 48]
    for p, gt in arithmetic_prompts():
        if gt != 10:
            continue
        x = torch.tensor([tok.encode(p) + [49, 48, eos]])
        for name in ("sft", "grpo"):
            with torch.no_grad():
                lp = F.log_softmax(models[name](x)[0, -4:-1], -1)
            assert lp[torch.arange(3), torch.tensor([49, 48, eos])].sum().exp().item() < 2e-5, (name, p)


# ---------------------------------------------------------------- 3.9 GRPO

def test_reward_function_edge_cases_and_bare_except(dlfs6_path, grpo_fns):
    r = grpo_fns["calculate_reward"]
    cases = {"7": 1.0, " 7": 1.0, "The answer is 7": 1.0, "3+4=7": 1.0, "8 7": 1.0, "07": 1.0,
             "7 8": 0.0, "7.0": 0.0, "-7": 0.0, "seven": 0.0, "": 0.0}
    assert {s: r(7, s) for s in cases} == cases
    assert r(12, "1 2") == 0.0 and r(12, "12") == 1.0
    broken = load_defs("ch03/09_grpo.py", ["calculate_reward"], keep_imports=False)["calculate_reward"]
    assert broken(7, "7") == 0.0                       # NameError: re 未定义，被裸 except 吞掉


def test_dataloader_cycle_replays_the_same_batches(dlfs6_path, tok, grpo_fns):
    from torch.utils.data import DataLoader
    torch.manual_seed(0)
    ds = grpo_fns["GRPODataset"](tok)
    it = cycle(DataLoader(ds, batch_size=32, shuffle=True))
    first = [tuple(next(it)[1].tolist()) for _ in range(3)]
    second = [tuple(next(it)[1].tolist()) for _ in range(3)]
    assert [len(b) for b in first] == [32, 32, 17]
    assert first == second             # itertools.cycle 记住了第一个 epoch，之后原样重放


def test_get_batch_never_contains_eos(dlfs6_path, tok, models, grpo_fns):
    ds = grpo_fns["GRPODataset"](tok)
    p = ALPACA.format("3+4=")
    ids, mask = ds.get_batch([p, p], ["7", "12"], "cpu")
    assert ids[0, -3:].tolist() == [10, 55, 0] and mask[0, -3:].tolist() == [0, 1, 0]
    assert ids[1, -3:].tolist() == [10, 49, 50] and mask[1, -3:].tolist() == [0, 1, 1]
    # 真实的回答来自原书的 generate_group（温度 1，采到 EOS 就停）：拼进 batch 之后同样没有 EOS
    torch.manual_seed(0)
    prompts, responses, _ = grpo_fns["generate_group"](models["sft"], tok, [p], [7], 8)
    ids, mask = ds.get_batch(prompts, responses, "cpu")
    assert 999 not in ids.flatten().tolist()              # EOS 不在 batch 里，也就从不进入 loss
    # 空回答（第一步就采到 EOS）：掩码全为 0，它对 loss 没有任何贡献
    ids, mask = ds.get_batch([p], [""], "cpu")
    assert mask.sum().item() == 0


def test_empty_answer_is_never_penalized(dlfs6_path, tok, models):
    """EOS 从不进入 loss，但它的概率没有被冻结：压低别的 token 时，softmax 把概率挪给了它。
    第一步就选 EOS 的“空回答”永远 0 分，却永远不会被惩罚。"""
    def p_empty(m, q):                                    # 第一个回答 token 就是 EOS 的概率
        with torch.no_grad():
            return F.softmax(m(torch.tensor([tok.encode(ALPACA.format(q))]))[0, -1], -1)[999].item()
    pe = {name: p_empty(models[name], "5+3=") for name in ("sft", "grpo")}
    print(f"官方模型，5+3= 回答为空的概率：SFT {pe['sft']:.4f} → GRPO {pe['grpo']:.4f}")
    assert pe["sft"] < 0.02 and pe["grpo"] > 0.15
    runs = {k: load_run_model(k) for k in ("run_sft", "run_grpo", "run_evalfix")}
    pr = {k: p_empty(m, "3+3=") for k, m in runs.items()}
    print("本书的运行，3+3= 回答为空的概率：", {k: round(v, 4) for k, v in pr.items()})
    assert pr["run_sft"] < 0.01 and pr["run_grpo"] > 0.99 and pr["run_evalfix"] < 0.01
    from codebot.utils import generate
    p = ALPACA.format("3+3=")
    assert generate(runs["run_grpo"], tok, p, temperature=0)[len(p):] == ""     # 贪心解码也是空回答


def _batch(tok, grpo_fns, responses, prompts):
    ds = grpo_fns["GRPODataset"](tok)
    return ds.get_batch(prompts, responses, "cpu")


def test_grpo_loss_at_ratio_one_is_reinforce_with_baseline(dlfs6_path, tok, models, grpo_fns):
    import copy
    model = copy.deepcopy(models["sft"]).eval()             # 关掉 dropout：权重相同 ⇒ ratio = 1
    old = models["sft"]
    prompts = [ALPACA.format("3+4=")] * 4 + [ALPACA.format("9+9=")] * 4
    responses = ["7", "8", "6", "12", "18", "17", "16", "13"]   # 都不是极小概率的 token（见下一个测试）
    rewards = torch.tensor([1., 0, 0, 0, 1, 0, 0, 0])
    adv = torch.cat([g - g.mean() for g in rewards.view(2, 4)])
    ids, mask = _batch(tok, grpo_fns, responses, prompts)
    lens = mask.sum(1)
    assert lens.tolist() == [1, 1, 1, 2, 2, 2, 2, 2]
    loss = grpo_fns["grpo_loss"](model, old, ids, mask, adv, 0.2)
    expected = -(adv * lens).sum() / len(responses)          # ratio = 1 时的数值：−Σ A·L / N
    assert abs(expected.item() - 0.03125) < 1e-12
    assert abs(loss.item() - expected.item()) < 1e-5
    loss.backward()
    g1 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    model.zero_grad()
    lp = F.log_softmax(model(ids)[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    reinforce = -(adv[:, None] * lp * mask[:, 1:]).sum() / len(responses)
    reinforce.backward()
    g2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    g1, g2 = g1.double(), g2.double()                         # 1100 万维的点积，用 float64 求余弦
    cos = (g1 @ g2 / (g1.norm() * g2.norm())).item()
    print(f"loss = {loss.item():+.6f}（−ΣA·L/N = {expected.item():+.6f}），‖g‖ = {g1.norm():.4f}，余弦 {cos:.8f}")
    assert g1.norm() > 0.1 and cos > 1 - 1e-6 and torch.allclose(g1, g2, atol=1e-6, rtol=1e-3)


def test_probability_space_ratio_is_biased_for_rare_tokens(dlfs6_path, tok, models, grpo_fns):
    p = ALPACA.format("2+3=")
    ids, mask = _batch(tok, grpo_fns, ["18"], [p])
    probs = grpo_fns["compute_probs"](models["sft"], ids)[0][mask[0, 1:].bool()]
    with torch.no_grad():
        ratio = probs / (probs + 1e-8)                       # 同一个模型、同一组 token：本应恰好为 1
    print("token 概率", [f"{x:.2e}" for x in probs.tolist()], "ratio", [f"{x:.4f}" for x in ratio.tolist()])
    assert probs.max().item() < 1e-5 and ratio.min().item() < 0.95


def test_numpy_grpo_loss_on_page_matches_book(dlfs6_path, grpo_fns):
    # 页面上的单元依赖前面单元定义的 np 与 advantages；这里先放进命名空间，再执行该单元的原文
    ns = {"np": np, "advantages": lambda r: np.asarray(r, dtype=float) - np.mean(r)}
    exec(verify_cell("3.9-e"), ns)
    grpo_loss_np = ns["grpo_loss_np"]

    class FixedLogits(torch.nn.Module):          # “模型”直接返回一组固定的 logits
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(logits.clone())

        def forward(self, ids):
            return self.logits

    g = torch.Generator().manual_seed(0)
    B, C, V = 6, 7, 5
    ids = torch.randint(0, V, (B, C), generator=g)
    mask = (torch.rand(B, C, generator=g) > 0.3).float()
    old_logits = torch.randn(B, C, V, generator=g)
    new_logits = old_logits + 0.4 * torch.randn(B, C, V, generator=g)    # ratio 明显偏离 1，两侧都会被截断
    adv = torch.randn(B, generator=g)
    loss = grpo_fns["grpo_loss"](FixedLogits(new_logits), FixedLogits(old_logits), ids, mask, adv, 0.2).item()
    cp = grpo_fns["compute_probs"]
    with torch.no_grad():
        p_new = cp(FixedLogits(new_logits), ids).numpy()
        p_old = cp(FixedLogits(old_logits), ids).numpy()
    ours = grpo_loss_np(p_new.astype(np.float64), p_old.astype(np.float64), mask.numpy(), adv.numpy(), 0.2)
    assert abs(ours - loss) < 1e-5


def test_dropout_makes_the_first_ratio_noisy(dlfs6_path, tok, models, grpo_fns):
    import copy
    exp = read_json("ratio_first_update.json")
    ds = grpo_fns["GRPODataset"](tok)
    prompts = [p for p, _ in ds.data[:32] for _ in range(8)]
    ids, mask = ds.get_batch(prompts, exp["responses"], "cpu")
    m = mask[:, 1:].bool()
    cp = grpo_fns["compute_probs"]
    policy = copy.deepcopy(models["sft"])
    with torch.no_grad():
        po = cp(models["sft"], ids)
        policy.eval()
        r_eval = (cp(policy, ids) / (po + 1e-8))[m]
        torch.manual_seed(1)
        policy.train()
        r_train = (cp(policy, ids) / (po + 1e-8))[m]
    assert r_eval.numel() == exp["summary"]["n_tokens"] == 283
    assert exp["lengths"] == [len(tok.encode(r)) for r in exp["responses"]] and sum(exp["lengths"]) == 283
    assert r_eval.min().item() > 0.99998 and r_eval.max().item() <= 1.0 + 1e-6   # 只剩 +1e-8 造成的微小偏差
    outside = ((r_train < 0.8) | (r_train > 1.2)).float().mean().item()
    print(f"train 模式：ratio 均值 {r_train.mean():.3f}，标准差 {r_train.std():.3f}，落在 [0.8, 1.2] 外 {outside:.1%}")
    assert outside > 0.3
    assert np.allclose(r_train.numpy(), np.array(exp["train"]), atol=1e-3)      # 与页面数据逐个一致
    # dropout 也改变了梯度方向：与“干净”（eval 模式）梯度的余弦明显小于 1
    adv = torch.tensor(exp["advantages"])
    rows = slice(0, 64)                                  # 前 8 道题的 64 条回答就足够看出方向偏了
    grads = []
    for mode in ("eval", "train"):
        torch.manual_seed(2)
        policy.zero_grad()
        getattr(policy, mode)()
        grpo_fns["grpo_loss"](policy, models["sft"], ids[rows], mask[rows], adv[rows], 0.2).backward()
        grads.append(torch.cat([p.grad.flatten().clone() for p in policy.parameters() if p.grad is not None]))
    g0, g1 = grads[0].double(), grads[1].double()
    cos = (g0 @ g1 / (g0.norm() * g1.norm())).item()
    print(f"前 8 道题的 64 条回答，一次 dropout 抽样（种子 2）的梯度与干净梯度的余弦：{cos:.3f}")
    assert 0.7 < cos < 0.8                                # 正文：“余弦只有 0.76”


def test_dropout_biases_the_logged_grpo_loss(dlfs6_path, tok, models, grpo_fns):
    """原脚本进度条上的 loss 大多是正的，并不贴着 0。同一批 256 条回答、权重相同（第一次更新）：
    eval 模式下 loss = −ΣA·L/N；train 模式（脚本的做法）下抽 20 次 dropout，loss 次次为正。
    同时数一数梯度的“有效权重” A·ρ（被裁剪掉的 token 记 0）：dropout 让推高一侧变弱，压低一侧几乎不变。"""
    import copy
    exp = read_json("ratio_first_update.json")
    ds = grpo_fns["GRPODataset"](tok)
    prompts = [p for p, _ in ds.data[:32] for _ in range(8)]
    ids, mask = ds.get_batch(prompts, exp["responses"], "cpu")
    adv = torch.tensor(exp["advantages"])
    m = mask[:, 1:].bool()
    A = adv[:, None].expand_as(m)[m]                       # 每个回答 token 的优势
    N = len(adv)
    cp, old = grpo_fns["compute_probs"], models["sft"]
    policy = copy.deepcopy(old)

    def clipped(r, eps=0.2):                               # 与 grpo_loss 相同的算式（逐 token 的比值 r）
        return (-torch.min(r * A, torch.clamp(r, 1 - eps, 1 + eps) * A).sum() / N).item()

    def weights(r):                                        # 梯度权重 A·ρ，被裁剪的 token 记 0
        live = ((A > 0) & (r <= 1.2)) | ((A < 0) & (r >= 0.8))
        w = torch.where(live, A * r, torch.zeros_like(r))
        return w[A > 0].sum().item(), w[A < 0].sum().item()

    with torch.no_grad():
        po = cp(old, ids)
        policy.eval()
        r_eval = (cp(policy, ids) / (po + 1e-8))[m]
        policy.train()
        draws = []
        for s in range(20):
            torch.manual_seed(100 + s)
            draws.append((cp(policy, ids) / (po + 1e-8))[m])
        torch.manual_seed(100)                             # 原书的 grpo_loss，与第 0 次抽样同一个种子
        book = grpo_fns["grpo_loss"](policy, old, ids, mask, adv, 0.2).item()
    assert abs(book - clipped(draws[0])) < 1e-5
    formula = (-(adv * mask[:, 1:].sum(1)).sum() / N).item()
    loss_clip = np.array([clipped(r) for r in draws])
    loss_noclip = np.array([clipped(r, eps=1e9) for r in draws])
    w_eval, w_train = weights(r_eval), np.array([weights(r) for r in draws])
    print(f"eval 模式：loss {clipped(r_eval):+.4f}（−ΣA·L/N = {formula:+.4f}）；有效权重 推高 {w_eval[0]:+.1f}，压低 {w_eval[1]:+.1f}")
    print(f"train 模式 20 次：loss 平均 {loss_clip.mean():+.4f}（{loss_clip.min():+.4f} 到 {loss_clip.max():+.4f}），"
          f"去掉裁剪 {loss_noclip.mean():+.4f}；有效权重 推高 {w_train[:, 0].mean():+.1f}，压低 {w_train[:, 1].mean():+.1f}")
    assert abs(clipped(r_eval) - formula) < 1e-4 and abs(formula - 0.0073) < 1e-4
    assert loss_clip.min() > 0 and 0.07 < loss_clip.mean() < 0.12          # 20 次全为正，平均约 +0.09
    assert 0.03 < loss_noclip.mean() < loss_clip.mean()                    # 不裁剪也偏正；裁剪的 min 又放大了偏差
    assert w_train[:, 0].mean() < 0.65 * w_eval[0]                         # 推高一侧少了三分之一以上……
    assert abs(w_train[:, 1].mean() - w_eval[1]) < 0.1 * abs(w_eval[1])    # ……压低一侧几乎不变
    assert np.all(w_train.sum(1) < 0)                                      # 每一次抽样，合力都是“往下压”


def test_local_runs_answer_distributions_and_greedy(dlfs6_path, tok, grpo_fns):
    """本书自己的 3060 Ti 运行（runs/<run>/work/ 下的 checkpoint）：
    answer_dists.json 里 run_* 三行可以从 checkpoint 重新算出；修复版的 SFT 起点就是原样运行产出的那一个；
    贪心解码答对 44 / 70 / 81 道（与进度条上的 54.3%…、86.4%、100% 一致）。"""
    import hashlib
    from codebot.utils import generate
    data = read_json("answer_dists.json")
    evalfix = json.loads((ROOT / "web/data/runs/codebot-grpo-evalfix-3060ti/report.json").read_text(encoding="utf-8"))
    assert data["run_checkpoints"]["run_sft"]["sha256"] == evalfix["seeded"][0]["sha256"]   # 同一个 SFT 起点
    eos = tok.end_token_id
    answers = [tok.encode(str(a)) + [eos] for a in range(19)]
    reward = grpo_fns["calculate_reward"]
    greedy = {}
    for key in RUN_CHECKPOINTS:
        m = load_run_model(key)
        assert hashlib.sha256(run_checkpoint(key).read_bytes()).hexdigest() == data["run_checkpoints"][key]["sha256"]
        greedy[key] = sum(reward(gt, generate(m, tok, p, temperature=0)[len(p):]) for p, gt in arithmetic_prompts())
        for k, (p, gt) in enumerate(arithmetic_prompts()):
            if k % 10:
                continue
            pid = tok.encode(p)
            x = torch.tensor([pid + a + [0] * (3 - len(a)) for a in answers])
            with torch.no_grad():
                lp = F.log_softmax(m(x)[:, len(pid) - 1:-1], -1)
            for a, ans in enumerate(answers):
                prob = lp[a, torch.arange(len(ans)), torch.tensor(ans)].sum().exp().item()
                assert abs(prob - data[key][k][a]) < 2e-5, (key, p, a)
    print("本书的三次运行，贪心解码答对的题数：", greedy)
    assert greedy == {"run_sft": 44, "run_grpo": 70, "run_evalfix": 81}


def test_optimizers_descend_by_default_but_can_ascend(dlfs6_path):
    """3.6.1：PyTorch 的优化器默认做梯度下降；SGD、Adam、AdamW 也都有 maximize 参数（默认 False）。"""
    for opt_cls in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        assert inspect.signature(opt_cls).parameters["maximize"].default is False
    steps = {}
    for maximize in (False, True):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=0.1, maximize=maximize)
        (2 * p).sum().backward()                     # J = 2θ，梯度为 +2
        opt.step()
        steps[maximize] = p.item()
    assert steps == {False: pytest.approx(-0.2), True: pytest.approx(0.2)}   # 默认往下走；maximize=True 往上走


def test_zero_gradient_still_moves_weights_through_weight_decay(dlfs6_path):
    """3.9.2：奖励函数写坏了（优势全为 0、梯度全为 0）时，AdamW 只剩权重衰减：每步把参数乘以 1 − 7e-8。"""
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=7e-6)          # 09_grpo.py 的学习率，weight_decay 默认 0.01
    assert opt.defaults["weight_decay"] == 0.01
    (p * 0).sum().backward()                        # 全零梯度（例如整批都是零方差组）
    opt.step()
    assert torch.allclose(p.detach(), torch.full((3,), 1 - 7e-6 * 0.01, dtype=torch.float64), rtol=0, atol=1e-15)
    for _ in range(999):                            # 500 轮 × 每轮 2 次更新 = 1000 步
        opt.zero_grad()
        (p * 0).sum().backward()
        opt.step()
    shrink = 1 - p[0].item()
    print(f"全零梯度下 1000 步：参数缩小了 {shrink:.2e}")
    assert abs(shrink - (1 - (1 - 7e-8) ** 1000)) < 1e-12 and shrink < 1e-4      # 不到万分之一


def test_recorded_grpo_runs(dlfs6_path):
    """页面引用的 CPU 对照实验记录（verify/_ch03_grpo_replicate.py 的输出，seed 0、1、2 各两条）：
    核对正文写到的数字。批量采样每条回答最多 40 个 token：77,760 条回答里只有 8 条写满上限被截断，
    贪心解码的评测里一条也没有。除了这 8 条，它与原书逐条调用 generate() 在统计上等价。"""
    runs = read_json("grpo_dropout_runs.json")
    assert sorted((r["seed"], r["dropout"]) for r in runs["runs"]) == [(s, d) for s in (0, 1, 2) for d in (False, True)]
    n_resp = n_trunc = 0
    for run in runs["runs"]:
        rows = [json.loads(line) for line in (DATA / run["file"]).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == run["iters"] == 60 and [r["it"] for r in rows] == list(range(60))
        assert run["env"].endswith(f"_ch03_grpo_replicate.py --iters 60 --seed {run['seed']} --tag "
                                   + ("base" if run["dropout"] else "nodrop --dropout-off"))
        accs = [r["greedy_acc"] for r in rows if "greedy_acc" in r]
        assert len(accs) == 7 and abs(accs[-1] - run["final_greedy_acc"]) < 0.05, run["file"]   # 第 0、10、…、50、59 轮
        assert [r["n_prompts"] for r in rows[:3]] == [32, 32, 17]                               # cycle 重放的三批
        n_resp += sum(r["n_prompts"] * 8 for r in rows)
        n_trunc += sum(r["truncated"] for r in rows)
        assert all(r["greedy_truncated"] == 0 for r in rows if "greedy_acc" in r), run["file"]
        assert 30 * 60 < run["seconds"] < 45 * 60                                                # 墙钟时间：33–41 分钟
    print(f"6 条运行一共采样 {n_resp:,} 条回答，写满 40 个 token 被截断的 {n_trunc} 条")
    assert (n_resp, n_trunc) == (77_760, 8)
