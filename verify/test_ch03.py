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

from _ch03_dlfs6 import ALPACA, CODEBOT, DLFS6, ROOT, arithmetic_prompts, load_defs, load_model, load_tokenizer, verify_cell

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
    assert (ids == 999).sum() == 21_835          # <|endoftext|>：每篇文档结尾一个
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
    with pytest.raises(RuntimeError):
        torch.multinomial(probs, num_samples=1)
    ok = F.softmax(logits / 0.01, dim=-1)          # 0.01 不会溢出，但也不等于 greedy
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


def test_sft_dataset_labels_padding_and_export(dlfs6_path, tok, tmp_path):
    """原书 SFTDataset 与页面数据 sft_tokens.json 逐位一致（抽查约 1/10 的样本，含 3+4=）；
    全量统计（23,635 个带 label 的位置等）由页面上的验证单元在这份数据上计算。"""
    data = json.loads((CODEBOT / "tiny_codes_sft.json").read_text(encoding="utf-8"))
    exp = read_json("sft_tokens.json")
    assert len(data) == len(exp["prompt_ids"]) == 1092
    assert exp["instructions"] == [d["instruction"] for d in data]
    k34 = exp["instructions"].index("3+4=")
    pick = sorted(set(range(0, 1092, 10)) | {k34, 1091})
    sub = tmp_path / "sft_subset.json"
    sub.write_text(json.dumps([data[k] for k in pick], ensure_ascii=False), encoding="utf-8")
    SFTDataset = load_defs("ch03/03_sft.py", ["SFTDataset"])["SFTDataset"]
    ds = SFTDataset(str(sub), tok, 256)
    for row, k in enumerate(pick):
        ids, labels = ds.samples[row]
        p, r = exp["prompt_ids"][k], exp["response_ids"][k]
        full = p + r
        ref_ids = (full[:-1] + [0] * 256)[:256]
        ref_labels = (([-100] * len(p) + r)[1:] + [-100] * 256)[:256]
        assert ids == ref_ids and labels == ref_labels      # 导出的 token 能逐位重建原书的样本
    ids, labels = ds.samples[pick.index(k34)]
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
    acc = {}
    for name in ("sft", "grpo"):
        acc[name] = sum(reward(gt, generate(models[name], tok, p, temperature=0)[len(p):])
                        for p, gt in arithmetic_prompts())
    print("81 道加法的 greedy 正确数：", acc)
    assert acc == {"sft": 44.0, "grpo": 67.0}


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
    assert first == second             # itertools.cycle 记住了第一轮，之后原样重放


def test_get_batch_never_contains_eos(dlfs6_path, tok, grpo_fns):
    ds = grpo_fns["GRPODataset"](tok)
    p = ALPACA.format("3+4=")
    ids, mask = ds.get_batch([p, p], ["7", "12"], "cpu")
    assert ids[0, -3:].tolist() == [10, 55, 0] and mask[0, -3:].tolist() == [0, 1, 0]
    assert ids[1, -3:].tolist() == [10, 49, 50] and mask[1, -3:].tolist() == [0, 1, 1]
    assert 999 not in ids.tolist()[0] + ids.tolist()[1]  # EOS 不在 batch 里，也就从不进入 loss


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
    print(f"dropout 梯度与干净梯度的余弦：{cos:.3f}")
    assert cos < 0.9


def test_zero_gradient_still_moves_weights_through_weight_decay(dlfs6_path):
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p], lr=7e-6)          # 09_grpo.py 的学习率，weight_decay 默认 0.01
    (p * 0).sum().backward()                        # 全零梯度（例如整批都是零方差组）
    opt.step()
    assert torch.allclose(p.detach(), torch.full((3,), 1 - 7e-6 * 0.01))


def test_recorded_grpo_runs(dlfs6_path):
    """页面引用的 GRPO 对照实验记录（研究阶段 seed 0 与本章复现 seed 2）：核对正文写到的数字。"""
    runs = read_json("grpo_dropout_runs.json")
    for run in runs["runs"]:
        rows = [json.loads(line) for line in (DATA / run["file"]).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == run["iters"]
        accs = [r["greedy_acc"] for r in rows if "greedy_acc" in r]
        assert abs(accs[-1] - run["final_greedy_acc"]) < 0.05, run["file"]
