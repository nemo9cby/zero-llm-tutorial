"""第 3.9 节的对照实验：GRPO 更新时开着 dropout（脚本原样）与关掉 dropout（更新前 model.eval()）。

GRPODataset / calculate_reward / compute_probs / grpo_loss 都是原书的函数，用 _ch03_dlfs6.load_defs
直接从 third_party/dlfs6/ch03/09_grpo.py 里取出。唯一的改动是批量采样（81 个 prompt 都是 22 个 token，
不需要 padding），统计上等价于逐条调用 256 次 generate()。超参数与脚本相同：lr 7e-6、G=8、batch 32、
eps 0.2、每批数据更新 2 次、clip_grad_norm 1.0、AdamW。每轮写一行 JSON 指标，每 10 轮加一次 greedy 准确率。

    CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch03_grpo_replicate.py --iters 60 --seed 2 --tag base
    CUDA_VISIBLE_DEVICES=-1 uv run python verify/_ch03_grpo_replicate.py --iters 60 --seed 2 --tag nodrop --dropout-off

页面上的 web/data/ch03/grpo_dropout/seed2_*.jsonl 就是这两条命令的输出（CPU，约 35 分钟一条）。
"""
import argparse, json, sys, time
from itertools import cycle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "verify"))
sys.path.insert(0, str(ROOT / "third_party" / "dlfs6"))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from _ch03_dlfs6 import CODEBOT, load_defs
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--dropout-off", action="store_true")
ap.add_argument("--tag", default="base")
ap.add_argument("--out", default=str(ROOT / "runs" / "ch03-grpo-dropout"))
args = ap.parse_args()
torch.set_num_threads(4)
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

g = load_defs("ch03/09_grpo.py", ["GRPODataset", "calculate_reward", "compute_probs", "grpo_loss"])
GRPODataset, calculate_reward, compute_probs, grpo_loss = (g[k] for k in ["GRPODataset", "calculate_reward", "compute_probs", "grpo_loss"])

torch.manual_seed(args.seed)
tok = BPETokenizer.load_from(str(CODEBOT / "merge_rules.pkl")); EOS = tok.end_token_id
model = GPT.load_from(str(CODEBOT / "model_sft.pt")); old_model = GPT.load_from(str(CODEBOT / "model_sft.pt")); old_model.eval()
opt = torch.optim.AdamW(model.parameters(), lr=7e-6)
ds = GRPODataset(tok)
data_iter = cycle(DataLoader(ds, batch_size=32, shuffle=True))


@torch.no_grad()
def sample(m, prompts, G, temperature, max_new_tokens=40):
    m.eval()
    ids = torch.tensor([tok.encode(p) for p in prompts for _ in range(G)])
    out = [[] for _ in range(len(ids))]; done = torch.zeros(len(ids), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits = m(ids)[:, -1, :]
        nxt = logits.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
        for k in torch.nonzero(~done).flatten().tolist():
            if nxt[k].item() == EOS: done[k] = True
            else: out[k].append(nxt[k].item())
        if done.all(): break
        ids = torch.cat([ids, nxt[:, None]], 1)
    return [tok.decode(o) for o in out], done


def greedy_acc(m):
    resp, _ = sample(m, [p for p, _ in ds.data], 1, 0)
    return sum(calculate_reward(gt, r) > 0 for r, (_, gt) in zip(resp, ds.data)) / len(ds.data) * 100


log = open(out_dir / f"grpo_{args.tag}_seed{args.seed}.jsonl", "w")
t0 = time.time()
print(f"[{args.tag}] start greedy acc {greedy_acc(model):.1f}%", flush=True)
for i in range(args.iters):
    prompts, gts = next(data_iter)
    old_model.load_state_dict(model.state_dict())
    responses, done = sample(old_model, prompts, 8, 1.0)
    all_prompts = [p for p in prompts for _ in range(8)]
    r = torch.tensor([calculate_reward(int(gts[k // 8]), responses[k]) for k in range(len(responses))])
    rg = r.view(-1, 8)
    adv = (rg - rg.mean(1, keepdim=True)).flatten()
    ids, mask = ds.get_batch(all_prompts, responses, "cpu")
    if args.dropout_off: model.eval()
    else: model.train()
    gns, losses = [], []
    for _ in range(2):
        opt.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, adv, 0.2)
        loss.backward()
        gns.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        opt.step(); losses.append(loss.item())
    with torch.no_grad():
        m = mask[:, 1:].bool()
        model.eval()
        lp_all = F.log_softmax(model(ids)[:, :-1, :], -1)
        lp_new = lp_all.gather(-1, ids[:, 1:, None]).squeeze(-1)[m]
        lp_old = torch.log(compute_probs(old_model, ids)[m])
        ratio = (lp_new - lp_old).exp()
        A_tok = adv[:, None].expand_as(m)[m]
        rec = dict(it=i, reward_mean=r.mean().item(), zero_var_frac=(rg.std(1) == 0).float().mean().item(),
                   loss0=losses[0], loss1=losses[1], grad_norm0=gns[0], grad_norm1=gns[1],
                   clipfrac=(((ratio > 1.2) & (A_tok > 0)) | ((ratio < 0.8) & (A_tok < 0))).float().mean().item(),
                   kl_k3=(ratio - 1 - (lp_new - lp_old)).mean().item(),
                   entropy=-(lp_all.exp() * lp_all).sum(-1)[m].mean().item(),
                   resp_len=m.sum(1).float().mean().item(), n_prompts=len(prompts))
    if i % 10 == 0 or i == args.iters - 1:
        rec["greedy_acc"] = greedy_acc(model)
        print(f"[{args.tag}] it {i:3d} acc {rec['greedy_acc']:5.1f}% r {rec['reward_mean']:.3f} zv {rec['zero_var_frac']:.2f} "
              f"loss {rec['loss1']:+.4f} gn {rec['grad_norm0']:.2f} H {rec['entropy']:.3f} t={time.time()-t0:.0f}s", flush=True)
    log.write(json.dumps(rec) + "\n"); log.flush()
print(f"[{args.tag}] done in {time.time()-t0:.0f}s", flush=True)
