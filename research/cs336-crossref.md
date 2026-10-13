# CS336 × DLFS6 cross-reference dossier

Scope: Stanford CS336 "Language Modeling from Scratch" (latest offering Spring 2026, plus Spring 2025 deltas) mapped against the code of 『ゼロから作るDeep Learning ❻ —LLM編』 (DLFS6, repo `oreilly-japan/deep-learning-from-scratch-6`, local clone `C:\Users\chenf\Projects\_refs\dlfs6`).
Researched 2026-10-11. Evidence tags used below:

- **[S]** read directly from a primary source (course site, handout PDF, repo code or tests). File or URL given.
- **[V]** we re-ran it ourselves. Script and command are in §9, and the numbers are pasted from the run.
- **[E]** napkin estimate. It still has to be measured on real hardware before the tutorial states it as fact.

---

## 0. TL;DR (the ten things a tutorial author needs)

1. **Latest offering = Spring 2026** (Hashimoto & Liang, Mar 30 – Jun 3 2026). It runs on **B200s and Modal** where 2025 used H100s and Together. The handouts were re-versioned 26.x, and A5 was rewritten: OLMo-2-0425-1B on GSM8K, with GRPO, Dr. GRPO, RFT, MaxRL, off-policy clipping and GSPO. [S]
2. **DLFS6 is visibly built on CS336 A1.**
   - StoryBot's pretraining config is *exactly* CS336 A1's TinyStories config: vocab 10 000, context 256, d_model 512, 16 heads, 4 layers, d_ff 1344, RoPE θ 10 000, 32×40 000×256 = **327 680 000 tokens**.
   - DLFS6 saves a checkpoint at iteration 5 000. The loop counter starts at 0, so the checkpoint has made 5 001 updates and seen 40 968 192 tokens. That is about the size of the handout's low-resource *reference run* (32 × 5 000 × 256 = 40 960 000 tokens) and just over the low-resource *budget*, which is 40 000 000 tokens (A1 v26.0.3 §7.2). Its learning rate is still at 88 % of the peak, so it is only a rough stand-in for a run whose schedule ends there (tutorial page cs336, verify X.1-a). [V]
3. **Same data, byte for byte.**
   - DLFS6's `storybot/tiny_stories_{train,valid}.txt` are the same files as CS336's `TinyStoriesV2-GPT4-{train,valid}.txt` (identical size and SHA-256).
   - DLFS6's `webbot/owt_valid.txt` is identical to CS336's `owt-sample/owt_valid.txt.gz` once decompressed.
   - So CS336's numeric targets apply to DLFS6 runs: val loss ≤ 1.45 on the full budget, ≤ 2.00 on the 40 M-token budget. [V]
4. **Same function, up to one ε.** We loaded CS336's A1 test-fixture weights into DLFS6's StoryBot `GPT` by renaming keys only. Its logits match CS336's `test_transformer_lm` snapshot to **1.05e-5**, but only after RMSNorm eps is set to 1e-5. As shipped, DLFS6 uses `nn.RMSNorm()` with default eps≈1.19e-7, and the difference is 4.6e-4, which *fails* the 1e-4 tolerance. [V]
5. **BPE: one tie-break key decides "passes CS336" or not.** DLFS6's ch04 trainer breaks ties by token-ID tuples. On CS336's `corpus.en` (vocab 500) the first 92 merges match the reference; it diverges at merge #93 (0-based index 92), with 170/243 positions equal. Switching the key to byte-string lexicographic order reproduces all 243 reference merges exactly. Speed sits right at CS336's 1.5 s bar and depends on how busy the machine is: two recorded sessions on this machine, both under load, 5 runs each, took 1.27–1.52 s and 0.95–1.11 s (`web/data/cs336/a1_results.json` → `speed_history`; tutorial ch04 §4.2.4). The test itself passed both times, at 1.29 s and 1.43 s. The single research-phase run in V1 took 1.04 s. [V]
6. **Other components verified against CS336 protocols.**
   - DLFS6 `RoPE` (interleaved pairs) matches CS336's `test_rope` snapshot to 3e-7.
   - DLFS6 `AdamW` matches `torch.optim.AdamW` to 4.8e-6 after 1 000 steps, which passes CS336's `test_adamw`.
   - DLFS6 `get_batch` is off by one: it never samples the last valid window (max start 91 against the expected 92), so it would fail CS336's `test_get_batch`. [V]
7. **DLFS6 GRPO in CS336 vocabulary** (`ch03/09_grpo.py`):
   - advantage = r − group mean, with **no std division**. That is Dr.-GRPO-style.
   - loss = Σ_tokens / #sequences, which is a "constant" normalizer.
   - 2 gradient steps per rollout batch with PPO clip ε = 0.2.
   - no KL term; G = 8; 32 prompts × 8 = 256 rollouts. The 256 rollouts are identical to CS336's `rollout_batch_size=256, group_size=8`.
   - Two numerical caveats, both verified. First, the ratio is computed in *probability* space as p/(p_old+1e-8), so at p = 1e-8 the ratio is 0.5. Second, the CodeBot policy keeps **dropout 0.1 active** while `old_model` is in eval mode. On the *first* "on-policy" step that gives mean|r−1| ≈ 0.05 and about 0.9 % of tokens outside the clip range, using random weights. [V] With the trained SFT weights it is about half of the tokens (§11).
8. **CS336 goes much deeper on systems, scaling, data and RL theory.** Its A2 to A5 have no DLFS6 counterpart: profiling, a Triton FlashAttention-2 kernel, DDP/ZeRO-1/FSDP, IsoFLOPs scaling-law fitting, Common Crawl filtering and dedup, and the RL estimator zoo. DLFS6 is gentler on tokenization and attention intuition, and gives small end-to-end bots that run on one GPU.
9. **CS336 "executable lectures"** are Python programs traced by **edtrace** (github.com/percyliang/edtrace, MIT). The pipeline is `python -m edtrace.execute -m lecture_XX`, which writes `var/traces/lecture_XX.json`, which a React viewer replays line by line.
   - `text()` emits markdown and `# @inspect x` shows a variable's value; plain `assert` statements carry the claims.
   - GPU lectures are executed on **Modal (`gpu="B200:4"`)** and the trace JSON is copied back.
   - This is a ready template for our "every claim is executable" goal.
10. **Windows caveats for the user's machine** [S/V]:
    - CS336 A1's tokenizer tests `import resource` at module top. On native Windows that raises `ModuleNotFoundError` [V], so the whole file errors.
    - A2's Triton kernels, A4's fasttext/resiliparse, and A5's vLLM/flash-attn are Linux-first.
    - Plan: CPU tests and small runs locally; WSL2 or Modal/RunPod/Lambda for anything else.

---

## 1. Offerings, sites, repos (URLs)

| What | URL | Notes |
|---|---|---|
| Course site (latest, Spring 2026) | https://cs336.stanford.edu/ | Mon/Wed 3:00–4:20pm, Skilling Auditorium. Instructors Tatsunori Hashimoto and Percy Liang; CAs Herman Brunborg, Marcel Rød, Steven Cao. [S] |
| Spring 2025 archive | https://cs336.stanford.edu/spring2025/ | Tue/Thu. Compute prices quoted for H100. [S] |
| Spring 2024 archive | https://cs336.stanford.edu/spring2024/ | |
| Site source | https://github.com/stanford-cs336/stanford-cs336.github.io | |
| 2026 lecture videos | https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV | Linked from the 2026 site. [S] |
| 2026 lectures repo (executable `.py` + PDF) | https://github.com/stanford-cs336/lectures | Last commit 2026-09-12. [S] |
| 2025 lectures repo | https://github.com/stanford-cs336/spring2025-lectures | Own `execute.py` + `trace-viewer/`. [S] |
| 2026 trace viewer | `https://cs336.stanford.edu/lectures/?trace=lecture_01` | Same pattern for 02, 02_recording, 06, 07, 10, 12, 13, 14, 17. [S] |
| 2025 trace viewer | `https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json` | [S] |
| edtrace (tracer + viewer) | https://github.com/percyliang/edtrace | MIT license, backend v0.1.20, pushed 2026-10-08. [S] |
| A1 | https://github.com/stanford-cs336/assignment1-basics (handout `cs336_assignment1_basics.pdf`, v26.0.3) | Tags `spring2025` and HEAD 2026-04-07. [S] |
| A1 leaderboard | https://github.com/stanford-cs336/assignment1-basics-leaderboard | |
| A2 | https://github.com/stanford-cs336/assignment2-systems (`cs336_assignment2_systems.pdf`, v26.1.3) | HEAD 2026-05-01. |
| A2 leaderboard | https://github.com/stanford-cs336/assignment2-systems-leaderboard | |
| A3 | https://github.com/stanford-cs336/assignment3-scaling (`cs336_assignment3_scaling.pdf`, v26.0.5) | Tags `spring2025`, `spring2026`. |
| A4 | https://github.com/stanford-cs336/assignment4-data (`cs336_assignment4_data.pdf`, v26.0.1) | |
| A5 | https://github.com/stanford-cs336/assignment5-alignment (`cs336_spring2026_assignment5_alignment.pdf` v26.0.0 + `cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`) | HEAD 2026-06-04. The `spring2025` tag holds the Qwen/MATH version. |
| Data | https://huggingface.co/datasets/roneneldan/TinyStories (`TinyStoriesV2-GPT4-*.txt`) and https://huggingface.co/datasets/stanford-cs336/owt-sample | Download commands are in the A1 README. |
| DLFS6 data | https://huggingface.co/datasets/koki0702/zero-llm-data | Contains the same TinyStories and OWT files (see §8, V8). |
| DLFS6 book page | https://www.oreilly.co.jp/books/9784814401611/ | 2026-06-03, 384 pp. [S] |

- **Newer assignments:** there is no A6. Every 2026 repo carries `AGENTS.md`/`CLAUDE.md` with *AI-assistant rules for enrolled students*: "teaching assistant, not solution generator", no code for TODOs. [S]
  - We only cross-reference topics and verify DLFS6 code against public test *protocols*.
  - The tutorial should **not** publish CS336 assignment solutions or adapter code.
- **Repo licenses:** MIT (Stanford). Short attributions are fine; prefer linking over copying handout text.

---

## 2. Lecture list — Spring 2026 (with 2025 deltas)

| # | Date | Lecturer | Format | Title → one-line content |
|---|---|---|---|---|
| 1 | Mar 30 | Percy | `.py` | **Overview, tokenization**. Course philosophy, then character → byte → word → BPE tokenizers. The lecture's `merge()`/`train_bpe()` is nearly line-for-line DLFS6 ch01. |
| 2 | Apr 1 | Percy | `.py` (+ recording trace) | **PyTorch (einops), resource accounting**. Tensors and dtypes (fp32/fp16/bf16/fp8) and memory; FLOPs of matmuls; 6·N·D; arithmetic intensity and roofline; gradient accumulation; activation checkpointing. |
| 3 | Apr 6 | Tatsu | PDF | **Architectures, hyperparameters**. Pre- vs post-norm, RMSNorm, dropping biases, SwiGLU, serial vs parallel layers, RoPE, d_ff/d_model ≈ 4 (8/3 for GLU), aspect ratio, vocab sizes, dropout vs weight decay, z-loss, QK-norm, logit soft-capping, SWA/NoPE hybrids. |
| 4 | Apr 8 | Tatsu | PDF | **Attention alternatives and MoE**. Linear attention and its recurrent form, Mamba-2, hybrids (MiniMax M1), mixture-of-experts. |
| 5 | Apr 13 | Tatsu | PDF | **GPUs, TPUs**. GPU anatomy, performance model, unpacking FlashAttention. |
| 6 | Apr 15 | Percy | `.py` | **Kernels, Triton**. Benchmarking and profiling; naive vs builtin vs compiled GeLU; Triton GeLU, softmax, row-sum and tiled matmul+ReLU; PTX. |
| 7 | Apr 20 | Percy | `.py` | **Parallelism**. Collectives, NCCL bandwidth, bare-bones data/tensor/pipeline parallel on deep MLPs. Multiprocessing is turned off while tracing and the stdout is linked separately. |
| 8 | Apr 22 | Tatsu | PDF | **Parallelism** (basics). Networking, the forms of parallelism (DP/ZeRO/FSDP/TP/PP/SP/EP), large-run case studies. |
| 9 | Apr 27 | Tatsu | PDF | **Scaling laws, basics**. History, Kaplan vs Chinchilla, IsoFLOPs. |
| 10 | Apr 29 | Percy | `.py` | **Inference**. Arithmetic intensity of prefill vs decode, KV-cache size reduction (GQA/MQA/MLA), quantization, pruning/distillation, speculative sampling, continuous batching, PagedAttention. |
| 11 | May 4 | Tatsu | PDF | **Scaling laws, details**. MiniCPM and DeepSeek recipes, muP, LR/batch scaling, WSD-style schedules. |
| 12 | May 6 | Percy | `.py` | **Evaluation**. Perplexity; exam, chat, agentic, reasoning and safety benchmarks; realism and validity. |
| 13 | May 11 | Percy | `.py` | **Data I: sources, datasets**. Common Crawl, Wikipedia, GitHub, arXiv; WebText, C4, The Pile, MassiveText, RefinedWeb, Dolma, DCLM, Nemotron-CC, Common Pile; copyright. |
| 14 | May 13 | Percy | `.py` | **Data II: filtering, dedup, mixing, synthetic data**. Classifier filtering, MinHash/LSH, mixture optimization, post-training data. |
| 15 | May 18 | Tatsu | PDF | **Mid/post-training: SFT/RLHF**. Instruction data, SFT, reward models, PPO-RLHF, DPO. |
| 16 | May 20 | Tatsu | PDF | **Post-training: RLVR**. PPO → GRPO; the std-normalization and length-normalization critique (Dr. GRPO); case studies DeepSeek-R1, Kimi K1.5, Qwen 3. |
| 17 | May 27 | Percy | `.py` | **Alignment: multimodality**. CLIP, SigLIP, LLaVA(-OneVision), Qwen-VL 1–3, Chameleon, omni models. |
| 18 | Jun 1 | Daniel Selsam | guest | — |
| 19 | Jun 3 | Dan Fu | guest | — |

- **Spring 2025 differences** [S]:
  - L4 was MoE only.
  - L6 (Kernels/Triton) was given by Tatsu.
  - L7 was a PDF ("Parallelism basics") and L8 was the executable parallelism lecture.
  - L17 was "Alignment: RL (Percy, .py)".
  - Guests were Junyang Lin and Mike Lewis.
  - Executable lectures were 01, 02, 06, 08, 10, 12, 13, 14, 17.

---

## 3. Assignments (2026), with 2025 deltas

Common harness [S]:
- Students fill `tests/adapters.py` with glue only; the tests are fixed.
- Most tests are **snapshot tests**: `NumpySnapshot.assert_match` compares against `tests/_snapshots/*.npz` with `np.testing.assert_allclose`. Defaults are `rtol=1e-4, atol=1e-2`, overridden per test, and `--snapshot-exact` forces 0 tolerance.
- Non-array snapshots are pickles.
- Environments are managed by `uv`. A1/A2 need Python 3.12–3.13 and `torch~=2.11`.

### A1 Basics: "Building a Transformer LM" (v26.0.3)

- **Rule:** no `torch.nn`, `torch.nn.functional` or `torch.optim` definitions, except `nn.Parameter`, containers and the `optim.Optimizer` base class. [S]
- **Adapters, in file order** [S]:
  - `run_linear`, `run_embedding`, `run_swiglu`, `run_scaled_dot_product_attention`, `run_multihead_self_attention`
  - `run_multihead_self_attention_with_rope`, `run_rope`, `run_transformer_block`, `run_transformer_lm`, `run_rmsnorm`
  - `run_silu`, `run_get_batch`, `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`
  - `get_adamw_cls`, `run_get_lr_cosine_schedule`, `run_save_checkpoint`, `run_load_checkpoint`, `get_tokenizer`, `run_train_bpe`
  - The 2025 list is identical.
- **Weight-name contract.** It comes from the fixture `tests/fixtures/ts_tests/model.pt`: vocab 10 000, ctx 16, d_model 64, 3 layers, 4 heads, d_ff 128. Keys:
  - `token_embeddings.weight`
  - `layers.{i}.attn.{q,k,v,output}_proj.weight`
  - `layers.{i}.ffn.{w1,w2,w3}.weight`, where w1 is the SiLU branch, w3 the linear branch and w2 the down-projection
  - `layers.{i}.ln{1,2}.weight`, `ln_final.weight`, `lm_head.weight`
  - There are no biases anywhere. [S/V]
- **How the tests check each component** [S]:
  - Model pieces are compared to snapshots with fixture weights. `atol` is 1e-5 for SwiGLU, SDPA (3-D and 4-D), MHA, MHA+RoPE and RoPE; 1e-4 for RMSNorm, block and LM, and the LM also uses rtol 1e-2. Linear and embedding use the default tolerance.
  - Softmax is checked against `F.softmax`, including inputs +100.
  - Cross-entropy is checked against `F.cross_entropy`, including logits ×1000, at atol 1e-4.
  - Gradient clipping is checked against `clip_grad_norm_` with one frozen parameter, at atol 1e-5.
  - AdamW must match `torch.optim.AdamW` at atol 1e-4 after 1 000 steps on `Linear(3,2)`, otherwise the snapshot.
  - The cosine schedule is compared to exact expected values, with warmup 7, cycle 21, max 1 and min 0.1.
  - `get_batch` is a statistical test: 1 000×32 draws on `arange(100)` with ctx 7. Every start index 0..92 must appear, counts must fall within ±5σ, and `cuda:99` must raise.
  - Checkpointing is a round-trip test.
  - `train_bpe` on `corpus.en` with vocab 500 must reproduce the reference merges exactly, the vocab key and value sets must match, and it must run in **< 1.5 s**.
  - `train_bpe` on `tinystories_sample_5M.txt` with vocab 1000 is a pickle snapshot, and no vocab entry may contain `<|`.
  - The tokenizer must round-trip, equal `tiktoken.get_encoding("gpt2")` on several fixtures, keep overlapping special tokens (`<|endoftext|><|endoftext|>` as one token), and pass an `encode_iterable` memory test (1 MB `RLIMIT_AS`, **Linux only**).
- **Data** [S]: TinyStoriesV2-GPT4 train/valid, and the OWT sample from HF `stanford-cs336/owt-sample`.
- **Configs and targets** [S]:
  - BPE on TinyStories: vocab 10k, ≤ 30 min and 30 GB RAM; the hint says under 2 min with multiprocessing.
  - §2.5 "Experimenting with BPE Tokenizer Training" parallelizes only pre-tokenization. Its "Optimizing the merging step" paragraph (an incremental pair-count index) ends: "the merging part of BPE training is not parallelizable in Python". Its "Low-Resource Tip: 'Downscaling'" says to debug the tokenizer on the TinyStories validation set ("22K documents instead of 2.12M") before the full set.
  - BPE on OWT: vocab **32k**, ≤ 12 h and 100 GB RAM.
  - TinyStories LM: vocab 10 000, ctx 256, d_model 512, d_ff 1344, θ 10 000, 4 layers, 16 heads, "~17 M non-embedding params", **327 680 000 tokens**, "20–30 min on 1 B200".
  - Deliverable: **val loss ≤ 1.45**.
  - Low-resource variant (the CPU/MPS tip in §7.2): **40 000 000 tokens**, target **≤ 2.00**. The staff's own reference run on an M4 Max used 32×5000×256 = 40 960 000 tokens: 36 min on MPS or 1 h 22 min on CPU, val 1.80 at step 5000. So 40 960 000 is that reference run, not the budget.
  - Ablations: drop RMSNorm, post-norm, NoPE, SwiGLU vs SiLU (SiLU with d_ff = 4·d).
  - OWT main run.
  - Leaderboard: ≤ **45 min on one B200**, beat the naive loss of 5.0.
- **Accounting problems** [S]:
  - FLOPs and params of a GPT-2-XL-shaped model: vocab 50 257, ctx 1024, 48 layers, d 1600, 25 heads, d_ff 4288.
  - AdamW memory and FLOPs.
  - An MFU exercise at 495 TF32 TFLOP/s on H100 with 50 % MFU.
- **Compute (2026)** [S]: learning_rate 2 B200-h; batch_size 1; each ablation 0.5; OWT 2; leaderboard 10.
- **2025 delta** [S]: H100s ("~30–40 min on 1 H100"), and the leaderboard allowed 1.5 H100-h (90 min).
  - The 2026 changelog also says "Fix AdamW disparity to paper (order of weight-decay vs gradient update)", "Assume B200s" and "Latex → Typst".

### A2 Systems (v26.1.3)

- **Problems** [S]:
  - benchmarking script (warm-up; `torch.cuda.synchronize`)
  - nsys profiling with NVTX
  - mixed-precision accumulation (fp16 vs fp32 sums) and autocast dtypes in a ToyModel
  - memory profiling (`torch.cuda.memory._record_memory_history`, pytorch.org/memory_viz)
  - **activation checkpointing** (new in 2026; includes the "memory-optimal checkpointing" analysis)
  - PyTorch attention benchmarks; `torch.compile`
  - **FlashAttention-2**: a PyTorch tiled forward, then a Triton forward with causal masking, then a backward via `torch.compile`, then benchmarking with `triton.testing.do_bench` from 128 to 65 536 tokens in bf16/fp32
  - optional Triton backward (Algorithm 2)
  - single-node all-reduce benchmarks; naive DDP; DDP that overlaps per-parameter communication
  - **optimizer-state sharding** (a simplified ZeRO-1)
  - **FSDP** (new in 2026)
  - analytical DP/FSDP/TP/2-D parallel calculations
- **Adapters** [S]:
  - `get_flashattention_autograd_function_pytorch`, `get_flashattention_autograd_function_triton`
  - `get_ddp`, `ddp_on_after_backward`
  - `get_fsdp`, `fsdp_on_after_backward`, `fsdp_gather_full_params`
  - `get_sharded_optimizer`
  - 2025 had `get_ddp_individual_parameters`, `get_ddp_bucketed` and `ddp_bucketed_on_train_batch_start` (bucketed DDP), and no FSDP.
- **Tests** [S]:
  - Flash forward/backward against a reference at rtol/atol 1e-2. The LSE tensor must be saved for backward. Triton tests are skipped without CUDA.
  - DDP, sharded-optimizer and FSDP tests use `mp.spawn` with world_size 2 on the **gloo/CPU** backend.
  - DDP runs on `ToyModel` and `ToyModelWithTiedWeights`.
  - FSDP is checked against non-parallel training at atol 1e-6 in fp32, or 1e-4 with fp16 compute.
- **Model sizes** [S]: small 768/3072/12/12; medium 1024/4096/24/16; large 1280/5120/36/20; xl 2560/10240/32/32; 10B 4608/12288/50/36. Vocab 10k, batch 4, ctx 512.
- **Leaderboard** [S]:
  - Fastest full training step on 2 B200s, batch 2, **ctx 32 768**.
  - Model: d 4096, 34 layers, 32 heads, d_ff 11 008, vocab 151 936, bf16, causal.
  - Must finish within 10 min from a cold cache, and beat the naive 10 s per step.
- **Practical notes** [S]:
  - The repo ships the **staff A1 solution** as `cs336-basics/`.
  - Its RoPE reads interleaved pairs but writes `concat(x1_rot, x2_rot)`, a half-split layout. Attention is unchanged because Q and K are permuted identically (see V3).
  - B200 figures quoted in the handout: 80 TFLOPS fp32 and 2500 TFLOPS fp16/bf16.

### A3 Scaling (v26.0.5)

- **2026 setup** [S]:
  - Students *query a training API*, `http://hyperturing.stanford.edu:8000`, reachable only from the Stanford network.
  - Training runs on B200s in JAX/Equinox. Data is DCLM tokenized with a 32K vocab, ctx 512, with no epoching and a fixed data order.
  - Budget: **12 B200-hours** of queries (43 200 s) to fit a scaling law for a **48 B200-hour** final run. Students submit a config and a predicted loss.
  - The API model is A1-like plus **QK-norm**, GQA (`num_key_value_heads`), RMSNorm eps 1e-6, RoPE θ = 1e6 and untied embeddings.
  - Example optimizer: AdamW β = (0.9, 0.95), wd 0.01, warmup 5 %, cosine to 10 %, clip 1.0.
- **Warm-up problem `chinchilla_isoflops`** [S]: fit N_opt(C) and D_opt(C) from `data/isoflops_curves.json`, extrapolate to 1e23 and 1e24 FLOPs, using C = 6ND. The handout suggests taking each budget's min-loss run rather than fitting parabolas.
  - **Reusable for us:** the data file is MIT-licensed, so we can build an in-browser IsoFLOPs widget on it.
- **2025** [S]: a FLOPs-budgeted API, a 1e19 target and 2e18 FLOPs for fitting.

### A4 Data (v26.0.1)

- **Adapters** [S]:
  - `run_extract_text_from_html_bytes`, which uses resiliparse with encoding detection
  - `run_identify_language` (fastText `lid.176.bin`; must return "en" and "zh")
  - `run_mask_emails`, `run_mask_phone_numbers`, `run_mask_ips`. These replace matches with `|||EMAIL_ADDRESS|||` and similar placeholders and return (text, count).
  - `run_classify_nsfw` and `run_classify_toxic_speech` (Dolma Jigsaw fastText models)
  - `run_classify_quality` (the student trains a fastText classifier on Wikipedia-linked pages vs random Common Crawl)
  - `run_gopher_quality_filter`, which removes a document if it has fewer than 50 or more than 100 000 words, mean word length outside 3–10, more than 30 % of lines ending in "...", or fewer than 80 % of words containing a letter
  - `run_exact_line_deduplication` (hash counts in a 2-pass scheme)
  - `run_minhash_deduplication` (n-gram MinHash plus LSH bands, a Jaccard threshold, normalization by lowercasing, stripping punctuation and whitespace, removing accents, and NFD; clusters are kept as one random member)
- **Tests** [S]: small fixtures. NSFW, toxicity and quality tests are explicitly "sanity checks".
- **Data and goal** [S]:
  - Input: 2 500 English-prefiltered WET files from CC-MAIN-2026-12.
  - Goal: minimize perplexity on Paloma **C4-100-domains**.
- **Training** [S]:
  - GPT-2 tokenizer; the model in `cs336_basics/train_config.py` is d 1024, 24 layers, 16 heads, d_ff 3072, ctx 512. The handout calls it "GPT-2 small-shaped, ~430 M".
  - 8×B200 DDP, batch 128 per device, 16 384 steps, about 8.6 B tokens, roughly 2 h.
  - lr 6e-4, warmup 3 %, wd 0.1, β = (0.9, 0.98), eps 1e-9, bf16, compile.
  - Training on Modal uses `gpu=f"B200:{N_GPUS}"`.

### A5 Alignment and reasoning RL: 2026 version (v26.0.0)

- **Setup** [S]:
  - Model and data: **OLMo-2-0425-1B** on **GSM8K**, with prompts `question_only` (boxed), `r1_zero` and `r1_zero_three_shot`.
  - Training uses HF transformers on GPU 0 and a vLLM server on GPU 1, with NCCL weight sync (`VLLMServer.sync_policy_weights`). On Modal this is `GPU="B200:2"`.
  - Sampling: temperature 1.0, top-p 1.0, max 512 tokens, stop at `</answer>`.
  - Grader: `drgrpo_grader.r1_zero_reward_fn`, binary, with no partial credit for format.
- **Adapters** [S]:
  - `run_tokenize_prompt_and_output`, which returns `input_ids`, `labels` and `response_mask`, with no special tokens between prompt and response
  - `run_get_response_log_probs`, with optional token entropy
  - `run_compute_rollout_rewards`
  - `run_compute_group_normalized_rewards(raw_rewards, group_size, baseline∈{mean,none}, advantage_eps=1e-6, advantage_normalizer∈{std,none,mean})`. It uses the unbiased `torch.std`, and eps is added to the normalizer.
  - `run_compute_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs, importance_reweighting_method∈{none,noclip,grpo,gspo}, old_log_probs, cliprange, response_mask)`, which returns a per-token loss
  - `run_aggregate_loss_across_microbatch(per_token_loss, mask, loss_normalization∈{sequence,constant}, normalization_constant)`
  - `run_grpo_train_step(...)`, which does gradient accumulation, clips to `max_grad_norm`, steps, and leaves grads at None
  - Supplement adapters: `get_packed_sft_dataset`, `run_iterate_batches`, `run_parse_mmlu_response`, `run_parse_gsm8k_response`, `run_compute_per_instance_dpo_loss`.
- **Default hyperparameters** [S]: n_train 6 400, n_val 1 024, 200 rollout steps, **lr 1e-5**, **rollout = train batch = 256 (32 prompts × G = 8)**, grad-accum 32, clip 1.0, **AdamW β = (0.9, 0.95), wd 0.0**. Evaluate every 10 steps on ≥1 024 examples, **4 seeds**.
  - **Target:** ≥ 25 % GSM8K validation accuracy, averaged over seeds.
- **Experiments and B200-hours** [S]: standard 2; LR sweep 4; prompt ablation 4; on-policy variants 8; off-policy 8; plus "try your own estimator".
- **Variant grid** [S]:

| name | baseline | advantage normalizer | loss normalization | importance reweighting |
|---|---|---|---|---|
| GRPO (standard) | mean | std (+eps) | sequence (mean over tokens, then sequences) | none (on-policy) |
| GRPO_constant | mean | std | constant c = B·L_max (L_max = 512) | none |
| Dr_GRPO | mean | none | constant | none |
| RFT (expert iteration) | none | none | constant | none (incorrect samples have zero weight and are pruned) |
| MaxRL | mean | **mean** (÷ group mean reward) | constant | none |
| offpolicy_naive / noclip / clip / gspo | (from chosen variant) | | | none / token-ratio / PPO clip ε=0.2 / sequence geometric-mean ratio, clip 3e-4 |

  - Off-policy setting: rollout 256, train batch 8, grad-accum 1, which gives **32 steps per inference batch**.
  - Zero-advantage sequences are pruned to save compute.
  - CISPO (an upper clip on the weight only) is optional.
- **Theory content** [S]: REINFORCE; baselines with a variance exercise; std normalization as a stability trick that changes the objective; sequence-length normalization biasing toward short answers; RFT vs Dr. GRPO; derived difficulty reweightings w(p) induced by each normalizer; token- vs sequence-level importance weights with surrogate objectives; PPO clipping; GSPO.
  - Literature summary (not the graded derivation): with binary reward and success rate p, no normalization gives ∇p; ÷std weights prompts by 1/√(p(1−p)), up-weighting very easy and very hard prompts (Dr. GRPO critique, arXiv 2503.20783); ÷mean gives ∇log p (MaxRL, arXiv 2602.02710).
- **Tests** [S]:
  - A tiny GPT-2 (n_embd 8, 1 layer) with `SGD(lr=1e-3)` inside the train-step tests.
  - Snapshots of loss and all parameters at rtol 1e-4, atol 1e-6.
  - Group normalization on `[1,0,0,1]` with G = 2.
  - Off-policy cliprange 0.1; constant normalizer 32 or 42.

### A5: 2025 version (tag `spring2025`), for the SFT / expert-iteration / GRPO history

- **Setup** [S]:
  - **Qwen2.5-Math-1.5B** on **MATH**. MATH is not publicly redistributable, and GSM8K is suggested off-cluster.
  - **SFT** on R1 reasoning traces (`sft.jsonl`): 2 H100-h, ≥ 15 % val.
  - **Expert iteration:** n_ei_steps 5, |D_b| ∈ {512, 1024, 2048}, vary G and SFT epochs, min_tokens 4; 6 H100-h, ≥ 15 %.
- **GRPO defaults** [S]: n_grpo_steps 200, lr 1e-5, advantage_eps 1e-6, rollout 256, **group_size 8**, temperature 1.0, max_tokens 1024, epochs_per_rollout_batch 1, train_batch 256, grad-accum 128 (microbatch 2 on H100), gpu_memory_utilization 0.85, `loss_type ∈ {no_baseline, reinforce_with_baseline (default), grpo_clip}`, `use_std_normalization=True`, AdamW(β = 0.9/0.95, wd 0), clip 1.0.
- **Ablations** [S]: LR sweep (6 H100-h, ≥ 25 % on MATH); baseline vs none; `masked_mean` vs `masked_normalize` (length normalization, with Lambert's 4-token vs 7-token example); std on or off; off-policy epochs and train-batch sweep (12 H100-h); GRPO-Clip vs No-Clip; prompt ablation; leaderboard of 4 h on 2 H100s.
- **KL** [S]: "omitting the KL term had no impact on performance while saving GPU memory" (their experiments plus Dr. GRPO).
- **2025 adapters** [S]:
  - `run_compute_naive_policy_gradient_loss`, `run_compute_grpo_clip_loss`, `run_compute_policy_gradient_loss`
  - `run_masked_mean`, `run_masked_normalize`
  - `run_sft_microbatch_train_step`, `run_grpo_microbatch_train_step`, `run_compute_entropy`
  - plus the 2026 ones.

### A5 supplement (safety / RLHF, optional)

- **Model and evaluation** [S]: **Llama 3.1 8B** base. Zero-shot MMLU, GSM8K, AlpacaEval (annotator Llama-3.3-70B-Instruct vs GPT-4-Turbo reference) and SimpleSafetyTests.
- **SFT** [S]: Alpaca template, packed 512-token sequences, 1 epoch, batch 32, lr 2e-5 cosine, warmup 3 %, wd 0.1, clip 1.0; 3 B200-h. Followed by red-teaming.
- **DPO** [S]: per-instance loss on Anthropic HH (4 subsets, single-turn only), β 0.1, lr 1e-6, **RMSprop**, effective batch 64, policy and reference on separate GPUs, 200-example validation measuring implicit-reward accuracy; 1 B200-h.
  - Test: tiny GPT-2 policy and reference, β = 0.5, expected loss 0.9104 (atol 1e-4).

---

## 4. Executable lectures: how they work, and what to copy

### Mechanism, 2026 (edtrace) [S]

- **Authoring.** A lecture is a normal Python module whose `main()` calls section functions in order, so the call list doubles as the table of contents.
  - Prose: `text("markdown…")`.
  - Media and links: `image(url|path, width=)`, `video()`, `link(title=, url=)` or `link(arxiv_url)` (auto-metadata from `references.py`), `link(SomeClass)` (link into the code).
  - Charts: `plot(vega_lite_spec)`, `graph(make_graph(nodes, edges))` (Cytoscape), `note()`, `system_text(["nvidia-smi"])`.
- **Line directives in comments:**
  - `# @inspect x y` shows variable values next to the line.
  - `# @stepover` keeps a helper's internals out of the trace.
  - `# @hide` hides a line.
  - `# @clear x` removes a shown value.
  - `@animate` reveals renderings one at a time.
- **Claims are plain `assert`s.** For example, `assert x.element_size() == 2` or `assert get_memory_usage(torch.empty(12288*4, 12288)) == 2304*1024*1024`. Lecture 2 has 21 asserts and 74 `@inspect`; lecture 10 has 13 and 22.
- **Tracing.** `python -m edtrace.execute -m lecture_02` runs the module under `sys.settrace`.
  - For each executed line it records the call stack, the inspected locals (tensors as lists, dataclasses as dicts, sympy numbers), captured stdout and the renderings emitted by that line.
  - It writes `var/traces/lecture_02.json`.
  - If any assert fails, trace generation fails. Publishing a lecture therefore means all its claims held on the recording machine.
- **Viewing.** The `edtrace/frontend` is React 19 + Vite + marked + highlight.js + vega-lite + cytoscape, with MathJax added in `index.html`.
  - It steps forward and back through the trace with the code beside the rendered output, and links into the code and arXiv.
  - It is hosted statically on GitHub Pages at `cs336.stanford.edu/lectures/?trace=lecture_XX`.
- **GPU lectures.** `modal_execute.py` builds an image from `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04`, installs the edtrace backend, and runs `python -m edtrace.execute -m <lecture>` on `gpu="B200:4"`. It then copies back `var/traces/<module>.json`, `var/*-ptx.txt` and `var/profiles.txt`.
  - Hardware constants are single-sourced in `facts.py`, e.g. `h100_flop_per_sec = 1979e12 / 2`.
- **Spring 2025** used the same idea with an in-repo `execute.py` / `execute_util.py` (`text`, `image`, `link`, `system_text`) and `trace-viewer/`, and a Slurm remote-execute script instead of Modal.

### Insights for our tutorial

1. **Claim = code + assert + recorded value.** Every numeric claim in our prose should come from a check that is either run live in the browser or replayed from a recorded trace. A failing check should block the build, as in edtrace.
2. **Two tiers of verification.**
   - Tier A runs live in the browser: tokenizer toys, softmax stability, RoPE relative property, ln V initial loss, ln 2 DPO at initialization, LR schedules, IsoFLOPs fitting on `isoflops_curves.json`.
   - Tier B is a recorded trace from `uv run` locally or on Modal, shown with provenance: GPU type, date, commit, seed, wall time.
3. **Ship golden values the way CS336 ships snapshots.** Store small `.npz`/JSON fixtures with an explicit per-claim tolerance; CS336 uses atol 1e-5 for exact ops and 1e-4 for whole models. Show the reader the tolerance and the measured error next to each other.
4. **`@inspect`-style margins.** Show tensor shapes and values beside code lines instead of in prose, which cuts the text needed. `@stepover` keeps the narrative focused.
5. **`main()` as the table of contents** maps neatly onto Saito's chapter/section rhythm. Each section function becomes one page anchor.
6. **Modal recipe for GPU traces.** The image, GPU spec, timeout and files-back pattern can be copied almost verbatim, with `gpu="H100"` or `"B200"` instead of `"B200:4"` for single-GPU DLFS6 runs.
7. **Weakness to avoid: replays are static.** Readers cannot change inputs. Add small live sliders for parameters (temperature, θ, group size) on top of the recorded traces.

---

## 5. File-by-file map: DLFS6 → CS336

DLFS6 structure (per the publisher's table of contents) [S]:
- Part I, CodeBot: ch1 tokenizer basics, ch2 model basics, ch3 training basics (pretraining, SFT, GRPO).
- Part II, StoryBot: ch4 fast BPE, ch5 RoPE/SwiGLU/RMSNorm/KV cache, ch6 AdamW, LR schedule, mixed precision, evaluation, DPO.
- Part III, WebBot: ch7 tokenizer at scale, ch8 FlashAttention/GQA/MoE (**no code in the repo**), ch9 gradient accumulation, DDP, ZeRO.
- Appendices A–E.

| DLFS6 file | CS336 counterpart | Delta (what differs / what to teach) |
|---|---|---|
| ch01/01_char_tokenizer.py | L1 `CharacterTokenizer`; A1 §2.1 `unicode1` | Same idea. CS336 adds `chr(0)`/repr questions. |
| ch01/02_byte_tokenizer.py | L1 `ByteTokenizer`; A1 §2.2 `unicode2` (why UTF-8; the bytewise-decode bug; invalid 2-byte sequences) | DLFS6 `decode` is strict; CS336 requires `errors='replace'`. DLFS6's later tokenizers use `replace`. |
| ch01/03_bpe_train.py | L1 `train_bpe`/`merge`/`count_adjacent_pairs` (near line-for-line); A1 `run_train_bpe` (naive) | Tie-break: `max(counts, key=counts.get)` returns the first maximum in dict order. CS336 A1 requires the lexicographically greatest *bytes* pair. |
| ch01/04_bpe_tokenizer.py | L1 `BPETokenizer` ("very slow: loops over all merges"); A1 `get_tokenizer` | ID-keyed `merge_rules {(id,id): id}`. CS336's interface is bytes-keyed: `vocab: dict[int, bytes]`, `merges: list[tuple[bytes, bytes]]`. DLFS6 therefore can't load GPT-2/tiktoken merges without a converter. |
| ch01/05_special_token.py | A1 "special tokens" (split before pretokenizing, never merged; `test_overlapping_special_tokens`) | DLFS6 supports one `end_token`, appended *last* (id = 256 + #merges). The CS336 reference vocab puts it at 0; the tests compare sets. Overlapping specials are not handled. |
| ch01/06_pretokenize.py | A1 GPT-2 regex `PAT` (the same string) | Uses `re.findall`; CS336 recommends `finditer`, which DLFS6 adopts in ch04/03. |
| ch01/07_tiny_codes.py | A1 `train_bpe_tinystories` (vocab 10k) | CodeBot vocab is 1000. |
| ch01/08_eval.py | A1 `tokenizer_experiments` (a)–(c), compression ratio in bytes per token; L1 `get_compression_ratio` | DLFS6 also compares with tiktoken gpt2/cl100k (commented out). |
| ch01/09_bpe_encode.py | A1 `tokenizer_experiments` (d), "why uint16" | Same. |
| ch02/01_soft_dict.py, 02_attn_math.py | — (CS336 assumes attention) | **DLFS6 is gentler**: attention as a soft dictionary, with a movie-ratings example. |
| ch02/03_attn_scaling.py | A1 §3.4.4 softmax max-subtraction (`run_softmax` +100 test); 1/√d_k | DLFS6 demonstrates the variance argument empirically. Var(q·k) ≈ d, measured over 10 000 samples. |
| ch02/06_attn_mask.py, 07_attn_value.py | A1 `run_scaled_dot_product_attention` (mask True = attend; 3-D and 4-D) | DLFS6: `tril` plus `masked_fill(-inf)`. CS336 forbids `F.softmax` and has students write their own. |
| ch02/08_multi_head.py | A1 `run_multihead_self_attention` (3 projection matmuls; head as a batch dim) | DLFS6 adds attention and output dropout. CS336 has no dropout. |
| ch02/09_norm_gelu.py | A1 RMSNorm/SwiGLU; A1 ablation "SwiGLU vs SiLU (d_ff = 4d)"; L3 history | DLFS6 teaches GPT-2-era LayerNorm and tanh-GELU first. The tanh approximation differs from erf-GELU by ≤ 4.7e-4 [V]. `codebot/model.py` actually uses `nn.GELU()` (erf) and `nn.LayerNorm`. |
| ch02/10_gpt2.py (= `codebot/model.py`) | A1 `run_transformer_lm` (contrast) | GPT-2 flavor: learned absolute positions, **tied** embeddings with an LM-head **bias**, biases in the FFN and LN, dropout 0.1, N(0, 0.02) init. CS336 has none of these (§7). |
| ch03/01_pretrain.py | A1 `training_together`, `data_loading`, `checkpointing`, `experiment_log` | **Gentler**: torch AdamW at constant lr 3e-4, no warmup, no clipping, no validation; a `Dataset` over every window with shuffling. 163.8 M tokens. |
| ch03/02_generate.py | A1 `decoding` (temperature plus **top-p**) | DLFS6 has temperature only (0 means argmax) and no top-p. The staff A2 `generate` has top-k. |
| ch03/03_alpaca.py, 03_sft.py | 2025 A5 SFT (`tokenize_prompt_and_output`, `response_mask`, `masked_normalize`); 2026 A5 `tokenize_prompt_and_output`; supplement SFT (Alpaca template) | Same `### Instruction:/### Response:` template; CS336 adds a preamble sentence. DLFS6 pads to 256 with id 0 and labels −100 and takes a token-mean over the batch. The CS336 supplement *packs* documents with no prompt masking. |
| ch03/04_chat.py | Supplement red-teaming and chat evaluation | — |
| ch03/09_grpo.py | 2026 A5 GRPO family; 2025 A5 GRPO | Toy task: 81 single-digit sums, with the eval set equal to the training set. See §7 row "GRPO" and V4(d). |
| ch04/01_bpe_optimize.py | A1 "pre-tokenize and count" | Same strategy. |
| ch04/02_bpe_cache.py | A1 "optimizing the merging step" (incremental pair-count index); `test_train_bpe_speed` < 1.5 s | **Right at the speed bar**: two recorded sessions under load took 1.27–1.52 s and 0.95–1.11 s on corpus.en (5 runs each; TL;DR 5). The test's docstring quotes 0.38 s for the staff reference on the author's laptop and about 3 s for the toy version [S/V]. Merges differ only through the tie-break [V]. |
| ch04/03_bpe_chunk.py | A1 `pretokenization_example.py::find_chunk_boundaries` | Essentially the same algorithm: 4 096-byte mini-reads, snapping chunk starts to `<\|endoftext\|>`. |
| ch04/04_bpe_parallel.py | A1 "parallelizing pre-tokenization" (< 2 min TinyStories hint) | `multiprocessing.Pool`, 8 processes by default. Only pre-tokenization runs in parallel; as A1 §2.5 puts it, "the merging part of BPE training is not parallelizable in Python". On the full TinyStories train set the book's `04` took 9.4 min wall on this machine under full load (3.2 min pre-tokenization, 6.1 min merging; about 1.5 GiB peak): well inside the 30 min / 30 GB limits, far from the 2-minute hint (tutorial ch04 §4.4). |
| ch04/06_encode_optimize.py | A1 `Tokenizer.encode` | Rank-based: repeatedly merge the lowest-rank pair. This equals "apply merges in creation order", which L1 calls "only loop over merges that matter". |
| ch04/07_encode_parallel.py | A1 `encode_iterable` (streaming; 1 MB memory test, Linux only) | DLFS6 takes a different route to the same goal: multiprocess chunk encoding, then disk cache, then a `np.memmap` of uint16. There is no generator API. |
| ch05/01_rope.py | A1 `run_rope` (`RotaryPositionalEmbedding(theta, d_k, max_seq_len)` plus `token_positions`) | **Numerically identical to CS336's snapshot (3e-7) [V].** Interleaved-pair layout. DLFS6 has no `token_positions` argument; it uses an `offset` in ch05/05. |
| ch05/02_swiglu.py | A1 `run_swiglu` (w1, w2, w3; d_ff ≈ 8/3·d rounded to a multiple of 64) | DLFS6 default `int(8/3·d)` has no rounding, e.g. 512 → 1365. StoryBot passes 1344 explicitly, CS336's value. |
| ch05/03_rmsnorm.py | A1 `run_rmsnorm` (eps 1e-5, **upcast to fp32**) | DLFS6's hand-written version has eps 1e-5 and no upcast. The bots use `nn.RMSNorm()`, where the default eps = finfo(dtype).eps (V2c, V6). |
| ch05/04_improved_gpt.py (= `storybot/model.py`) | A1 `run_transformer_block`/`run_transformer_lm` | **Same function as CS336's reference** with eps = 1e-5: max\|Δlogits\| = 1.05e-5 [V]. The key-rename map is in V6. |
| ch05/05_kvcache.py | L10 inference (prefill vs decode, KV-cache arithmetic, GQA/MQA/MLA, speculative decoding, PagedAttention) | No CS336 assignment (A5 uses vLLM). Beyond `max_context_len`, DLFS6 clamps the RoPE offset in StoryBot and truncates the cache in WebBot, so positions saturate. |
| ch06/02_adamw.py | A1 `get_adamw_cls`, `adamw_accounting` | **Matches torch AdamW to 4.8e-6 [V]**, which passes CS336's test. Decoupled weight decay is computed from the old θ, which is algebraically equal to PyTorch's "decay first". |
| ch06/03_lr_scheduler.py, lr_graph.py | A1 `run_get_lr_cosine_schedule` | DLFS6 uses **linear warmup, then linear decay to 0 ("D2Z")**; Bergsma et al. 2025, arXiv 2502.15938. CS336 uses cosine to min_lr. Values differ, e.g. at iteration 14: 0.5 vs 0.55 [V]. |
| ch06/04_mixed_precision.py | L2 dtypes; A2 `mixed_precision_accumulation`, `benchmarking_mixed_precision` | DLFS6 demonstrates fp16 under/overflow and bf16 range. CS336 adds autocast dtype tracing in a ToyModel plus timing and memory. |
| ch06/05_pretrain.py | A1 `learning_rate`, `training_together` on TinyStories | **Same config and token budget as CS336** [V]. bf16 autocast, clip 1.0, lr 1e-3 with warmup 200 then D2Z, full-validation-set evaluation every 500 iterations, checkpoints at 500 and 5000. Target from CS336: ≤ 1.45 (≤ 2.00 at 40 M tokens). |
| ch06/06_generate.py | A1 `generate` | — |
| ch06/07_llm_judge.py, 09_llm_judge.py | L12 evaluation; supplement AlpacaEval (Llama-3.3-70B judge) | DLFS6 judges with gpt-4o-mini on coherence/grammar, and pairwise with position randomization. CS336 relies on perplexity and benchmarks, plus a local 70B judge. |
| ch06/08_dpo.py | Supplement `dpo_loss` / `dpo_training`; L15 | Same loss. DLFS6: β 0.1, AdamW lr 5e-6, batch 8, 1 000 steps, reference = **pretrained** StoryBot, synthetic "happy ending" pairs. CS336: β 0.1, RMSprop lr 1e-6, effective batch 64, reference = **SFT** model, HH data. Loss = ln 2 at π = π_ref [V]. |
| ch07/01_bpe_train.py, 02_bpe_encode.py | A1 `train_bpe_expts_owt` (vocab **32k**; 12 h and 100 GB RAM warning) | DLFS6 uses vocab **50k** on the same OWT text. Memory-heavy, since all pretoken counts are held in RAM. |
| (ch08, no code) FlashAttention, GQA, MoE | A2 FlashAttention-2 in Triton; L4 (attention alternatives, MoE); L5 (FlashAttention explained); L6 (Triton); L10 (GQA for the KV cache) | DLFS6 *uses* `F.scaled_dot_product_attention(..., enable_gqa=True)` in `webbot/model.py`; CS336 has students write the kernel. |
| ch09/01_pretrain.py (+ `webbot/model.py`) | A2 DDP / sharded optimizer / FSDP; L7/L8 parallelism; A1 OWT main and leaderboard; A4 `train.py` (multi-GPU) | DLFS6 runs torchrun + NCCL DDP with `no_sync()` gradient accumulation (32×4), `torch.compile`, wandb and GQA 12/4. CS336 implements DDP, ZeRO-1 and FSDP by hand and analyzes TP/PP. |
| ch09/02_generate.py | A1 decoding | Temperature 0.5. |
| codebot / storybot / webbot `tokenizer.py`, `utils.py` | A1 tokenizer and decoding | `utils.generate` in StoryBot/WebBot uses the KV cache; CodeBot recomputes the full context. |

---

## 6. Where DLFS6 is gentler, and where CS336 goes deeper

**DLFS6 is gentler on:**
- **Intuition-first attention:** a soft dictionary, then the √d variance demonstration.
- **Incremental tokenizer builds** across 9 small files plus 6 optimization steps, each runnable.
- **Uses PyTorch library pieces freely.** `nn.Linear`, `nn.Embedding`, `F.softmax`, `F.cross_entropy`, torch AdamW and `F.scaled_dot_product_attention` are all allowed. Only selected parts are rebuilt by hand: LayerNorm, GELU, RMSNorm, SwiGLU, RoPE, AdamW. CS336 A1 forbids all of `nn`/`F`/`optim`.
- **Three self-contained bots** small enough for one consumer GPU: CodeBot 11.1 M params, StoryBot 22.7 M (§10).
- **RL on a toy task:** GRPO on sums of two digits with a model trained from scratch, against CS336's OLMo-1B on GSM8K needing 2×B200 plus vLLM.
- **DPO on a 22 M model** with synthetic preferences, against Llama-8B on HH.
- **Evaluation** is by LLM-as-judge through an API, not a benchmark harness.

**CS336 goes deeper on:**
- **Systems.** nsys/NVTX profiling, memory snapshots, activation-checkpointing theory, a Triton FlashAttention-2 forward with optional backward, bf16/fp16 accumulation, `torch.compile`, all-reduce benchmarks, DDP overlap, ZeRO-1, FSDP with compute-dtype casting, and TP/PP/2-D calculations. There is also an 8B-model step-time leaderboard at 32k context.
- **Resource accounting.** FLOP and parameter formulas, AdamW memory, and MFU (A1 problems; L2).
- **Scaling laws.** Chinchilla IsoFLOPs fits plus a budgeted training API. Non-embedding N ≈ 12·L·d².
- **Data.** WARC/WET, HTML extraction, langid, PII masking, NSFW/toxicity, Gopher rules, a quality classifier, exact and MinHash/LSH dedup, and filtering to optimize Paloma C4-100.
- **RL theory and practice.** Policy-gradient derivations, baseline variance, the std-normalization debate, length normalization, RFT vs GRPO, MaxRL, token- vs sequence-level importance sampling, PPO clip, GSPO and CISPO. Every result uses multi-seed (4) curves with confidence intervals.
- **Breadth lectures:** MoE and linear attention, inference (speculative decoding, PagedAttention, quantization), evaluation methodology, multimodality.

---

## 7. Disagreements (design choices and hyperparameters)

| Topic | DLFS6 | CS336 | Status / note |
|---|---|---|---|
| BPE tie-break | lexicographic on **token-ID** tuple (ch04); dict-first (ch01) | lexicographically greater **bytes** pair | [V] diverges at merge #93 (0-based index 92) on `corpus.en`; comparing bytes instead gives an exact match (keep an id → bytes table so the key stays cheap) |
| Tokenizer API | ID-keyed `merge_rules`; special token id last | bytes-keyed vocab and merges; special tokens anywhere; overlapping specials | [S] |
| Normalization | CodeBot: LayerNorm (bias); Story/Web: `nn.RMSNorm()` with eps = finfo eps ≈ 1.19e-7 for fp32 | RMSNorm eps 1e-5 **with fp32 upcast** | [V] eps matters at test tolerance (4.6e-4 vs 1.05e-5 logits error) and for tiny activations. Under bf16 autocast the residual stream stays fp32, so the norms see fp32 inputs (V7b, CPU autocast). |
| Positional encoding | CodeBot learned absolute; Story/Web RoPE θ = 1e4, interleaved pairs | A1 RoPE θ = 1e4, interleaved (snapshot); A3 API θ = 1e6; staff A2 code writes half-split output | [V] all layouts give identical q·k. Weights are not interchangeable across layouts without a permutation. |
| FFN | CodeBot 4·d GELU with bias; Story/Web SwiGLU, default `int(8/3·d)` | SwiGLU, d_ff ≈ 8/3·d rounded to 64 | [S] the configs coincide (1344) |
| Bias terms | CodeBot: FFN, LN and LM-head biases; Story/Web: none | none anywhere | [S] |
| Weight tying | CodeBot ties embedding and LM head (keeping the head bias); Story/Web untied | untied (the staff code has tying commented out); the leaderboard tip says tying needs a smaller init std | [V] CodeBot `embed.weight is unembed.weight` |
| Init | N(0, 0.02) for every Linear and Embedding (no residual-proj scaling) | TruncN(0, 2/(d_in+d_out)) ±3σ for Linear; TruncN(0, 1) ±3 for Embedding | [V] initial loss ≈ ln V + Var(logits)/2: 9.307 (logit std 0.45) vs 9.257 (0.31) |
| Dropout | CodeBot 0.1 (attention weights, attention output, FFN, embeddings); Story/Web 0 | 0 (L3: modern LMs drop dropout) | [V] in GRPO this noise contaminates the importance ratio |
| Optimizer | torch AdamW defaults β = (0.9, 0.999), eps 1e-8, wd 0.01 on all params | A1: student-tuned, notes (0.9, 0.95) as LLM-typical; A3 API (0.9, 0.95), wd 0.01; A4 (0.9, 0.98), eps 1e-9, wd 0.1; A5 (0.9, 0.95), wd 0 | [S] |
| LR schedule | warmup then linear to 0 (D2Z), max lr 1e-3 (Story), 6e-4 (Web) | cosine to min_lr (A1 test min = 0.1·max); A3 warmup 5 % then cosine to 10 % | [V] the curves are shown in V2e |
| Precision | Story/Web bf16 autocast; CodeBot fp32 | A1 fp32 (TF32 allowed on CUDA); A2 studies autocast; A4 trains in bf16 | [S] |
| Data sampling | random windows `randint(len − ctx − 1)` (ch06/ch09) | every start 0..len − ctx − 1 must be reachable | [V] DLFS6 misses the last start (91 vs 92). ch03's `TokenDataset` is correct. |
| Gradient clipping | 1.0 (Story, Web, GRPO); none (CodeBot pretraining and SFT) | 1.0 everywhere | [S] |
| Vocab sizes | 1k / 10k / 50k | 10k TinyStories / 32k OWT / 50 257 GPT-2 (A4) / 32k DCLM (A3) | [S] |
| SFT loss | per-example padding, −100 mask, token-mean over the batch | 2025: `masked_normalize` (sum ÷ constant); supplement: packed, unmasked | [S] |
| **GRPO** | adv = r − mean (no std) → **Dr. GRPO advantage**; loss = Σ_tokens ÷ #sequences → **constant** normalizer; ratio p/(p_old + 1e-8) in prob space; PPO clip 0.2; 2 updates per rollout batch (the 2nd is off-policy); no KL; G = 8; 256 rollouts; lr 7e-6; temp 1.0; `max_new_tokens` 1000 (default); reward = last integer equals the answer; eval on the training prompts | standard: ÷std + eps, sequence-mean normalization, on-policy; variants grid (§3); ratio = exp(logp − logp_old); clip 0.2 (GSPO 3e-4); no KL; G = 8; 256 rollouts; lr 1e-5; max 512 tokens with stop `</answer>`; r1_zero grader; held-out validation, 4 seeds | [V] constant loss scale is ~irrelevant under Adam (1/512 scale changes weights by 1.6e-6); prob-space ε bias reaches ratio 0.5 at p = 1e-8; dropout gives \|r − 1\| ≈ 0.05 even at the first step |
| DPO | β 0.1, AdamW 5e-6, batch 8, 1k steps, reference = pretrained | β 0.1, RMSprop 1e-6, effective batch 64, 1 epoch, reference = SFT, implicit-reward accuracy on validation | [V] ln 2 at initialization in DLFS6 |
| KV cache beyond context | StoryBot clamps the RoPE offset (positions saturate at max − 1); WebBot truncates the cache and resets the offset | not assigned; vLLM PagedAttention in A5 | [S] read from code; affects generations longer than `max_context_len` |

---

## 8. Our verification runs (all CPU, seconds each)

- **Location:** scratch scripts in `C:\Users\chenf\AppData\Local\Temp\claude\C--Users-chenf\2766cf54-7521-48a1-be58-647c0a0601b0\scratchpad\cs336\verify\`. The read-only CS336 clones are in the parent folder.
- **Helper:** `dlfs6_loader.py`. It AST-extracts only the `import`/`def`/`class` nodes from a DLFS6 script, so module-level training loops never run.
- **Command pattern** (Git Bash): `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run --project C:/Users/chenf/Projects/_refs/dlfs6 --no-sync python -I <script>`
  - Environment: DLFS6 venv, torch 2.9.1+cpu, numpy 1.26.4, Python 3.12.9.
  - The scripts read CS336 fixtures and snapshots as data and never import CS336 code.

| ID | Script | Result |
|---|---|---|
| V1 | `v1_bpe_tiebreak.py` | DLFS6 `ch04/02_bpe_cache.train_bpe(corpus.en, 500)`: 243 merges, **170 position-identical, first divergence at merge #93 (0-based index 92)** (DLFS6 `(b' c', b'om')` vs reference `(b't', b'h')`), 1.04 s in this one run. With the bytes-lexicographic key: **243/243 identical**. Later recorded timings: TL;DR 5. A bytes key that rebuilt the byte strings with a function call on every comparison took 1.80–2.48 s and failed the 1.5 s bar; with an id → bytes table it took 0.98–1.11 s, about 10 % more CPU time than the id key (`a1_results.json` `speed_history`; tutorial page cs336, verify X.2-d). |
| V2 | `v2_adamw_and_misc.py` | (a) DLFS6 AdamW vs torch AdamW, CS336 protocol: max\|Δw\| = **4.77e-6** (passes atol 1e-4). (b) Adam with loss vs loss/512: **1.55e-6**. (c) `nn.RMSNorm(8).eps` is None, i.e. 1.19e-7: on 1e-3-scale inputs it differs from eps = 1e-5 by **1.30**; on unit-scale inputs by **1.3e-5**. (d) tanh-GELU vs erf-GELU: **4.74e-4**. (e) Cosine (CS336 test) vs D2Z at iterations 0,4,…,24: `[0, .571, .989, .745, .355, .111, .1]` vs `[0, .571, .929, .643, .357, .071, 0]`. |
| V3 | `v3_rope_layout.py` | DLFS6 RoPE vs the CS336 `test_rope.npz` snapshot: **2.98e-7**. The half-split layout differs elementwise by 5.36, but q·k agrees to **5.7e-6**. score(5,2) = score(40,37) = 0.74422, while score(5,3) = 0.89683. |
| V4 | `v4_models_rl.py` | (a) StoryBot total 22 696 448; embeddings 5.12 M; LM head 5.12 M; blocks 12 455 936; **non-embedding including the LM head = 17 576 448**, CS336's "~17 M". CodeBot 11 121 640 (tied, head bias). Tokens 327 680 000 and 40 960 000 (the latter is 32 × 5 000 × 256, the handout's low-resource reference run; the budget is 40 000 000). (b) Initial loss 9.353 vs ln 10 000 = 9.210. (c) **DPO(π = π_ref) = 0.693147 = ln 2**. (d) GRPO ratio: train-mode policy vs eval-mode old policy gives mean\|r − 1\| = **0.052**, 0.9 % outside [0.8, 1.2]; eval mode gives 1.0e-5. ε bias: p = 1e-6 → 0.990; **p = 1e-8 → 0.500**. |
| V5 | `v5_get_batch.py` | DLFS6 `get_batch` under the CS336 protocol: min start 0, **max start 91**, 92 distinct; CS336 expects max 92 and 93 distinct, so the test would fail. |
| V6 | `v6_storybot_vs_cs336_lm.py` | CS336 fixture weights → DLFS6 StoryBot (key rename only; no missing or unexpected keys). As shipped (`nn.RMSNorm` default eps): max\|Δlogits\| **4.63e-4** (fails). eps = 1e-5: **1.05e-5** full and 9.5e-6 truncated (**passes**). |
| V7 | `v7_init_and_autocast.py` | Initial loss vs ln V + Var/2: DLFS6 init 9.3074 vs 9.3121; CS336 init 9.2567 vs 9.2578. Under CPU bf16 autocast: Linear gives bf16; fp32-residual + bf16 → fp32; RMSNorm on an fp32 residual gives fp32. finfo eps: bf16 7.8e-3, fp32 1.19e-7. |
| V8 | HF API + streamed hash | DLFS6 `storybot/tiny_stories_train.txt` = TinyStoriesV2-GPT4-train.txt (2 227 753 162 B, sha256 6418d412de72888f…); valid = TinyStoriesV2-GPT4-valid (22 502 601 B, 6874bae9a4c1a4e7…). DLFS6 `webbot/owt_valid.txt` = gunzip(CS336 `owt_valid.txt.gz`): 289 998 753 B, sha256 `2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660`. OWT train is the same size class (11.92 GB) but was not hashed. |
| V9 | `python -I -c "import resource"` on Windows | `ModuleNotFoundError`, so CS336 A1 `tests/test_tokenizer.py` cannot even be collected natively on Windows. |

**Tutorial-ready minimal checks** distilled from these:
- ln V initial loss
- ln 2 DPO at initialization
- RoPE relative-position equality
- the √d variance law
- the tie-break flip
- D2Z vs cosine values
- Adam's invariance to loss scale
- RMSNorm eps sensitivity on tiny inputs

All of these fit in a browser (JS or Pyodide), using numbers small enough to show.

---

## 9. Compute and hardware plan implications

### CS336's own price table, single GPU [S]

- **2026, B200, as of 2026-03-28:** Modal $6.25/h (plus $30/month free); Lambda $6.69; RunPod $4.99; Nebius $5.50 ($3.05 preemptible); Together $7.49 (minimum 8 GPUs).
- **2025, H100 80GB, as of 2025-06-06:** RunPod $1.99–2.99; Lambda $2.49–3.29; Paperspace $2.24; Together $2.85 (minimum 8 GPUs).
- **Advice from the course:** debug on CPU, then use GPUs for A1, A4 and A5 training and for A2 benchmarking.

### DLFS6 runs, napkin math [E]

FLOPs ≈ (6·N_nonemb + 6·L·ctx·d)·tokens. Every figure below needs measuring.

| Run | Params (verified) | Tokens | ≈FLOPs | RTX 3060 Ti 8GB (est. ~13 TFLOP/s bf16 effective) | 1×H100 (est. ~400 TFLOP/s effective) |
|---|---|---|---|---|---|
| CodeBot pretraining (fp32) | 11.1 M (tied) | 163.8 M | ~1.2e16 | ~40–60 min (fp32 path) | minutes |
| StoryBot pretraining | 22.7 M (17.6 M non-embedding) | 327.7 M | ~3.6e16, plus ~40 % for full-validation evals every 500 iterations | ~1–1.5 h | ~5–10 min. CS336 quotes 20–30 min on a B200 for its fp32 reference |
| WebBot pretraining | 152.3 M (113.9 M non-embedding) | 13.1 B per GPU at 100 k iterations | ~9.7e18 | ~9 days, not feasible | ~7 h on 1×H100; 8×H100 with max_iters/8 ≈ 1 h |
| GRPO (CodeBot) | 11.1 M | — | — | Generation-bound: 256 sequential, un-batched, no-KV-cache generations per iteration × 500 iterations. Kernel-launch overhead dominates; batching rollouts is the obvious speedup. | — |

Measured since (`web/data/runs/`, see §11): CodeBot pretraining 41.9 min and GRPO 89.7 min on the 3060 Ti; StoryBot pretraining 16.2 min on an H100; WebBot about 10.4 h on one H100, extrapolated from a 200-step probe (8.8 h of steps plus 101 full evaluations of about 56 s each).

**Platform caveats** [S]:
- **Triton (A2):** Linux or WSL2. A community `triton-windows` build exists.
- **vLLM 0.19.1 + flash-attn (A5):** Linux x86_64 with CUDA 12.9 wheels, two GPUs.
- **fasttext/resiliparse (A4):** Linux-first.
- **The user's 3060 Ti (Ampere, sm_86)** supports bf16 but not FP8 or TMA. The A2 leaderboard tips (TMA) need Hopper or Blackwell.

---

## 10. Suggested tutorial hooks (how to weave CS336 in)

- **Per DLFS6 chapter, add a "CS336 对照" box** with three parts: (1) the CS336 lecture or assignment it corresponds to, (2) one *verifiable* difference (from §7, with our check), and (3) an "if you want to go deeper" link (handout section or lecture trace).
- **Use CS336's targets as acceptance criteria** for our end-to-end StoryBot: val ≤ 2.00 on the 40 M-token low-resource budget (the DLFS6 iteration-5000 checkpoint, about 40.97 M tokens, is the closest stand-in) and ≤ 1.45 at the full 327.68 M tokens. Report wall time and GPU, like CS336's experiment log.
  - Done: on a Modal H100 the unchanged `ch06/05_pretrain.py` reached full-validation loss 1.662 at step 5 000 and 1.304 at the end, in 16 minutes (run `storybot-h100`; tutorial page cs336, verify X.1-b).
- **Turn the weight-rename experiment (V6) into a page:** "our hand-built model == Stanford's reference, within 1e-5". This is a strong trust anchor. Present it as architectural equivalence, not as assignment solutions.
- **RL chapter:** present DLFS6's GRPO as the starting point, then show CS336's variant grid as a table of single-switch changes. That covers baseline, normalizer, loss normalization and importance reweighting, plus the dropout and log-space fixes. Each switch gets a tiny in-browser numeric demo (advantages on `[1,0,0,1]`; Lambert's 4-vs-7-token gradient example).
- **Systems sidebar** for ch8/ch9: CS336 A2's FlashAttention-2 tiling and online softmax (Algorithm 1) as an interactive tile stepper, with an optional Modal GPU trace.

---

## 11. Open items (not verified)

- Whether OWT *train* matches byte for byte (4.6 GB download). The validation split matches.
- Real throughput and wall time of DLFS6 runs on 3060 Ti, H100 or B200. The DLFS6 venv has CPU-only torch, so measure on Modal or with a CUDA build.
  - Since measured (`web/data/runs/`): CodeBot pretraining 41.9 min on the RTX 3060 Ti (`codebot-3060ti`); StoryBot pretraining 16.2 min on an H100 (`storybot-h100`); WebBot 0.317 s per step on an H100 over a 200-step probe, extrapolated to about 10.4 h for the full run (`webbot-probe`). B200 was not tried.
- The dropout effect on GRPO ratios was measured with *random* weights. Re-measure with the trained `codebot/model_sft.pt` (downloadable from `koki0702/zero-llm-data`).
  - Since measured with the trained `model_sft.pt`: on the first update, with identical weights, about half of the response tokens fall outside [0.8, 1.2] (53.9 % in `research/repo-ch03-train-align.md`, 56.9 % in tutorial ch03 verify 3.9-g), far more than the 0.9 % seen with random weights.
- CUDA-autocast behaviour of `nn.RMSNorm`. Only CPU autocast was checked; type promotion keeps the residual stream in fp32, so the same result is expected.
- The 2025 YouTube playlist URL was not found on the official site.
