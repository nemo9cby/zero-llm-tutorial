/* The GPU handbook's cost model in plain JS: a line-by-line mirror of gpu/costs.py, plus the rows of the
   calculator's multi-GPU comparison (multiGpuVariants, which mirrors gpu/jobs.py's webbot_settings).
   tests/js/gpu-costmodel.test.mjs runs it on tests/js/fixtures/gpu/cases.json, which
   verify/test_gpu.py regenerates from the Python version, so the widget, the page's verify cells
   and the Modal launcher all compute the same numbers. No DOM here. */

export function pricePerGpuHour(offer) {
  return 'usd_per_gpu_second' in offer ? offer.usd_per_gpu_second * 3600 : offer.usd_per_gpu_hour;
}

/** [all parameters, matmul weights n_mm] from the architecture fields of bots.json. */
export function paramCounts(b) {
  const E = b.d, L = b.layers, ff = b.ff, V = b.vocab, T = b.ctx;
  const kv = b.kv_heads * Math.floor(E / b.heads);
  const attn = 2 * E * E + 2 * E * kv;
  const gpt2 = b.ffn === 'gelu';
  const ffn = (gpt2 ? 2 : 3) * E * ff;
  const block = attn + ffn + (gpt2 ? ff + E + 4 * E : 2 * E);
  let params = V * E + (b.learned_pos ? T * E : 0) + L * block + (gpt2 ? 2 * E : E);
  params += b.tied ? 0 : V * E;
  params += gpt2 ? V : 0;
  return [params, L * (attn + ffn) + V * E];
}

/** Evaluations in `for i in range(steps): if i % every == 0 or i == maxIters - 1`. */
export function nEvals(maxIters, every, steps) {
  if (!every) return 0;
  const run = steps || maxIters;
  const last = maxIters - 1 < run && (maxIters - 1) % every ? 1 : 0;
  return Math.ceil(run / every) + last;
}

export function evalWindows(validTokens, ctx) {
  return Math.max(0, Math.ceil((validTokens - ctx - 1) / ctx));
}

export function runShape(b, { world = 1, maxIters = null, batch = null, accum = null, steps = null } = {}) {
  const it = maxIters || b.max_iters;
  const run = steps ? Math.min(steps, it) : it;
  const mb = batch || b.batch;
  const acc = accum || b.accum;
  const perRank = mb * acc * b.ctx;
  const evals = nEvals(it, b.eval_every, run);
  return {
    world, max_iters: it, steps: run, batch: mb, accum: acc,
    tokens_per_step_per_rank: perRank,
    tokens_per_step: perRank * world,
    train_tokens: perRank * world * run,
    evals,
    eval_tokens: evalWindows(b.valid_tokens, b.ctx) * b.ctx * evals,
  };
}

export function flops(b, shape) {
  const attn = b.layers * b.ctx * b.d;
  const perTokTrain = 6 * b.n_mm + 6 * attn;
  const perTokEval = 2 * b.n_mm + 2 * attn;
  return [perTokTrain * shape.train_tokens, perTokEval * shape.eval_tokens];
}

export function peakTflops(b, gpu) {
  return b.precision === 'bf16' ? gpu.bf16 : gpu.fp32;
}

/** FLOPs-based plan; evaluation runs on rank 0 while the other ranks wait (and are billed). */
export function estimate(b, gpu, usdPerGpuHour, shape, mfu, floorMs = 0) {
  const peak = peakTflops(b, gpu);
  if (peak == null) throw new Error(`${gpu.name} has no published ${b.precision} peak`);
  const eff = peak * 1e12 * mfu;
  const [trainF, evalF] = flops(b, shape);
  const perTok = 6 * b.n_mm + 6 * b.layers * b.ctx * b.d;
  const stepS = (shape.tokens_per_step_per_rank * perTok) / eff;
  const wallS = shape.steps * Math.max(stepS, floorMs / 1e3) + evalF / eff;
  const gpuHours = (wallS * shape.world) / 3600;
  return {
    train_flops: trainF, eval_flops: evalF, step_ms: stepS * 1e3, floor_bound: stepS * 1e3 < floorMs,
    wall_hours: wallS / 3600, gpu_hours: gpuHours, usd: gpuHours * usdPerGpuHour,
  };
}

/** Probe-based ETA: what tqdm measured beats any MFU guess. */
export function estimateMeasured({ steps, sPerIt, evals, evalS, world, usdPerGpuHour }) {
  const wallS = steps * sPerIt + evals * evalS;
  const gpuHours = (wallS * world) / 3600;
  return { wall_hours: wallS / 3600, gpu_hours: gpuHours, usd: gpuHours * usdPerGpuHour };
}

/** Ways to run a single-GPU job of `baseIters` steps on n GPUs (runShape arguments, WebBot only):
 *  the job itself, the script started unchanged on n GPUs, and gpu/jobs.py's two ways to keep the
 *  single-GPU token budget: scale "iters" (max_iters ÷ n) and scale "batch" (per-GPU batch ÷ n).
 *  Always derived from the single-GPU job, never from a preset that already divided something. */
export function multiGpuVariants(b, n, baseIters = b.max_iters) {
  const rows = [
    { key: 'single', label: '单卡，脚本原样', args: { world: 1, maxIters: baseIters } },
    { key: 'trap', label: `${n} 卡，脚本原样（陷阱）`, args: { world: n, maxIters: baseIters } },
    { key: 'iters', label: `${n} 卡，max_iters ÷ ${n}`, args: { world: n, maxIters: Math.round(baseIters / n) } },
  ];
  const seqs = (b.batch * b.accum) / n;            // sequences per GPU per optimizer step
  if (Number.isInteger(seqs)) {
    const micro = seqs >= b.batch ? b.batch : seqs;
    rows.push({ key: 'batch', label: `${n} 卡，每卡 batch ÷ ${n}`,
      args: { world: n, maxIters: baseIters, batch: micro, accum: seqs / micro } });
  }
  return rows;
}
