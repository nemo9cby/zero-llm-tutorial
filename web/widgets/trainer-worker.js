/* Web Worker that trains the tiny GPT of _tinygpt.js, so the page stays responsive while it learns.
   Messages in:  {type:'init', trainURL, valURL, config, seed, settings}  {type:'start'}  {type:'pause'}
                 {type:'reset'}  {type:'set', settings:{lr, maxSteps, mode, temperature, simd}}  {type:'sample'}
   Messages out: {type:'ready', ...facts}  {type:'progress', step, losses, evals, seconds}
                 {type:'sample', step, prompt, text, val, overfit}  {type:'state', running, step, done}
                 {type:'reset', initialLoss, initialVal}  {type:'simd', on}  {type:'error', message} */
import { CharTokenizer, createSession, paramCount, ngramBaselines, simdSupported, setSimd } from './_tinygpt.js';

let S = null;
const post = (msg) => self.postMessage(msg);

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`读取 ${url} 失败（HTTP ${r.status}）`);
  return r.text();
}

function reset() {
  const { trainIds, valIds, cfg, seed, settings } = S;
  S.session = createSession({ trainIds, valIds, config: cfg, seed, batch: settings.batch, lr: settings.lr });
  S.session.mode = settings.mode;
  S.initialLoss = S.session.fixedBatchLoss();
  S.lastVal = S.session.heldOutLoss(); // the whole held-out text, every character predicted once
  S.seconds = 0;
}

async function init({ trainURL, valURL, config, seed = 0, settings }) {
  const [train, val] = await Promise.all([fetchText(trainURL), fetchText(valURL)]);
  const tok = new CharTokenizer(train);
  const trainIds = Uint8Array.from(tok.encode(train)), valIds = Uint8Array.from(tok.encode(val));
  const cfg = { ...config, vocabSize: tok.vocabSize };
  S = { tok, trainIds, valIds, cfg, seed, settings: { ...settings }, running: false };
  reset();
  post({
    type: 'ready', vocabSize: tok.vocabSize, params: paramCount(cfg), lnV: Math.log(tok.vocabSize),
    trainTokens: trainIds.length, valTokens: valIds.length, baselines: ngramBaselines(trainIds, valIds, tok.vocabSize),
    simd: simdSupported, initialLoss: S.initialLoss, initialVal: S.lastVal,
  });
}

const OVERFIT_PROMPT = 8; // in overfit mode, sample from the first 8 characters of the memorized batch

function sample() {
  const { tok, settings, session, cfg } = S;
  const overfit = session.mode === 'overfit';
  const prompt = overfit ? Array.from(session.fixed.x.subarray(0, OVERFIT_PROMPT)) : tok.encode(settings.prompt);
  const ids = session.sample(prompt, {
    maxNewTokens: overfit ? cfg.ctx - OVERFIT_PROMPT : settings.sampleChars, temperature: settings.temperature,
    endId: tok.endId, stopAtEnd: false,
  });
  post({ type: 'sample', step: session.step, prompt: tok.decode(prompt), text: tok.decode(ids), val: S.lastVal, overfit });
}

function loop() {
  if (!S.running) return;
  const { settings, session } = S;
  const t0 = performance.now(), losses = [], evals = [];
  let sampleNow = false;
  // Train for about 100 ms, then report, so that "pause" and the chart stay responsive.
  while (performance.now() - t0 < 100 && session.step < settings.maxSteps) {
    const s = session.step;
    losses.push([s, session.trainStep()]);
    if (session.step % settings.evalEvery === 0) { S.lastVal = session.heldOutLoss(); evals.push([session.step, S.lastVal]); }
    if (session.step % settings.sampleEvery === 0) { sampleNow = true; break; }
  }
  S.seconds += (performance.now() - t0) / 1000;
  post({ type: 'progress', step: session.step, losses, evals, seconds: S.seconds });
  const done = session.step >= settings.maxSteps;
  if (sampleNow || done) sample();
  if (done) {
    S.running = false;
    post({ type: 'state', running: false, step: session.step, done: true });
  } else {
    setTimeout(loop, 0); // yield, so that pause/set messages get through
  }
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') { await init(msg); return; }
    if (!S) throw new Error('训练器还没有初始化');
    if (msg.type === 'start') {
      if (!S.running && S.session.step < S.settings.maxSteps) {
        S.running = true;
        post({ type: 'state', running: true, step: S.session.step });
        loop();
      }
    } else if (msg.type === 'pause') {
      S.running = false;
      post({ type: 'state', running: false, step: S.session.step });
    } else if (msg.type === 'reset') {
      S.running = false;
      reset();
      post({ type: 'reset', initialLoss: S.initialLoss, initialVal: S.lastVal });
    } else if (msg.type === 'set') {
      Object.assign(S.settings, msg.settings);
      if ('lr' in msg.settings) S.session.opt.lr = msg.settings.lr; // takes effect at the next step
      if ('mode' in msg.settings) S.session.mode = msg.settings.mode;
      if ('simd' in msg.settings) post({ type: 'simd', on: setSimd(msg.settings.simd) });
    } else if (msg.type === 'sample') {
      sample();
    }
  } catch (err) {
    if (S) S.running = false;
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
