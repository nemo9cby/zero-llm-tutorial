// Rebuild the WebAssembly SIMD kernels of web/widgets/_gpt.js from their source, web/data/playground/kernels.wat.
//
//   npm install --no-save wabt@1.0.39          (or set WABT=<path to a wabt package folder>)
//   node tests/js/fixtures/playground/compile_kernels.cjs [--write]
//
// Prints the module size and whether it equals the base64 string KERNELS_WASM embedded in _gpt.js;
// --write replaces that string. tests/js/playground-gpt.test.mjs runs the same comparison when wabt is installed.
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../../../..');
const WAT = path.join(ROOT, 'web/data/playground/kernels.wat');
const GPT = path.join(ROOT, 'web/widgets/_gpt.js');

function loadWabt() {
  const where = process.env.WABT || 'wabt';
  try {
    return require(where);
  } catch (err) {
    return null;
  }
}

/** Compile kernels.wat; resolves to a Uint8Array, or null when wabt is not installed. */
async function compileKernels() {
  const init = loadWabt();
  if (!init) return null;
  const wabt = await init();
  const mod = wabt.parseWat('kernels.wat', fs.readFileSync(WAT, 'utf8'), { simd: true });
  mod.validate();
  return new Uint8Array(mod.toBinary({}).buffer);
}

/** The base64 string currently embedded in _gpt.js (the concatenated string literal). */
function embeddedBase64() {
  const src = fs.readFileSync(GPT, 'utf8');
  const m = src.match(/const KERNELS_WASM = ([\s\S]*?);\n/);
  if (!m) throw new Error('KERNELS_WASM not found in _gpt.js');
  return [...m[1].matchAll(/'([^']*)'/g)].map((x) => x[1]).join('');
}

module.exports = { compileKernels, embeddedBase64, WAT, GPT };

if (require.main === module) {
  (async () => {
    const bytes = await compileKernels();
    if (!bytes) {
      console.error('wabt is not installed: npm install --no-save wabt@1.0.39 (or set WABT=<path>)');
      process.exit(2);
    }
    const b64 = Buffer.from(bytes).toString('base64');
    const same = b64 === embeddedBase64();
    console.log(`kernels.wat -> ${bytes.length} bytes; identical to _gpt.js: ${same}`);
    if (!same && process.argv.includes('--write')) {
      const lines = b64.match(/.{1,116}/g).map((s, i) => (i ? `  + '${s}'` : `'${s}'`)).join('\n');
      const src = fs.readFileSync(GPT, 'utf8').replace(/const KERNELS_WASM = [\s\S]*?;\n/, `const KERNELS_WASM = ${lines};\n`);
      fs.writeFileSync(GPT, src);
      console.log('updated KERNELS_WASM in _gpt.js');
    }
  })();
}
