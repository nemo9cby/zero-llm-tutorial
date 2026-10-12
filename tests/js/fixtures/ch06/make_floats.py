"""Generate tests/js/fixtures/ch06/floats.json (run: uv run python tests/js/fixtures/ch06/make_floats.py tests/js/fixtures/ch06/floats.json): expected fp16 / bf16 bits computed with NumPy.

fp16: np.float16(x) for a Python float x (double -> half, round-to-nearest-even).
bf16: fp32 rounding first, then round-to-nearest-even on the fp32 bits (the chapter's to_bf16, which
      verify/test_ch06.py checks bit-for-bit against PyTorch).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])


def to_bf16_bits(x):
    u = int(np.asarray(x, dtype=np.float32).view(np.uint32))
    if (u & 0x7FFFFFFF) > 0x7F800000:
        return ((u >> 16) & 0x8000) | 0x7FC0
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


rng = np.random.default_rng(0)
xs = list(rng.standard_normal(3000) * 10.0 ** rng.uniform(-10, 6, 3000))
xs += list(rng.standard_normal(1000) * 10.0 ** rng.uniform(-45, 38, 1000))
specials = [0.0, -0.0, 1.0, -1.0, 1000.0, 0.01, 1e-8, 3e-8, 2.0 ** -25, 3 * 2.0 ** -25, 2.0 ** -24, 2.0 ** -14,
            6.1e-5, 65504.0, 65519.0, 65520.0, 70000.0, -70000.0, math.pi, 1 / 3, 1e-45, 3.4e38, -3.4e38,
            1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 1 + 2.0 ** -8, 1 + 3 * 2.0 ** -8, 2.0 ** 15 * (2 - 2.0 ** -11),
            float("inf"), float("-inf")]
xs += specials
rows = []
with np.errstate(over="ignore"):
    for x in xs:
        x = float(x)
        h = int(np.float16(x).view(np.uint16))
        b = to_bf16_bits(x) if math.isfinite(x) or math.isinf(x) else None
        rows.append([repr(x).replace("inf", "Infinity"), h, b])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"note": "x as a string (Number(x) in JS), fp16 bits, bf16 bits; generated with NumPy "
                           + np.__version__, "rows": rows}), encoding="utf-8")
print(f"wrote {len(rows)} rows to {out}")
