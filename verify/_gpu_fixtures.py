"""Expected outputs of the GPU cost model, generated from gpu/costs.py.

tests/js/fixtures/gpu/cases.json holds them so the widget's JS mirror (web/widgets/gpu-costmodel.js)
is tested against the Python original; verify/test_gpu.py fails when the fixture is stale.

    uv run python verify/_gpu_fixtures.py        # rewrite the fixture after changing the model or data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "js" / "fixtures" / "gpu" / "cases.json"
if str(ROOT / "gpu") not in sys.path:
    sys.path.insert(0, str(ROOT / "gpu"))
import costs  # noqa: E402
import jobs  # noqa: E402

EXTRA_OFFERS = ["modal:h100", "modal:t4", "runpod-community:rtx4090", "lambda:h100x8"]
EVAL_CASES = [(40000, 500, None), (100000, 1000, None), (12500, 1000, None), (100000, 1000, 200),
              (23000, 1000, None), (20000, 0, None), (1000, 1000, None), (1001, 1000, None)]


def build() -> dict:
    gpus, bots = costs.load()
    cases = []
    for s in bots["scenarios"]:
        b = bots["bots"][s["bot"]]
        kw = {"world": s["world"], "max_iters": s.get("max_iters"), "batch": s.get("batch"),
              "accum": s.get("accum"), "steps": None}
        for offer_id in dict.fromkeys([s["offer"], *EXTRA_OFFERS]):
            offer = costs.find(gpus["offers"], offer_id)
            gpu = costs.find(gpus["gpus"], offer["gpu"])
            if costs.peak_tflops(b, gpu) is None:
                continue
            for floor in (0.0, 15.0):
                shape = costs.run_shape(b, **kw)
                est = costs.estimate(b, gpu, costs.price_per_gpu_hour(offer), shape, s["mfu"], floor)
                cases.append({"scenario": s["id"], "bot": s["bot"], "offer": offer_id, "args": kw,
                              "mfu": s["mfu"], "floor_ms": floor, "shape": shape, "estimate": est})
    probe = costs.run_shape(bots["bots"]["webbot"], steps=200)
    return {
        "params": {k: list(costs.param_counts(v)) for k, v in bots["bots"].items()},
        "n_evals": [[it, every, steps, costs.n_evals(it, every, steps)] for it, every, steps in EVAL_CASES],
        "probe_shape": probe,
        "measured": {"args": {"steps": 100000, "sPerIt": 0.3, "evals": 101, "evalS": 90.0, "world": 1,
                              "usdPerGpuHour": 3.9492},
                     "expected": costs.estimate_measured(100000, 0.3, 101, 90.0, 1, 3.9492)},
        # gpu/jobs.py's WebBot settings on n GPUs: the calculator's comparison rows must match them
        "multi_gpu": [{"n": n, "scale": scale, "settings": {**jobs.WEB_DEFAULTS, **jobs.webbot_settings(n, None, scale)}}
                      for n in (2, 4, 8) for scale in ("iters", "batch")],
        "cases": cases,
    }


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
