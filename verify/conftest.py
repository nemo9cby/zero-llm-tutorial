"""Shared helpers for the local verification suite.

The suite checks the tutorial's claims against the real DLFS6 code in third_party/dlfs6
(pinned git submodule). Run everything with:  uv run pytest
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"

# Importing the book's packages must not write __pycache__ into the submodule.
sys.dont_write_bytecode = True


def run_dlfs6_script(rel: str, timeout: int = 300) -> list[str]:
    """Run a DLFS6 script exactly as a reader would (on CPU); return its stdout lines."""
    path = DLFS6 / rel
    if not path.is_file():
        pytest.skip(f"{rel} missing — did you run `git submodule update --init`?")
    # "-1" rather than "": Windows drops empty environment variables, which would leave the GPU visible.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, str(path)], cwd=DLFS6, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    assert proc.returncode == 0, f"{rel} exited with {proc.returncode}:\n{proc.stderr[-2000:]}"
    return proc.stdout.splitlines()


@pytest.fixture(scope="session")
def dlfs6_path():
    """Make `import codebot`, `import storybot`, ... work like the book's scripts do."""
    if not (DLFS6 / "codebot").is_dir():
        pytest.skip("third_party/dlfs6 is empty — run `git submodule update --init`")
    sys.path.insert(0, str(DLFS6))
    yield DLFS6
    sys.path.remove(str(DLFS6))
