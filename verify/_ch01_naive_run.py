"""Record one full run of the book's own BPE trainer on TinyCodes (ch01/07_tiny_codes.py), for 1.7.3.

    uv run python verify/_ch01_naive_run.py      # about an hour on one CPU core

ch01/07 is executed as the book wrote it, with one line changed: the book pickles the rules to
codebot/merge_rules.pkl inside the pinned submodule, so this copy writes them to
build/ch01-naive-run/merge_rules.pkl instead. The record goes to web/data/ch01/naive_run.json:
commits, the patched line, Python and pickle versions, platform and CPU, wall-clock and CPU time,
tqdm's last status line, and the sha256 of the output next to that of the repository's file.
verify/test_ch01.py::test_naive_run_record checks the record against the repository.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True          # importing codebot.tokenizer must not write __pycache__ into the submodule

ROOT = Path(__file__).resolve().parent.parent
DLFS6 = ROOT / "third_party" / "dlfs6"
SCRIPT = "ch01/07_tiny_codes.py"
OUT = ROOT / "build" / "ch01-naive-run" / "merge_rules.pkl"
RECORD = ROOT / "web" / "data" / "ch01" / "naive_run.json"
ORIGINAL = 'with open("codebot/merge_rules.pkl", "wb") as f:'
PATCHED = 'with open(OUT, "wb") as f:   # OUT = build/ch01-naive-run/merge_rules.pkl'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


class KeepTail:
    """Pass stderr through unchanged and keep its tail, to recover tqdm's last status line ('\\r'-rewritten)."""

    def __init__(self, stream):
        self.stream, self.tail = stream, ""

    def write(self, s):
        self.tail = (self.tail + s)[-4000:]
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def last_line(self) -> str:
        lines = [seg.strip() for seg in re.split(r"[\r\n]", self.tail) if seg.strip()]
        return lines[-1] if lines else ""

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main() -> None:
    script = DLFS6 / SCRIPT
    src = script.read_text(encoding="utf-8")
    if src.count(ORIGINAL) != 1:
        sys.exit(f"{SCRIPT} no longer contains {ORIGINAL!r}: update the patch")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    patched = src.replace(ORIGINAL, PATCHED)
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tee = KeepTail(sys.stderr)
    sys.stderr = tee
    cwd = os.getcwd()
    w0, c0 = time.perf_counter(), time.process_time()
    try:
        # the script chdirs to the DLFS6 root and imports codebot.tokenizer, as when the reader runs it
        exec(compile(patched, str(script), "exec"), {"__name__": "__main__", "__file__": str(script), "OUT": str(OUT)})
    finally:
        sys.stderr = tee.stream
        os.chdir(cwd)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0

    repo_pkl = DLFS6 / "codebot" / "merge_rules.pkl"
    ours, theirs = pickle.loads(OUT.read_bytes()), pickle.loads(repo_pkl.read_bytes())
    record = {
        "run": "ch01-bpe-naive",
        "what": "ch01/07_tiny_codes.py on the whole of codebot/tiny_codes.txt (vocab_size 1000, codebot.tokenizer.train_bpe),"
                " unchanged except the line that writes the pickle",
        "command": "uv run python verify/_ch01_naive_run.py",
        "dlfs6_commit": git_head(DLFS6),
        "script": SCRIPT,
        "script_sha256": sha256(script),
        "patch": {"line": src[:src.index(ORIGINAL)].count("\n") + 1, "from": ORIGINAL, "to": PATCHED},
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu": cpu_name(),
                 "logical_cpus": os.cpu_count()},
        "pickle_default_protocol": pickle.DEFAULT_PROTOCOL,
        "started": started,
        "seconds": round(wall, 1),
        "cpu_seconds": round(cpu, 1),
        "tqdm_last_line": tee.last_line(),
        "merges": len(ours),
        "output_bytes": OUT.stat().st_size,
        "output_sha256": sha256(OUT),
        "repo_sha256": sha256(repo_pkl),
        "same_rules_same_order": list(ours.items()) == list(theirs.items()),
        "byte_identical": OUT.read_bytes() == repo_pkl.read_bytes(),
    }
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
