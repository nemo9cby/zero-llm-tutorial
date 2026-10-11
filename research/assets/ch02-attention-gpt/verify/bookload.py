# Helpers shared by the ch02 verification scripts.
#   load(rel_path): import a book script (e.g. ch02/06_attn_mask.py) as a module without polluting stdout.
#   checkpoint():   path of the trained codebot checkpoint (scratch download first, else the repo's codebot/ copy).
import contextlib, importlib.util, io, os, sys

REPO = r"C:\Users\chenf\Projects\_refs\dlfs6"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(rel_path, name=None):
    path = os.path.join(REPO, rel_path)
    name = name or "book_" + os.path.basename(path).replace(".py", "")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    os.chdir(cwd)  # 09/10 chdir to the repo root on import; undo it
    return mod


def checkpoint():
    candidates = [os.path.join(os.path.dirname(HERE), "dl_codebot", "model_pretrain.pt"),
                  os.path.join(REPO, "codebot", "model_pretrain.pt")]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("model_pretrain.pt not found; run codebot/download.sh (44,516,353 bytes) first. Tried: "
                            + ", ".join(candidates))
