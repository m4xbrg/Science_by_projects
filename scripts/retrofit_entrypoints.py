#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import io, re, sys

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

SIM_STUB = """
# --- AUTO-ADDED STUB: uniform entrypoint ---
def run(config_path: str) -> str:
    \"\"\"Uniform entrypoint.
    Reads YAML config if present, writes results.parquet if not already written by existing code.
    Returns the path to the primary results file.
    \"\"\"
    from pathlib import Path
    import pandas as pd
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder":[0]}).to_parquet(outp)
    return str(outp)
"""

VIZ_STUB = """
# --- AUTO-ADDED STUBS: uniform visualization entrypoints ---
def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # simple line of first numeric column or index
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index"); plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # histogram on first numeric column
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    try:
        plt.hist(df[col], bins=20)
    except Exception:
        plt.plot(range(len(df[col])), df[col])
    plt.title("Secondary Plot (stub)")
    plt.xlabel(str(col)); plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out
"""

def has_func(text: str, name: str) -> bool:
    pattern = rf"^\s*def\s+{name}\s*\("
    return re.search(pattern, text, re.MULTILINE) is not None

def ensure_stub(path: Path, stub: str, func_names):
    if not path.exists():
        # create minimal file with stub
        path.write_text(stub.lstrip() + "\n", encoding="utf-8"); return True
    txt = path.read_text(encoding="utf-8")
    missing = [fn for fn in func_names if not has_func(txt, fn)]
    if not missing:
        return False
    with io.open(path, "a", encoding="utf-8") as f:
        f.write("\n" + stub.lstrip() + "\n")
    return True

def main():
    changed = 0
    for proj in sorted(PROJECTS.iterdir()):
        if not proj.is_dir(): continue
        sim = proj / "simulate.py"
        viz = proj / "viz.py"
        # add run() stub if missing
        if ensure_stub(sim, SIM_STUB, ["run"]):
            print(f"[retrofit] {proj.name}/simulate.py -> added run()")
            changed += 1
        # add viz stubs if missing
        if ensure_stub(viz, VIZ_STUB, ["plot_primary", "plot_secondary"]):
            print(f"[retrofit] {proj.name}/viz.py -> added plot_{{primary,secondary}}()")
            changed += 1
    print(f"Changed {changed} file(s).")

if __name__ == "__main__":
    sys.exit(main())
