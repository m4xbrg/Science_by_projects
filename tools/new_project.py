#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
TEMPLATE = ROOT / "template_project"  # if you have one; else we synthesize

SKELETON = {
    "README.md": "# {title}\n\nSee meta.json for ontology; run simulate/viz via entrypoints.\n",
    "config.yaml": "seed: 0\npaths:\n  results: results.parquet\n",
    "requirements.txt": "numpy\npandas\nmatplotlib\npyyaml\npyarrow\nscipy\n",
    "simulate.py": """# minimal entrypoint; replace with real
def run(config_path: str) -> str:
    from pathlib import Path
    import yaml, pandas as pd
    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    out = (cfg.get("paths") or {}).get("results", "results.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not Path(out).exists():
        pd.DataFrame({"placeholder":[0]}).to_parquet(out)
    return str(out)
""",
    "viz.py": """def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd, matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure(); plt.plot(range(len(df)), df.iloc[:,0]); plt.title("Primary (stub)")
    out = str(Path(outdir)/"primary.png"); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close(); return out
def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd, matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure(); df.iloc[:,0].hist(bins=20); plt.title("Secondary (stub)")
    out = str(Path(outdir)/"secondary.png"); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close(); return out
""",
    "figs/.gitkeep": "",
    "meta.json": json.dumps({
        "title": "{title}",
        "domain": ["{domain}"],
        "math_core": ["{math_core}"],
        "computational_tools": ["NumPy","SciPy","Pandas","Matplotlib","PyYAML","PyArrow"],
        "visualization_types": ["curve_plots"],
        "portfolio_links": [],
        "stage": "scaffolded",
        "created": "{created}"
    }, indent=2) + "\n"
}

def create_from_skeleton(dst: Path, title: str, domain: str, math_core: str):
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    dst.mkdir(parents=True, exist_ok=False)
    for rel, content in SKELETON.items():
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            content.format(title=title, domain=domain, math_core=math_core, created=created),
            encoding="utf-8"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="folder name under projects/, e.g., heat_equation_lab")
    ap.add_argument("--title", required=True)
    ap.add_argument("--domain", default="mathematics")
    ap.add_argument("--math-core", default="ode")
    args = ap.parse_args()

    dst = PROJECTS / args.slug
    if dst.exists():
        print(f"Project already exists: {dst}", file=sys.stderr); sys.exit(1)

    # If you maintain a richer template_project, copy it and patch meta later
    if TEMPLATE.exists():
        shutil.copytree(TEMPLATE, dst)
        # ensure simulate/viz stubs exist
        for req in ["simulate.py","viz.py"]:
            if not (dst/req).exists():
                (dst/req).write_text(SKELETON[req], encoding="utf-8")
        if not (dst/"meta.json").exists():
            create_from_skeleton(dst, args.title, args.domain, args.math_core)
    else:
        create_from_skeleton(dst, args.title, args.domain, args.math_core)

    print(dst)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
