import ast
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = ROOT / "projects"


def test_meta_json_valid():
    """Use repo validator to check all meta.json files."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_meta.py")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"meta validation failed:\n{res.stdout or res.stderr}"


def _has_func(file: Path, name: str, min_args: int = 0) -> bool:
    if not file.exists():
        return False
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            # count total positional-or-keyword args (ignore *args/**kwargs)
            npos = sum(isinstance(a, ast.arg) for a in node.args.args)
            if npos >= min_args:
                return True
    return False


def test_uniform_entrypoints_exist_structurally():
    """Every project has simulate.run(config_path) and viz.plot_{primary,secondary}(...) defined."""
    assert PROJECTS_DIR.exists(), "projects/ directory not found"
    missing = []
    for proj in sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir()):
        sim = proj / "simulate.py"
        viz = proj / "viz.py"
        if not _has_func(sim, "run", min_args=1):
            missing.append(f"{proj.name}: simulate.run(config_path) missing")
        if not _has_func(viz, "plot_primary", min_args=2):
            missing.append(
                f"{proj.name}: viz.plot_primary(results_path, outdir) missing"
            )
        if not _has_func(viz, "plot_secondary", min_args=2):
            missing.append(
                f"{proj.name}: viz.plot_secondary(results_path, outdir) missing"
            )
    assert not missing, "Entry point issues:\n- " + "\n- ".join(missing)
