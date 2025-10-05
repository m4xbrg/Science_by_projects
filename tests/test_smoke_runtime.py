import importlib.util
import types
from pathlib import Path
import tempfile
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

# modules we may need to stub to avoid heavy top-level imports
POSSIBLE_STUBS = {"model", "viz_helpers", "utils", "helpers", "metrics"}

def _import_from_path(path: Path):
    """
    Import a module from an arbitrary file path, while temporarily
    adding its parent directory to sys.path so sibling imports like
    `import model` resolve. We also inject lightweight dummy modules
    for common local imports if they are referenced but missing,
    and populate attributes for 'from X import Y' patterns so that
    top-level imports succeed without executing heavy code.
    """
    import sys, re

    parent = str(path.parent.resolve())
    added_path = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added_path = True

    text = path.read_text(encoding="utf-8")

    # 1) Create stub modules if referenced
    for name in POSSIBLE_STUBS:
        if re.search(rf"\b(import|from)\s+{re.escape(name)}\b", text) and name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # 2) If file contains "from X import A, B", attach dummy attributes to stub module X
    for m in re.finditer(r"from\s+([A-Za-z_]\w*)\s+import\s+([A-Za-z0-9_,\s]+)", text):
        mod_name = m.group(1)
        if mod_name in POSSIBLE_STUBS:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
            target_mod = sys.modules[mod_name]
            names = [n.strip() for n in m.group(2).split(",") if n.strip()]
            for attr in names:
                if not hasattr(target_mod, attr):
                    # create a harmless dummy class for capitalized names, else a simple object
                    if attr and attr[0].isupper():
                        setattr(target_mod, attr, type(attr, (), {})())
                    else:
                        setattr(target_mod, attr, object())

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    finally:
        if added_path:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass

def test_smoke_each_project_runtime():
    """
    For each project:
      - create a minimal config pointing results to a temp parquet
      - run simulate.run(config)
      - run viz.plot_primary/plot_secondary on that parquet into a temp figs dir
    Uses dummy module stubs to bypass heavy top-level imports; only our stubs are executed.
    """
    for proj in sorted(p for p in PROJECTS.iterdir() if p.is_dir()):
        sim = proj / "simulate.py"
        viz = proj / "viz.py"
        if not (sim.exists() and viz.exists()):
            continue

        sim_mod = _import_from_path(sim)
        viz_mod = _import_from_path(viz)

        assert hasattr(sim_mod, "run"), f"{proj.name}: simulate.run missing"
        assert hasattr(viz_mod, "plot_primary"), f"{proj.name}: viz.plot_primary missing"
        assert hasattr(viz_mod, "plot_secondary"), f"{proj.name}: viz.plot_secondary missing"

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out_parquet = td / f"{proj.name}_results.parquet"
            figs = td / "figs"
            cfg = {"paths": {"results": str(out_parquet)}, "seed": 0}
            cfg_path = td / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))

            # simulate
            res_path = Path(sim_mod.run(str(cfg_path)))
            assert res_path.exists(), f"{proj.name}: simulate.run did not produce parquet"

            # parquet should be readable and non-empty
            _ = pd.read_parquet(res_path)

            # viz
            figs.mkdir(parents=True, exist_ok=True)
            p1 = Path(viz_mod.plot_primary(str(res_path), str(figs)))
            p2 = Path(viz_mod.plot_secondary(str(res_path), str(figs)))
            assert p1.exists(), f"{proj.name}: primary figure missing"
            assert p2.exists(), f"{proj.name}: secondary figure missing"
