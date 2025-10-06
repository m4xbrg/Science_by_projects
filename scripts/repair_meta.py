#!/usr/bin/env python3
"""
Repairs meta.json files under projects/ to satisfy meta.schema.json.

Fixes:
- Ensure required keys exist: title, domain[], math_core[], computational_tools[], visualization_types[]
- Convert scalar strings to arrays
- Normalize domain strings like "math/statistics" or "biology (epidemiology)" -> ["math","statistics"] / ["biology","epidemiology"]
- If portfolio_links is an object, flatten to array of strings
- Add stage (default: "scaffolded") and created (ISO timestamp) if missing
- Title fallback from directory name ("snake_case" -> "Snake Case")
- Minimal default values:
    computational_tools: ["NumPy","SciPy","Pandas","Matplotlib","PyYAML","PyArrow"]
    visualization_types: ["curve_plots"] (if unknown)
"""

from __future__ import annotations
import json, re, sys, datetime, pathlib
from typing import Any, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

REQ_ARRAY_FIELDS = ["domain", "math_core", "computational_tools", "visualization_types"]
DEFAULTS = {
    "computational_tools": [
        "NumPy",
        "SciPy",
        "Pandas",
        "Matplotlib",
        "PyYAML",
        "PyArrow",
    ],
    "visualization_types": ["curve_plots"],
    "stage": "scaffolded",
}

SEP_PATTERN = re.compile(r"[\/,&]| and |;")  # split common delimiters
PARENS = re.compile(r"\(([^)]+)\)")


def snake_to_title(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in s.split())


def norm_domain_value(x: Any) -> List[str]:
    """
    Normalize domain-like strings:
      "mathematics/statistics" -> ["mathematics","statistics"]
      "biology (epidemiology)" -> ["biology","epidemiology"]
    Preserve existing arrays.
    """
    if isinstance(x, list):
        # also normalize each element
        out: List[str] = []
        for item in x:
            out.extend(norm_domain_value(item))
        # deduplicate while preserving order
        seen = set()
        res = []
        for t in out:
            if t not in seen:
                seen.add(t)
                res.append(t)
        return res
    if isinstance(x, str):
        s = x.strip()
        # pull out parenthetical tokens
        par_tokens = PARENS.findall(s)
        s = PARENS.sub("", s).strip()
        parts = [p.strip() for p in SEP_PATTERN.split(s) if p.strip()]
        for pt in par_tokens:
            parts.extend([p.strip() for p in SEP_PATTERN.split(pt) if p.strip()])
        # also split on " - " and long dashes in compound words (e.g., dynamical-systems)
        more = []
        for p in parts:
            more.extend([q.strip() for q in re.split(r"[-–—]", p) if q.strip()])
        parts = more or parts
        # lowercase normalized tokens
        return [p.lower() for p in parts] or [s.lower()] if s else []
    # unknown type -> ignore
    return []


def to_array(
    value: Any, normalizer=None, fallback: List[str] | None = None
) -> List[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, list):
        return value if value else list(fallback or [])
    if isinstance(value, str):
        return normalizer(value) if normalizer else [value]
    # coercion for odd types
    return list(fallback or [])


def flatten_portfolio_links(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        # already an array of strings? accept; if it's objects, stringify
        out = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for k, vals in item.items():
                    if isinstance(vals, list) and vals:
                        out.extend([f"{k}: {x}" for x in vals])
                    elif isinstance(vals, str):
                        out.append(f"{k}: {vals}")
            else:
                out.append(str(item))
        return out
    if isinstance(v, dict):
        out = []
        for k, vals in v.items():
            if isinstance(vals, list):
                out.extend([f"{k}: {x}" for x in vals])
            elif isinstance(vals, str):
                out.append(f"{k}: {vals}")
            elif vals is None:
                continue
            else:
                out.append(f"{k}: {vals}")
        return out
    # other scalars
    return [str(v)]


def repair_one(meta_path: pathlib.Path, apply: bool, verbose: bool) -> bool:
    data = json.loads(meta_path.read_text())
    before = json.dumps(data, sort_keys=True)

    # 1) title
    if (
        "title" not in data
        or not isinstance(data.get("title"), str)
        or not data["title"].strip()
    ):
        data["title"] = snake_to_title(meta_path.parent.name)

    # 2) arrays
    # domain with special normalization
    data["domain"] = to_array(
        data.get("domain"), normalizer=norm_domain_value, fallback=[]
    )

    # math_core: allow string or list; split on delimiters; lowercase tokens
    def norm_mc(x: Any) -> List[str]:
        if isinstance(x, str):
            parts = [p.strip().lower() for p in SEP_PATTERN.split(x) if p.strip()]
            more = []
            for p in parts:
                more.extend(
                    [q.strip().lower() for q in re.split(r"[-–—]", p) if q.strip()]
                )
            return more or parts or [x.lower()]
        if isinstance(x, list):
            out = []
            for item in x:
                out.extend(norm_mc(item))
            # dedupe
            seen = set()
            res = []
            for t in out:
                if t not in seen:
                    seen.add(t)
                    res.append(t)
            return res
        return []

    data["math_core"] = to_array(data.get("math_core"), normalizer=norm_mc, fallback=[])

    # computational_tools (default if missing)
    data["computational_tools"] = to_array(
        data.get("computational_tools"),
        normalizer=lambda s: [s],
        fallback=DEFAULTS["computational_tools"],
    )

    # visualization_types (default if missing)
    data["visualization_types"] = to_array(
        data.get("visualization_types"),
        normalizer=lambda s: [s],
        fallback=DEFAULTS["visualization_types"],
    )

    # 3) portfolio_links flatten
    if "portfolio_links" in data:
        data["portfolio_links"] = flatten_portfolio_links(data.get("portfolio_links"))

    # 4) stage + created
    data.setdefault("stage", DEFAULTS["stage"])
    if "created" not in data:
        data["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

    after = json.dumps(data, sort_keys=True)
    changed = before != after
    if changed and verbose:
        print(f"[FIX] {meta_path}")
    if changed and apply:
        meta_path.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes to disk")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    metas = sorted(PROJECTS.glob("*/meta.json"))
    if not metas:
        print("No meta.json files under projects/")
        sys.exit(1)

    changes = 0
    for mp in metas:
        try:
            if repair_one(mp, apply=args.apply, verbose=args.verbose):
                changes += 1
        except Exception as e:
            print(f"[ERROR] {mp}: {e}", file=sys.stderr)
    print(f"Changed {changes} file(s).")
    sys.exit(0)


if __name__ == "__main__":
    main()
