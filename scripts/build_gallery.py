#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
README = ROOT / "README.md"


def row(slug, meta):
    domain = ", ".join(meta.get("domain", []))
    math_core = ", ".join(meta.get("math_core", []))
    stage = meta.get("stage", "")
    title = meta.get("title", slug.replace("_", " ").title())
    return f"| `{slug}` | {title} | {domain} | {math_core} | {stage} |"


def main():
    metas = []
    for p in sorted(PROJECTS.iterdir()):
        if not p.is_dir():
            continue
        m = p / "meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            metas.append((p.name, meta))
    lines = [
        "# Science by Projects",
        "",
        "Unified, ontology-ready computational science portfolio.",
        "",
        "## Project Gallery",
        "",
        "| Project | Title | Domain | Math Core | Stage |",
        "|---|---|---|---|---|",
    ]
    for slug, meta in metas:
        lines.append(row(slug, meta))
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
