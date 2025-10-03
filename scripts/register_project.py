"""
Register a project meta.json into ontology/master_index.csv
Usage:
  python scripts/register_project.py --meta projects/001_lotka_volterra/meta.json
"""
import argparse, json, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True)
    ap.add_argument("--index", type=str, default="ontology/master_index.csv")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    idx_path = Path(args.index)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    row = {
        "id": meta.get("id", ""),
        "title": meta.get("title", ""),
        "domain": meta.get("domain", ""),
        "math_core": meta.get("math_core", ""),
        "tools": ";".join(meta.get("tools", [])),
        "viz_types": ";".join(meta.get("viz_types", [])),
        "related_projects": ";".join(meta.get("related_projects", [])),
    }

    df = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame(columns=row.keys())
    # replace existing row with same id
    df = df[df["id"] != row["id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(idx_path, index=False)
    print(f"Registered project {row['id']} into {idx_path}")

if __name__ == "__main__":
    main()