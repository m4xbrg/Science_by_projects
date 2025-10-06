#!/usr/bin/env python3
import json, sys, pathlib
from jsonschema import validate, Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "meta.schema.json").read_text())


def main():
    projects = sorted((ROOT / "projects").glob("*/meta.json"))
    if not projects:
        print("No meta.json files found under projects/", file=sys.stderr)
        sys.exit(1)
    errors = 0
    for p in projects:
        data = json.loads(p.read_text())
        v = Draft202012Validator(SCHEMA)
        errs = sorted(v.iter_errors(data), key=lambda e: e.path)
        if errs:
            errors += 1
            print(f"[INVALID] {p}", file=sys.stderr)
            for e in errs:
                loc = "/".join(map(str, e.path))
                print(f"  - {loc or '<root>'}: {e.message}", file=sys.stderr)
        else:
            print(f"[OK] {p}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
