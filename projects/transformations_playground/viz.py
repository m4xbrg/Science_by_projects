"""
viz.py — Two complementary visualizations:
1) Overlay original vs. transformed polygon (last step).
2) Deformation grid (unit lattice) under the composed transform.

Usage:
    python viz.py --cfg config.yaml

Notes:
- We close polygons visually for plotting (repeat first vertex).
- Axes equal and labeled; legend shown.
"""
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import compose, apply

def _close(V: np.ndarray) -> np.ndarray:
    """Return a closed polygon for plotting (append first vertex if needed)."""
    if not np.allclose(V[0], V[-1]):
        return np.vstack([V, V[0]])
    return V

def _load_vertices(cfg):
    return np.array(cfg["polygon"]["vertices"], dtype=float)

def _load_ops(cfg):
    return cfg.get("ops", [])

def plot_overlay(original: np.ndarray, transformed: np.ndarray, out_path: str):
    orig_c = _close(original)
    trans_c = _close(transformed)

    plt.figure()
    plt.plot(orig_c[:,0], orig_c[:,1], marker='o', label='Original')
    plt.plot(trans_c[:,0], trans_c[:,1], marker='o', linestyle='--', label='Transformed')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polygon: Original vs. Transformed")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_deformation_grid(A: np.ndarray, out_path: str, n: int = 10, span: float = 2.0):
    """
    Plot an n×n lattice before/after transform to visualize deformation.
    """
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)

    # Horizontal lines
    plt.figure()
    for y in ys:
        line = np.stack([xs, np.full_like(xs, y)], axis=1)
        line_t = apply(A, line)
        plt.plot(line[:,0], line[:,1], linewidth=0.5)
        plt.plot(line_t[:,0], line_t[:,1], linestyle='--', linewidth=0.8)

    # Vertical lines
    for x in xs:
        line = np.stack([np.full_like(ys, x), ys], axis=1)
        line_t = apply(A, line)
        plt.plot(line[:,0], line[:,1], linewidth=0.5)
        plt.plot(line_t[:,0], line_t[:,1], linestyle='--', linewidth=0.8)

    # Axes & aesthetics
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Deformation Grid: Before (solid) vs. After (dashed)")
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.outdir or cfg.get("output_dir", "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    V0 = _load_vertices(cfg)
    ops = _load_ops(cfg)
    A = compose(ops)
    V1 = apply(A, V0)

    plot_overlay(V0, V1, os.path.join(out_dir, "overlay.png"))
    plot_deformation_grid(A, os.path.join(out_dir, "deformation_grid.png"))

    print("Saved overlay.png and deformation_grid.png to", out_dir)

if __name__ == "__main__":
    main()
