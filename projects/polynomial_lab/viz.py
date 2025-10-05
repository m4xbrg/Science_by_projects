from __future__ import annotations
import yaml, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from model import PolynomialLab

def plot_real_curve_with_roots(lab: PolynomialLab, x_min=-5, x_max=5, num=1000, root_tol=1e-8, out_path="figs/real_curve.png"):
    xs = np.linspace(x_min, x_max, num)
    ys = np.real(lab.polyval(xs))
    plt.figure()
    plt.plot(xs, ys, label="P(x)")
    plt.axhline(0, linestyle='--', linewidth=1)
    rr = lab.roots_companion().roots
    real_roots = [r.real for r in rr if abs(r.imag) < root_tol]
    if real_roots:
        plt.scatter(real_roots, [0]*len(real_roots), marker='x', label="real roots")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend()
    plt.title("Polynomial Curve with Real Roots")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()

def plot_complex_roots(roots: np.ndarray, out_path="figs/complex_roots.png"):
    plt.figure()
    plt.scatter(np.real(roots), np.imag(roots))
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Complex Root Map")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    coeffs = cfg["polynomial"]["coeffs"]
    io = cfg["io"]
    viz_cfg = cfg["viz"]
    lab = PolynomialLab(coeffs)
    res = lab.roots_companion()

    plot_real_curve_with_roots(
        lab,
        x_min=viz_cfg["real_curve"]["x_min"],
        x_max=viz_cfg["real_curve"]["x_max"],
        num=viz_cfg["real_curve"]["num"],
        root_tol=viz_cfg["real_curve"]["root_tol"],
        out_path=io["real_curve_png"],
    )
    plot_complex_roots(res.roots, out_path=io["complex_map_png"])

    r_div = viz_cfg.get("synthetic_division_r", None)
    if r_div is not None:
        _, _, b = lab.synthetic_division(r_div)
        plt.figure()
        plt.plot(np.arange(len(b)), np.real(b), marker='o')
        plt.xlabel("k (Horner index)")
        plt.ylabel("Re(b_k)")
        plt.title(f"Horner Table (synthetic division) at r={r_div}")
        Path("figs").mkdir(parents=True, exist_ok=True)
        plt.savefig("figs/horner_table.png", bbox_inches="tight", dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
