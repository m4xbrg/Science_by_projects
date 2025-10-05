import argparse, os, pandas as pd, matplotlib.pyplot as plt

def plot_error_rates(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for method in ["Bonferroni", "BH"]:
        sub = df[df["method"] == method]
        plt.plot(sub["alpha"], sub["mean_FDR"], marker="o", label=f"{method} FDR")
        plt.plot(sub["alpha"], sub["FWER"], marker="x", label=f"{method} FWER")
    plt.xlabel("Nominal level (alpha)")
    plt.ylabel("Rate")
    plt.title("Error Rates vs Alpha (Monte Carlo)")
    plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, "error_rates_vs_alpha.png")
    plt.savefig(p, dpi=160); plt.close()
    return p

def plot_power(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for method in ["Bonferroni", "BH"]:
        sub = df[df["method"] == method]
        plt.plot(sub["alpha"], sub["TPR"], marker="o", label=f"{method} TPR")
    plt.xlabel("Nominal level (alpha)")
    plt.ylabel("True Positive Rate (Power)")
    plt.title("Power vs Alpha (Monte Carlo)")
    plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, "power_vs_alpha.png")
    plt.savefig(p, dpi=160); plt.close()
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results.csv")
    ap.add_argument("--outdir", type=str, default="figs")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    print(plot_error_rates(df, args.outdir))
    print(plot_power(df, args.outdir))

if __name__ == "__main__":
    main()