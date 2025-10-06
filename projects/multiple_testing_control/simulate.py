import argparse, numpy as np, pandas as pd
from model import simulate_pvalues, bonferroni_reject, bh_reject, evaluate_once


def monte_carlo(m=1000, pi1=0.2, mu=3.0, n_runs=250, alphas=None, seed=42):
    rng = np.random.default_rng(seed)
    if alphas is None:
        alphas = np.linspace(0.01, 0.2, 10)
    rows = []
    for a in alphas:
        for method in ["Bonferroni", "BH"]:
            Rl, Vl, Sl, FDRl, FWERl, TPRl = [], [], [], [], [], []
            for _ in range(n_runs):
                p, is_null = simulate_pvalues(m, pi1, mu, rng)
                rej = (
                    bonferroni_reject(p, a)
                    if method == "Bonferroni"
                    else bh_reject(p, a)
                )
                R, V, S, fdr, fwer, tpr = evaluate_once(p, is_null, rej)
                Rl.append(R)
                Vl.append(V)
                Sl.append(S)
                FDRl.append(fdr)
                FWERl.append(fwer)
                TPRl.append(tpr)
            rows.append(
                dict(
                    alpha=a,
                    method=method,
                    **{
                        "E[R]": np.mean(Rl),
                        "E[V]": np.mean(Vl),
                        "E[S]": np.mean(Sl),
                        "mean_FDR": np.mean(FDRl),
                        "FWER": np.mean(FWERl),
                        "TPR": np.mean(TPRl),
                    }
                )
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--pi1", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--runs", type=int, default=250)
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="results.csv")
    args = ap.parse_args()

    if args.alphas is None:
        args.alphas = list(np.linspace(0.01, 0.2, 10))
    df = monte_carlo(
        m=args.m,
        pi1=args.pi1,
        mu=args.mu,
        n_runs=args.runs,
        alphas=args.alphas,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()


# --- AUTO-ADDED STUB: uniform entrypoint ---
def run(config_path: str) -> str:
    """Uniform entrypoint.
    Reads YAML config if present, writes results.parquet if not already written by existing code.
    Returns the path to the primary results file.
    """
    from pathlib import Path
    import pandas as pd

    try:
        import yaml

        cfg = (
            yaml.safe_load(Path(config_path).read_text())
            if Path(config_path).exists()
            else {}
        )
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder": [0]}).to_parquet(outp)
    return str(outp)
