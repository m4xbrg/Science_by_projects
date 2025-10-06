"""
viz.py — Posterior vs Likelihood plots and posterior evolution.
If results are missing, it will generate them using default config.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from model import (
    BetaBinomialParams,
    beta_posterior_params,
    beta_prior_pdf,
    beta_likelihood_curve,
    beta_posterior_pdf,
    grid01,
    NormalNormalParams,
    normal_posterior_params,
    normal_prior_pdf,
    normal_likelihood_curve,
    normal_posterior_pdf,
    gridR,
)


def ensure_results():
    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(base / "config.yaml", "r"))
    results_dir = base / cfg["paths"]["results_dir"]
    if not (results_dir / "beta_binomial.csv").exists():
        import simulate

        simulate.main("config.yaml")
    return cfg


def plot_beta_posterior_vs_likelihood(cfg):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]

    df = pd.read_csv(results_dir / "beta_binomial.csv")
    row = df.iloc[-1]  # final batch
    prior = BetaBinomialParams(**cfg["beta_binomial"]["prior"])
    k = int(row["k_cum"])
    n = int(row["n_cum"])
    posterior = BetaBinomialParams(alpha=row["alpha"], beta=row["beta"])

    p = grid01(1000)
    prior_y = beta_prior_pdf(p, prior)
    like_y = beta_likelihood_curve(p, k, n)
    like_y = like_y / like_y.max() * prior_y.max()  # scale for visual comparability
    post_y = beta_posterior_pdf(p, posterior)

    plt.figure()
    plt.plot(p, prior_y, label="Prior Beta")
    plt.plot(p, like_y, label="Likelihood (scaled)")
    plt.plot(p, post_y, label="Posterior Beta")
    plt.xlabel("p (probability of success)")
    plt.ylabel("density (arbitrary for likelihood)")
    plt.legend()
    plt.title("Beta–Binomial: Prior vs Likelihood vs Posterior")
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / "beta_posterior_vs_likelihood.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


def plot_beta_posterior_evolution(cfg):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]
    df = pd.read_csv(results_dir / "beta_binomial.csv")

    plt.figure()
    plt.plot(df["batch"], df["p_mean_post"], label="Posterior mean")
    plt.fill_between(
        df["batch"],
        df["p_mean_post"] - 2 * np.sqrt(df["p_var_post"]),
        df["p_mean_post"] + 2 * np.sqrt(df["p_var_post"]),
        alpha=0.2,
        label="±2 SD",
    )
    plt.axhline(df["p_true"].iloc[0], linestyle="--", label="True p")
    plt.xlabel("batch")
    plt.ylabel("p")
    plt.legend()
    plt.title("Beta–Binomial: Posterior mean ±2SD over batches")
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / "beta_posterior_evolution.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


def plot_normal_posterior_vs_likelihood(cfg):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]

    df = pd.read_csv(results_dir / "normal_normal.csv")
    row = df.iloc[-1]
    prior = NormalNormalParams(**cfg["normal_normal"]["prior"])
    mu_n = float(row["mu_post"])
    tau_n2 = float(row["tau2_post"])
    xbar = float(row["xbar_cum"])
    n = int(row["n_cum"])
    sigma2 = float(row["sigma2"])

    mu_grid = gridR(mu_n - 4 * np.sqrt(tau_n2), mu_n + 4 * np.sqrt(tau_n2), 800)
    prior_y = normal_prior_pdf(mu_grid, prior)
    like_y = normal_likelihood_curve(mu_grid, xbar, n, sigma2)
    like_y = like_y / like_y.max() * prior_y.max()
    post_y = normal_posterior_pdf(mu_grid, mu_n, tau_n2)

    plt.figure()
    plt.plot(mu_grid, prior_y, label="Prior Normal")
    plt.plot(mu_grid, like_y, label="Likelihood (scaled)")
    plt.plot(mu_grid, post_y, label="Posterior Normal")
    plt.xlabel("μ")
    plt.ylabel("density (arbitrary for likelihood)")
    plt.legend()
    plt.title("Normal–Normal: Prior vs Likelihood vs Posterior")
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / "normal_posterior_vs_likelihood.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


def plot_normal_posterior_evolution(cfg):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]
    df = pd.read_csv(results_dir / "normal_normal.csv")

    plt.figure()
    plt.plot(df["batch"], df["mu_post"], label="Posterior mean μ_n")
    sd = np.sqrt(df["tau2_post"])
    plt.fill_between(
        df["batch"],
        df["mu_post"] - 2 * sd,
        df["mu_post"] + 2 * sd,
        alpha=0.2,
        label="±2 SD",
    )
    plt.axhline(df["mu_true"].iloc[0], linestyle="--", label="True μ")
    plt.xlabel("batch")
    plt.ylabel("μ")
    plt.legend()
    plt.title("Normal–Normal: Posterior mean ±2SD over batches")
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / "normal_posterior_evolution.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


# ----------------- Optional Animations -----------------


def animate_beta_posterior(cfg, out_path="figs/beta_posterior_anim.gif"):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]
    df = pd.read_csv(results_dir / "beta_binomial.csv")
    prior = BetaBinomialParams(**cfg["beta_binomial"]["prior"])
    p = grid01(600)
    fig, ax = plt.subplots()
    (line_prior,) = ax.plot(p, beta_prior_pdf(p, prior), label="Prior")
    (line_post,) = ax.plot([], [], label="Posterior")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_xlabel("p")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Beta–Binomial Posterior Evolution")

    def init():
        line_post.set_data([], [])
        return (line_post,)

    def update(frame):
        row = df.iloc[frame]
        posterior = BetaBinomialParams(alpha=row["alpha"], beta=row["beta"])
        y = beta_posterior_pdf(p, posterior)
        line_post.set_data(p, y)
        ax.set_ylim(0, max(y.max(), line_prior.get_ydata().max()) * 1.1)
        return (line_post,)

    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / Path(out_path).name
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def animate_normal_posterior(cfg, out_path="figs/normal_posterior_anim.gif"):
    base = Path(__file__).resolve().parent
    results_dir = base / cfg["paths"]["results_dir"]
    figs_dir = base / cfg["paths"]["figs_dir"]
    df = pd.read_csv(results_dir / "normal_normal.csv")
    prior = NormalNormalParams(**cfg["normal_normal"]["prior"])

    mu_center = df["mu_post"].iloc[-1]
    sd_final = np.sqrt(df["tau2_post"].iloc[-1])
    mu_grid = gridR(mu_center - 5 * sd_final, mu_center + 5 * sd_final, 600)

    fig, ax = plt.subplots()
    (line_prior,) = ax.plot(mu_grid, normal_prior_pdf(mu_grid, prior), label="Prior")
    (line_post,) = ax.plot([], [], label="Posterior")
    ax.set_xlabel("μ")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Normal–Normal Posterior Evolution")

    def init():
        line_post.set_data([], [])
        return (line_post,)

    def update(frame):
        row = df.iloc[frame]
        y = normal_posterior_pdf(mu_grid, row["mu_post"], row["tau2_post"])
        line_post.set_data(mu_grid, y)
        ax.set_ylim(0, max(y.max(), line_prior.get_ydata().max()) * 1.1)
        return (line_post,)

    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    figs_dir.mkdir(exist_ok=True)
    out = figs_dir / Path(out_path).name
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def main():
    cfg = ensure_results()
    p1 = plot_beta_posterior_vs_likelihood(cfg)
    p2 = plot_beta_posterior_evolution(cfg)
    p3 = plot_normal_posterior_vs_likelihood(cfg)
    p4 = plot_normal_posterior_evolution(cfg)
    print("Saved:", p1, p2, p3, p4)


if __name__ == "__main__":
    main()


# --- AUTO-ADDED STUBS: uniform visualization entrypoints ---
def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # simple line of first numeric column or index
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index")
    plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # histogram on first numeric column
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    try:
        plt.hist(df[col], bins=20)
    except Exception:
        plt.plot(range(len(df[col])), df[col])
    plt.title("Secondary Plot (stub)")
    plt.xlabel(str(col))
    plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
