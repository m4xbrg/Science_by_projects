# Least-Squares Fitter: Residual Diagnostics & Leverage Plot

## Math → Code
- Model: y = Xβ + ε. Solve β via QR/SVD; compute residuals, σ̂², leverage h_ii (hat diag), standardized residuals, Cook’s D.

## Usage
1) Configure `config.yaml` (data source or simulation, solver, ridge λ, plots).
2) `python simulate.py` → writes `results.parquet`, `beta.json`.
3) `python viz.py` → writes `figs/residuals_vs_fitted.png`, `figs/leverage_plot.png`.

## Outputs
- Parquet: per-observation y, y_hat, residual, leverage, std_resid, cooks_d.
- Figures: residuals vs fitted; leverage vs std-resid² with reference lines.

## Notes
- QR preferred; SVD for ill-conditioned X or rank-deficiency.
- Avoid forming full hat matrix; use thin-Q or Cholesky route.
