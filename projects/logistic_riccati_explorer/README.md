# Logistic/Riccati Explorer: Analytic vs. Numeric Solutions; Parameter Sensitivity

This project compares analytic and numeric solutions of the logistic equation (a special Riccati ODE) and implements a general Riccati RHS. It includes forward sensitivity ODEs for the logistic model and utilities for parameter sweeps.

## Structure
- `meta.json` – ontology tags
- `config.yaml` – parameters and solver settings
- `model.py` – mathematical models (`rhs_logistic`, `rhs_riccati`, analytic solution for logistic, sensitivities)
- `simulate.py` – runs `solve_ivp` w/ optional sensitivity integration; writes `results.parquet`
- `viz.py` – plots time series and a 1D phase portrait proxy (\(x\) vs. \(\dot x\)); saves figures to `figs/`

See the docstrings in each module for details.
