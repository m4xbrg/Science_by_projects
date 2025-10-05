# Lotka–Volterra Studio: Nullclines, Fixed Points, Limit Cycles, and Parameter Continuation

This repository implements a clean, modular workflow for the classical two-species predator–prey (Lotka–Volterra) model. It includes:
- **Model API** (`model.py`): `rhs(t, x, params)`; Jacobian; fixed points; nullclines.
- **Simulation** (`simulate.py`): deterministic ODE integration (SciPy `solve_ivp`) with YAML-configurable parameters and optional parameter continuation (1D sweep).
- **Visualization** (`viz.py`): time series, phase portrait with nullclines/fixed points, and a simple bifurcation-style summary from a parameter sweep.
- **Data I/O**: results saved as Parquet for reproducibility and portability.

See `config.yaml` for parameters, seeds, and sweep configuration.
