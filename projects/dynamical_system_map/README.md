# Discrete Linear Dynamical Systems: `x_{k+1} = A x_k`

## Math → Code
- Map: `x_{k+1} = A x_k` implemented in `model.step(k, x, params)`.
- Classification via eigenvalues/eigenvectors in `model.eig_info`.

## Run
```bash
python simulate.py
python viz.py
```

Outputs:
- `results.parquet`, `run_meta.json`
- `figs/time_series.png`, `figs/phase_portrait.png`

## Notes
- 2D config by default; supports nD (time series always; phase portrait shown for n≥2 using first two components).
