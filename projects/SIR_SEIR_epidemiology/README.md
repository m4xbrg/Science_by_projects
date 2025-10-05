# SIR/SEIR Epidemiology: Interventions, R₀ Estimation, and Uncertainty Ribbons

This project implements SIR/SEIR compartmental models with time-varying transmission (`beta(t)`) to simulate interventions, estimate basic/effective reproduction numbers (R₀, R_t), and propagate parameter uncertainty via Monte Carlo to produce ribbons.

**Structure:** Title → Math → Pseudocode → Viz → Reflection → Integration (see repo files).

## Quickstart

```bash
pip install -r requirements.txt
python simulate.py
python viz.py
```

Artifacts: `results.parquet`, `results_mc.parquet`, `figs/time_series.png`, `figs/phase_portrait.png`.
