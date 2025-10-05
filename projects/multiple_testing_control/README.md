# Multiple Testing Control: Bonferroni vs. BH-FDR

**Universal structure: Title → Math → Pseudocode → Viz → Reflection → Integration**

## 1. Title & Abstract
We simulate large-scale hypothesis testing under a Gaussian two-groups model to compare Bonferroni (FWER control) to Benjamini–Hochberg (FDR control). Outcomes: empirical FDR, FWER, and power as functions of nominal level.

## 2. Mathematical Formulation
- z-scores: N(0,1) under nulls, N(μ,1) under non-nulls (fraction π1).
- p-values: p_i = 2(1-Φ(|Z_i|)).
- Bonferroni: reject p_i ≤ α/m; BH: largest k with p_(k) ≤ (k/m)α.

## 3. Algorithm (Pseudocode)
Generate truths → z-scores → p-values → apply Bonferroni & BH for a grid of α → aggregate (E[R], E[V], E[S], FDR, FWER, TPR).

## 4. How to Run
```bash
python simulate.py --m 1000 --pi1 0.2 --mu 3.0 --runs 250 --alphas 0.01 0.02 ... 0.2 --seed 42
python viz.py --input results.csv --outdir figs
```

## 5. Visualization
- `figs/error_rates_vs_alpha.png` and `figs/power_vs_alpha.png`

## 6. Reflection
- Independence assumption; BH controls FDR (not FWER), Bonferroni controls FWER (often conservative).

## 7. Integration
- Domain: statistics | Math core: probability | Tools: NumPy/pandas/Matplotlib