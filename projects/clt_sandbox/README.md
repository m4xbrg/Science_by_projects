# Central Limit Sandbox: Visualizing Convergence of Standardized Sums

This project explores the Central Limit Theorem (CLT) by simulating standardized sums of i.i.d. random variables from multiple distributions and quantifying convergence to the standard normal.

## 1. Title & Abstract
**Title:** Central Limit Sandbox: sum various i.i.d. variables; visualize convergence to Normal.  
**Abstract:** We study the distribution of standardized partial sums \(Z_n = (\sum_{i=1}^n X_i - n\mu)/(\sigma\sqrt{n})\) for i.i.d. \(X_i\) drawn from configurable distributions. Using Monte Carlo, we measure distances to \(\mathcal N(0,1)\) (KS, AD) and visualize convergence via histograms, Q–Q plots, and KS curves. Expected outcome: a modular sandbox to test CLT behavior, including edge cases (heavy tails) with clear, reproducible metrics.

## 2. Mathematical Formulation
- **Core equation:** \( Z_n = (S_n - n\mu)/(\sigma\sqrt{n}),\quad S_n = \sum_{i=1}^n X_i \).
- **Assumptions:** \(\{X_i\}\) are i.i.d., with finite mean \(\mu\) and variance \(\sigma^2\). For heavy-tailed Pareto, require \(\alpha>2\).
- **Method:** Probability (limit theorem) studied via Monte Carlo sampling.
- **Variants:** Use population moments (classical CLT) or per-trial studentization (approximate Studentized CLT).

## 3. Algorithm Design (Pseudocode)
Inputs: distribution name & parameters; list of n; trials per n; random seed; use_population_moments flag.  
Process: for each n, draw trials×n matrix of i.i.d. values; compute standardized sums; record KS/AD statistics and summary quantiles; subsample Z for visualization.  
Outputs: `results.parquet` table and `samples.json` subset for plots.

## 4. Python Implementation
- `model.py`: distribution registry, population moments, and `standardized_sum_samples`.
- `simulate.py`: runs simulation across n, writes `results.parquet` and `samples.json`.
- `viz.py`: renders histogram with normal PDF, Q–Q plot, and KS convergence curve.

## 5. Visualization
- Histogram with standard normal overlay (`figs/histogram.png`)
- Q–Q plot (`figs/qqplot.png`)
- KS convergence curve (`figs/ks_convergence.png`)

## 6. Reflection
- **Limitations:** CLT rate depends on distribution (Berry–Esseen); extremely heavy tails (infinite variance) violate assumptions; finite trials introduce Monte Carlo error.
- **Extensions:** Lindeberg/Feller conditions; triangular arrays; self-normalized sums; bootstrap CIs for KS; Edgeworth corrections; stable-law limits for infinite variance.
- **Real-world:** Aggregation of independent noise sources; error of sample means; binomial/Poisson approximations via normal.

## 7. Integration
See `meta.json` for ontology tags; `config.yaml` for parameters; `requirements.txt` for environment.

---

## Usage

```bash
python simulate.py --config config.yaml
python viz.py
```
Artifacts land in `results.parquet`, `samples.json`, and `figs/`.