# Bayesian Updating with Conjugate Priors: Beta–Binomial and Normal–Normal

This project demonstrates Bayesian sequential updating with two classic conjugate pairs:
1) **Beta–Binomial** for probability of success `p` in Bernoulli/Binomial trials, and
2) **Normal–Normal (known variance)** for an unknown mean `μ` with Gaussian observations of fixed variance.

It includes clean modular code for priors, likelihoods, posteriors, a simple simulation harness that writes results to Parquet/CSV, and visualization utilities for **posterior vs. likelihood** and **evolution of posterior summaries**. Optional animation helpers are included.

## Structure (Universal)
- **Title & Abstract** → README top
- **Mathematical Formulation** → below
- **Pseudocode** → below
- **Python Implementation** → `model.py`, `simulate.py`
- **Visualization** → `viz.py`, `figs/`
- **Reflection** → below
- **Integration/Ontology** → `meta.json`

---

## Mathematical Formulation (Summary)

### Beta–Binomial
- Prior: \(p \sim \mathrm{Beta}(\alpha, \beta)\).
- Likelihood: \(X \mid p \sim \mathrm{Binomial}(n, p)\).
- Posterior: \(p \mid X=k \sim \mathrm{Beta}(\alpha+k, \beta+n-k)\).

### Normal–Normal (known variance)
- Prior: \(\mu \sim \mathcal{N}(\mu_0, \tau_0^2)\).
- Likelihood: \(x_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)\) i.i.d.
- Posterior after \(n\) obs with mean \(\bar{x}\):
  \[
    \mu \mid x_{1:n} \sim \mathcal{N}\left(\mu_n, \tau_n^2\right),\quad
    \tau_n^{-2} = \tau_0^{-2} + n\sigma^{-2},\quad
    \mu_n = \tau_n^2\left(\mu_0\tau_0^{-2} + n\bar{x}\,\sigma^{-2}\right).
  \]

---

## Pseudocode (Summary)

- See `simulate.py` for the end-to-end driver and `model.py` for modular model APIs.

---

## Quickstart

```bash
pip install -r requirements.txt
python simulate.py  # generates results in results/
python viz.py       # saves plots in figs/
```

---

## Reflection (Summary)

- Conjugacy yields closed-form posteriors and constant-time parameter updates.
- Limitations: model misspecification (e.g., unknown variance for Normal–Normal), independence assumptions, and finite-sample sensitivity to priors.
