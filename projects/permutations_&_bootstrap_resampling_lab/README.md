# Resampling Lab: Bootstrap & Permutation — Confidence Intervals and Coverage

## 1. Title & Abstract
**Title.** Resampling Confidence Intervals for Mean/Median with Coverage Experiments  
**Abstract.** We study nonparametric resampling methods—bootstrap and permutation—to construct confidence intervals (CIs) for the mean and median and to estimate empirical coverage under diverse data-generating processes. We implement a modular simulation harness that varies sample size, statistic, distribution family, and CI method, producing reproducible artifacts (Parquet results + Matplotlib figures) suitable for ontology-ready portfolios.

## 2. Mathematical Formulation
- Let i.i.d. data \(X_1,\dots,X_n\sim F\). Target parameter \(	heta=	heta(F)\) is the mean \(\mu\) or median \(m\).
- Statistic: \(T_n = t(X_{1:n})\) (sample mean or sample median).
- **Bootstrap percentile CI** (one-sample): draw \(B\) bootstrap resamples with replacement, compute \(T_n^{*(b)}\); let \(q_lpha, q_{1-lpha}\) be empirical quantiles; CI: \([q_lpha, q_{1-lpha}]\).
- **Bootstrap basic CI**: \([2\,T_n - q_{1-lpha},\ 2\,T_n - q_{lpha}]\).
- **Two-sample difference** (optional): for samples A,B of sizes \(n_a,n_b\), estimate \(	heta=	heta(F_A)-	heta(F_B)\) using paired bootstrap over groups.
- **Coverage**: repeat \(M\) experiments; generate data from \(F\) with true \(	heta\); a CI covers if \(	heta\in[\ell,u]\). Empirical coverage \(\hat c = M^{-1}\sum 1\{	heta\in CI_m\}\).

Assumptions: i.i.d., finite moments as needed for statistics; bootstrap consistency depends on \(F\) and statistic (percentile median robust under heavy tails; mean under skewness may require larger \(B\)).

Core method: probability/statistical resampling + Monte Carlo simulation.

## 3. Algorithm Design (Pseudocode)

**Inputs**: distribution spec (family, params), \(n\), reps \(M\), \(B\) (bootstrap draws), \(lpha\), statistic in {mean, median}, ci_method in {percentile, basic}.  
**Process**:
1. For each condition (n, statistic, ci_method, family):
   - For m in 1..M:
     - Sample X ~ F of size n.
     - Compute T = t(X).
     - For b in 1..B: sample with replacement; compute T^{*(b)}.
     - Compute CI based on method.
     - Record: (covers = 1[θ_true ∈ CI], ci_width, T, CI, seed state).
2. Aggregate coverage = mean(covers) and summaries by condition.
**Outputs**: `results.parquet` with row-level records + aggregated CSV-like table; figures under `figs/`.

Complexity: per condition O(M * (n + B*n)) time, O(B) memory for bootstrap statistics (streaming quantiles optional).

## 4. Python Implementation
See `model.py` (resampling engines), `simulate.py` (experiment runner), `viz.py` (plots). All functions are documented and parameters configurable via `config.yaml`.

## 5. Visualization
- Plot 1: Coverage vs. sample size (by statistic & method).
- Plot 2: CI width distribution across methods/statistics.  
Interactive extension idea: a small dashboard with sliders for n, B, alpha, distribution; recompute bootstrap CI live.

## 6. Reflection
Assumptions: i.i.d., stationarity across reps, B large enough for stable quantiles. Limitations: percentile/basic may be under/over-cover under strong skew/heavy tails; median CI may be discrete for small n; permutation CIs via inversion not implemented by default (added as extension).

Extensions: BCa or studentized bootstrap; permutation test inversion for two-sample CIs; heteroskedastic two-sample CI; real-data plug-in; streaming quantile estimators for large B; vectorized JAX/NumPyro backends.

## 7. Integration
- **Domain**: statistics / data science
- **Math Core**: probability; Monte Carlo
- **Computational Tools**: NumPy, Pandas, Matplotlib, PyArrow
- **Visualization Type**: curve plots; histograms/ECDF
- **Portfolio Links**: nonparametric estimation; Monte Carlo lab; robust statistics

---

### Usage
```bash
python simulate.py --config config.yaml
python viz.py --results results.parquet --outdir figs
```
