# Distribution Gallery: PDFs/PMFs, CDFs, and Sample Moments

## 1. Title & Abstract
**Title**: Distribution Gallery — Normal, Exponential, and Poisson  
**Abstract**: This project builds a reproducible gallery of common probability distributions, generating samples, analytical PDF/PMF and CDF curves, and empirical moment estimates. The domain is probability and statistical computing. The outcome is a parameterized pipeline that compares theoretical and empirical properties and renders publication-ready figures.

## 2. Mathematical Formulation
We consider three canonical distributions with parameters θ:
- Normal: X ~ N(μ, σ²) with pdf f(x)= (1/(σ√(2π))) exp(-(x-μ)²/(2σ²)), cdf Φ((x-μ)/σ).
- Exponential: X ~ Exp(λ) with pdf f(x)= λ e^{-λ x} for x≥0; cdf F(x)= 1 - e^{-λ x}.
- Poisson: K ~ Pois(λ) with pmf P(K=k)= e^{-λ} λ^k / k!, k∈{0,1,2,...}; cdf is sum of pmf.

Assumptions: iid sampling; correct parameterization; finite moments as applicable.  
Boundary/initial conditions: support constraints (R, [0,∞), N₀).  
Math core: probability; analytical moments included in `model.py`.

## 3. Algorithm Design (Pseudocode)
**Inputs**: config.yaml (seed, distributions, parameters, plotting ranges).  
**Process**:
1. For each distribution spec (family, kind, params):
   - Instantiate model; draw `n_samples` iid samples via RNG.
   - Compute sample moments (mean, var, skew, kurtosis_excess).
   - Compute theoretical moments and errors.
   - Persist results to Parquet; cache raw samples as `.npy`.
2. Visualization:
   - Plot PDF/PMF with histogram/normalized counts.
   - Plot theoretical CDF vs empirical ECDF.
   - Optional: mean absolute error bar chart.
**Outputs**: `results.parquet`; figures in `figs/`.

Complexity: Sampling O(n); histogram and ECDF O(n log n) due to sort; memory O(n) per distribution.

## 4. Python Implementation
See `model.py`, `simulate.py`, `viz.py`. Functions have docstrings; parameters are read from YAML to avoid hardcoding.

## 5. Visualization
- Figures: `*_pdf_pmf.png`, `*_cdf.png`, plus `mean_error.png` summary.
- Axes and legends labeled; units implicit from variable support.
- Interactive extension: add a lightweight `streamlit` or `ipywidgets` app with sliders for μ, σ, and λ to live-update PDFs and resample.

## 6. Reflection
- Assumptions & limits: iid draws; SciPy parameterizations; large-n approximations for skew/kurtosis; ECDF requires sorting (O(n log n)).
- Extensions: add Gamma/Binomial/Student-t; MLE fitting from real data; QQ plots; Bayesian priors/posteriors.
- Scientific grounding: connects analytical probability with Monte Carlo validation and numerical estimation.

## 7. Integration
- Domain: mathematics
- Math Core: probability
- Computational Tools: NumPy, SciPy, Pandas, Matplotlib, PyYAML, PyArrow
- Visualization Types: curve plots; ECDF vs CDF; histogram/PMF overlay
- Portfolio Links: (fill as applicable)

## Usage
```bash
python simulate.py      # writes results.parquet and samples/*.npy using config.yaml
python viz.py           # reads samples & results, writes figs/*.png
```
