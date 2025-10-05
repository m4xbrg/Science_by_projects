# Outlier Detective: Robust vs Classical Detection under Contamination

## 1. Title & Abstract
**Title:** Outlier Detective — Comparing Classical and Robust Outlier Detectors under Data Contamination  
**Abstract:** This project builds a reproducible pipeline to compare classical z-score and IQR-based outlier detection with robust alternatives (median–MAD z and adjusted boxplot fences) on contaminated univariate data. We formalize estimators, derive decision rules, and evaluate detection quality as contamination increases, producing performance curves and interpretable plots.

## 2. Mathematical Formulation
- **Data model:** i.i.d. samples \(x_1,\dots,x_n \in \mathbb{R}\) from a contaminated distribution \(F_\epsilon = (1-\epsilon) F_0 + \epsilon G\).\
  \(F_0\) is "inlier" (e.g., \(\mathcal{N}(\mu,\sigma^2)\)), \(G\) produces outliers.
- **Classical estimators:** sample mean \(\hat{\mu}=\frac{1}{n}\sum x_i\), sample std \(\hat{\sigma}=\sqrt{\frac{1}{n-1}\sum (x_i-\hat{\mu})^2}\).\
  Classical z-score: \( z_i = \frac{x_i-\hat{\mu}}{\hat{\sigma}}\); flag if \(|z_i| > \tau\).
- **Robust estimators:** median \(\tilde{x}\), MAD \( \mathrm{MAD} = \mathrm{median}_i |x_i-\tilde{x}|\).\
  Robust z-score (Gaussian consistency): \( z^{\mathrm{rob}}_i = \frac{x_i-\tilde{x}}{1.4826\,\mathrm{MAD} + \delta} \); flag if \(|z^{\mathrm{rob}}_i| > \tau\). (\(\delta>0\) protects against zero MAD.)
- **IQR fences:** \(Q_1, Q_3\) quartiles, \(IQR=Q_3-Q_1\).\
  Classical: lower \(=Q_1 - k \cdot IQR\), upper \(=Q_3 + k \cdot IQR\) with \(k=1.5\).\
  Adjusted \(k\) permitted to study sensitivity.
- **Assumptions:** i.i.d., univariate, fixed threshold \(\tau\) (z) or \(k\) (IQR). Boundary/initial conditions are not applicable.
- **Method:** probability & robust statistics; decision rules from estimators.

## 3. Algorithm Design (Pseudocode)
**Inputs** → data vector `x`, thresholds (`tau`, `k`), robust constants, RNG seed, contamination parameters.  
**Process** → compute estimators (classical & robust), compute scores/fences, flag outliers, evaluate precision/recall/F1 against known labels.  
**Outputs** → tabular metrics across contamination levels; annotated figures.

```
FUNCTION detect_outliers(x, method, params):
    IF method == "z":
        mu = mean(x); s = std(x, ddof=1)
        z = (x - mu) / s
        mask = abs(z) > params.tau
    ELIF method == "z_robust":
        med = median(x); mad = median(abs(x - med))
        s = 1.4826 * mad + params.eps
        z = (x - med) / s
        mask = abs(z) > params.tau
    ELIF method == "iqr":
        q1, q3 = quantile(x, 0.25), quantile(x, 0.75)
        iqr = q3 - q1
        lo, hi = q1 - params.k*iqr, q3 + params.k*iqr
        mask = (x < lo) OR (x > hi)
    RETURN mask, auxiliary_stats

FUNCTION simulate(n, epsilon, base_dist, out_dist, seed):
    x0 ~ base_dist(size = floor((1-eps)*n))
    x1 ~ out_dist(size = n - len(x0))
    x = concat(x0, x1), y = [0]*len(x0) + [1]*len(x1)
    shuffle (x,y) with seed
    FOR method IN METHODS: mask = detect_outliers(x, method, params)
        precision, recall, f1 = PRF(mask, y)
    RETURN metrics_row
```

**Complexity:** All methods are \(O(n)\) time, \(O(1)\) space beyond input (quantiles/median are \(O(n)\) with typical selection algorithms; using sort is \(O(n\log n)\) in practice).

## 4. Python Implementation
See `model.py`, `simulate.py`, and `viz.py`. Each function has docstrings and configurable parameters via `config.yaml`.

## 5. Visualization
- **Performance curves:** F1 vs contamination \(\epsilon\) for each method.
- **Annotated sample:** Data cloud with flags (classical vs robust) for a chosen \(\epsilon\).  
Interactive extension: add sliders (epsilon, tau, k) with `ipywidgets` or a small Streamlit app.

## 6. Reflection
- **Assumptions/limits:** i.i.d. univariate; fixed thresholds; contamination independent of inliers. Real data may be heteroskedastic or multimodal.
- **Extensions:** multivariate robust covariance (MCD), depth-based fences, time-series outliers, heavy-tail baselines, heteroskedastic contamination, adaptive thresholds via FDR.
- **Principles:** compares breakdown points (0% for mean/std vs 50% for median/MAD); studies robustness and efficiency trade-offs under model misspecification.

## 7. Integration
- **Domain:** data science / statistics
- **Math Core:** probability, robust statistics
- **Computational Tools:** NumPy, Pandas, Matplotlib, PyYAML
- **Visualization Types:** curve plots, annotated scatter/box
- **Portfolio Links:** (fill as you connect related projects)
