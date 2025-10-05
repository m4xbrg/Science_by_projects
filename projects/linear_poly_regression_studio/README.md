# Linear & Polynomial Regression Studio

## 1. Title & Abstract
**Title:** Linear/Polynomial Regression Studio: K-Fold Training & Bias–Variance Analysis

**Abstract:** This project implements a unified pipeline for linear and polynomial regression with \(k\)-fold cross‑validation, regularization, and bias–variance curve estimation. The domain is statistical learning / computational data analysis. Outcomes include a reproducible API (`model.py`), a configurable experiment runner (`simulate.py`), and visualization utilities (`viz.py`) producing learning curves and bias–variance tradeoff plots.

## 2. Mathematical Formulation
### Core Equations
Given inputs \(X \in \mathbb{R}^{n \times d}\) and target \(y \in \mathbb{R}^n\), with polynomial feature map \(\Phi_d: \mathbb{R}^{d}\to\mathbb{R}^{p}\) of degree \(m\):
- Hypothesis: \( \hat{y} = \Phi(X)\, w \).
- Ridge-regularized ERM: \( \min_w \; \frac{1}{n}\Vert \Phi(X)w - y \Vert_2^2 + \lambda \Vert w \Vert_2^2 \).
- Closed form (when \(\Phi^\top\Phi\) invertible): \( w^* = (\Phi^\top \Phi + n\lambda I)^{-1}\Phi^\top y \).
- Mean squared error: \( \operatorname{MSE} = \frac{1}{n}\Vert \hat{y} - y \Vert_2^2 \).

**Symbols:** \(X\): inputs; \(y\): targets; \(\Phi\): polynomial feature matrix (with optional bias); \(w\): coefficients; \(\lambda\): ridge strength; \(m\): degree; \(n\): sample size; \(p\): feature count.

**Assumptions:** i.i.d. samples; additive noise \(\varepsilon\) with \(\mathbb{E}[\varepsilon]=0\) and finite variance; feature matrix has full column rank for stability or ridge used; scaling optional.

**Method:** Optimization / linear algebra.

### Alternative Formulations
- **Unregularized OLS:** \(\lambda=0\).
- **Lasso (not implemented):** \(\ell_1\) penalty (would require coordinate descent).
- **Stochastic:** SGD approximation of ERM (extension).

## 3. Algorithm Design (Pseudocode)
```
inputs: X, y, degrees, lambdas, k_folds, scale, include_bias, seed
process:
  for each degree m in degrees:
    Phi = poly_features(X, degree=m, include_bias)
    if scale: Phi = standardize(Phi) using train fold stats
    for each lambda in lambdas:
      kfold split (seeded)
      for each fold:
        train on (Phi_train, y_train) via ridge closed-form
        predict on train/val; record MSE_train, MSE_val
outputs:
  results table with columns: degree, lambda, fold, mse_train, mse_val
  best hyperparameters by avg val MSE
Optional (bias–variance on synthetic data):
  repeat T trials: resample train; fit model; predict on fixed test grid
  compute per-x: bias^2 = (mean(pred) - f_true)^2; var = var(pred); noise = sigma^2
  aggregate curves across x
```
**Complexity:** Forming \(\Phi\): \(O(ndm)\) in 1D or \(O(nd^m)\) worst‑case; solve: \(O(p^3)\) (Cholesky) per fold per \(\lambda\). Memory \(O(np + p^2)\).

## 4. Python Implementation
See `model.py` (core), `simulate.py` (runner), `viz.py` (plots). Parameters configurable via `config.yaml`.

## 5. Visualization
- Validation MSE vs degree (per \(\lambda\)); learning curves.
- Bias–variance decomposition on synthetic ground truth.
Interactive extension: `viz.py` exposes a function to generate plots for any config; can be wrapped in a small Streamlit/Voila dashboard with sliders for degree and \(\lambda\).

## 6. Reflection
- **Limitations:** Polynomial blow-up in high dimension; sensitivity to scaling; closed‑form \(O(p^3)\) scaling; bias–variance decomposition requires known ground truth or strong assumptions.
- **Extensions:** Lasso/ElasticNet; orthogonal polynomial bases; kernel ridge; robust losses; real datasets with k-fold nested CV; uncertainty via bootstrap.
- **Principles:** Empirical risk minimization, regularization, cross‑validation, and the bias–variance decomposition of generalization error.

## 7. Integration
- **Domain:** Data Science / Applied Mathematics
- **Math Core:** Optimization & Linear Algebra (statistical learning)
- **Computational Tools:** NumPy, Pandas, Matplotlib, PyYAML, PyArrow (Parquet)
- **Visualization Type:** Curve plots (learning curves; bias–variance)
- **Portfolio Links:** Regression Basics → Regularization → Kernel Methods (future)
```
