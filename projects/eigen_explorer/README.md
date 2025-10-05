# Eigen Explorer: Eigen-Decomposition & Invariant Axes Visualization

## 1. Title & Abstract
**Title:** Eigen Explorer — dominant eigenpairs via the Power Method, vs. `numpy.linalg.eig`, with invariant-axis visualization.

**Abstract:** We study eigen-decomposition of real square matrices and the geometry of invariant axes. The project implements a modular power method and compares its convergence to NumPy’s direct eigen-decomposition. We visualize invariant directions and the linear transform of the unit circle to reveal principal stretch/compression.

## 2. Mathematical Formulation
- **Core problem:** Given a matrix $A\in\mathbb{R}^{n\times n}$, find $\lambda\in\mathbb{C}$ and nonzero $v\in\mathbb{C}^n$ such that $A v = \lambda v$.
- **Rayleigh quotient:** $\rho(x)=\dfrac{x^\top A x}{x^\top x}$ (for real symmetric $A$). Used as an estimator of the dominant eigenvalue in the power method.
- **Power method iteration:** Starting with $x_0\neq 0$,
  $$y_k = A x_{k-1},\quad x_k = \dfrac{y_k}{\|y_k\|},\quad \lambda_k = \rho(x_k).$$
- **Residual:** $r_k = \|A x_k - \lambda_k x_k\|$ monitors convergence.
- **Assumptions:** Dominant eigenvalue is simple and strictly larger in magnitude than others; initial vector has nonzero component along the dominant eigenvector.

**Method class:** Linear algebra / numerical linear algebra.

## 3. Algorithm Design (Pseudocode)

**Inputs → Processes → Outputs**

- **Inputs:** matrix `A (n×n)`, initial guess `x0`, tolerance `tol`, `max_iter`, normalization norm (`l2`), random seed (optional).
- **Process:** iterative power method; compute Rayleigh quotients and residuals; compute reference eigendecomposition via NumPy; compare.
- **Outputs:** dominant eigenvalue/vector estimate per iteration; residual series; final comparison table; figures.

**Pseudocode**
```
function POWER_METHOD(A, x0, tol, max_iter):
    x ← x0 / ||x0||
    for k in 1..max_iter:
        y ← A x
        λ ← (xᵀ y)/(xᵀ x)   # Rayleigh quotient
        r ← ||A x − λ x||
        record (k, λ, r, x)
        if r < tol: break
        x ← y / ||y||
    return λ, x, history

function EIG_INVARIANT_AXES(A):
    compute eigenvalues/eigenvectors via numpy.linalg.eig(A)
    return Λ, V

simulate():
    load config.yaml
    run POWER_METHOD and EIG_INVARIANT_AXES
    write results (history, eigs) to results.parquet (fallback CSV)
```

**Complexity:** Each power step: `O(n^2)` for dense `A` (matrix–vector). Storage: `O(n)` per step plus history if saved (`O(k n)`).

## 4. Python Implementation

See `model.py` (algorithms), `simulate.py` (experiment + I/O), `viz.py` (plots).

## 5. Visualization
- **Convergence plot:** residual $\|A x_k - \lambda_k x_k\|$ vs iteration; and Rayleigh quotient sequence.
- **Invariant axes:** plot eigenvectors (normalized) and the image of the unit circle under `A` to show principal stretching.
- **Interactive extension:** add sliders (e.g., with `ipywidgets`/`panel`) to edit `A` entries and re-run power iterations with live plots.

## 6. Reflection
- **Assumptions:** simple spectral gap; non-defective matrices for clean geometry.
- **Limitations:** power method finds only the *dominant* eigenpair; slow when spectral gap is small; sensitive if initial vector is orthogonal to dominant eigenvector.
- **Extensions:** shifted/inverse power; deflation to recover multiple eigenpairs; symmetric/Hermitian specialization; stochastic matrices (Perron–Frobenius); real-data PCA link (covariance eigenpairs).

## 7. Integration
- **Domain:** Applied Mathematics / Numerical Linear Algebra
- **Math Core:** Linear algebra; iterative methods
- **Computational Tools:** NumPy, Pandas, Matplotlib, PyYAML
- **Visualization Type:** Convergence curves; geometric transform plots
- **Portfolio Links:** PCA Explorer; Spectral Graph Explorer (if present)

## How to run
```bash
pip install -r requirements.txt
python simulate.py --config config.yaml
python viz.py --results results.parquet   # or results.csv fallback
```