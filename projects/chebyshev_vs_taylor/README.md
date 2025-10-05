# Chebyshev vs. Taylor on [a,b]: Uniform Approximation and Max-Error Comparisons

## 1. Title & Abstract
**Title:** Chebyshev vs. Taylor on \([a,b]\): Uniform Approximation and Max-Error Comparisons  

**Abstract:** We study polynomial approximation of a scalar function \(f:[a,b]\to\mathbb{R}\) using truncated Chebyshev series mapped from \([-1,1]\) and truncated Taylor polynomials about a center \(c\in[a,b]\). We implement a modular pipeline to compute both approximants and compare their **maximum absolute error** on \([a,b]\). Outputs include reproducible error curves and convergence plots over degree \(n\).

---

## 2. Mathematical Formulation
**Mapping**
\[
t=\frac{2x-(a+b)}{b-a}\in[-1,1],\quad x=\frac{(b-a)t+(a+b)}{2}.
\]

**Chebyshev (degree n)** with \(T_k(t)=\cos(k\arccos t)\):
\[
g(t)=f(x(t)),\quad g_n(t)=\sum_{k=0}^n a_k T_k(t),\quad p_n^{(\mathrm{cheb})}(x)=g_n(t(x)).
\]

**Taylor (degree n) at \(c\)**
\[
p_n^{(\mathrm{taylor})}(x)=\sum_{k=0}^n \frac{f^{(k)}(c)}{k!}(x-c)^k.
\]

**Errors**
\[
E_{\max}=\max_{x\in[a,b]}|f(x)-p(x)|\ \text{(estimated on a dense grid)}.
\]

**Assumptions:** \(f\) continuous on \([a,b]\); for Taylor, \(f\) is \(n\)-times differentiable at \(c\).

---

## 3. Algorithm Design (Pseudocode)
**Inputs:** \(f,[a,b],n,c,N_{\text{fit}},N_{\text{eval}}\).  
**Process:** compute Chebyshev fit; compute Taylor coefficients by FD+Richardson; evaluate on dense grid; derive errors and \(E_{\max}\); optional sweep over degrees.  
**Outputs:** parquet files with samples and errors; figures of function vs approximants, error curves, convergence.

---

## 4. Implementation (Files)
- `model.py`: core approximation routines (Chebyshev & Taylor), reusable API.
- `simulate.py`: reads `config.yaml`, runs single `n` and optional `n_list` sweep, writes parquet outputs.
- `viz.py`: loads parquet, produces two plots (function/approximants; error curves) and an optional convergence plot.
- `config.yaml`: parameters (interval, degree, center, grid sizes).
- `meta.json`: ontology tags.
- `requirements.txt`: NumPy, Pandas, Matplotlib, PyYAML, PyArrow.

> **Note:** This adapts the universal template (originally ODE-centric) to a *static approximation* problem. No `solve_ivp` is used; instead we compute polynomial approximants.

---

## 5. Visualization
- **Plot A:** \(f(x)\), Chebyshev, and Taylor approximants on \([a,b]\).
- **Plot B:** absolute error curves for both; legends include \(E_{\max}\).
- **Optional:** \(E_{\max}\) vs degree \(n\) (semilogy).  
Outputs saved to `figs/`.

---

## 6. Reflection
- Chebyshev is global and near-minimax for analytic \(f\); Taylor is local and can degrade away from \(c\).
- Max-error is grid-approximated; increase `N_eval` for sharper peaks.
- Extensions: Remez (true minimax), piecewise domains, automatic differentiation for Taylor derivatives, rational (Padé) approximants.

---

## 7. Integration (Ontology)
- **Domain:** Applied Mathematics / Numerical Analysis
- **Math Core:** Approximation Theory; Orthogonal Polynomials; Series Expansions
- **Computational Tools:** NumPy, Pandas, Matplotlib
- **Visualization Type:** Curve plots; Error curves; Convergence plot
- **Portfolio Links:** Remez Minimax Polynomial; Spectral Methods (Chebyshev); Padé vs. Taylor