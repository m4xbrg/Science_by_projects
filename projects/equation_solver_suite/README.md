# Equation Solver Suite — Linear Systems & Quadratics

## 1. Title & Abstract
**Title:** Equation Solver Suite — Linear Systems (Gaussian Elimination) & Quadratics (Quadratic Formula) with Stepwise Trace

**Abstract:** This project implements (i) linear systems \(A\mathbf{x}=\mathbf{b}\) via Gaussian elimination with partial pivoting and (ii) single-variable quadratics \(ax^2+bx+c=0\) via a numerically stable quadratic formula. Outcomes: modular APIs, stepwise provenance logging, diagnostics, and standard visualizations for residuals and sensitivity.

---

## 2. Mathematical Formulation
**Linear system:** Row-reduction with multipliers \(m_{ik}=a_{ik}/a_{kk}\) for forward elimination; back substitution \(x_i = (b_i - \sum_{j>i} a_{ij}x_j)/a_{ii}\).  
**Quadratic:** Discriminant \(\Delta=b^2-4ac\); stable branch using \(q=-\tfrac12(b+\operatorname{sign}(b)\sqrt\Delta)\) when real.

**Assumptions:** Square matrix, partial pivoting; complex roots supported for quadratics; tolerance \(\varepsilon\) for near-zero pivots.

**Method Type:** Direct numerical linear algebra; closed-form polynomial roots.

---

## 3. Algorithm Design (Pseudocode)
### Linear (Gaussian Elimination, partial pivoting)
```
Input: A (n×n), b (n), eps
Aug ← [A|b]
for k in 0..n-1:
  swap pivot row with max |Aug[i,k]|
  if |Aug[k,k]|<eps: singular
  for i in k+1..n-1:
    m ← Aug[i,k]/Aug[k,k]
    Row_i ← Row_i − m·Row_k
Back-substitute to recover x
Output: x, steps (trace)
```

### Quadratic (Stable quadratic formula)
```
Input: a,b,c, eps
Δ ← b^2 − 4ac
if Δ≥0:
  if b≠0: q ← −0.5 (b + sign(b) √Δ); x1 ← q/a; x2 ← c/q
  else:    x1 ←  √Δ/(2a); x2 ← −√Δ/(2a)
else:
  x1, x2 ← complex roots via √Δ
Output: (x1,x2), type, steps
```

**Complexity:** Linear: Θ(n³) time, Θ(n²) space; Quadratic: Θ(1).

---

## 4. Python Implementation
See `model.py` for solvers with step tracing and `simulate.py` for a reproducible run that writes `data/results.parquet`.

---

## 5. Visualization
- `viz.py` generates two figures to `figs/`:
  1. Residual norm vs. back-substitution step (linear system).
  2. Quadratic root sensitivity sweep over **b** for fixed (a,c).

---

## 6. Reflection
**Limitations:** Conditioning dominates accuracy; partial pivoting not universally stable; quadratic cancellation mitigated by stable branch but scaling may still be needed.  
**Extensions:** LU/QR APIs, growth-factor diagnostics, interval arithmetic, stochastic perturbation analysis.

---

## 7. Integration
See `meta.json` for ontology tags; related interactive notebook/utility in `notebooks/` optional.
