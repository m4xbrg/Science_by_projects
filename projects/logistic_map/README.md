# Logistic Map Chaos: Bifurcation Diagram & Lyapunov Exponent

This project implements a unified, ontology-ready pipeline to study the logistic map
\(x_{n+1} = r x_n (1 - x_n)\). It generates a bifurcation diagram and estimates
the maximal Lyapunov exponent across a range of control parameters \(r\).

**Structure** (Universal):
- Title → Math → Pseudocode → Visualization → Reflection → Integration

## 1. Title & Abstract
**Title:** Logistic Map Chaos: Bifurcation & Lyapunov Spectrum  
**Abstract:** We investigate the dynamics of the logistic map in the domain of nonlinear
dynamics and chaos. Using a configurable, reproducible pipeline, we compute the
bifurcation diagram and the maximal Lyapunov exponent across \(r\), producing
publication-quality figures and reusable code.

## 2. Mathematical Formulation
- **Map:** \(x_{n+1} = f(x_n; r) = r x_n (1 - x_n)\), \(r \in [0, 4]\), \(x_n \in [0,1]\)
- **Jacobian / derivative:** \(f'(x; r) = r (1 - 2x)\)
- **Lyapunov exponent (maximal):**
  \[\lambda(r) = \lim_{N \to \infty} \frac1N \sum_{n=0}^{N-1} \log |f'(x_n; r)|\]
- **Assumptions:** double precision; uniform grid of \(r\); discard transients before statistics.
- **Initial condition:** configurable (default \(x_0 = 0.5\)).
- **Boundary/constraints:** values clipped to \([0,1]\) numerically when needed.

## 3. Algorithm Design (Pseudocode)
```
inputs: r_grid, x0, n_transient, n_iter, n_keep
for each r in r_grid:
    x <- x0
    // burn-in
    repeat n_transient:
        x <- r * x * (1 - x)
    // statistics
    lyap_sum <- 0
    repeat n_iter:
        x <- r * x * (1 - x)
        lyap_sum += log(|r * (1 - 2x)|)
        if last n_keep steps: store (r, x) for bifurcation
    lambda <- lyap_sum / n_iter
    save lambda(r)
outputs: bifurcation points (r,x), lyapunov curve (r, lambda)
```
Time complexity: O(|r-grid| (n_transient + n_iter)).  
Space: dominated by stored bifurcation samples, O(|r-grid| * n_keep).

## 4. Python Implementation
See `model.py`, `simulate.py`, and `viz.py`. All parameters are in `config.yaml` and
metadata in `meta.json`.

## 5. Visualization
- **Bifurcation diagram:** scatter of (r, x) after transient.
- **Lyapunov exponent:** curve of \(\lambda(r)\) with zero-line overlay.
- Interactive extensions: add sliders for r-range and iteration counts with ipywidgets or a
simple dashboard (e.g., Panel/Streamlit).

## 6. Reflection
- **Assumptions/limitations:** finite sampling, sensitivity to x0, numerical rounding,
finite N for \(\lambda\) introduces bias/variance. Periodic windows exist where
\(\lambda < 0\) even at large r.
- **Extensions:** parameter noise, random maps, higher-dimensional coupled maps, Feigenbaum
delta estimation, measure of chaos via invariant measure approximation, correlation dimension.

## 7. Integration
- **Domain:** Physics / Applied Math (Nonlinear Dynamics)
- **Math Core:** Discrete-time dynamical systems, ergodic averages
- **Computational Tools:** NumPy, Pandas, Matplotlib, PyYAML, PyArrow
- **Visualization Types:** Scatter (bifurcation), curve (Lyapunov)
- **Portfolio Links:** precursor projects on 1D maps, chaos indicators, Feigenbaum constants.

---

### Usage

1) Edit parameters in `config.yaml`  
2) Run simulation:
```
python simulate.py
```
This writes `bifurcation.parquet` and `lyapunov.parquet` to the project root.

3) Generate figures:
```
python viz.py
```
Figures are saved under `figs/`.
