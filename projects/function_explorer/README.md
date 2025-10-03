# Function Explorer — Symbolic–Numeric Analysis of Real-Valued Functions

## 1. Title & Abstract
This project builds a computational widget that ingests a user-typed mathematical expression \(f(x)\), plots it, and automatically computes its domain, range, and intercepts.
Domain: applied mathematics & computational analysis. Outcomes: reusable Python library, visualization routines, and an interactive dashboard.

## 2. Mathematical Formulation
- **Function:** \(y=f(x),\ x \in \mathbb{R}\)
- **Domain:** exclude denominators = 0, log args \(\le 0\), even-root radicands \(< 0\)
- **Range:** estimated from extrema candidates + boundaries + sampling
- **Intercepts:** solve \(f(x)=0\) (x), evaluate \(f(0)\) if defined (y)
- **Method:** hybrid symbolic (SymPy) + numeric (NumPy sampling)

## 3. Algorithm Design (Pseudocode)
```
INPUT: expr_str, [xmin,xmax], N samples
1. Parse to symbolic f(x)
2. Domain constraints (denoms, logs, even roots)
3. Singularities, intercepts, derivative zeros
4. Numeric sampling over domain intervals
5. Range from samples + critical points + boundaries
OUTPUT: {domain, intercepts, singularities, range, samples}
```

## 4. Python Implementation
- `model.py`: thin wrapper exposing `FunctionExplorer` (from `function_explorer.py`)
- `viz.py`: plotting helpers (`plot_analysis`, `plot_value_hist`)
- `simulate.py`: config-driven run → `results.parquet`
- Configurable parameters in `config.yaml`

## 5. Visualization
- Curve plot of \(f(x)\) with intercepts + singularities
- Histogram of sampled values (distribution of \(f(x)\))
- Interactive Streamlit app: `streamlit_app.py`

## 6. Reflection
- **Assumptions:** real-valued only, principal branches, finite window
- **Limitations:** symbolic solving may fail; range approximation is window-dependent
- **Extensions:** adaptive sampling; exact ranges for special families; parameter sweeps
- **Principles:** real analysis; hybrid symbolic–numeric computation

## 7. Integration
- **Domain:** Mathematics → Analysis; Scientific Computing
- **Math Core:** Real analysis, symbolic algebra, optimization (extrema)
- **Tools:** SymPy, NumPy, Matplotlib, Streamlit, Pandas
- **Visualizations:** Curve plot, histogram
- **Links:** Rational Function Analyzer; Derivative & Curvature Explorer; Parameter Sweep Dashboard
