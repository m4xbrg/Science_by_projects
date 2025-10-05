# Damped Oscillator Lab: Phase Portraits & Energy Decay

## 1. Title & Abstract
**Title:** Damped Harmonic Oscillator — Dynamics, Phase Portraits, and Energy Decay  
**Abstract:** We model and simulate the linear damped oscillator governed by \(m\ddot x + c\dot x + kx = 0\). The goal is to quantify transient behavior across damping regimes, visualize phase trajectories, and study energy dissipation. Outputs include time-series, phase portraits, and energy-decay curves with a reproducible pipeline.

## 2. Mathematical Formulation
**Core ODE:**  
\[ m\ddot x(t) + c\dot x(t) + k x(t) = 0. \]

**Definitions:**  
- \(x(t)\): displacement [m]  
- \(\dot x(t) = v(t)\): velocity [m/s]  
- \(\ddot x(t)\): acceleration [m/s^2]  
- \(m>0\): mass [kg]  
- \(c\ge 0\): damping coefficient [N·s/m]  
- \(k>0\): stiffness [N/m]

**Assumptions/ICs/BCs:**  
- Linear, single-DOF, time-invariant system; no external forcing.  
- Initial conditions: \(x(0)=x_0\), \(\dot x(0)=v_0\).  
- No boundary constraints beyond initial value problem.

**Method:** First-order ODE system: \(\dot y = f(t,y)\) with \(y=[x, v]^\top\). Numerical integration via `solve_ivp` (Runge–Kutta).

**Energy:** \(E(t) = \tfrac12 m v^2 + \tfrac12 k x^2\). For \(c>0\), \(\dot E(t) = -c v^2 \le 0\).

**Damping regimes (\(\zeta = c/(2\sqrt{km})\)):** under (\(\zeta<1\)), critical (\(\zeta=1\)), over (\(\zeta>1\)).

## 3. Algorithm Design (Pseudocode)

**Inputs → Processes → Outputs**

- **Inputs:** `params={m,c,k}`, `IC=[x0,v0]`, `t_span=[0,t_final]`, `dt`, solver options.  
- **Processes:** define `rhs(t,y,params)`; integrate with `solve_ivp` on a fixed time grid; compute energy; write Parquet.  
- **Outputs:** `results.parquet` with `t, x, v, E` and `figs/*.png`.

**Pseudocode:**  
```
function rhs(t, y, params):
    x = y[0]; v = y[1]
    dxdt = v
    dvdt = -(c/m)*v - (k/m)*x
    return [dxdt, dvdt]

function simulate(params, IC, t_span, dt):
    grid = linspace(t_span[0], t_span[1], ceil((t1-t0)/dt)+1)
    sol = solve_ivp(rhs, t_span, IC, t_eval=grid, args=(params,), rtol, atol)
    E = 0.5*m*sol.y[1]^2 + 0.5*k*sol.y[0]^2
    write parquet {t, x, v, E}
    return dataframe

function make_plots(df):
    plot t vs x
    plot phase portrait x vs v with arrows or color by time
    (optional) plot t vs E
```

**Complexity:** Each step = O(1); time steps ≈ N = t_final/dt ⇒ O(N) time, O(N) memory for stored trajectory.

## 4. Python Implementation
See `model.py`, `simulate.py`, `viz.py`. API is consistent across projects:
- `rhs(t, y, params)`, `energy(x, v, params)` in `model.py`
- `run_simulation(config_path)` in `simulate.py`
- `plot_timeseries(df, outpath)`, `plot_phase(df, outpath)`, `plot_energy(df, outpath)` in `viz.py`

## 5. Visualization
- Time evolution: x(t)  
- Phase portrait: (x, v)  
- Energy decay: E(t)  
Axes labeled with SI units; legends included. Interactive extension: add a small Streamlit app or ipywidgets sliders for `m, c, k, x0, v0`.

## 6. Reflection
**Assumptions/Limitations:** linearity, no forcing, constant parameters, numerical dissipation from solver.  
**Extensions:** add forcing (LTI/LTV), nonlinear damping (e.g., quadratic), param sweeps & bifurcations, stochastic forcing (Langevin), identification from real data, multi-DOF mass–spring networks.

## 7. Integration (Ontology)
- **Domain:** physics → classical mechanics → vibrations  
- **Math Core:** ODE (linear, 2D system)  
- **Computational Tools:** NumPy, SciPy, Pandas, Matplotlib, PyYAML, PyArrow  
- **Visualization Type:** curve plots, phase portraits, energy curves  
- **Portfolio Links:** Simple Harmonic Oscillator; Forced Oscillator; Nonlinear Duffing Oscillator (future).

---

### How to run

```
pip install -r requirements.txt
python simulate.py --config config.yaml
python viz.py --results results.parquet --outdir figs
```

Artifacts: `results.parquet`, `figs/` (PNG).