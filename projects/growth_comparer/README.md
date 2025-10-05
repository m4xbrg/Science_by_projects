# Growth Comparer — Linear vs. Exponential vs. Logistic (with crossover analysis)

**Abstract.** Unified comparison of linear, exponential, and logistic growth with crossover-time analysis. We offer a clean ODE API (`rhs(t, y, params)`), a solver pipeline (`solve_ivp`), parquet outputs, and two visualizations (time series, phase portrait).

---

## Title → Math → Pseudocode → Viz → Reflection → Integration

### Title & Abstract
- **Title:** Growth Comparer — Linear vs. Exponential vs. Logistic
- **Purpose:** Compare dynamics and quantify takeover/crossover times.
- **Expected outcomes:** Reusable code, results.parquet, figures, slider-ready design.

### Mathematical Formulation
- Linear: \( y(t)=y_0 + a t,\; dy/dt=a \).
- Exponential: \( y(t)=y_0 e^{rt},\; dy/dt=r y \).
- Logistic: \( dy/dt=r y (1-y/K) \), closed form \( y(t)=K/(1+A e^{-rt}), A=(K-y_0)/y_0 \).
- IC: \( y(0)=y_0>0 \), constants \( a,r,K \), \( 0<y_0<K \) for logistic.
- Core method: ODE + root-finding for crossovers (numeric bisection on \( y_i(t)-y_j(t) \)).

### Pseudocode
1. Load config (seed, `t_final`, `dt`, `y0`, params).
2. For each model m ∈ {linear, exponential, logistic}: integrate `dy/dt = rhs(t,y,params_m)` via `solve_ivp` with `t_eval` grid.
3. Assemble DataFrame with columns `[t, y_linear, y_exponential, y_logistic]` and write parquet.
4. For each pair, compute first **nontrivial** crossover time by scanning sign changes + bisection; write `crossovers.json`.
5. Plot: (a) time series; (b) 1D phase portrait `y` vs `dy/dt` along trajectory.

### Visualization
- Axes labeled, legends included; saved to `figs/`.
- Interactive extension: wrap `rhs` + solver with widgets/Streamlit sliders for `(y0, a, r, K, t_final)`.

### Reflection
- Assumptions: deterministic, constant params, single carrying capacity; crossovers are horizon-dependent.
- Limitations: sharp transitions may require finer `dt`; non-identifiability for small horizons.
- Extensions: stochasticity (SDE), time-varying rates, generalized logistic/Gompertz, multi-population models, data fitting.

### Integration
- See `meta.json` for ontology tags.
