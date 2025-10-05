
# Linear Systems via Matrix Exponential

## 1. Title & Abstract
**Title:** Linear Systems via Matrix Exponential: Eigenstructure Classification & Trajectory Visualization

**Abstract:** We study continuous-time linear time-invariant (LTI) systems \(\dot{x}(t)=Ax(t)\) through the matrix exponential \(e^{At}\). We derive and compute \(e^{At}\), relate trajectories to the eigenstructure (real/complex, defective/diagonalizable), and visualize state evolution with time-series and phase portraits, including optional animations. Outcomes include a reusable code scaffold, eigenstructure classifier, and visualization utilities.

---

## 2. Mathematical Formulation

- **Core system:** \(\dot{x}(t)=A x(t),\quad x(0)=x_0\), with \(A\in\mathbb{R}^{n\times n}\).
- **Solution:** \(x(t)=e^{At}x_0\), where \(e^{At}=\sum_{k=0}^{\infty} \frac{(At)^k}{k!}\).
- **Transpose flow:** \(e^{A^\top t}=(e^{At})^\top\).
- **Eigen-decomposition (diagonalizable):** if \(A=V\Lambda V^{-1}\), then \(e^{At}=V e^{\Lambda t} V^{-1}\) with \(e^{\Lambda t}=\operatorname{diag}(e^{\lambda_i t})\).
- **Jordan form (defective):** if \(A=VJV^{-1}\), \(e^{At}=V e^{Jt} V^{-1}\), and each Jordan block contributes polynomial-in-\(t\) factors multiplying \(e^{\lambda t}\).

**Symbols:**  
\(A\): system matrix; \(x(t)\): state; \(x_0\): initial state; \(\lambda_i\): eigenvalues; \(v_i\): eigenvectors; \(J\): Jordan form; \(V\): generalized eigenvectors; \(t\): time.

**Assumptions:** LTI system; real-valued state; numerical routines return stable results for moderate \(\|A\|, t\).  
**Boundary/Initial conditions:** \(x(0)=x_0\).  
**Method class:** ODE (linear, autonomous).

**Deterministic vs numerical:** Closed form via \(e^{At}\) vs numerical ODE integration (e.g., `solve_ivp`) providing a cross-check.

---

## 3. Algorithm Design (Pseudocode)

**Inputs:** matrix `A` (n×n), time grid `t_grid`, list of initial conditions `ICs`, config params.  
**Process:**  
1. Compute eigenvalues/eigenvectors; assess diagonalizability (rank of eigenvectors).  
2. Classify stability/type (stable/unstable/center; node/saddle/spiral; defective).  
3. Compute \(e^{At}\) for each `t` using `scipy.linalg.expm( A * t )`.  
4. For each `x0` in `ICs`, propagate `x(t)=e^{At} x0` (or integrate `rhs`).  
5. Visualize: (a) time series per component; (b) phase portrait with multiple trajectories and vector field.  
6. (Optional) Animate trajectories with `matplotlib.animation.FuncAnimation`.

**Outputs:** classification report, trajectories array, saved figures/animation.

**Complexity:**  
- `expm` is \(\mathcal{O}(n^3)\) per time step (Padé/Schur). For `m=|t_grid|`, total \(\mathcal{O}(m n^3)\).  
- Memory: storing trajectories \(\mathcal{O}(m n |ICs|)\).

---

## 4. Python Implementation
See `model.py`, `simulate.py`, and `viz.py` for modular code.

---

## 5. Visualization
- Time series of \(x_i(t)\), with legend and units.
- 2D phase portrait (quiver + trajectories) when `n==2`.
- Optional interactive extension: slider-controlled entries of `A` or a small Dash/Panel app; animation of trajectories over time.

---

## 6. Reflection
- **Assumptions/limits:** LTI only; numerical sensitivity in nearly defective cases; visualization mainly 2D.  
- **Extensions:** time-varying \(A(t)\) via chronological exponentials; stochastic perturbations; controllability Gramians (\(W(t)=\int_0^t e^{A\tau}BB^\top e^{A^\top\tau} d\tau\)); higher-dimensional embeddings; parameter inference from data.  
- **Principles:** linear stability theory, spectral mapping theorem, Jordan calculus, numerical linear algebra (Schur/Padé).

---

## 7. Integration (Ontology)
- **Domain:** Applied Mathematics / Dynamical Systems  
- **Math Core:** ODE (linear systems), Linear Algebra  
- **Computational Tools:** NumPy, SciPy (`expm`, `eig`), Matplotlib, PyYAML, Pandas/Parquet (I/O)  
- **Visualization Type:** Time series; Phase portraits; Animations  
- **Portfolio Links:** Precursor—“2D Linear Phase Portraits”; Extension—“Controllability/Observability Gramians”

