# Parameter Identification for ODEs from Noisy Data (Pendulum Case Study)

## 1. Title & Abstract
**Title:** Robust Parameter Identification for Nonlinear ODEs from Noisy Observations — A Pendulum Case Study  
**Abstract:** We study parameter identification for nonlinear dynamical systems governed by ODEs using noisy data. Focusing on the damped pendulum, we compare multiple estimators—trajectory non-linear least squares (NLS), collocation/derivative-matching, and a Laplace-approximated MAP—under controlled noise and sampling regimes. Outcomes include a unified codebase, reproducible experiments, and estimator comparisons (bias/variance, runtime, sensitivity).

## 2. Mathematical Formulation
**Model (damped simple pendulum):**
\[
\dot x_1 = x_2,\quad
\dot x_2 = -\theta_1 \sin(x_1) - \theta_2 x_2
\]
- State: \(x=[x_1, x_2]^\top\) with angle (rad) and angular velocity (rad/s).  
- Parameters: \(\theta = [\theta_1,\theta_2]\) where \(\theta_1=g/L\) (s\(^{-2}\)), \(\theta_2=c\) damping (s\(^{-1}\)).
- Observations: \(y_k = x_1(t_k) + \epsilon_k\), with \(\epsilon_k\sim\mathcal N(0,\sigma^2)\).

**Assumptions:** (i) Continuous-time Markovian dynamics; (ii) Gaussian, iid measurement noise; (iii) known initial condition or estimated jointly.  
**Initial condition:** \(x(0)=x_0\).  
**Boundary conditions:** none (IVP).  
**Core math:** nonlinear ODE; estimation by optimization (NLS) and collocation; uncertainty via Fisher/Laplace.

**Alternative formulations:** small-angle linearization (\(\sin x_1\approx x_1\)), process noise via SDE (EKF/UKF), full Bayesian MCMC.

## 3. Algorithms (High-Level)
### A. Trajectory NLS (simulate-and-compare)
- Input: sampling grid \(\{t_k\}\), observations \(y_k\), initial guess \(\theta^{(0)}\), IC \(x_0\).
- Process: For candidate \(\theta\), integrate ODE \(x(t;\theta)\) and compute residuals \(r_k = y_k - x_1(t_k;\theta)\). Minimize \(\|r\|_2^2\) via Gauss–Newton/LM.
- Output: \(\hat\theta_{\text{NLS}}\), Jacobian, covariance \((J^\top J)^{-1}\hat\sigma^2\).

### B. Collocation / Derivative-Matching
- Smooth \(y_k\) with cubic smoothing spline to get \(\tilde x_1(t)\) and \(\frac{d}{dt}\tilde x_1\). Estimate \(x_2\approx \frac{d}{dt}\tilde x_1\).
- Build residuals: \(r(t)=\frac{d}{dt}\tilde x_2(t)+\theta_1\sin(\tilde x_1(t))+\theta_2\tilde x_2(t)\).
- Minimize \(\int r(t)^2\,dt\) via discrete least squares on \(\{t_k\}\).

### C. Laplace-Approximated MAP
- With Gaussian prior \(\theta\sim \mathcal N(\mu_0,\Sigma_0)\), objective is NLS + quadratic penalty. At optimum, posterior \(\mathcal N(\hat\theta, (J^\top J/\hat\sigma^2+\Sigma_0^{-1})^{-1})\).

## 4. Files
- `meta.json`, `config.yaml`, `requirements.txt`
- `model.py` — ODE RHS and simulators
- `simulate.py` — experiment driver, synthetic data
- `estimate.py` — NLS, collocation, MAP, metrics
- `viz.py` — time series, phase portrait, residual and sensitivity plots
- `figs/` — outputs
- `notebooks/` — optional demos

## 5. Usage
```bash
pip install -r requirements.txt
python simulate.py --config config.yaml
python estimate.py --config config.yaml --method nls
python estimate.py --config config.yaml --method collocation
python estimate.py --config config.yaml --method map
python viz.py --config config.yaml
```

## 6. Reflection & Extensions
- Assumptions: iid Gaussian measurement noise, correct model structure, known ICs. Violations bias estimates.
- Extensions: joint state-parameter filtering (EKF/UKF), process noise SDEs, partial observability, Bayesian MCMC, multiple trajectories, experiment design (Fisher information).

## 7. Integration
See `meta.json` for ontology tags.
