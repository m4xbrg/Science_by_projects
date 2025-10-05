# Logistic Regression Playground — Synthetic Classification, ROC/PR, Calibration

## 1. Title & Abstract
Probabilistic binary classification on synthetic datasets using L2-regularized logistic regression. We evaluate discrimination (ROC/PR) and calibration (reliability diagrams, Brier) and support post-hoc Platt/isotonic scaling.

## 2. Mathematical Formulation
- Model: p(y=1|x)=σ(w^T x + b)
- Loss: NLL + (λ/2)||w||^2
- Metrics: ROC/PR/AUC/AUPRC; calibration via binning; Brier score
- Calibration: Platt (parametric) and Isotonic (non-parametric)

## 3. Pseudocode
See `simulate.py` docstring and the main loop.

## 4. Implementation
- `model.py`: training and calibration utilities
- `simulate.py`: experiment runner, metrics & serialization
- `viz.py`: ROC/PR, calibration plots, optional decision boundaries

## 5. Visualization
Generated in `results/`: `roc.png`, `pr.png`, `calibration.png`, (`decision_boundary.png` if 2D).

## 6. Reflection
- Assumptions: logit-linear decision surface, IID data, class priors stationary.
- Limitations: nonlinearity only via features; threshold selection is application-specific.
- Extensions: polynomial/RBF features; Bayesian LR; multinomial; real datasets; cost-sensitive learning; calibration under shift.

## 7. Integration
See `meta.json` for ontology tags.
