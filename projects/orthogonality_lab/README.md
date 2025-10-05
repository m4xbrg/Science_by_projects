# Orthogonality Lab — Gram–Schmidt

## 1. Title & Abstract
Numerical exploration of classical vs. modified Gram–Schmidt with reorthogonalization, focusing on loss of orthogonality under finite precision. Outputs: metrics, factorization quality, and animations.

## 2. Math → Code
- Factorization: A = Q R, with GS variants (CGS, MGS).
- Metrics: ‖QᵀQ − I‖_F, relative residual ‖A − QR‖_F/‖A‖_F, max off-diagonal of G = QᵀQ, angles, cond₂(A).

## 3. Pseudocode
See main doc; implemented in `model.py`.

## 4. Implementation
- `model.py`: algorithms + metrics + matrix generators (gaussian, correlated, hilbert, vandermonde).
- `simulate.py`: batched runs; writes Parquet (CSV fallback).
- `viz.py`: scatter & bar plots; optional 2D/3D MGS animation.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python simulate.py --config config.yaml
python viz.py --results results/results.parquet --animate --dim 2
```

Artifacts in `results/` and `figs/`.
