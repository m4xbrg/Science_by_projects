# Matrix Workbench — Rank, Determinant, LU/QR; Linear Map Effects on Unit Square/Cube

## Abstract
This repository provides a consistent API to compute rank (SVD-based), determinant (pivoted LU), LU and QR (Householder) factorizations, and to visualize the geometric action of linear maps on canonical sets. See `model.py` for the core algorithms, `simulate.py` for data products, and `viz.py` for figures.

## Mathematical Formulation
- Linear map \(f_A(x)=Ax\); rank, determinant, LU with partial pivoting \(PA=LU\), Householder QR \(A=QR\).
- Rank via SVD threshold \( \epsilon = \max(n,m)\,\sigma_{\max}\,\mathrm{eps} \).
- Geometric images of the unit square/cube reveal shear/rotation/scaling and orientation flips.

## Pseudocode
See the "Algorithm Design" section in the project brief and docstrings in `model.py`.

## Usage
```bash
# install
pip install -r requirements.txt

# run a quick 2D demo
python simulate.py
python viz.py
```

## Outputs
- `results/report.json` — rank, tol, determinant, condition number.
- `figs/square_transform.png`, `figs/circle_to_ellipse.png`, `figs/cube_transform.png`.

## Notes
- All routines are NumPy-based and avoid SciPy requirements. Swap in `scipy.linalg` for production-grade LU/QR if desired.
