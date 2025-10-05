# Inequality Regions in the Plane: Half-Space Intersection, Vertex Extraction, and Visualization

> **Universal structure:** Title → Math → Pseudocode → Python → Visualization → Reflection → Integration

## 1. Title & Abstract
This project computes feasible regions defined by 2D linear inequalities, extracts extreme points (vertices), and visualizes the region.
Domain: computational geometry / convex analysis. Outcomes: ordered vertex list, feasible polygon plot, and feasibility heatmap.

## 2. Mathematical Formulation
- Feasible set: \( \mathcal{F} = \bigcap_{i=1}^m \{(x,y) : a_i x + b_i y \le c_i \}\). ('>=' is canonicalized by multiplying both sides by -1.)
- Candidate vertices: intersections of boundary lines \(a_i x + b_i y = c_i\) that satisfy all inequalities.
- Ordering: convex hull (Andrew's monotone chain).
- Boundedness: shoelace area > 0.

**Symbols:** \(A\in\mathbb{R}^{m\times 2}\), \(b\in\mathbb{R}^m\) s.t. \(A[x\ y]^T \le b\). Tolerance \(\varepsilon\) for numeric decisions.

## 3. Algorithm (Pseudocode)
```
CANONICALIZE(ineq s.t. sense in {<=,>=}) -> (a,b,c) with <=
SOLVE_REGION(inequalities, bbox=None):
  C <- canonicalize all + optional bbox
  A,b <- stack(C)
  S <- {}
  for all pairs (i,j) in C:
    p <- line_intersection(i,j)
    if p and A p <= b + eps: S <- S ∪ {p}
  S <- unique(S, tol)
  if |S|<=2: hull <- S; bounded <- |S|>0
  else: hull <- convex_hull(S); bounded <- area(hull)>0
  return {A,b,hull,S,bounded}
```

## 4. Python Implementation
- `model.py` exposes:
  - `normalize_ineq`, `line_intersection`, `feasible_region_2d`, `plot_feasible_region`, `plot_feasibility_heatmap`.
- `simulate.py` reads `config.yaml`, computes region, writes:
  - `results/vertices.parquet`, `results/raw_points.parquet` (if any), and `results/meta.json`.
- `viz.py` renders:
  - `figs/feasible_region.png` and `figs/feasibility_heatmap.png`.

## 5. Visualization
- Plot 1: boundary lines + shaded feasible polygon + labeled vertices.
- Plot 2: feasibility heatmap (count of satisfied inequalities per grid point).

## 6. Reflection
Assumptions: linear constraints; deterministic; numeric tolerance for near-degeneracy.
Limitations: unbounded sets visualize as open half-planes unless a view window is chosen; strict inequalities not shown.
Extensions: objective overlays (LP), higher dimensions, unions of polyhedra, interactive sliders.

## 7. Integration
See `meta.json` for ontology tags. Update `config.yaml` to change inequalities and plot window.

---

### Quickstart
```bash
pip install -r requirements.txt
python simulate.py
python viz.py
```
Artifacts land in `results/` and `figs/`.
