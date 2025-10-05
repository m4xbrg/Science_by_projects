# Analytic Geometry Toolkit

**Purpose:** Compute distances, midpoints, and line/circle intersections in \(\mathbb{R}^2\); plot scenes and distance matrices.

## 1. Title & Abstract
See `meta.json` for structured metadata.

## 2. Mathematical Formulation
- Points: (x, y). Lines: ax + by + c = 0 (normalized). Circles: (x-x0)^2 + (y-y0)^2 = r^2.
- Core equations implemented in `model.py` with numerical tolerance `epsilon` from `config.yaml`.

## 3. Algorithm Design (Pseudocode)
- Inputs: points, lines, circles (see `config.yaml`).
- Processes: compute distances/midpoints; L–L / L–C / C–C intersections; assemble tables.
- Outputs: `results.parquet` (distances & intersections), figures in `figs/`.
See `simulate.py` for the pipeline.

## 4. Python Implementation
- `model.py`: clean API for primitives; deterministic, closed-form computations.
- `simulate.py`: loads config, runs computations, writes Parquet.
- `viz.py`: builds the two plots into `figs/`.

## 5. Visualization
- `scene.png`: objects + intersections.
- `distance_matrix.png`: pairwise point distances heatmap.

## 6. Reflection
- Assumptions: Euclidean 2D; floating-point tolerance; lines are infinite.
- Limitations: no segments/rays; near-degeneracies depend on epsilon.
- Extensions: segments/rays; robust predicates; 3D generalization; shapely/GEOS interop.

## 7. Integration
- Ontology tags in `meta.json`.
- Parameters in `config.yaml`.
- Requirements in `requirements.txt`.
