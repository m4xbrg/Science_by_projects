# Transformations Playground — 2D Affine Operations on Polygons

## 1. Title & Abstract
See repository title and summary above.

## 2. Mathematical Formulation
- Homogeneous 2D points; affine 3×3 matrices.
- Ops: translation, rotation, reflection (Householder), composition.
- See `model.py` docstrings for equations → code mapping.

## 3. Algorithm Design (Pseudocode)
- Build matrices from op specs; compose; apply to vertices; record steps.
- Complexity: O(Nk); space O(N) (+ O(Nk) if recording).

## 4. Python Implementation
- `model.py`: Transform2D + pipeline.
- `simulate.py`: runs pipeline; writes `artifacts/results.parquet` & `A.npy`.
- `viz.py`: overlay plot; deformation grid.

## 5. Visualization
- Figures written to `artifacts/overlay.png` and `artifacts/deformation_grid.png`.

## 6. Reflection
- See below for assumptions, limitations, and extensions.

## 7. Integration
- See `meta.json` for ontology tags; `requirements.txt` lists dependencies.

### Quickstart
```bash
pip install -r requirements.txt
python simulate.py
python viz.py --cfg config.yaml
```
