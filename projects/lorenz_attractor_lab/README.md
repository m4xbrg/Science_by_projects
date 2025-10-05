# Lorenz Attractor Lab: Integration, Poincaré Sections, and Sensitivity to Initial Conditions

See the main chat for the structured writeup. This repo follows: Title → Math → Pseudocode → Viz → Reflection → Integration.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python simulate.py --config config.yaml --out results.parquet --format parquet  # or: --format csv
python viz.py --infile results.parquet --plane x=0 --direction increasing
```
