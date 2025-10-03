# Portfolio Ontology Schema

**Per-project fields** (stored in `projects/*/meta.json` and indexed in `ontology/master_index.csv`):

- `id` (string): unique short ID (e.g., "001").
- `title` (string): project title.
- `domain` (enum): physics | biology | chemistry | math | computing | earth_science | astronomy | interdisciplinary | ethics.
- `math_core` (enum): ODE | PDE | probability | optimization | graph | linear_algebra | calculus | statistics | simulation.
- `tools` (list[str]): libraries or engines used.
- `viz_types` (list[str]): e.g., time series | phase portrait | heatmap | network | histogram.
- `related_projects` (list[str]): list of project IDs.