# Semantic Plasticity Explainability Report

## Adapter Overview
- Adapter ID: `semantic_plasticity_m4`
- Embedding dimension: 4
- Concepts tracked: 3
- Mode: warmup

## Concept Metrics
| Concept | Status | Cosine | Magnitude Ratio | ΔL2 | Observations | Anchor |
|---------|--------|--------|-----------------|-----|-------------|--------|
| `concept://coloring` | stable | 0.998 | 1.042 | 0.062 | 2 | ec4758c503d8bb9938c480da2543b4ecfd5506b7b12793a8d957a67fc356fcd5 |
| `concept://explainer` | stable | 0.981 | 1.058 | 0.127 | 2 | 9725269d97c719dd95d9665d1640aa4477a03d2874e07da322def6a7c7061215 |
| `concept://ontology` | stable | 0.995 | 1.168 | 0.047 | 2 | 5a5d894981c2f9a97434b9773553dcfad2d95cd5d4bb98410ae955994901bc37 |

## Recent Adaptation Events
- `concept://coloring` @ 1750356000000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | post-coherence adjustment | source=telemetry:orchestrator
- `concept://ontology` @ 1750356005000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | bridge alignment refresh | source=telemetry:ontology
- `concept://explainer` @ 1750356010000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | attention rollout | source=telemetry:explainability
- `concept://coloring` @ 1750359600000 → status: stable, cosine=0.998, magnitude_ratio=1.042, ΔL2=0.062
  - notes: baseline alignment maintained | sparse-to-dense reconciliation | source=telemetry:orchestrator
- `concept://ontology` @ 1750359610000 → status: stable, cosine=0.995, magnitude_ratio=1.168, ΔL2=0.047
  - notes: baseline alignment maintained | semantic drift mitigation | source=telemetry:ontology
- `concept://explainer` @ 1750360000000 → status: stable, cosine=0.981, magnitude_ratio=1.058, ΔL2=0.127
  - notes: baseline alignment maintained | drift remediation | source=telemetry:explainability
