# Semantic Plasticity Explainability Report

## Adapter Overview
- Adapter ID: `semantic_plasticity_m4`
- Embedding dimension: 4
- Concepts tracked: 3
- Mode: warmup

## Concept Metrics
| Concept | Status | Cosine | Magnitude Ratio | ΔL2 | Observations | Anchor |
|---------|--------|--------|-----------------|-----|-------------|--------|
| `concept://coloring` | stable | 0.998 | 1.054 | 0.062 | 2 | 20e711be932f9bf35aecc9175fb992491c85a3ddb13d3f5a8f52bcae2466e004 |
| `concept://explainer` | stable | 0.990 | 1.036 | 0.091 | 2 | 2cf74ada6bb575866cd0da12bd8d2b34a52fe19cb6b10a03442d27e501cd45f5 |
| `concept://ontology` | stable | 0.994 | 1.121 | 0.039 | 2 | 724496f913624e8df44d6628878f5c22618e6aaadcb1ec4dee12a0411388e6ef |

## Recent Adaptation Events
- `concept://coloring` @ 1751356000000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | baseline snapshot | source=telemetry:orchestrator
- `concept://ontology` @ 1751356005000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | bridge alignment | source=telemetry:ontology
- `concept://coloring` @ 1751359600000 → status: stable, cosine=0.998, magnitude_ratio=1.054, ΔL2=0.062
  - notes: baseline alignment maintained | post-coherence adjustment | source=telemetry:orchestrator
- `concept://ontology` @ 1751359610000 → status: stable, cosine=0.994, magnitude_ratio=1.121, ΔL2=0.039
  - notes: baseline alignment maintained | alignment refresh | source=telemetry:ontology
- `concept://explainer` @ 1751360000000 → status: stable, cosine=1.000, magnitude_ratio=1.000, ΔL2=0.000
  - notes: baseline alignment maintained | attention rollout | source=telemetry:explainability
- `concept://explainer` @ 1751363600000 → status: stable, cosine=0.990, magnitude_ratio=1.036, ΔL2=0.091
  - notes: baseline alignment maintained | drift remediation | source=telemetry:explainability
