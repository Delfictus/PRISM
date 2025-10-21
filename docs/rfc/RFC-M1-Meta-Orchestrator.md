# RFC M1 — Meta-Orchestrator MVP

**Status**: Draft  
**Author**: Meta Engineering Program  
**Audience**: GPU/quantum leads, neuromorphic leads, governance council  
**Created**: 2025-10-22

---

## 1. Objective

Deliver the first operational release of the Meta Evolutionary Compute (MEC) orchestrator. The orchestrator must:

1. Deterministically generate variant genomes using low-discrepancy sampling (Halton sequence + Sobol refinement).
2. Evaluate variants via a multi-objective functional capturing thermodynamic energy, chromatic optimality, and free-energy divergence.
3. Select the next generation using replicator dynamics with adaptive temperature control and KL-regularized exploitation.
4. Emit determinism manifests with meta hashes (`meta_genome_hash`, `meta_merkle_root`, etc.) and compliance telemetry (`meta_variant` schema).

The implementation must be algorithmically rigorous—no simplified heuristics.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│      MetaOrchestrator (Phase M1 Runtime)                   │
│                                                            │
│  1. Variant Sampling                                       │
│     - Halton base generation                               │
│     - Sobol low-discrepancy refinement                     │
│     - Deterministic jitter seeded per meta flag manifest   │
│                                                            │
│  2. Simulation & Scoring                                   │
│     - Thermodynamic proxy (Ising-like energy)              │
│     - Chromatic loss proxy (statistical surrogate)         │
│     - Free-energy deviation (KL divergence vs. baseline)   │
│                                                            │
│  3. Selection                                              │
│     - Replicator dynamics (continuous-time discretization) │
│     - KL penalty to canonical genome                       │
│     - Adaptive temperature annealing                       │
│                                                            │
│  4. Artifact Emission                                      │
│     - Determinism manifest with meta hashes                │
│     - Telemetry `meta_variant` manifest                    │
│     - Compliance summary for Phase M1                      │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Variant Sampling

### 3.1 Halton + Sobol Hybrid

- Base sequence: 5-D Halton (primes `[2,3,5,7,11]`).
- Refinement: nested Sobol points to break correlation when `population > 32`.
- Deterministic jitter: XOR-shift on base seed combined with meta Merkle root.

### 3.2 Genome Encoding

| Parameter Key              | Range / Choices            | Description                                    |
|----------------------------|----------------------------|------------------------------------------------|
| `annealing.beta`           | [0.25, 5.0] (continuous)    | Governs simulated annealing schedule           |
| `ensemble.replicas`        | [64, 4096] (discrete)       | Ensemble size                                  |
| `fusion.strategy`          | {density\_aware, thermo, quantum\_bias, neuromorphic\_phase} | Fusion selector  |
| `refinement.iterations`    | [512, 8192] (discrete)      | Gradient refinement steps                      |
| `mutation.strength`        | [0.01, 0.45] (continuous)   | Exploration amplitude                          |
| `mutation.temperature`     | [0.75, 1.75] (continuous)   | Dynamic temperature parameter                  |
| `tensor_core_prefetch`     | Bool (probabilistic)        | Enables WMMA prefetch within dense path        |
| `use_quantum_bias`         | Bool                         | Activates quantum bias heuristics              |
| `enable_neuromorphic_feedback` | Bool                    | Hooks neuromorphic feedback loop               |

Hash: `SHA256(seed || parameters || toggles)` for deterministic IDs.

---

## 4. Evaluation Function

### 4.1 Surrogate Metrics

Because variants cannot call the full GPU pipeline yet, we employ deterministic surrogates:

1. **Thermodynamic energy**: simulate an Ising-like lattice using Metropolis-Hastings seeded by genome hash.
2. **Chromatic surrogate**: evaluate variant parameters against a statistical surrogate trained on historical runs (stored coefficients, deterministic evaluation).
3. **Free-energy divergence**: compute KL divergence between variant distribution and canonical baseline.

### 4.2 Multi-objective Aggregation

Score vector:  
`S = ( -E_normalized, -ChromaticLoss, -KL_divergence )`  
Weights adapt via trust-region update based on replicator payoffs.

Deterministic aggregator:  
`scalar = wᵀS - λ * ||θ - θ₀||²`  
where `θ` is vector of parameters, `θ₀` canonical baseline, `λ` KL penalty coefficient.

---

## 5. Selection Algorithm

### 5.1 Replicator Dynamics

Update rule for population probability `p_i`:

```
p_i(t+1) = p_i(t) * exp(η * (score_i - ⟨p, score⟩))
normalize(p)
```

- Learning rate `η` adapts based on variance of scores; clamp to [1e-3, 0.2].
- After update, Boltzmann resampling at temperature `τ` yields next generation seeds.

### 5.2 Deterministic Reseeding

Seed of child genome: `mix(seed_parent, base_seed, generation_index)` using SHA256.
If meta feature `free_energy_snapshots` enabled, enforce conservative step size to respect divergence constraints.

---

## 6. Determinism & Telemetry

### 6.1 Manifest Attachment

`MetaDeterminism` struct attaches:

- `meta_genome_hash`: best genome hash from selection.
- `meta_merkle_root`: from feature registry snapshot.
- `ontology_hash`: optional (populated after Phase M2).
- `free_energy_hash`: optional (Phase M3).

Recorder interface:

```rust
let mut recorder = DeterminismRecorder::new(master_seed);
recorder.record_input(&plan)?;
recorder.record_intermediate("meta.selection", &selection_state)?;
recorder.record_output(&best_genome)?;
recorder.attach_meta(meta_det);
let proof = recorder.finalize();
```

### 6.2 Telemetry Emission

Telemetry sink writes JSON lines conforming to `telemetry/schema/meta_v1.json`:

```json
{
  "meta_variant": {
    "genome_hash": "...",
    "determinism_manifest": "...",
    "flags": ["meta_generation", "ontology_bridge"],
    "free_energy": {
      "lattice_norm": 0.82,
      "mode_confidence": 0.91,
      "divergence": 0.07
    },
    "ontology": {
      "concept_id": "chromatic_phase",
      "version": "..."
    }
  }
}
```

---

## 7. Compliance & Artifacts

1. `meta_bootstrap` CLI emits baseline manifest (`artifacts/mec/M1/meta_manifest.json`).
2. Reset script prints registry snapshot to console.
3. Compliance validator ensures Merkle roots, entropy hash, and manifest presence.

Artifacts:

- `artifacts/mec/M1/evolution_plan.json`
- `artifacts/mec/M1/selection_report.json`
- `artifacts/mec/M1/determinism_manifest_meta.json`

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Surrogate drift vs. real GPU pipeline | Validate surrogates against historical metrics, enforce variance bounds. |
| Replicator instability | Adaptive η + temperature clamping, verify with deterministic unit tests. |
| Telemetry volume | Reuse durable sink, compress meta payload. |

---

## 9. Approval

| Role | Name | Status |
|------|------|--------|
| Platform Engineering | TBD | Pending |
| Governance Council | TBD | Pending |
| Compliance | TBD | Pending |
| Observability | TBD | Pending |

Once approved, proceed with implementation in `src/meta/orchestrator/`.
