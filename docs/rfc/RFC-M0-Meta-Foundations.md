# RFC M0 — Meta Evolution Foundations

**Status**: Draft  
**Author**: Meta Engineering Program  
**Audience**: Platform governance, GPU/quantum engineering, compliance  
**Created**: 2025-10-22

---

## 1. Problem Statement

PRISM-AI requires an adaptive meta layer that can:

1. Generate, evaluate, and mutate architectural variants deterministically.
2. Anchor every variant decision to a governed ontology.
3. Provide reflexive feedback (free-energy lattice) before production rollout.

Mission objectives demand that these capabilities ship without sacrificing deterministic reproducibility, governance enforcement, or observability. Phase M0 delivers the scaffolding—feature flag registry, telemetry schema vNext, service surface—for higher phases (M1–M6).

---

## 2. Scope

| Component | Description | Owner |
|-----------|-------------|-------|
| Meta Feature Registry | Merkle-backed registry for MEC feature flags (`meta_generation`, `ontology_bridge`, etc.). | Core Platform |
| Telemetry vNext | Schema extension capturing variant genome hashes, ontology digests, and free-energy lattice metrics. | Telemetry Guild |
| Service Skeletons | High-assurance scaffolds for `meta::orchestrator` and `meta::ontology` modules. | MEC Program |
| Compliance Hooks | Validator updates ensuring flags + determinism manifest parity. | Governance Ops |

Out of scope: actual variant evolution algorithms (Phase M1+) and ontology population (Phase M2+).

---

## 3. Architecture Overview

```
┌──────────────────────────────┐
│        Meta Registry         │
│  (meta/meta_flags.json +     │
│   Merkle Ledger)             │
└──────────────┬───────────────┘
               │
               │ emits manifests + telemetry
               ▼
┌──────────────────────────────┐
│     Meta Orchestrator        │
│  (Phase M1 implementation)   │
├──────────────────────────────┤
│ - Genome definition          │
│ - Variant selection pipeline │
│ - Determinism replay hooks   │
└──────────────┬───────────────┘
               │
               │ consults ontology / telemetry
               ▼
┌──────────────────────────────┐
│       Ontology Bridge        │
│  (Phase M2 implementation)   │
├──────────────────────────────┤
│ - Semantic embeddings        │
│ - Governance approvals       │
│ - Change audit trail         │
└──────────────────────────────┘
```

Phase M0 lays down the registry + telemetry to support the orchestrator and ontology services.

---

## 4. Feature Flag Model

| Flag | Semantics | Activation Rule |
|------|-----------|------------------|
| `meta_generation` | Allows MEC orchestrator to produce variants. | Requires determinism replay + governance approval. |
| `ontology_bridge` | Enforces ontology alignment per variant. | Requires `meta_generation` + ontology approval checksum. |
| `free_energy_snapshots` | Publishes reflexive lattice metrics. | Requires `meta_generation` + telemetry evaluations. |
| `semantic_plasticity` | Enables representation learners & explainability. | Requires `ontology_bridge`. |
| `federated_meta` | Allows distributed MEC replicas. | Requires `meta_generation` & `ontology_bridge`. |
| `meta_prod` | Full production enablement. | Requires all previous flags + H-I-L vote. |

Each transition is Merkle-audited, monotonic, and logged to telemetry.

---

## 5. Telemetry Schema Delta

New JSON schema (`telemetry/schema/meta_v1.json`) captures:

```json
{
  "meta_variant": {
    "genome_hash": "sha256",
    "parent_hash": "sha256",
    "determinism_manifest": "sha256",
    "flags": ["meta_generation", "..."],
    "free_energy": {
      "lattice_norm": "float32",
      "mode_confidence": "float32",
      "divergence": "float32"
    },
    "ontology": {
      "concept_id": "string",
      "version": "sha256"
    }
  }
}
```

Telemetry durability (P1-11) ensures fsync + missing stage alerts cover the new schema as well.

---

## 6. Compliance Integration

1. **Validator**: `scripts/compliance_validator.py` will require:
   - `meta_generation` flag disclosure when MEC modules are built.
   - determinism manifests to report `meta_variant_hash`, `ontology_hash`, `free_energy_hash`.
2. **Sprint Gates**: `SPRINT-GATES.md` update adds `MECEnabled` gate triggered when `meta_generation` flips to shadow/gradual/enable.
3. **Reset Script**: `scripts/reset_context.sh` persists meta flag snapshots after each run.

---

## 7. Implementation Plan

1. Implement `src/features/meta_flags.rs` (Merkle-backed registry).
2. Introduce telemetry schema file and log pipeline.
3. Scaffold `src/meta/orchestrator` & `src/meta/ontology` with deterministic APIs and placeholder integration tests.
4. Update validators + reset scripts to surface meta state.

Every step ships with determinism manifests and compliance reports stored under `artifacts/mec/M0`.

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Registry corruption | Orchestrator disabled / inconsistent | Merkle verification & append-only ledger |
| Flag misuse | Governance bypass | Monotonic transitions + compliance validator |
| Telemetry volume | Storage pressure | Compression + fsync cadence from P1-11 |
| Human-in-the-loop delay | Slower rollout | Shadow + gradual states with audit evidence |

---

## 9. Open Questions

1. Should `meta_prod` require multi-party approval (H-I-L quorum)? (Proposed: yes, specify in constitution addendum.)
2. How will ontology diffs be reviewed during M2? (Proposed workflow: RFC + governance vote.)
3. Do we need GPU residency for free-energy lattice instrumentation? (Research in Phase M3.)

---

## 10. Approval

| Role | Name | Status |
|------|------|--------|
| Platform Engineering | TBD | Pending |
| Governance Council | TBD | Pending |
| Compliance | TBD | Pending |
| Observability | TBD | Pending |

Once approved, Phase M0 implementation may commence.
