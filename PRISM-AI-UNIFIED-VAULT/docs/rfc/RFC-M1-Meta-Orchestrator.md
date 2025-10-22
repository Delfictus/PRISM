# PRISM-AI RFC: Meta Orchestrator MVP (Phase M1)

**Status:** Draft  
**Owner:** Meta Evolution Lead  
**Last Updated:** 2025-10-21

## Context

Phase M1 builds on M0 by delivering the first working version of the meta orchestrator. The orchestrator manages candidate generation, evaluation, and selection cycles under strict determinism.

## Goals

1. Implement the meta variant registry and genome scaffold.
2. Provide deterministic evaluation loops with scoring heuristics.
3. Persist determinism manifests and telemetry payloads specific to meta runs.
4. Integrate the `ci-meta-orchestrator` pipeline and governance hooks.
5. Offer a bootstrap CLI for initializing orchestrator worktrees.
6. Emit reflexive feedback lattice snapshots and hash them into determinism metadata.

## Deliverables

- `src/meta/registry.rs`
- `src/meta/orchestrator/mod.rs`
- `src/meta/reflexive/mod.rs`
- `artifacts/mec/M1/selection_report.json`
- `artifacts/mec/M3/lattice_report.json`
- `src/bin/meta_bootstrap.rs`
- Pipeline wiring in `scripts/run_full_check.sh`

## Acceptance Criteria

- Orchestrator runs deterministic twin executions with matching hashes and emits a reflexive lattice snapshot whose hash is mirrored in the determinism manifest.
- CI pipeline blocks merges without up-to-date manifest + selection reports + lattice report.
- Telemetry logger captures stage coverage for meta phases and records the current reflexive mode (`strict`/`explore`/`recovery`).

## Reflexive Feedback Telemetry

- `MetaOrchestrator::run_generation` attaches a `ReflexiveController` that produces a lattice snapshot, history, and mode classification.
- `MetaDeterminism` stores `lattice_hash`, `reflexive_mode`, `lattice_stability`, and `lattice_entropy` alongside ontology metadata.
- Governance gate `ReflexiveFeedbackGate` validates that the determinism manifest matches `artifacts/mec/M3/lattice_report.json` before approving meta phase promotions.
