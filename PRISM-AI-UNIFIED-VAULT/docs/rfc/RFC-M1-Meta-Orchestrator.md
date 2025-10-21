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

## Deliverables

- `src/meta/registry.rs`
- `src/meta/orchestrator/mod.rs`
- `artifacts/mec/M1/selection_report.json`
- `src/bin/meta_bootstrap.rs`
- Pipeline wiring in `scripts/run_full_check.sh`

## Acceptance Criteria

- Orchestrator runs deterministic twin executions with matching hashes.
- CI pipeline blocks merges without up-to-date manifest + selection reports.
- Telemetry logger captures stage coverage for meta phases.

