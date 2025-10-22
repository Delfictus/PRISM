# Meta Rollout Runbook

**Purpose:** Guide the staged enablement and rollback of the Meta Evolution Cycle (MEC) in production environments.

## Prerequisites
- Phase M0–M4 tasks are complete and passing compliance.
- Governance approvals recorded in `01-GOVERNANCE/META-GOVERNANCE-LOG.md`.
- Determinism manifests and Merkle anchors published under `artifacts/mec/`.

## Rollout Checklist
1. Verify `meta_generation` feature flag remains disabled in production.
2. Run `python3 PRISM-AI-UNIFIED-VAULT/scripts/run_full_check.sh --strict`.
3. Execute `python3 PRISM-AI-UNIFIED-VAULT/03-AUTOMATION/master_executor.py phase --name M6 --strict`.
4. Audit cognitive ledger: `python3 PRISM-AI-UNIFIED-VAULT/scripts/ledger_audit.py --block <latest_hash>`.
5. Review dashboards (governance, telemetry) for regressions.
6. Obtain sign-off from governance lead and SRE.

## Knowledge Evolution (Phase M4)
- Extract telemetry into a representation dataset: `python3 PRISM-AI-UNIFIED-VAULT/meta/representation/pipelines/extract_telemetry.py --input telemetry/semantic_plasticity.jsonl --output PRISM-AI-UNIFIED-VAULT/meta/representation/dataset.json`
- Generate semantic plasticity explainability report: `python3 PRISM-AI-UNIFIED-VAULT/03-AUTOMATION/master_executor.py --skip-build --skip-tests --skip-benchmarks --strict`
- Review `artifacts/mec/M4/explainability_report.md` for drift status of ontology adapters.
- Confirm `tests/meta/semantic_plasticity.rs` passes locally before promoting Phase M4 changes.
- Ensure `scripts/compliance_validator.py --strict` reports `plasticity:*` checks as `PASS`.
- Investigate any `plasticity:warning_concepts` finding (drift approaching threshold) and block promotion if `plasticity:drifted_concepts` is raised.

## Rollback Plan
- Use `master_executor.py --rollback M6` to revert to prior Merkle anchor.
- Disable `meta_*` feature flags via `meta/meta_flags.json`.
- Restore previous determinism manifests from `artifacts/mec/M5`.
- Document actions in `META-GOVERNANCE-LOG.md` with `ROLLBACK` entry.

## Contacts
- Meta Evolution Lead
- Governance Engineering
- SRE On-call
