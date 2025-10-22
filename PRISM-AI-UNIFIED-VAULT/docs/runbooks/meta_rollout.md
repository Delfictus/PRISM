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
4. **Federated signatures:**
   - `cargo run --bin federated_sim -- --verify-summary PRISM-AI-UNIFIED-VAULT/artifacts/mec/M5/simulations/epoch_summary.json --verify-ledger PRISM-AI-UNIFIED-VAULT/artifacts/mec/M5/ledger`
   - Repeat for any labeled scenarios (e.g., `edge_failover`, `validator_loss`).
   - Record summary signature + ledger Merkle root in `META-GOVERNANCE-LOG.md` entry.
5. Audit cognitive ledger: `python3 PRISM-AI-UNIFIED-VAULT/scripts/ledger_audit.py --block <latest_hash>`.
6. Review dashboards (governance, telemetry) for regressions.
7. Obtain sign-off from governance lead and SRE.

## Rollback Plan
- Use `master_executor.py --rollback M6` to revert to prior Merkle anchor.
- Disable `meta_*` feature flags via `meta/meta_flags.json`.
- Restore previous determinism manifests from `artifacts/mec/M5`.
- Re-validate federated signatures post-rollback (`federated_sim --verify-summary … --verify-ledger …`).
- Document actions in `META-GOVERNANCE-LOG.md` with `ROLLBACK` entry.

## Housekeeping
- After verification, remove transient build outputs (`deps/`, `libonnxruntime*`, `libprism_ai*`, `comprehensive_gpu_benchmark`) to keep the repository clean, then rerun `run_full_check.sh` so regenerated artifacts remain under version control only when needed.

## Contacts
- Meta Evolution Lead
- Governance Engineering
- SRE On-call
