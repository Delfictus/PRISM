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

## Rollback Plan
- Use `master_executor.py --rollback M6` to revert to prior Merkle anchor.
- Disable `meta_*` feature flags via `meta/meta_flags.json`.
- Restore previous determinism manifests from `artifacts/mec/M5`.
- Document actions in `META-GOVERNANCE-LOG.md` with `ROLLBACK` entry.

## Contacts
- Meta Evolution Lead
- Governance Engineering
- SRE On-call
