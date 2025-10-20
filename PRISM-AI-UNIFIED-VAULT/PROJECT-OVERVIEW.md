# **PRISM-AI PROJECT OVERVIEW**

This overview stays pinned for quick orientation across the entire development timeline. It links the governance contracts, phased roadmap, and automation tooling now in the vault.

---

## **Key Artifacts**

- **Roadmap**: `05-PROJECT-PLAN/MULTI-PHASE-TODO.md`
- **Task Manifest**: `05-PROJECT-PLAN/tasks.json`
- **Task Monitor**: `scripts/task_monitor.py`
- **Governance Validator**: `scripts/compliance_validator.py`
- **Master Executor**: `03-AUTOMATION/master_executor.py`
- **Continuous Monitor**: `scripts/continuous_monitor.sh`

---

## **How to Track Progress**

```bash
# View task summary (all phases)
python3 scripts/task_monitor.py --once

# Filter by phase or status
python3 scripts/task_monitor.py --phase phase1 --status pending

# Watch tasks and run compliance in strict mode every 5 minutes
python3 scripts/task_monitor.py --watch 300 --run-compliance --strict
```

- Update task statuses directly in `tasks.json` (`pending`, `in_progress`, `blocked`, `done`).
- The monitor validates statuses and can trigger the compliance validator on demand.

---

## **Phase Snapshot**

| Phase | Focus | Exit Criteria (excerpt) |
|-------|-------|-------------------------|
| Phase 0 | Governance foundations | A-DoD gates active, automation scripts deployed |
| Phase 1 | Harden (Adjustments) | Determinism replay, benchmark manifest, device guards, telemetry contract |
| Phase 2 | Optimize | Advanced GPU kernels, CUDA Graph capture, Nsight roofline proof |
| Phase 3 | Learn | RL/ADP + GNN integration, adaptive fusion, learning uplift ≥10% |
| Phase 4 | Explore | World-record DSJC1000.5 ≤ 82 colors, reproducibility audit |
| Phase 5 | Continuous Ops | Monitoring, determinism rotation, benchmark upkeep |

---

## **Compliance Workflow**

1. Modify implementation according to the roadmap tasks.
2. Update `tasks.json` to reflect progress.
3. Run `python3 03-AUTOMATION/master_executor.py --strict` for a full governed execution.
4. Capture reports in `reports/`, `artifacts/`, and `audit/` directories.

This document should remain up to date for the entire multi-phase development effort.
