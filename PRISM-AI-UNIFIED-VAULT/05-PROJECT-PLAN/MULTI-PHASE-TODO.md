# **PRISM-AI MULTI-PHASE IMPLEMENTATION ROADMAP**
## **Active Task Tracker (A-DoD Compliant)**

This roadmap decomposes the unified vault directives into ordered phases. Each task references its source contract and is mirrored in `tasks.yaml` for programmatic tracking.

---

## **Phase 0 · Governance Foundations**

| # | Task | Source | Status |
|---|------|--------|--------|
| 0.1 | Finalize AdvancedDoD, Roofline, Ablation, DeviceGuard, Protein gates in governance engine | `01-GOVERNANCE/AUTOMATED-GOVERNANCE-ENGINE.md` | done |
| 0.2 | Wire CUDA Graph capture + persistent kernel bitmap export into telemetry | `00-CONSTITUTION`, Article X §10.5 | done |
| 0.3 | Stand up compliance automation (master executor + validator + monitor scripts) | `03-AUTOMATION/AUTOMATED-EXECUTION.md` | done |
| 0.4 | Publish determinism manifest schema + governance dashboard panels | `00-CONSTITUTION`, Article X §10.7 | done |

---

## **Phase 1 · Sprint 1 HARDEN (Remove Limits)**

Ordered tasks address the six production gaps, then high-leverage upgrades.

| # | Task | Source | Status |
|---|------|--------|--------|
| 1.1 | Implement determinism replay system with Merkle audit trail | `04-ADJUSTMENTS/DETERMINISM-REPLAY.md` | done |
| 1.2 | Create benchmark manifest with checksums & regression gates | `04-ADJUSTMENTS/BENCHMARK-MANIFEST.md` | done |
| 1.3 | Add device-aware guards & path decision telemetry | `04-ADJUSTMENTS/DEVICE-AWARE-GUARDS.md` | done |
| 1.4 | Enforce unified telemetry contract across adapters | `04-ADJUSTMENTS/TELEMETRY-CONTRACT.md` | done |
| 1.5 | Activate sprint gate feature-lock automation | `04-ADJUSTMENTS/SPRINT-GATES.md` | ☐ pending |
| 1.6 | Ship protein acceptance harness + AUROC thresholds | `04-ADJUSTMENTS/PROTEIN-TESTS.md` | ☐ pending |
| 1.7 | Enable WMMA pad-and-scatter safeguards (Tensor Core) | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |
| 1.8 | Bridge CI gates to governance violation pipeline | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |
| 1.9 | Embed determinism manifest in all result payloads | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |
| 1.10 | Harden protein numerics (AUROC ≥0.7) | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |
| 1.11 | Add telemetry durability (fsync + alerts) | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |

Exit criteria: no hard limits, ≥64-color support, dense path guarded, determinism replay passes.

---

## **Phase 2 · Sprint 2 OPTIMIZE (Performance)**

| # | Task | Source | Status |
|---|------|--------|--------|
| 2.1 | Implement sparse GPU coloring upgrades (warp bitonic, 128-bit masks, Philox RNG) | `00-CONSTITUTION`, Article X §10.3 | ☐ pending |
| 2.2 | Deliver WMMA dense path with feasibility guards + residual validation | `00-CONSTITUTION`, Article X §10.3 | ☐ pending |
| 2.3 | Capture CUDA Graph for ensemble→fusion→coloring→SA | `00-CONSTITUTION`, Article X §10.2 | ☐ pending |
| 2.4 | Persist work-stealing attempts kernel & device reduction | `00-CONSTITUTION`, Article X §10.2 | ☐ pending |
| 2.5 | Integrate SA tempering pipelines + swap telemetry | `00-CONSTITUTION`, Article X §10.4 | ☐ pending |
| 2.6 | Implement diffusion label propagation acceptance logic | `00-CONSTITUTION`, Article X §10.4 | ☐ pending |
| 2.7 | Validate Nsight roofline metrics (occupancy, SM eff., bandwidth/FLOP) | `01-GOVERNANCE` gates | ☐ pending |

Exit criteria: ≥2× speedup, determinism variance ≤10%, memory peak ≤8 GB.

---

## **Phase 3 · Sprint 3 LEARN (RL/GNN Integration)**

| # | Task | Source | Status |
|---|------|--------|--------|
| 3.1 | Connect RL AdaptiveDecisionProcessor with GPU telemetry feedback | `foundation/adp` | ☐ pending |
| 3.2 | Embed GNN predictions (ColoringGNN) into ensemble priorities | `src/cma/neural.rs` | ☐ pending |
| 3.3 | Implement adaptive coherence fusion (projected gradient weights) | `00-CONSTITUTION`, Article X §10.3 | ☐ pending |
| 3.4 | Build online hyperparameter tuning loop driven by determinism manifests | `00-CONSTITUTION`, Article X §10.7 | ☐ pending |
| 3.5 | Validate learning uplift (≥10% improvement on held-out set) | Sprint 3 gates | ☐ pending |

---

## **Phase 4 · Sprint 4 EXPLORE (World Record Push)**

| # | Task | Source | Status |
|---|------|--------|--------|
| 4.1 | Enable world-record mode (extreme optimization flags) | `03-AUTOMATION/AUTOMATED-EXECUTION.md` | ☐ pending |
| 4.2 | Integrate quantum PIMC + neuromorphic consensus for DSJC1000.5 | `02-IMPLEMENTATION/MODULE-INTEGRATION.md` | ☐ pending |
| 4.3 | Scale distributed ensemble attempts (replica exchange telemetry) | `00-CONSTITUTION`, Article X §10.3 | ☐ pending |
| 4.4 | Run protein voxel overlays alongside coloring for bio benchmarks | `00-CONSTITUTION`, Article X §10.3 | ☐ pending |
| 4.5 | Produce reproducible world-record package (manifest + audit) | `01-GOVERNANCE` + `03-AUTOMATION` | ☐ pending |

Exit criteria: DSJC1000.5 ≤ 82 colors, reproducibility certified.

---

## **Phase 5 · Continuous Operations & Compliance**

| # | Task | Source | Status |
|---|------|--------|--------|
| 5.1 | Deploy continuous monitor with governance dashboard updates | `scripts/continuous_monitor.sh` | ☐ pending |
| 5.2 | Automate determinism replay daily + publish variance reports | `04-ADJUSTMENTS/DETERMINISM-REPLAY.md` | ☐ pending |
| 5.3 | Maintain benchmark manifest & regression history | `04-ADJUSTMENTS/BENCHMARK-MANIFEST.md` | ☐ pending |
| 5.4 | Sustain telemetry durability audits (fsync alerts) | `04-ADJUSTMENTS/HIGH-LEVERAGE-IMPROVEMENTS.md` | ☐ pending |
| 5.5 | Keep protein acceptance suite in weekly CI rotation | `04-ADJUSTMENTS/PROTEIN-TESTS.md` | ☐ pending |

---

### **How to Update**

1. Modify statuses in `05-PROJECT-PLAN/tasks.yaml` (`pending` → `in_progress` → `done`).
2. Run `python3 03-AUTOMATION/master_executor.py --strict` for compliance.
3. View live status with `python3 scripts/task_monitor.py --summary`.

All changes are governed by the Advanced Definition of Done (Article X).
