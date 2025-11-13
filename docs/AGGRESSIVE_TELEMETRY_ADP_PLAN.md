# Aggressive Telemetry-Guided ADP Loop Plan

**Goal:** Build a closed-loop system that (1) captures detailed telemetry, (2) diagnoses when a phase is underperforming, (3) proposes aggressive yet safe parameter adjustments, (4) injects those changes via ADP in real time, and (5) restarts the full pipeline using the latest best solution as the new baseline until convergence stalls.

---

## 1. System Overview

```
┌──────────────┐    metrics     ┌──────────────────┐    actions     ┌───────────────────────┐
│  Pipeline    │──────────────▶│ Telemetry Engine  │──────────────▶│ Aggressive ADP Engine │
│ (Phases 0-5) │               │  (analysis)       │               │ (controllers)         │
└──────┬───────┘               └──────────┬───────┘               └──────────┬────────────┘
       │                                   │ telemetry log                     │ control msgs
       │ best solution                     ▼                                  ▼
       │                              Knowledge Base                   Pipeline Controller
       │                                   │                                 │
       ▼                                   │                                 ▼
  Iterative Runner ◀───────────────────────┴──── best_solution ───────────────┘
   (restart loop)
```

Key building blocks:
- **Telemetry Engine:** High-frequency metrics capture + feature extraction + anomaly/underperformance detection.
- **Aggressive ADP Engine:** Decision layer that translates telemetry insights into parameter adjustments (thermo temps, TE weights, quantum batch sizes, etc.).
- **Pipeline Controller:** Applies adjustments via ADP hooks, persists best solutions, and triggers iterative restarts using the most recent best as initialization.

---

## 2. Telemetry Enhancements

### 2.1 Metrics Capture
Augment existing `RunMetric` with:
- `phase_progress` (0–1) for each phase.
- `conflict_gradient` (Δconflicts/Δtime).
- `chromatic_gradient`.
- `acceptance_rate` (phase-specific).
- `resource_usage` (GPU SM%, mem, CPU load).
- `param_snapshot` (key knobs for the phase).
- `best_so_far` (global best chromatic number).

Emit metrics at:
- Every TE iteration (after ordering).
- Each thermo temperature exchange.
- Every quantum reduction attempt.
- Memetic generation boundaries.

### 2.2 Feature Extraction
Telemetry engine computes rolling windows:
- `conflict_decay_half_life` per phase.
- `time_since_last_improvement`.
- `GPU_efficiency = conflicts_reduced / GPU_time`.
- `parameter_sensitivity` (approx derivative of conflicts wrt parameter change; computed via finite differences using telemetry history).

### 2.3 Underperformance Detection
Define rule sets:
- **Stalled Phase:** `time_since_last_improvement > threshold` *and* `conflict_gradient ≈ 0`.
- **Inefficient GPU use:** `GPU_efficiency < min_efficiency`.
- **Thermo stuck:** acceptance rate < target for multiple temperatures.
- **Quantum failure:** sequential reduction attempts fail to reduce colors.

Trigger events: `TelemetryEvent::PhaseStalled { phase, reason }`.

---

## 3. Aggressive ADP Engine

### 3.1 Inputs
- Telemetry events (stalls, low efficiency).
- Parameter sensitivity estimates.
- Config guardrails (min/max values, restart-required flags).

### 3.2 Action Catalogue
For each phase define aggressive adjustments with priorities:

| Phase | Action | Effect | Safety |
|-------|--------|--------|--------|
| TE | Increase TE weight, increase histogram bins, tighten geodesic blend | Re-order vertices more aggressively | Low |
| Thermo | Add replicas, expand temperature range, increase steps per temp | Improve escape probability | Medium (VRAM) |
| Quantum | Reduce target colors by 2, increase iterations, enlarge batch size | Force deeper search | High (time) |
| Memetic | Increase mutation rate, depth, population | Exploration | Medium |

Each action includes:
- Preconditions (phase stalled, GPU headroom).
- Cost estimate (runtime, VRAM).
- Expected benefit (from sensitivity model).
- Rollback conditions.

### 3.3 Decision Logic
Pseudo-code:
```rust
fn evaluate_actions(events: &[TelemetryEvent], state: &PhaseState) -> Vec<ADPAction> {
    events.iter()
        .filter_map(|event| match event {
            TelemetryEvent::PhaseStalled { phase, .. } => {
                let actions = ACTIONS_FOR_PHASE[phase]
                    .iter()
                    .filter(|a| a.preconditions_met(state))
                    .take(2); // top aggressive actions
                Some(actions.collect())
            }
            TelemetryEvent::LowEfficiency { phase, .. } => { /* ... */ }
            _ => None
        })
        .flatten()
        .collect()
}
```

- Score each action by `(expected_benefit / cost)` using historical sensitivity data.
- Pick the top-scoring action per phase, apply via ADP interface.
- Log action in telemetry with outcome (success/fallback).

### 3.4 Integration with ADP
- Extend ADP to accept external control messages:
  ```rust
  enum ADPControl {
      AdjustThermoTemps { delta: i32 },
      SetQuantumBatch { size: usize },
      SetTeWeight { value: f64 },
      ...
  }
  ```
- Aggressive ADP engine sends `ADPControl` via async channel.
- ADP module merges actions with its existing Q-learning policy (e.g., treat external actions as high-priority overrides with decay).

---

## 4. Iterative Pipeline Controller

### 4.1 Flow
1. Start pass with current configuration and optional initial solution.
2. During run, telemetry+ADP adjust parameters dynamically.
3. At pass end:
   - Compare `best_solution` to previous pass.
   - If improvement ≥ `iterative.min_delta` (e.g., 1 color), persist and schedule next pass.
   - If no improvement or max passes/time reached, stop.
4. On restart:
   - Use `best_solution` as initial coloring (skip TE reordering if desired).
   - Optionally tighten parameters based on last pass (e.g., narrower thermodynamic temps focusing near successful region).

### 4.2 Controller Implementation

```rust
struct IterativeController {
    current_solution: Option<ColoringSolution>,
    best_solution: Option<ColoringSolution>,
    pass: usize,
}

impl IterativeController {
    async fn run(&mut self, pipeline: &mut Pipeline) -> Result<()> {
        while self.pass < config.iterative.max_passes {
            let result = pipeline.run(self.current_solution.clone()).await?;
            telemetry.record_pass_summary(self.pass, &result);

            if self.should_continue(&result) {
                self.current_solution = Some(result.clone());
                self.best_solution = Some(result.clone());
                self.pass += 1;
            } else {
                break;
            }
        }
        Ok(())
    }
}
```

### 4.3 Parameter Reset Strategy

- After each pass, optionally reset certain parameters to defaults (e.g., TE weights) while keeping others (thermo temps) if they proved beneficial (use telemetry evidence).
- Provide config flags:
  ```toml
  [iterative]
  enabled = true
  max_passes = 3
  min_delta = 1
  reset_te = true
  carry_thermo = true
  ```

---

## 5. Knowledge Base & Learning

Maintain persistent store (SQLite/Parquet) of actions and outcomes:
- Columns: `run_id`, `pass`, `phase`, `action`, `params_before`, `params_after`, `delta_conflicts`, `delta_chromatic`, `duration`.
- Use this data to:
  - Update action benefit estimates.
  - Train lightweight recommendation models (e.g., random forest) to predict best action per context.
  - Provide GUI suggestions (“In similar runs, increasing thermodynamic replicas +8 yielded 2-color improvements”).

---

## 6. Safety & Guardrails

- Define max/min bounds per parameter; aggressive engine never exceeds them.
- Rate-limit actions per phase (e.g., no more than one aggressive change per 30 s).
- Provide “kill switch” to revert to defaults if metrics degrade.
- Log every adjustment with before/after values for audit.

---

## 7. Integration Steps

1. **Telemetry upgrades** (Section 2).
2. **Event detection module** producing `TelemetryEvent`s.
3. **Action catalogue** with metadata, guardrails, scoring functions.
4. **ADP control channel** in pipeline.
5. **Iterative controller** hooking into pipeline run loop.
6. **Knowledge base** persistence (SQLite).
7. **CLI/GUI hooks**:
   - Toggle aggressive mode on/off (`prism-config set hypertune.aggressive=true`).
   - Show action history and next suggested actions (GUI panel).
   - Allow manual approval if desired.

---

## 8. Acceptance Criteria

1. Telemetry logs include new diagnostics (gradients, acceptance rates) and flag stalls automatically.
2. Aggressive ADP engine applies parameter changes during runs; telemetry confirms effect (e.g., conflict drop).
3. Iterative controller completes multiple passes when enabled, each starting from last best solution.
4. Knowledge base records action/outcome pairs and can report which adjustments historically yielded improvements.
5. GUI/CLI surfaces action history and allows enabling/disabling aggressive mode.
6. Safety guards prevent runaway parameter values; system reverts if metrics worsen beyond threshold.

---

## 9. Future Enhancements

- Plug in Bayesian optimization or RL agent trained on telemetry history.
- Support distributed runs with per-node telemetry aggregation.
- Provide simulation mode to test aggressive strategies offline before applying live.

---
