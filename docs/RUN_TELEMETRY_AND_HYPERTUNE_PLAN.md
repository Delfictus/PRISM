# Run Telemetry, Hypertuning, and Adaptive Loop Plan

**Objective:** Build an end-to-end observability and adaptive-control layer for the PRISM pipeline that (1) streams live metrics for every phase, (2) surfaces them in both CLI and web dashboards, (3) feeds hypertuning/ADP logic that actively nudges phases toward lower chromatic numbers, and (4) enables multi-pass pipeline runs with improved initialization (beyond greedy) and optional iterative restarts.

---

## 1. Architecture Summary

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| Telemetry Publisher | Emit structured metrics/events from each phase | `telemetry::RunMetrics`, per-phase hooks |
| Storage & Transport | Persist and stream metrics | JSONL file, ring-buffer channel, optional WebSocket |
| CLI Dashboard | Terminal monitoring | `prism-config monitor` |
| Web Dashboard | Browser-based live view + history | Axum (Rust) or Actix server serving Tailwind/HTMX UI |
| Hypertuning Engine | Consume telemetry + adjust parameters mid-run | Enhanced ADP + heuristics |
| Initialization Strategies | Provide alternative starting solutions | Spectral ordering, multi-source heuristics |
| Iterative Pipeline Controller | Re-run pipeline using prior best solution | Pipeline orchestrator loop |

---

## 2. Telemetry System Design

### 2.1 Data Model

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct RunMetric {
    pub timestamp: DateTime<Utc>,
    pub phase: PhaseName,              // Enum: TE, Thermo, Quantum, Memetic, etc.
    pub step: String,                  // e.g., "thermo_temp_12", "quantum_target_112"
    pub chromatic_number: usize,
    pub conflicts: usize,
    pub gpu_mode: PhaseExecMode,       // gpu_success / cpu_fallback / cpu_disabled
    pub duration_ms: f64,
    pub parameters: serde_json::Value, // Hyper-params snapshot (temp, batch size, etc.)
    pub notes: Option<String>,         // Free-form info (e.g., ADP action taken)
}
```

Additional structs:
- `PhaseExecMode` reused from GPU status.
- `RunSummary` aggregate emitted at end (total time, best colors).

### 2.2 Publishers

- Wrap each major phase block with helper `telemetry::record(PhaseName::Thermo, &metrics)`.
- For inner loops (thermo temperatures, quantum color reductions, memetic improvements), emit per-step metrics when:
  - Conflicts drop
  - Chromatic number improves
  - GPU fallback occurs
  - ADP action applied

Implementation:
- `telemetry::TelemetryHandle` holds `crossbeam_channel::Sender<RunMetric>` and file writer.
- Instantiated at pipeline start; passed via `Arc<TelemetryHandle>` to phases.
- Non-blocking send (drop on overflow) to avoid slowing pipeline.

### 2.3 Persistence

- Append newline-delimited JSON to `target/run_artifacts/live_metrics.jsonl`.
- Optionally rotate files per run (timestamp-based).
- Keep an in-memory circular buffer (e.g., `VecDeque<RunMetric>` limited to 1,000 entries) for live dashboards/WebSocket.

### 2.4 Transport Options

1. **File-only (baseline):** CLI tails JSONL.
2. **WebSocket server:** Telemetry handle pushes to broadcaster (`tokio::sync::broadcast::Sender`). Web dashboard subscribes for live updates.
3. **REST endpoints:** Provide `/api/metrics/live` (last N entries) and `/api/metrics/history?run_id=` (load file).

---

## 3. CLI Monitoring

### 3.1 Command: `prism-config monitor`

Subcommands:
- `--tail`: stream latest metrics (similar to `tail -f`).
- `--summary`: show aggregated stats (best chromatic per phase, GPU usage).
- `--follow`: curses/TTY UI (optional).

Implementation steps:
1. Parse JSONL entries using `serde_json::from_str`.
2. Display as table:
   ```
   [TE][GPU] t=12.3s colors=118 conflicts=5600 params={"geodesic_weight":0.3}
   ```
3. Provide filtering (`--phase thermo`).

### 3.2 Alerts

- Add optional thresholds (e.g., `--alert conflicts>1000`) to beep/log when conditions met.

---

## 4. Web Dashboard

### 4.1 Stack

- Rust web server (Axum or Actix) running within PRISM binary (feature-gated `telemetry_web`).
- Static frontend: Tailwind CSS + HTMX or small React bundle.
- Live charts via WebSocket (Plotly.js or Chart.js).

### 4.2 Features

1. **Live Timeline Chart:** Chromatic number & conflicts per phase vs. time.
2. **Phase Status Grid:** GPU vs CPU indicator, latest parameters, duration.
3. **ADP Actions Feed:** Show last N ADP decisions with outcomes.
4. **Hyperparameter Panel:** Display current key params (thermo temps, quantum batch size).
5. **Run Controls (optional):** Buttons to pause/resume or trigger iterative restart (see Section 7).

### 4.3 API Endpoints

- `GET /` -> dashboard HTML.
- `GET /api/metrics/live` -> JSON of last 200 metrics.
- `WS /ws` -> push RunMetrics as JSON.
- `POST /api/control/restart` -> request new pipeline iteration (secured).

### 4.4 Deployment

- Run server on `localhost:7878` by default; allow `PRISM_DASH_PORT` override.
- Document how to enable: `PRISM_TELEMETRY_WEB=1 cargo run ...`.

---

## 5. Hypertuning & ADP Enhancements

### 5.1 Telemetry-Driven ADP

- Extend ADP loop to subscribe to telemetry stream, enabling:
  - Dynamic learning rate adjustments when conflict reductions stall.
  - Temperature ladder tweaking based on recent thermo improvement rate.
  - Quantum batch size adjustments if acceptance rate falls below threshold.

Implementation:
- `HypertuneController` listens to metrics channel.
- Maintains rolling windows (e.g., last 5 thermo temps). If no improvement, emits ADP action (increase temps, adjust memetic mutation).
- Actions logged in telemetry (`RunMetric.notes`).

### 5.2 Parameter Surrogates

- Fit lightweight regression/GP model on-the-fly using metrics to predict which parameter tweaks yield improvements.
- Use prior runs’ JSONL files to seed initial model (offline training).

### 5.3 Control Hooks

- `PhaseControl` trait exposing functions like `adjust_temperature(delta)`, `set_quantum_batch(size)`, `boost_memetic_mutation(rate)`.
- Phases check for pending control messages at safe points (e.g., between temps).

---

## 6. Alternative Initial Coloring Strategies

Current greedy initialization can be complemented with:

1. **Spectral Ordering:** Compute Laplacian eigenvectors (GPU or CPU) to order vertices before greedy.
2. **Community-aware ordering:** Use Louvain clustering to color dense subgraphs first.
3. **Randomized multi-start:** Run multiple greedy variants with different heuristics; pick best.
4. **Transfer learning:** (Optional) Load precomputed vertex embeddings (e.g., GraphSAGE) to guide ordering.

Implementation plan:
- Add `InitialColoringStrategy` enum to config (default `Greedy`).
- Implement strategies under `initial_coloring.rs`.
- Telemetry logs which strategy was used and its initial chromatic/conflicts.
- CLI command `prism-config set init.strategy=spectral`.

---

## 7. Iterative Pipeline Restarts

Goal: after completing all phases once, restart pipeline using best solution as new baseline to squeeze additional improvements.

### 7.1 Controller Workflow

1. Run pipeline (phases 0–5) as usual, producing `best_solution`.
2. If `iterative.enabled` and time budget remains:
   - Persist `best_solution` to disk (e.g., `target/run_artifacts/best_solution.iter1.json`).
   - Reinitialize pipeline with:
     - `initial_solution = best_solution`.
     - Adjusted parameters (e.g., fewer thermodynamic temps, more aggressive quantum target).
3. Repeat until `iterative.max_passes` reached or no improvement after pass.

### 7.2 Implementation Notes

- Add `PipelineController` with loop:
  ```rust
  for pass in 0..config.iterative.max_passes {
      let result = pipeline.run_with_initial_solution(current_solution)?;
      if result.chromatic_number < current_solution.chromatic_number {
          current_solution = result;
          telemetry.record_pass_summary(pass, &result);
      } else {
          break;
      }
  }
  ```
- Ensure phases respect `initial_solution` (skip TE if not needed, or use as guidance).
- Telemetry should log `pass` field for metrics to differentiate iterations.

### 7.3 CLI Control

- `prism-config set iterative.enabled=true`.
- Dashboard shows pass progress and best colors per pass.

---

## 8. Resource & Performance Considerations

- **Telemetry overhead:** Use async writer; buffer to avoid disk contention.
- **Web dashboard:** Runs in same process; consider throttling updates to 10 Hz to limit CPU usage.
- **ADP controls:** Ensure locks are non-blocking; prefer message-passing (channels).
- **Iterative runs:** Respect global time budget (`config.pipeline.max_runtime_hours`). Telemetry should record per-pass durations.

---

## 9. Implementation Checklist

| Task | Files | Notes |
|------|-------|-------|
| Telemetry core module | `foundation/prct-core/src/telemetry/mod.rs` | Metric struct, handle, writers |
| Phase instrumentation | `world_record_pipeline.rs` + phase modules | Insert `telemetry.record(...)` calls |
| CLI monitor command | `src/bin/prism_config_cli.rs` | JSON tail, filters |
| Web dashboard | `telemetry_web.rs`, static assets | Feature-gated server |
| Hypertune controller | `hypertune/controller.rs` | Consumes telemetry, sends control actions |
| Initial coloring strategies | `initial_coloring.rs` | Configurable strategies |
| Iterative controller | `pipeline_controller.rs` | Multi-pass orchestration |
| Config updates | `GpuConfig`, `WorldRecordConfig`, registry | Flags for telemetry, dashboards, iterative |
| Tests | Unit tests for telemetry serialization, controller logic; integration test verifying iterative pass reduces colors | Use `cargo test telemetry::*` |

---

## 10. Acceptance Criteria

1. **Telemetry:** During a run, `live_metrics.jsonl` fills with per-phase entries; `prism-config monitor --tail` shows live updates without lag.
2. **Web dashboard:** Accessible at configured port, shows live chromatic/conflict charts and phase status.
3. **Hypertuning:** Metrics demonstrate automatic parameter adjustments (logged actions) leading to improved chromatic numbers vs. static runs (validate via A/B test).
4. **Alternative initial coloring:** Configurable strategy produces different initial chromatic numbers; telemetry reflects choice.
5. **Iterative controller:** When enabled, pipeline performs multiple passes, achieving strictly improving chromatic numbers until convergence or budget.
6. **Documentation:** README/handbook updated to describe telemetry dashboard, CLI commands, and iterative mode.

---

## 11. Notes & Future Enhancements

- Integrate with Prometheus/Grafana via exporter for long-term monitoring.
- Add API for remote control (e.g., start/stop pipeline) with auth.
- Persist telemetry to SQLite for easier offline analysis.
- Use telemetry-driven Bayesian optimization offline to propose new default configs.

---
