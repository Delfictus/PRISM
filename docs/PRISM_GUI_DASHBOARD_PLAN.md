# PRISM GUI Dashboard & Visualization Plan

**Goal:** Transform the current CLI-centric PRISM experience into a fully featured graphical interface with live visualizations (neural maps, solver/process graphs, node/service maps), complete parameter control, and tight integration with the telemetry/hypertuning subsystems.

---

## 1. Product Goals & Use Cases

1. **Live Run Monitoring:** Display real-time phase status, chromatic number trajectory, conflicts, GPU utilization.
2. **Interactive Tuning:** Expose every registry parameter (toggles, sliders, numeric inputs) with validation and immediate effect.
3. **Visualization Suites:**
   - **Neural Map Firing:** Visualize TE/Active Inference/Reservoir activity (per-neuron or per-vertex).
   - **Solver/Process Map:** Live dependency graph for pipeline phases and sub-tasks, showing current execution state.
   - **Node/Service Map:** Display compute resources (GPU devices, kernels, threads) with health indicators.
4. **Iterative Control:** Start/stop runs, trigger iterative restarts, snapshot states, compare passes.
5. **Reporting & Playback:** Replay past runs, inspect telemetry timeline, export data.

---

## 2. Architecture Overview

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| Backend API Server | Serve REST/WebSocket APIs, proxy to CLI/registry | Axum (Rust) |
| State Store | Maintain live telemetry, config, run state | Tokio tasks, shared `Arc<RwLock<>>`, or lightweight DB (SQLite) |
| Frontend App | UI components (controls, charts, visualizations) | TypeScript SPA (React + Tailwind + D3.js, or Svelte Kit) |
| Visualization Engine | Render graphs/maps (WebGL/D3) | D3.js + WebGL shaders (via regl or deck.gl) |
| Telemetry Ingest | Stream metrics from pipeline to backend | WebSocket bridge or local channel |
| Control Interface | Send commands (start, stop, tune) to pipeline | HTTP POST / WebSocket RPC |

**App Flow:**
1. Pipeline emits telemetry → ingested via channel → API server updates in-memory state → pushes to frontend via WebSocket.
2. User adjusts parameters in GUI → API server writes to config registry & notifies pipeline control channel.
3. Visualization components subscribe to live data streams to animate activity maps.

---

## 3. Backend Design

### 3.1 API Server

- Framework: Axum + Tokio.
- Mount routes under `/api/*` and `/ws`.
- Use `tower::ServiceBuilder` for middleware (logging, auth hooks, CORS).

#### REST Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET /api/config` | Fetch current registry parameters (structured tree) |
| `PATCH /api/config` | Update subset of parameters (validate & persist) |
| `POST /api/run/start` | Start pipeline run (with optional overrides) |
| `POST /api/run/stop` | Request graceful shutdown |
| `POST /api/run/restart` | Trigger iterative pass |
| `GET /api/telemetry/live?limit=500` | Return latest metrics |
| `GET /api/telemetry/history/:run_id` | Stream stored metrics |
| `GET /api/visualization/neural-map` | Snapshot of latest neural activity graph |
| `GET /api/visualization/process-map` | Current solver graph state |
| `GET /api/resources` | GPU/CPU utilization, VRAM usage |

#### WebSocket Channels
| Path | Payload |
|------|---------|
| `/ws/telemetry` | `RunMetric` JSON stream |
| `/ws/control` | Bidirectional messages (`{"cmd":"pause"}`, `{"status":"phase_started"}`) |

### 3.2 State & Persistence

- **Telemetry Buffer:** `Arc<RwLock<VecDeque<RunMetric>>>` (size limit 10k). Persist to JSONL/SQLite.
- **Config Store:** Wrap existing registry; provide atomic patches with validation (re-use CLI logic).
- **Run State:** Track phases, iterative pass counters, best solution summary.
- **Auth/Security (future):** Add token-based auth for remote access; for now assume local use.

### 3.3 Integration with Pipeline

- Start API server in same binary (feature `gui_dashboard`). Provide handle for pipeline to register telemetry/control channels.
- Control channel uses `tokio::sync::mpsc` to send commands (start, stop, adjust parameters) to pipeline orchestrator.

---

## 4. Frontend Design

### 4.1 Tech Stack

- Build with Vite + React + TypeScript + Tailwind CSS.
- Use Zustand or Redux Toolkit for state management.
- Charting: `visx` or `recharts` for standard charts, `d3-force`/WebGL for network maps.

### 4.2 Layout

```
┌────────────────────────────┬────────────────────────────┐
│ Phase Status Cards         │ Parameter Control Panel    │
│ (TE, Thermo, Quantum, ...)│ (toggles, sliders, inputs)  │
├────────────────────────────┴────────────────────────────┤
│ Timeline (chromatic, conflicts, GPU)                    │
├─────────────────────────────────────────────────────────┤
│ Visualization Tabs: [Neural Map][Process Map][Node Map] │
├─────────────────────────────────────────────────────────┤
│ Run Controls & Iteration Panel                          │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Components

1. **Phase Cards:** Show current mode (GPU/CPU), time in phase, last improvement.
2. **Parameter Panel:** Tree view of registry; inline editing with validation & reset to defaults. Include toggles, sliders, numeric inputs with units.
3. **Timeline Charts:** Multi-line chart for chromatic number, conflicts, acceptance rates.
4. **Neural Map Visualization:**
   - Graph representing neurons/vertices; nodes change color/brightness based on activity from telemetry (e.g., TE scores, reservoir activations).
   - Use D3 force layout or WebGL for performance. Provide time scrubber to replay.
5. **Solver/Process Map:**
   - Directed graph showing pipeline phases/subtasks. Nodes highlight when active; edges animate to show data flow.
   - Option to collapse/expand subgraphs (e.g., Thermo replicates).
6. **Node/Service Map:**
   - Infra view: GPU devices, CPU threads, CUDA streams. Show utilization (color-coded), memory usage.
7. **Run Controls:**
   - Buttons: Start, Pause, Resume, Stop, Restart Pass, Snapshot.
   - Display iterative pass progress and current best solution.
8. **Notifications/Logs:** Toasts for errors, fallback warnings, ADP actions.

### 4.4 Visual Enhancements

- Animations for “firing” (glowing nodes, pulsing edges).
- Color themes matching PRISM branding (dark mode default).
- Sparklines on cards for mini-trends.
- Keyboard shortcuts for quick toggles (e.g., space to pause).

---

## 5. Visualizations: Technical Details

### 5.1 Neural Map Firing

- Data source: telemetry events containing per-node activation scores (from TE, reservoir, or Active Inference). Extend telemetry schema with optional `activations: Vec<f32>`.
- Rendering pipeline:
  1. Layout nodes (fixed positions based on graph geometry or auto layout).
  2. Map activation to color/size (e.g., gradient from blue (idle) to orange (firing)).
  3. Animate edges using stroke width or glowing particles to represent flow.
  4. Provide controls for:
     - Timeslice selection
     - Auto-rotate 3D view (optional)
     - Highlight top-k active nodes

### 5.2 Solver/Process Map

- Build DAG of phases/subtasks (Thermo temps, Quantum reduction targets, Memetic restarts).
- Map statuses:
  - Idle (gray), Running (blue), Completed (green), Error/Fallback (red).
- Live tokens: small circles moving along edges to show pipeline progress.
- On click: show detail drawer with metrics, ADP actions, links to logs.

### 5.3 Node/Service Map

- Represent GPU devices, kernels, threads.
- Use concentric circles or “star map” layout.
- Color-coded health:
  - Utilization thresholds (green < 70%, yellow 70–90%, red > 90%).
  - Memory usage bars.
- Optionally show network connections if running distributed (future).

---

## 6. Parameter Control & Validation

- Fetch full config schema from backend (re-used from CLI registry).
- Each parameter includes:
  - Display name, description, type (bool, int, float, enum, string).
  - Min/max constraints, step size.
  - Category tags (GPU, Thermo, Quantum, etc.).
- UI controls:
  - Toggles for booleans.
  - Sliders with numeric input for ranges.
  - Dropdowns for enums/strategies (e.g., initial coloring strategy).
- Live validation:
  - On change, call `/api/config/validate` (or full PATCH with dry-run).
  - Show warnings if change requires restart.
- Bulk actions:
  - Save/load config profiles (.toml).
  - Undo/redo parameter changes.
  - Diff view vs. defaults.

---

## 7. Integration with Telemetry & Hypertuning

### 7.1 Live Feedback Loop

- GUI subscribes to telemetry feed; highlights when ADP/hypertune adjusts parameters.
- Provide panel showing “Suggested tweaks” from hypertuning engine; user can accept/reject.
- Option to enable “auto-apply” for certain parameter categories.

### 7.2 Historical Analysis

- Timeline scrubber to replay past runs.
- Comparative view: overlay metrics from two runs (before vs after parameter change).
- Export data (CSV/JSON) for offline analysis.

---

## 8. Iterative Pipeline Control

- GUI exposes controls to:
  - Configure `iterative.max_passes`, `iterative.cooldown`.
  - Trigger manual restart with new initial solution.
  - View per-pass summary (duration, best chromatic, parameter modifications).
- Visual indicator (progress bar) showing current pass out of max.
- Provide option to schedule automatic restarts based on time or lack of improvement.

---

## 9. Implementation Plan

### 9.1 Milestones

| Milestone | Deliverables | Target |
|-----------|--------------|--------|
| M1 Backend skeleton | Axum server, config endpoints, telemetry feed | Week 1 |
| M2 CLI ↔ API integration | Registry PATCH, run controls, telemetry ingestion | Week 2 |
| M3 Frontend MVP | Phase cards, timelines, parameter panel | Week 3 |
| M4 Visualizations | Neural map, process map, node map | Week 5 |
| M5 Hypertune linkage | UI for ADP actions, suggestions | Week 6 |
| M6 Iterative control | Restart controls, pass summaries | Week 7 |
| M7 Polish | Theming, auth hooks, docs | Week 8 |

### 9.2 Task Breakdown

1. **Backend Core**
   - Setup Axum project structure (`src/gui_api/mod.rs` etc.).
   - Implement telemetry broadcaster bridging existing `RunMetrics`.
   - Build config endpoints (reusing CLI registry validation).
2. **Pipeline Hooks**
   - Ensure pipeline orchestrator exposes start/stop/pause/resume handles.
   - Extend telemetry to include activation data for neural map.
3. **Frontend Development**
   - Scaffold Vite project (`ui/` directory).
   - Implement shared API client, WebSocket hooks.
   - Build components iteratively per milestone.
4. **Visualization Engine**
   - Create D3/Canvas layers for activity maps.
   - Optimize for thousands of nodes (use WebGL for neuron map if needed).
5. **Testing & CI**
   - Add backend unit tests (Axum routes).
   - Frontend component tests (Vitest/Jest).
   - End-to-end smoke test (playwright) to ensure dashboards load.

---

## 10. Enhancements & Bells/Whistles

- **Themes:** Dark/Light mode, customizable color palettes.
- **Keyboard shortcuts & command palette** (Ctrl+K) to jump to parameters/commands.
- **Annotations:** Users can pin notes on timelines (e.g., “Applied ADP action here”).
- **Alerts:** Configure thresholds (conflicts, GPU temp) → desktop notifications/webhooks.
- **Multi-run comparison:** Pick two runs, show diff of parameters + outcomes.
- **Embeddable widgets:** Provide mini dashboards to embed elsewhere (e.g., Confluence).
- **Authentication (future):** OAuth or API tokens for remote teams.
- **Plugin system:** Allow custom visualizations via WebAssembly or Python backend.

---

## 11. Dependencies & Tooling

- **Rust crates:** `axum`, `tokio`, `serde`, `serde_json`, `tower`, `tokio-stream`, `tokio-tungstenite`.
- **Frontend libs:** `react`, `react-router`, `tailwindcss`, `d3`, `chart.js` or `visx`, `zustand`.
- **Build tooling:** `pnpm` or `yarn`, `vite`.
- **Packaging:** Provide `cargo feature gui_dashboard` to bundle API + serve frontend (static files embedded via `include_dir` or served from `/static`).
- **Dev workflow:** `cargo run --features gui_dashboard` launches backend; `pnpm dev` runs frontend with proxy to backend.

---

## 12. Acceptance Criteria

1. GUI displays live metrics, visualizations, and phase status during a run without dropping data.
2. Users can change parameters via GUI; changes propagate to pipeline immediately (with validation and feedback).
3. Visualizations (neural map, process map, node map) animate live activity and are performant (60 FPS target).
4. Iterative restart controls work end-to-end from GUI (start next pass, view results).
5. Telemetry + hypertuning suggestions visible; user can accept/decline and see effect in real time.
6. System retains compatibility with CLI workflows (GUI optional).
7. Documentation provides setup instructions and screenshots.

---

## 13. Future Directions

- Integrate with remote clusters (multi-node status).
- Provide drag-and-drop workflow editing (reorder phases).
- Implement VR or AR visualizations for large-scale demos.
- Offer REST/webhook APIs for external automation.

---
