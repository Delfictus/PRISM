# CAFA6 Implementation Roadmap (PRISM Platform)

**Worktree:** `worktrees/cafa6-plan` (branch `cafa6-plan`)  
**Scope:** Fully wire GPU Phases 1–3 + Active Inference (Options A & B), add GPU verification & CLI configurability, and extend PRISM for CAFA 6 protein function prediction (GO terms + optional text).

---

## 1. Program Structure

| Layer | Purpose | Key Files / Modules | Primary Owners |
|-------|---------|---------------------|----------------|
| Core Pipeline | Phase orchestration, ADP, ensemble | `foundation/prct-core/src/world_record_pipeline.rs`, `world_record_pipeline_gpu.rs` | Core team |
| GPU Kernels | CUDA kernels for TE, Thermo, Quantum, Active Inference | `foundation/kernels/*.cu`, `foundation/cuda/*.cu` | GPU team |
| Config & CLI | Registry, schema, CLI | `shared_types/config_registry`, `src/bin/prism_config_cli.rs`, `config_wrapper.rs` | Platform team |
| Domain Layer | Protein / CAFA-specific ingestion & featurization | New modules under `foundation/prct-core/src/biochem` | Bio team |
| Evaluation | Metric runners, CAFA harness | `tools/cafa_eval`, new scripts | Evaluation team |

---

## 2. Workstream Breakdown

### 2.1 Build & Toolchain Prep
1. **git worktree**: `git worktree add worktrees/cafa6-plan cafa6-plan` (already created).  
2. **NVCC/PTX** (`foundation/prct-core/build.rs`):
   - When `CARGO_FEATURE_CUDA` set, compile and copy PTX to `$OUT_DIR/ptx` and `target/ptx`:
     | Kernel Source | Output PTX | Notes |
     |---------------|------------|-------|
     | `foundation/kernels/transfer_entropy.cu` | `transfer_entropy.ptx` | Include histogram kernels |
     | `foundation/kernels/thermodynamic.cu` | `thermodynamic.ptx` | Oscillator suite |
     | `foundation/kernels/quantum_evolution.cu` | `quantum_evolution.ptx` | Trotter-Suzuki + Hamiltonians |
     | `foundation/kernels/active_inference.cu` | `active_inference.ptx` | Policy gradient kernels |
   - Emit build-log lines with absolute PTX paths for CLI diagnostics.
3. **Feature gates**: ensure `Cargo.toml` exposes `cuda` feature + optional `cuda_tests` flag for GPU unit tests.

### 2.2 Phase GPU Wiring (Option A)

#### 2.2.1 Transfer Entropy (Phase 1)
- **Module**: `foundation/prct-core/src/gpu_transfer_entropy.rs` (new).
- **API**:
  ```rust
  pub fn compute_transfer_entropy_ordering_gpu(
      device: &Arc<CudaDevice>,
      graph: &Graph,
      kuramoto_state: &KuramotoState,
      geodesic_features: Option<&GeodesicFeatures>,
      geodesic_weight: f64,
      te_vs_kuramoto: f64,
  ) -> Result<Vec<usize>>;
  ```
- **Implementation Steps**:
  1. Lazy-load `transfer_entropy.ptx` via `device.load_ptx(.., "te_module", &[... kernels ...])`.
  2. Cache `CudaFunction` handles (`compute_minmax_kernel`, `build_histogram_3d_kernel`, `build_histogram_2d_kernel`, `compute_transfer_entropy_kernel`, `build_histogram_1d_kernel`, `build_histogram_2d_xp_yp_kernel`).
  3. Prepare data:
     - Convert adjacency to dense `Vec<u8>` (or CSR) expected by kernels.
     - Generate synthetic time-series via existing CPU helper (`generate_vertex_time_series`) or move to GPU version if feasible.
  4. Launch kernels sequentially with `LaunchConfig::for_num_elems(...)`.
  5. Copy TE matrix back, compute ordering identical to CPU fallback.
- **Pipeline Integration** (`world_record_pipeline.rs:1855-1906`):
  - Under `cfg(feature = "cuda")`, call GPU path when `self.config.gpu.enable_te_gpu` and `self.cuda_device.is_some()`.
  - Set `phase_gpu_status.phase1 = GpuStatus::GpuSuccess`.
  - On error/disabled: log `[PHASE 1][GPU→CPU FALLBACK] <reason>` and set `GpuStatus::CpuFallback`.
- **Testing**:
  - `tests/transfer_entropy_gpu.rs`: compare GPU vs CPU ordering for deterministic graph (set `rand` seed).
  - Add CLI flag `--verify gpu-te` to run the test via `prism_config_cli verify`.

#### 2.2.2 Thermodynamic Replica Exchange (Phase 2)
- **API**: extend `ThermodynamicEquilibrator`:
  ```rust
  pub fn equilibrate_gpu(
      device: &Arc<CudaDevice>,
      graph: &Graph,
      initial_solution: &ColoringSolution,
      target_chromatic: usize,
      t_min: f64,
      t_max: f64,
      num_temps: usize,
      steps_per_temp: usize,
  ) -> Result<Self>;
  ```
- **GPU Flow**:
  1. Load `thermodynamic.ptx`.
  2. Allocate device buffers: `positions`, `velocities`, `phases`, `forces`, `coupling_matrix`.
  3. Launch `initialize_oscillators_kernel`.
  4. For each temperature:
     - Launch `compute_coupling_forces_kernel`.
     - Loop `steps_per_temp` times calling `evolve_oscillators_kernel`.
     - Optionally compute metrics (energy/order parameter) with `compute_energy_kernel`/`compute_order_parameter_kernel`.
     - Copy state, convert to `ColoringSolution` (respect vertex color mapping).
  5. Mirror CPU logic for ADP interactions & ensemble additions.
- **Pipeline Changes**:
  - Replace CPU call (line ~1960) with GPU branch.
  - Update ADP temperature adjustments to feed GPU path.
  - Log GPU usage & fallback.
- **Tests**:
  - `tests/thermodynamic_gpu.rs`: assert GPU & CPU produce equal best chromatic number on small graph (allow tolerance).

#### 2.2.3 Quantum-Classical Hybrid (Phase 3)
- **QuantumColoringSolver** refactor:
  - `find_coloring_cpu` (existing body).
  - `find_coloring_gpu`:
    1. Load `quantum_evolution.ptx`.
    2. Build Hamiltonians (`build_tight_binding_hamiltonian`, `build_ising_hamiltonian`) directly on GPU.
    3. Execute `trotter_suzuki_step` for `self.adp_quantum_iterations`.
    4. Convert final amplitudes to color assignments (via measurement heuristics).
  - `find_coloring` chooses GPU path when `self.gpu_device.is_some()` and `enable_quantum_gpu`.
- **QuantumClassicalHybrid**:
  - Store `Arc<CudaDevice>` and propagate to solver.
  - Continue to merge outputs into DSATUR + Active Inference as today.
- **Testing**:
  - `tests/quantum_gpu.rs`: run hybrid solver on toy graph, ensure GPU path reduces colors vs baseline.

### 2.3 Option B – Active Inference GPU
- **Module**: `foundation/prct-core/src/gpu_active_inference.rs`.
- **API**:
  ```rust
  pub fn active_inference_policy_gpu(
      device: &Arc<CudaDevice>,
      graph: &Graph,
      coloring: &ColoringSolution,
      kuramoto_state: &KuramotoState,
  ) -> Result<ActiveInferencePolicy>;
  ```
- **Implementation**:
  1. Load `active_inference.ptx`.
  2. Execute kernels for expected free energy, policy gradients, posterior updates.
  3. Return policy struct identical to CPU version.
- **Config**:
  - Add `enable_active_inference_gpu: bool` to `GpuConfig` (`world_record_pipeline.rs`, `config_wrapper.rs`, defaults to `true`).
  - Register parameter in config registry and expose via CLI.
- **Pipeline**:
  - In Phase 1 after TE ordering call GPU policy if flag set.
  - Update GPU status struct and logging.
- **Tests**:
  - `tests/active_inference_gpu.rs`: compare GPU vs CPU policy vectors.

### 2.4 GPU Usage Telemetry & CLI Integration
- **PhaseGpuStatus struct** (new module `foundation/prct-core/src/phase_gpu_status.rs`):
  ```rust
  pub enum PhaseExecMode { GpuSuccess, CpuFallback { reason: String }, CpuDisabled }
  pub struct PhaseGpuStatus { te: PhaseExecMode, thermo: PhaseExecMode, quantum: PhaseExecMode, active_inference: PhaseExecMode, timestamp: SystemTime }
  ```
- Persist JSON to `target/run_artifacts/phase_gpu_status.json` after each pipeline run.
- CLI (`src/bin/prism_config_cli.rs`):
  - Add `prism-config verify --gpu-status` to display last run, highlight fallbacks, and exit non-zero if any `CpuFallback` occurred while GPU flag enabled.
  - Add `prism-config gpu --set <phase>=<on/off>` to toggle registry parameters quickly.

### 2.5 Testing Matrix

| Command | Purpose | Notes |
|---------|---------|-------|
| `cargo fmt` | Formatting | Run before commits |
| `cargo clippy --features cuda` | Lint | Allow `-A clippy::too_many_arguments` where needed |
| `cargo check --features cuda` | Build validation | Must pass |
| `cargo test transfer_entropy_gpu --features cuda` | TE parity | Add CI guard |
| `cargo test thermodynamic_gpu --features cuda` | Thermo parity |  |
| `cargo test quantum_gpu --features cuda` | Quantum solver | May need `#[cfg_attr(not(feature="cuda_tests"), ignore)]` |
| `cargo test active_inference_gpu --features cuda` | Policy parity |  |
| `prism-config verify --gpu-status` | CLI verification | Should flag fallbacks |

Document how to run GPU tests locally vs CI (e.g., set `CUDA_VISIBLE_DEVICES`).

---

## 3. CAFA 6 Adaptation Roadmap

### 3.1 Data Ingestion
1. **Datasets**: Download CAFA training FASTA + GO annotations under `data/cafa6/{train,test_superset}`.
2. **Parsers**:
   - Add `foundation/prct-core/src/biochem/fasta_loader.rs` for streaming FASTA.
   - Add `go_terms.rs` to parse GO OBO + term metadata (weights, IC).
3. **Feature Extraction**:
   - Integrate ESM/OpenFold embeddings (via ONNX or external service) into `biochem/features.rs`.
   - Optional: incorporate structural templates (AlphaFold) cached locally.

### 3.2 Model Integration
1. **Sequence Encoder Phase** (new Phase -1):
   - Preprocess sequences, compute embeddings, store in `Graph`-like structure consumed by Phase 0+.
2. **GO Prediction Heads**:
   - Post-phase ensemble: logistic heads per GO term, or hierarchical decoder.
   - Use thermodynamic + quantum phases as meta-optimizers for label assignment.
3. **Text Generation**:
   - Extend `prism_config_cli` with `text-predict` subcommand calling LLM (local or API) and referencing pipeline outputs for evidence.

### 3.3 Evaluation Harness
1. Implement CAFA metric runner: `tools/cafa_eval/main.rs`.
2. Simulate prospective evaluation via temporal splits (train <= year N, test > year N).
3. Track experiments with CLI config snapshots + registry digests.

### 3.4 Compliance & Logging
1. Add `docs/CAFA6_VALIDATION_CHECKLIST.md` (future work) capturing dataset provenance, CLI commands, QA steps.
2. Ensure `phase_gpu_status` + config registry hash stored alongside predictions for reproducibility.

---

## 4. Milestones & Owners

| Milestone | Deliverables | Owner(s) | Target |
|-----------|--------------|----------|--------|
| M1 | Build script updates + PTX artifacts verified | Platform | Week 1 |
| M2 | TE/Thermo/Quantum GPU wiring + parity tests | GPU Team | Week 2 |
| M3 | Active Inference GPU + CLI flags | GPU + Platform | Week 3 |
| M4 | Phase GPU telemetry + CLI reporting | Platform | Week 3 |
| M5 | CAFA data ingestion + evaluation harness | Bio | Week 4 |
| M6 | Integrated CAFA pipeline dry run (historical split) | All | Week 5 |

---

## 5. Acceptance Criteria
1. All GPU-enabled phases run without CPU fallback on supported hardware; fallbacks are logged + surfaced via CLI.
2. `cargo check --features cuda` and all GPU tests pass.
3. CLI accurately toggles new GPU flags and reports last-run status.
4. CAFA training pipeline ingests official datasets, emits GO predictions + CAFA metrics locally.
5. Documentation (this roadmap + future validation checklist) kept in branch `cafa6-plan`.

---

**Next Steps:**  
- Assign owners per milestone, start with build script changes, then progress through Option A/B, telemetry, and CAFA-specific modules.  
- Keep roadmap updated in this worktree; merge back to `main` once implementation stabilizes.

