# GPU Active Inference Pipeline Integration Plan

**Objective:** Extend the existing GPU Active Inference module (Phase 1) so that its outputs (expected free energy, uncertainty, policy guidance) are produced once per run on the GPU, preserved in device memory, and consumed by every downstream phase (Thermo, Quantum, Memetic, Iterative passes). Ensure GPU stream management honors per-phase configuration and maximizes concurrency whenever the `enable_active_inference_gpu` flag is on.

---

## 1. Current State Recap

- Phase 1 already computes Active Inference on GPU, producing:
  - `expected_free_energy: Vec<f64>`
  - `uncertainty_scores: Vec<f64>`
  - Policy metadata (optional)
- Phase 3 (CPU) consumes EFE for DSATUR tie-breaking.
- Other phases are not yet using these signals.

Key upgrades needed:
1. Persist GPU-resident tensors (EFE, uncertainty) for reuse by later phases.
2. Provide normalized views (host & device) accessible to each phase.
3. Wire phases 2–4 to incorporate EFE/uncertainty in their logic.
4. Manage CUDA streams so Active Inference computations run on dedicated streams and do not block other kernels when the GPU is fully available.

---

## 2. Data Lifecycle & Memory Plan

### 2.1 GPU Buffers

- Allocate device buffers once per run:
  - `d_expected_free_energy: DeviceBuffer<f32>`
  - `d_uncertainty: DeviceBuffer<f32>`
  - `d_activation_map` (optional 2D map for visualization)
- Keep them alive in the pipeline struct (`PipelineGpuState`).
- Provide host mirrors (`Vec<f32>`) for phases that prefer CPU consumption.

### 2.2 Production Flow (Phase 1)

1. Phase 1 runs GPU Active Inference:
   - Returns device pointers + host copies.
   - Normalizes values (0–1) and stores min/max for reference.
2. Registers metadata in telemetry (`RunMetric.parameters["active_inference"] = {...}`).

### 2.3 Consumption Flow (Downstream Phases)

- Phase modules request references via `pipeline_gpu_state.get_active_inference()` which yields:
  ```rust
  struct ActiveInferenceView<'a> {
      device_efe: &'a DeviceBuffer<f32>,
      host_efe: &'a [f32],
      device_uncertainty: &'a DeviceBuffer<f32>,
      host_uncertainty: &'a [f32],
      normalization: NormalizationStats,
  }
  ```
- Each phase decides whether to use device or host data (prefer GPU when kernels exist).

---

## 3. Phase Integration

### 3.1 Phase 2: Thermodynamic Replica Exchange

