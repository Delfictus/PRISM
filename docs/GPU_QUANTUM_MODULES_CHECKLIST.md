# GPU Quantum Modules Checklist (cudarc 0.9)

This checklist standardizes GPU module design across PRISM’s quantum stack (gpu_coloring.rs, gpu_tsp.rs, platform integrations) using cudarc 0.9. Follow it for new code and when reviewing changes.

## Device Ownership
- Create exactly one `Arc<CudaDevice>` per module; pass it into constructors. No per‑iteration device creation.
- Prefer passing `&CudaDevice`/`Arc<CudaDevice>` down call chains; avoid globals.
- Streams are optional. Only create if you need ordering; otherwise omit from `LaunchConfig`.

## Kernel Loading & Function Resolution
- Load PTX/CUBIN once and resolve all functions up front; reuse for launches.
- API pattern:
  - `device.load_ptx(ptx, "module", &["sym1","sym2"])` (or `load_cubin`)
  - `let f = device.get_func("module", "sym")?;`
- Store `CudaFunction` fields or in a small `HashMap<&'static str, CudaFunction>`.
- Do not call `load_ptx`/`get_func` inside hot loops.

## Memory Operations
- Allocate on device: `device.alloc_zeros::<T>(n)?`.
- Host→Device: `device.htod_copy(&host)?` or `device.htod_copy_into(&host, &mut d)?`.
- Device→Host: `device.dtoh_sync_copy(&d)?`.
- Zero set: `device.memset_zeros(&mut d)?`.
- Keep H2D/D2H outside hot loops; reuse buffers where feasible.

## Kernel Launch Pattern
- Build a `LaunchConfig { grid_dim, block_dim, shared_mem_bytes, ..Default::default() }` (add `stream` if used).
- Launch with tuple args (scalars by value; device buffers by reference):
  ```rust
  unsafe { func.clone().launch(cfg, (scalar1, scalar2, &d_in, &mut d_out))?; }
  ```
- Ensure tuple order matches kernel signature exactly.

## Error Typing & Safety
- Use module `Result` alias or `PRCTError`; no `anyhow` in core paths.
- No `unwrap()/expect()` in runtime code. For `f64::partial_cmp`, use `unwrap_or(Ordering::Equal)` or `total_cmp` if appropriate.
- Provide clear error messages; avoid panics except in debug assertions.

## Feature Gating & CPU Fallbacks
- Gate GPU modules/exports under `#[cfg(feature = "cuda")]`.
- For non‑CUDA builds (`#[cfg(not(feature = "cuda"))]`), provide:
  - A CPU path (preferred), or
  - A clear `PRCTError` explaining the GPU feature requirement.
- Never add silent CPU fallbacks in GPU‑required code paths.

## Module Structure Guidelines
- gpu_coloring.rs: constructor loads PTX and resolves all kernels; methods only launch kernels and perform device copies.
- gpu_tsp.rs: same pattern; no per‑iteration kernel (re)loading.
- platform.rs: import GPU types under `#[cfg(feature = "cuda")]` and provide CPU equivalents/fallbacks under `#[cfg(not(feature = "cuda"))]`.

## Verification (Tools)
- Build with CUDA: `{"SUB":"cargo_check_cuda"}` via prism‑policy.
- Stubs/unwraps: `{"SUB":"stubs"}` — runtime paths must be clean.
- CUDA gates: `{"SUB":"cuda_gates"}` — gating must be present where required.
- GPU info (sanity): `{"SUB":"gpu_info"}`.

## Self‑Audit Commands
- Legacy CUDA API usage:
  ```bash
  rg -n 'CudaContext|ContextHandle|Module\b|DeviceBuffer<' foundation
  rg -n 'launch_builder\(|default_stream\(|memcpy_' foundation
  ```
- Kernel symbol/launch audit:
  ```bash
  rg -n 'load_ptx\(|get_func\(|compile_ptx\(' foundation
  ```
- Runtime unwraps (exclude tests/examples):
  ```bash
  rg -n '(unwrap\(|expect\()' foundation/prct-core/src foundation/neuromorphic/src -g '!**/tests/**' -g '!**/examples/**'
  ```

## Acceptance Criteria
- No legacy `CudaContext`/`ContextHandle`/`Module`/legacy stream builder usage.
- H2D/D2H exclusively via `device.*` methods; kernel launches via `func.launch`.
- Kernels loaded once; functions reused; no per‑iteration device/kernel setup.
- Runtime code free of `unwrap/expect`; `partial_cmp` guarded.
- CUDA gates present; CPU fallbacks or clear errors provided for non‑CUDA builds.
- Workspace builds with `--features cuda`; prism‑policy checks pass.

## References
- PRISM GPU‑First Integration Contract: `AGENTS.md`
- Quantum GPU Migration Patterns: this file
- Config I/O and Validation: `foundation/prct-core/configs/README.md`, `foundation/prct-core/COMPREHENSIVE_CONFIG_COMPLETE.md`
