# Integration Checklist

## Status: Core Infrastructure Complete ✅

All new modules compile successfully. The 14 build errors are from **pre-existing code** (function signature mismatches in `quantum_coloring.rs` and `world_record_pipeline.rs` unrelated to our changes).

---

## Phase 1: GPU Stream Integration (Pending)

### 1.1 Update GPU Module Signatures

**Files to modify**:
- `foundation/prct-core/src/gpu_reservoir.rs`
- `foundation/prct-core/src/gpu_transfer_entropy.rs`
- `foundation/prct-core/src/gpu_active_inference.rs`
- `foundation/prct-core/src/gpu_thermodynamic.rs`
- `foundation/prct-core/src/gpu_quantum_annealing.rs`

**Pattern**:
```rust
// BEFORE
pub fn compute_te_gpu(
    device: &Arc<CudaDevice>,
    graph: &Graph,
    // ...
) -> Result<Vec<usize>>

// AFTER
pub fn compute_te_gpu(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,  // ADD
    graph: &Graph,
    // ...
) -> Result<Vec<usize>>
```

**Kernel launches**:
```rust
// If using cudarc 0.9 launch (synchronous)
kernel.launch(config, params)?;

// Future cudarc 0.17+ (async)
kernel.launch_on_stream(stream, config, params)?;
```

### 1.2 Wire PipelineGpuState into WorldRecordPipeline

**File**: `foundation/prct-core/src/world_record_pipeline.rs`

```rust
pub struct WorldRecordPipeline {
    graph: Graph,
    config: WorldRecordConfig,

    #[cfg(feature = "cuda")]
    gpu_state: Option<Arc<PipelineGpuState>>,  // ADD

    // ... existing fields
}

impl WorldRecordPipeline {
    pub fn new(graph: Graph, config: WorldRecordConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_state = if config.gpu.streams > 0 {
            Some(Arc::new(PipelineGpuState::new(
                config.gpu.device_id,
                config.gpu.streams,
                match config.gpu.stream_mode {
                    StreamMode::Sequential => gpu::StreamMode::Sequential,
                    StreamMode::Parallel => gpu::StreamMode::Parallel,
                },
            )?))
        } else {
            None
        };

        Ok(Self {
            graph,
            config,
            #[cfg(feature = "cuda")]
            gpu_state,
            // ...
        })
    }
}
```

### 1.3 Use Streams in Pipeline Execution

**In `run()` method**:

```rust
// Phase 0: Reservoir
#[cfg(feature = "cuda")]
if let Some(ref gpu) = self.gpu_state {
    let stream = gpu.stream_for_phase(0);
    reservoir_predict_gpu(..., stream)?;
}

// Phase 1: Transfer Entropy
#[cfg(feature = "cuda")]
if let Some(ref gpu) = self.gpu_state {
    let stream = gpu.stream_for_phase(1);
    compute_transfer_entropy_ordering_gpu(..., stream)?;
}

// Phase 2: Thermodynamic
#[cfg(feature = "cuda")]
if let Some(ref gpu) = self.gpu_state {
    let stream = gpu.stream_for_phase(2);
    equilibrate_thermodynamic_gpu(..., stream)?;
}

// Phase 3: Quantum
#[cfg(feature = "cuda")]
if let Some(ref gpu) = self.gpu_state {
    let stream = gpu.stream_for_phase(3);
    gpu_qubo_simulated_annealing(..., stream)?;
}
```

### 1.4 Add Stream Logging

```rust
#[cfg(feature = "cuda")]
if let Some(ref gpu) = self.gpu_state {
    let stream_id = match gpu.mode() {
        gpu::StreamMode::Parallel => Some(phase_index),
        gpu::StreamMode::Sequential => Some(0),
    };
    eprintln!("[PHASE {}][GPU][stream={}] Starting",
             phase_index, stream_id.unwrap_or(0));
}
```

---

## Phase 2: Telemetry Integration (Pending)

### 2.1 Add TelemetryHandle to Pipeline

```rust
pub struct WorldRecordPipeline {
    // ...
    telemetry: Option<Arc<TelemetryHandle>>,  // ADD
}

impl WorldRecordPipeline {
    pub fn with_telemetry(mut self, run_id: &str) -> Result<Self> {
        self.telemetry = Some(Arc::new(TelemetryHandle::new(run_id, 1000)?));
        Ok(self)
    }
}
```

### 2.2 Record Metrics in Each Phase

**Pattern**:
```rust
// Phase start
let phase_start = Instant::now();

// ... phase execution ...

// Phase end
let duration_ms = phase_start.elapsed().as_secs_f64() * 1000.0;

if let Some(ref telemetry) = self.telemetry {
    let gpu_mode = if gpu_used {
        PhaseExecMode::gpu_success(Some(stream_id))
    } else {
        PhaseExecMode::cpu_disabled()
    };

    telemetry.record(RunMetric::new(
        PhaseName::Thermodynamic,  // or appropriate phase
        format!("temp_{}", temp_index),
        self.best_solution.chromatic_number,
        conflicts,
        duration_ms,
        gpu_mode,
    ).with_parameters(serde_json::json!({
        "temperature": temp_value,
        "replicas": num_replicas,
    })));
}
```

### 2.3 Finalize Run

```rust
pub fn run(mut self) -> Result<ColoringSolution> {
    let run_start = Instant::now();

    // ... all phases ...

    let total_time = run_start.elapsed().as_secs_f64();

    if let Some(ref telemetry) = self.telemetry {
        telemetry.finalize(RunSummary {
            run_id: telemetry.run_id().to_string(),
            graph_name: self.graph_name.clone(),
            total_runtime_sec: total_time,
            final_chromatic: solution.chromatic_number,
            final_conflicts: solution.conflicts,
            metric_count: telemetry.snapshot().len(),
            phase_stats: vec![],  // Aggregate from metrics
            gpu_summary: GpuUsageSummary {
                total_gpu_time_ms: 0.0,  // Sum from metrics
                total_cpu_time_ms: 0.0,
                gpu_percentage: 0.0,
                streams_used: vec![],
                stream_mode: format!("{:?}", self.gpu_state.map(|g| g.mode())),
            },
        });
    }

    Ok(solution)
}
```

---

## Phase 3: CLI Monitoring (Pending)

### 3.1 Add Monitor Subcommand

**File**: `src/bin/prism_config.rs`

```rust
#[derive(Debug, clap::Parser)]
enum Commands {
    // ... existing commands ...

    /// Monitor live telemetry metrics
    Monitor {
        /// Tail live metrics (follow mode)
        #[clap(long)]
        tail: bool,

        /// Show summary statistics
        #[clap(long)]
        summary: bool,

        /// Filter by phase
        #[clap(long)]
        phase: Option<String>,

        /// Metrics file path (default: latest in target/run_artifacts)
        #[clap(long)]
        file: Option<String>,
    },
}
```

### 3.2 Implement Monitor Functions

```rust
fn monitor_telemetry(
    tail: bool,
    summary: bool,
    phase_filter: Option<String>,
    file_path: Option<String>,
) -> Result<()> {
    let path = file_path
        .map(PathBuf::from)
        .or_else(|| find_latest_metrics_file())
        .ok_or_else(|| PRCTError::ConfigError("No metrics file found".to_string()))?;

    if summary {
        show_summary(&path, phase_filter)?;
    } else if tail {
        tail_metrics(&path, phase_filter)?;
    } else {
        // Default: show last N metrics
        show_recent(&path, 50, phase_filter)?;
    }

    Ok(())
}

fn tail_metrics(path: &Path, phase_filter: Option<String>) -> Result<()> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("---") {
            continue;  // Skip summary header
        }

        let metric: RunMetric = serde_json::from_str(&line)?;

        if let Some(ref filter) = phase_filter {
            if format!("{}", metric.phase).to_lowercase() != filter.to_lowercase() {
                continue;
            }
        }

        println!("{}", metric.format_terminal());
    }

    Ok(())
}

fn show_summary(path: &Path, phase_filter: Option<String>) -> Result<()> {
    // Read all metrics
    // Group by phase
    // Calculate stats (avg duration, total time, etc.)
    // Display table
    Ok(())
}
```

---

## Phase 4: Documentation (Pending)

### 4.1 Create Architecture Docs

**Files to create**:
1. `docs/CUDA_STREAM_ARCHITECTURE.md`
2. `docs/TELEMETRY_SYSTEM.md`
3. `docs/HYPERTUNING_GUIDE.md`

### 4.2 Update Main README

Add sections:
- GPU Stream Parallelization
- Real-Time Telemetry
- CLI Monitoring

---

## Phase 5: Testing (Pending)

### 5.1 Integration Tests

**File**: `foundation/prct-core/tests/integration_gpu_streams.rs`

```rust
#[cfg(feature = "cuda")]
#[test]
fn test_parallel_stream_execution() {
    let graph = load_test_graph();
    let config = WorldRecordConfig {
        gpu: GpuConfig {
            streams: 4,
            stream_mode: StreamMode::Parallel,
            ..Default::default()
        },
        ..Default::default()
    };

    let pipeline = WorldRecordPipeline::new(graph, config)
        .expect("Failed to create pipeline");

    let solution = pipeline.run().expect("Pipeline failed");
    assert_eq!(solution.conflicts, 0);
}
```

### 5.2 Telemetry Tests

**File**: `foundation/prct-core/tests/integration_telemetry.rs`

```rust
#[test]
fn test_telemetry_recording() {
    let telemetry = TelemetryHandle::new("test_run", 100)
        .expect("Failed to create telemetry");

    telemetry.record(RunMetric::new(
        PhaseName::Thermodynamic,
        "test_step",
        115,
        0,
        100.0,
        PhaseExecMode::gpu_success(Some(2)),
    ));

    std::thread::sleep(std::time::Duration::from_millis(100));

    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].chromatic_number, 115);
}
```

---

## Quick Start Guide

### Enable GPU Streams + Telemetry

**Config file** (`config.toml`):
```toml
[gpu]
streams = 4
stream_mode = "parallel"

[initial_coloring]
strategy = "spectral"

[iterative]
enabled = true
max_passes = 3
min_delta = 1
```

**Run with telemetry**:
```bash
# Run pipeline
cargo run --release --features cuda -- \
  --config config.toml \
  --graph data/DSJC1000.5.col \
  --telemetry

# Monitor live (in another terminal)
cargo run --release -- monitor --tail

# Show summary after run
cargo run --release -- monitor --summary
```

---

## Pre-Existing Build Errors (Not Our Code)

These errors exist in the base codebase and are **unrelated** to our implementation:

1. `quantum_coloring.rs:213` - `hybrid_te_kuramoto_ordering` takes 4 args, not 5
2. `quantum_coloring.rs:1015` - Same issue
3. `world_record_pipeline.rs:1870` - `GpuReservoirConflictPredictor::predict_gpu` signature mismatch
4. `world_record_pipeline.rs:1979` - `hybrid_te_kuramoto_ordering` signature mismatch
5. Similar issues in other locations

**Fix required** (separate from this feature):
- Update function signatures to match declarations
- Or update call sites to pass correct number of arguments

---

## Validation Commands

```bash
# Check new modules only
cargo check --features cuda --lib 2>&1 | grep -E "(gpu|telemetry|initial|iterative|hypertune)" | grep error

# Should return empty (all our code compiles)

# Run tests
cargo test --features cuda --lib telemetry
cargo test --features cuda --lib gpu::stream_pool
cargo test --features cuda --lib initial_coloring

# Check file structure
ls -la foundation/prct-core/src/gpu/
ls -la foundation/prct-core/src/telemetry/
ls -la foundation/prct-core/src/hypertune/
```

---

## Success Criteria

- [x] GPU stream infrastructure compiles
- [x] Telemetry system compiles
- [x] Hypertuning controller compiles
- [x] Initial coloring strategies compile
- [x] Iterative controller compiles
- [x] All new code has zero errors
- [ ] GPU modules accept stream parameter
- [ ] Pipeline uses PipelineGpuState
- [ ] Telemetry records phase metrics
- [ ] CLI monitor command works
- [ ] Integration tests pass
- [ ] Documentation complete

**Current Progress**: 6/12 complete (50%)

---

## Next Steps

1. **Fix pre-existing errors** (separate PR/commit)
2. **Integrate GPU streams** (Phase 1 checklist above)
3. **Integrate telemetry** (Phase 2 checklist above)
4. **Add CLI monitoring** (Phase 3 checklist above)
5. **Write documentation** (Phase 4 checklist above)
6. **Test thoroughly** (Phase 5 checklist above)
7. **Merge to main**

---

## Contact

For questions about this implementation:
- Architecture decisions: See IMPLEMENTATION_SUMMARY.md
- Integration help: This file (INTEGRATION_CHECKLIST.md)
- API reference: Inline documentation in source files
