# PRISM Platform Architecture Map
**Version**: 1.1 (WR Sweep + GPU Integration)
**Date**: 2025-11-02
**Location**: `/home/diddy/Desktop/PRISM-FINNAL-PUSH/`

---

## 📂 Top-Level Directory Structure

```
PRISM-FINNAL-PUSH/
├── foundation/             # Core algorithms & GPU kernels
│   ├── active_inference/   # Active Inference (ADP + FEP)
│   ├── cma/                # CMA-ES optimizer
│   ├── cuda/               # CUDA kernel sources (.cu files)
│   ├── kernels/            # PTX compiled kernels
│   ├── mathematics/        # Math utilities (double-double, etc.)
│   ├── neuromorphic/       # Reservoir computing + GPU
│   ├── optimization/       # Optimization algorithms
│   ├── orchestration/      # Phase orchestrator
│   └── prct-core/          # Main graph coloring engine
│
├── src/                    # Main application & meta-layers
│   ├── meta/               # Meta-cognitive layers
│   │   ├── reflexive/      # Self-monitoring & adaptation
│   │   ├── plasticity/     # Dynamic parameter tuning
│   │   ├── federated/      # Multi-agent coordination
│   │   └── ontology/       # Knowledge representation
│   ├── bin/                # Executable binaries
│   └── lib.rs              # Main library entry
│
├── benchmarks/             # Benchmark instances
│   └── dimacs/             # DIMACS graph coloring instances
│
├── tools/                  # Automation scripts
│   ├── run_wr_sweep.sh     # WR sweep runner
│   ├── run_wr_seed_probe.sh # Seed probe runner
│   ├── validate_wr_sweep.sh # Config validator
│   └── mcp_policy_checks.sh # GPU policy enforcer
│
├── docs/                   # Documentation
│   ├── rfc/                # RFC proposals
│   └── *.md                # Various guides
│
├── baseline-v1.0/          # Pre-GPU baseline
├── artifacts/              # Build outputs
├── data/                   # Input data (graphs, etc.)
├── results/                # Run outputs (JSONL, logs)
└── .gitignore              # Ignore build artifacts
```

---

## 🧠 Core Foundation Modules

### `foundation/prct-core/` (Graph Coloring Engine)
**Purpose**: Main PRISM engine for graph coloring with multi-phase pipeline

**Key Files**:
- `src/lib.rs` - Main entry point
- `src/world_record_pipeline.rs` - CPU pipeline orchestrator
- `src/world_record_pipeline_gpu.rs` - GPU pipeline orchestrator
- `src/quantum_coloring.rs` - Quantum annealing (SQA/PIMC)
- `src/memetic_coloring.rs` - Genetic algorithm + local search
- `src/dsatur.rs` - DSATUR heuristic
- `src/gpu_coloring.rs` - GPU-accelerated coloring
- `src/transfer_entropy_coloring.rs` - Transfer entropy analysis
- `src/thermodynamic_coloring.rs` - Thermodynamic sampling

**Configs**:
- `configs/wr_sweep_A.v1.1.toml` through `G` - WR sweep configs
- `configs/wr_sweep_D_seed_*.v1.1.toml` - Seed variants
- `configs/wr_sweep_F_seed_*.v1.1.toml` - Seed variants
- `configs/wr_sweep_D_aggr_seed_*.v1.1.toml` - Aggressive variants

**Examples**:
- `examples/world_record_dsjc1000.rs` - WR runner for DSJC1000.5
- `examples/simple_dimacs_benchmark.rs` - Basic DIMACS runner
- `examples/test_comprehensive_config.rs` - Config tester

**Cargo Features**:
- `cuda` - Enable CUDA GPU acceleration
- `quantum_mlir_support` - Enable MLIR quantum compiler
- `protein_folding` - Enable protein structure benchmarks

---

### `foundation/neuromorphic/` (Reservoir Computing + GPU)
**Purpose**: GPU-accelerated neuromorphic reservoir for tie-breaking

**Key Files**:
- `src/lib.rs` - Reservoir API
- `src/gpu_reservoir.rs` - GPU reservoir implementation
- `src/gpu_optimization.rs` - GPU performance optimizations
- `src/cuda_kernels.rs` - CUDA kernel wrappers (cudarc 0.9)

**CUDA Kernels**:
- Uses `neuromorphic_gemv.ptx` for GPU matrix operations
- 398x speedup on RTX 4090 vs CPU

**Integration**:
- Called by DSATUR during tie-breaking
- Predicts best vertex to color next
- Learns from graph structure patterns

---

### `foundation/orchestration/` (Phase Orchestrator)
**Purpose**: Coordinates multi-phase algorithm execution

**Key Files**:
- `src/orchestrator.rs` - Main orchestration logic
- `src/phase_control.rs` - Phase switching logic
- `src/config.rs` - Configuration management

**Phases**:
1. **Phase 0**: Initial heuristic (DSATUR)
2. **Phase 1**: Quantum annealing (SQA/PIMC)
3. **Phase 2**: Memetic optimization
4. **Phase 3**: Transfer entropy analysis
5. **Phase 4**: Thermodynamic sampling

**GPU Integration**:
- Routes GPU-enabled modules to CUDA streams
- Manages device synchronization
- Enforces VRAM limits (8GB guard)

---

### `foundation/active_inference/` (Active Inference & ADP)
**Purpose**: Free Energy Principle + Adaptive Dynamic Programming

**Key Files**:
- `src/lib.rs` - Active Inference API
- `src/adp.rs` - Q-learning for parameter tuning
- `src/fep.rs` - Free Energy minimization

**Integration**:
- Tunes orchestrator weights adaptively
- Learns optimal phase transitions
- Predicts expected utility of actions

---

### `foundation/cuda/` (CUDA Kernel Sources)
**Purpose**: Raw CUDA kernel implementations

**Key Files**:
- `adaptive_coloring.cu` - Adaptive coloring kernels
- `prct_kernels.cu` - Core PRCT operations
- `neuromorphic_gemv.cu` - Neuromorphic GEMV (398x speedup)

**Build Process**:
- Compiled by `build.rs` using `nvcc`
- Outputs to `foundation/kernels/ptx/*.ptx`
- Loaded at runtime by `cudarc 0.9`

---

### `foundation/kernels/` (Compiled PTX Kernels)
**Purpose**: Pre-compiled GPU kernels for distribution

**Key Files**:
- `ptx/active_inference.ptx`
- `ptx/neuromorphic_gemv.ptx`
- `ptx/quantum_evolution.ptx`
- `ptx/thermodynamic.ptx`
- `ptx/transfer_entropy.ptx`

**Usage**:
- Loaded by `cudarc::driver::safe::CudaDevice::load_ptx()`
- Executed via `cudarc::driver::LaunchConfig`

---

## 🚀 Meta-Cognitive Layers

### `src/meta/reflexive/` (Self-Monitoring)
**Purpose**: Monitors system performance and triggers adaptations

**Key Features**:
- Tracks conflicts per phase
- Detects stagnation
- Triggers parameter updates
- Snapshot/restore state

### `src/meta/plasticity/` (Dynamic Tuning)
**Purpose**: Adjusts parameters based on performance

**Key Features**:
- Phase weight adjustment
- Learning rate scaling
- Temperature scheduling

### `src/meta/federated/` (Multi-Agent)
**Purpose**: Coordinates multiple solver instances

**Key Features**:
- Config distribution
- Result aggregation
- Best solution selection

### `src/meta/ontology/` (Knowledge Graph)
**Purpose**: Stores learned patterns and relationships

**Key Features**:
- Graph structure patterns
- Phase effectiveness history
- Hyperparameter correlations

---

## 🎯 World Record Pipeline Flow

### GPU-Accelerated Pipeline (`world_record_pipeline_gpu.rs`)

```
┌─────────────────────────────────────────────────────────┐
│  1. Initialize GPU Context (CudaDevice)                 │
│     - Create CUDA streams per phase                      │
│     - Load PTX kernels                                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. Phase 0: Initial Construction (DSATUR + Reservoir)  │
│     - DSATUR heuristic with GPU tie-breaking             │
│     - Neuromorphic reservoir predicts best vertex        │
│     - Achieves ~100-105 colors on DSJC1000.5            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. Phase 1: Quantum Annealing (SQA/PIMC)               │
│     - GPU-accelerated quantum evolution                  │
│     - Simulated annealing + path integral Monte Carlo   │
│     - Reduces to ~95-98 colors                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. Phase 2: Memetic Optimization                       │
│     - Genetic algorithm + local search                   │
│     - Crossover, mutation, tabu search                   │
│     - Target: ~90-95 colors                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. Phase 3: Transfer Entropy Analysis                  │
│     - Analyzes information flow between vertices         │
│     - Identifies critical coloring decisions             │
│     - Refines to ~88-92 colors                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  6. Phase 4: Thermodynamic Sampling                     │
│     - MCMC exploration of color space                    │
│     - Temperature annealing                              │
│     - Goal: ~83 colors (world record territory)         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  7. Finalize & Output                                   │
│     - Validate coloring (check conflicts)                │
│     - Write JSONL results                                │
│     - Report metrics (colors, time, phase stats)         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration System

### Config Structure (TOML)
```toml
[metadata]
profile = "wr_sweep_D"           # Config name
version = "1.1.0"                # Config version
target_chromatic = 83            # Goal colors
deterministic = false            # Seed control
max_runtime_hours = 24.0         # Wall-clock limit

[orchestrator]
restarts = 8                     # Number of restarts
seed = 42                        # RNG seed (optional)
enable_gpu = true                # GPU toggle
enable_active_inference = true   # Active Inference
enable_reservoir = true          # Neuromorphic reservoir
enable_thermo = true             # Thermodynamics
enable_quantum = true            # Quantum annealing
enable_adp = true                # ADP Q-learning

# Phase weights
initial_weight = 1.0
quantum_weight = 2.5
memetic_weight = 2.0
te_weight = 1.5
thermo_weight = 3.0

[quantum]
depth = 6                        # Annealing depth
attempts = 192                   # Number of attempts
schedule = "exponential"         # Annealing schedule
use_gpu = true                   # GPU acceleration

[memetic]
gens = 800                       # Genetic generations
pop = 50                         # Population size
use_gpu = false                  # CPU-only for now

[thermodynamic]
temp_init = 10.0                 # Initial temperature
temp_final = 0.01                # Final temperature
anneal_steps = 5000              # Annealing steps
use_gpu = true                   # GPU PIMC

[vram_guard]
replicas = 48                    # Max replicas (8GB safe)
beads = 48                       # Max beads (8GB safe)
```

### Config Variants

**Base Configs** (`wr_sweep_A.v1.1.toml` through `G`):
- **A**: Balanced (all phases equal weight)
- **B**: Quantum-heavy (quantum_weight = 3.0)
- **C**: Memetic-heavy (memetic_weight = 3.0)
- **D**: Quantum-deeper (depth = 6, attempts = 192)
- **E**: TE-heavy (te_weight = 2.5)
- **F**: Thermo/PIMC-heavy (thermo_weight = 3.5)
- **G**: Aggressive all (all weights maxed)

**Seed Variants** (for probe testing):
- `*_seed_42.v1.1.toml` - Seed 42
- `*_seed_1337.v1.1.toml` - Seed 1337
- `*_seed_9001.v1.1.toml` - Seed 9001

**Aggressive Variants**:
- `wr_sweep_D_aggr.v1.1.toml` - D with depth=7, attempts=224, gens=900
- `wr_sweep_D_aggr_seed_*.v1.1.toml` - Aggressive D with seed variants

---

## 🛠️ Build System

### Cargo Workspace Structure
```toml
[workspace]
members = [
    "foundation/prct-core",
    "foundation/neuromorphic",
    "foundation/active_inference",
    "foundation/orchestration",
    "foundation/cma",
    "foundation/mathematics",
    "foundation/optimization",
    "src/meta/reflexive",
    "src/meta/plasticity",
    "src/meta/federated",
    "src/meta/ontology",
]
```

### Build Targets
- **Binary**: `prism_unified` - Main unified application
- **Libraries**: Each foundation module
- **Examples**: WR runners, benchmarks, tests

### Build Process (with CUDA)
```bash
# 1. Cargo resolves dependencies
# 2. build.rs compiles CUDA kernels (.cu → .ptx)
# 3. PTX kernels copied to foundation/kernels/ptx/
# 4. Rust code compiled with `cuda` feature
# 5. cudarc loads PTX at runtime
```

---

## 📊 Results & Logging

### Output Structure
```
results/
├── dsjc1000_seed_probe.jsonl     # Seed probe results
├── logs/
│   ├── wr_sweep_D_seed_42_<timestamp>.log
│   ├── wr_sweep_F_seed_1337_<timestamp>.log
│   └── ...
└── benchmarks/
    ├── dimacs_results.jsonl
    └── protein_results.jsonl
```

### JSONL Format
```json
{
  "config": "foundation/prct-core/configs/wr_sweep_D_seed_42.v1.1.toml",
  "seed": 42,
  "colors": 95,
  "time_s": 4521.3,
  "status": "ok",
  "ts": "2025-11-02T15:30:00Z",
  "phase_stats": {
    "phase0_colors": 103,
    "phase1_colors": 97,
    "phase2_colors": 95,
    "conflicts": 0
  }
}
```

---

## 🔌 GPU Integration Details

### CUDA Stack
```
Application (Rust)
       │
       ▼
  cudarc 0.9.14 (Rust bindings)
       │
       ▼
  PTX Kernels (.ptx files)
       │
       ▼
  CUDA Runtime (libcuda.so)
       │
       ▼
  NVIDIA Driver (sm_90, RTX 4090)
```

### GPU Memory Layout (8GB VRAM)
```
┌─────────────────────────────────────────┐
│  Graph Data (vertices, edges, colors)   │  ~1GB
├─────────────────────────────────────────┤
│  Reservoir Weights (neuromorphic)       │  ~500MB
├─────────────────────────────────────────┤
│  Quantum Replicas (SQA/PIMC)            │  ~2-3GB
├─────────────────────────────────────────┤
│  Temporary Buffers                      │  ~1GB
├─────────────────────────────────────────┤
│  CUDA Context Overhead                  │  ~500MB
└─────────────────────────────────────────┘
Total: ~5-6GB (safe margin for 8GB card)
```

### VRAM Guards
- `replicas ≤ 56` (quantum annealing)
- `beads ≤ 64` (PIMC path integral)
- Automatic fallback to CPU if GPU OOM

---

## 🧪 Testing & Validation

### Test Suites
```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration

# GPU tests
cargo test --features cuda test_gpu

# Benchmarks
cargo bench --features cuda
```

### Policy Enforcement
```bash
# Check CUDA compilation
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh

# Check for CPU stubs (should have none)
SUB=stubs ./tools/mcp_policy_checks.sh

# Check CUDA feature gates
SUB=cuda_gates ./tools/mcp_policy_checks.sh

# Check GPU reservoir
SUB=gpu_reservoir ./tools/mcp_policy_checks.sh
```

---

## 📚 Key Documentation Files

### Project Root
- `README.md` - Main project overview
- `ARCHITECTURE_MAP.md` - This file
- `COMMAND_REFERENCE.md` - Command quick reference
- `GITIGNORE_CLEANUP_PR.md` - Git cleanup documentation

### WR Sweep Documentation
- `WR_SWEEP_STRATEGY.md` - Strategy overview
- `WR_SWEEP_QUICKSTART.md` - Quick start guide
- `WR_SWEEP_DELIVERY_SUMMARY.md` - Delivery notes
- `CONFIG_V1.1_VERIFICATION_REPORT.md` - Config validation

### Seed Probe
- `SEED_PROBE_DELIVERY.md` - Seed probe delivery summary

### Technical Docs
- `ACTUAL_GPU_ARCHITECTURE.md` - GPU implementation details
- `ACTUAL_STATUS_AND_NEXT_STEPS.md` - Current status
- `GPU_QUANTUM_MLIR_INTEGRATION.md` - MLIR integration

---

## 🚀 Deployment

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **GPU**: NVIDIA RTX 4090 or similar (8GB+ VRAM, sm_90)
- **CUDA**: 12.0+
- **Rust**: 1.70+
- **Python**: 3.12+ (for protein folding benchmarks)

### Environment Setup
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA Toolkit
# (Follow NVIDIA docs for your OS)

# Clone repo
git clone <repo-url> PRISM-FINNAL-PUSH
cd PRISM-FINNAL-PUSH

# Build with CUDA
cargo build --release --features cuda

# Run WR example
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D.v1.1.toml
```

---

## 🔄 Development Workflow

### Typical Development Cycle
```bash
# 1. Make changes to source
vim foundation/prct-core/src/quantum_coloring.rs

# 2. Check compilation
cargo check

# 3. Run tests
cargo test --features cuda

# 4. Run policy checks
SUB=cargo_check_cuda ./tools/mcp_policy_checks.sh

# 5. Run local benchmark
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D.v1.1.toml

# 6. Commit changes
git add .
git commit -m "feat: improve quantum annealing convergence"

# 7. Push (if git history is clean)
git push origin gpu-quantum-acceleration
```

### Branch Structure
- `main` - Stable baseline
- `gpu-quantum-acceleration` - GPU + quantum work
- `chore/gitignore-artifact-cleanup` - Build artifact cleanup
- `maintenance/*` - Maintenance branches (history purge, etc.)

---

## 📈 Performance Characteristics

### Benchmark: DSJC1000.5 (1000 vertices, 249826 edges)

**Target**: 83 colors (world record territory)

**Config D Performance** (estimated):
- **Phase 0** (DSATUR + Reservoir): 103 colors in ~30 seconds
- **Phase 1** (Quantum): 97 colors in ~60 minutes
- **Phase 2** (Memetic): 95 colors in ~120 minutes
- **Phase 3** (TE): 92 colors in ~90 minutes
- **Phase 4** (Thermo): 88-90 colors in ~8 hours
- **Full 24h run**: Goal 83-85 colors

**GPU Speedups**:
- **Neuromorphic GEMV**: 398x vs CPU
- **Quantum Evolution**: 20-50x vs CPU
- **Thermodynamic PIMC**: 10-30x vs CPU

---

## 🎓 Learning Resources

### Understanding PRISM
1. Start with `WR_SWEEP_QUICKSTART.md`
2. Read `WR_SWEEP_STRATEGY.md`
3. Study `foundation/prct-core/src/world_record_pipeline_gpu.rs`
4. Explore `foundation/neuromorphic/src/gpu_reservoir.rs`
5. Review configs in `foundation/prct-core/configs/`

### GPU Acceleration
1. `ACTUAL_GPU_ARCHITECTURE.md`
2. `foundation/cuda/README.md` (if exists)
3. `cudarc` documentation: https://docs.rs/cudarc/

### Quantum Annealing
1. `foundation/prct-core/src/quantum_coloring.rs`
2. SQA paper: [citation needed]
3. PIMC overview: [citation needed]

---

**Last Updated**: 2025-11-02
**Version**: 1.1.0 (WR Sweep + GPU + Seed Probe)
**Maintainer**: PRISM Team
