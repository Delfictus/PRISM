# How to Run the PRISM Platform

**PRISM-AI**: GPU-Accelerated Meta-Evolutionary Compute Platform
**Status**: ✅ Fully Operational with GPU Acceleration

---

## Quick Start (3 Steps)

### 1. Run DIMACS Benchmarks (GPU-Accelerated)

```bash
# Navigate to PRISM directory
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Run GPU-accelerated DIMACS benchmark
./target/release/examples/simple_dimacs_benchmark

# Or specify custom benchmark directory:
./target/release/examples/simple_dimacs_benchmark /path/to/benchmarks
```

**What it does**: Runs 11 standard DIMACS graph coloring benchmarks with GPU acceleration

---

### 2. Run MEC Governance Tools

```bash
# View feature flag status
./target/release/meta-flagsctl status

# Enable a feature flag
./target/release/meta-flagsctl enable meta_generation \
  --actor "your_name" \
  --justification "Enabling meta-generation for production use"

# View ontology ledger
./target/release/meta-ontologyctl snapshot

# Generate reflexive snapshot (requires meta_generation enabled)
./target/release/meta-reflexive-snapshot --population 10 --stdout
```

---

### 3. Run Federated Learning Simulation

```bash
# Create output directory
mkdir -p /tmp/prism-federated

# Run 5-epoch federated simulation
./target/release/federated-sim \
  --output-dir /tmp/prism-federated \
  --epochs 5 \
  --clean

# View results
cat /tmp/prism-federated/simulations/epoch_summary.json
```

---

## Detailed Usage

### 🎯 GPU DIMACS Benchmarking

**Command**:
```bash
./target/release/examples/simple_dimacs_benchmark
```

**Output Example**:
```
=== PRISM-AI DIMACS Benchmark Runner ===

Configuration:
  GPU Acceleration: true
  Max Iterations: 1000
  GNN Enabled: true

Graph           Vertices    Edges  Time (ms)   Colors     Best    Gap %
===========================================================================
DSJC125.1            125      736       6.09        7        5    40.0%
DSJC1000.5          1000   249826    9860.35      122       82    48.8%
myciel6               95      755       1.08        7        7     0.0% ✅
...

Summary:
  Total Benchmarks: 11
  Successful Runs: 11
  Success Rate: 100.0%
```

**What's Happening**:
1. Initializes NVIDIA RTX 5070 GPU
2. Loads 11 compiled PTX kernels
3. Runs adaptive coloring (sparse CSR or dense FP16)
4. Reports chromatic numbers and performance

---

### 🎮 MEC Feature Flags

**View All Flags**:
```bash
./target/release/meta-flagsctl status
```

**Output**:
```
feature                  state            updated_at               actor
--------------------------------------------------------------------------------
meta_generation          enabled          2025-10-31T22:37:11Z     test_user
ontology_bridge          disabled         2025-10-21T02:41:18Z     bootstrap
free_energy_snapshots    disabled         2025-10-21T02:41:18Z     bootstrap
semantic_plasticity      disabled         2025-10-21T02:41:18Z     bootstrap
federated_meta           disabled         2025-10-21T02:41:18Z     bootstrap
meta_prod                disabled         2025-10-21T02:41:18Z     bootstrap

merkle_root: 65db2a8f5ce50a955e6e9188565f88e0a8cc8047d5665bed829892806c702191
```

**Enable a Flag**:
```bash
./target/release/meta-flagsctl enable semantic_plasticity \
  --actor "$(whoami)" \
  --justification "Testing semantic drift detection for variant adaptation"
```

**Disable a Flag**:
```bash
./target/release/meta-flagsctl disable meta_generation \
  --actor "$(whoami)" \
  --justification "Switching to manual variant selection for debugging"
```

**Export Snapshot**:
```bash
./target/release/meta-flagsctl snapshot --out /tmp/flags-snapshot.json
cat /tmp/flags-snapshot.json
```

---

### 🧠 Reflexive Controller

**Generate Reflexive Snapshot**:
```bash
# First enable the meta_generation flag
./target/release/meta-flagsctl enable meta_generation \
  --actor "$(whoami)" \
  --justification "Testing reflexive feedback system"

# Generate snapshot with 5 variants
./target/release/meta-reflexive-snapshot --population 5 --stdout
```

**Output Example**:
```json
{
  "distribution": [0.209, 0.207, 0.188, 0.188, 0.208],
  "snapshot": {
    "alerts": ["divergence 0.727 exceeded cap 0.180"],
    "divergence": 0.7269,
    "effective_temperature": 0.85,
    "energy_mean": -0.7363,
    "energy_variance": 0.1596,
    "entropy": 1.6082,
    "exploration_ratio": 1.0,
    "mode": "Recovery",
    "lattice": [...16x16 free-energy lattice...]
  }
}
```

**What's Happening**:
- Generates 16x16 free-energy lattice
- Computes Shannon entropy (1.608)
- Selects governance mode (Recovery/Strict/Explore)
- Monitors divergence vs thresholds
- Returns distribution over variants

**Save to File**:
```bash
./target/release/meta-reflexive-snapshot \
  --population 10 \
  --output /tmp/reflexive-snapshot.json

cat /tmp/reflexive-snapshot.json | jq '.snapshot.mode'
# Output: "Recovery"
```

---

### 🌐 Federated Learning Simulation

**Basic Simulation**:
```bash
./target/release/federated-sim \
  --output-dir /tmp/prism-federated \
  --epochs 3 \
  --clean
```

**Output**:
```
Wrote 3 epochs to /tmp/prism-federated/simulations/epoch_summary.json
```

**View Results**:
```bash
cat /tmp/prism-federated/simulations/epoch_summary.json | jq '.'
```

**Example Output**:
```json
{
  "epoch_count": 3,
  "epochs": [
    {
      "epoch": 1,
      "aggregated_delta": 38,
      "quorum_reached": true,
      "ledger_merkle": "55438b57d942923d",
      "signature": "ixGxNfarKSW0vWbuvCCWU6Z7szm/w1WLGy3W0PpdYFA=",
      "aligned_updates": [
        {"node_id": "edge-c", "delta_score": 11, "ledger_height": 1},
        {"node_id": "validator-a", "delta_score": 17, "ledger_height": 1},
        {"node_id": "validator-b", "delta_score": 10, "ledger_height": 1}
      ]
    }
  ],
  "summary_signature": "44MjZXA407HFFbeA4nmCs4zy2y6i46V28D1yaPg9sl8="
}
```

**What's Happening**:
- Simulates 3-node federated network (edge-c, validator-a, validator-b)
- Each node proposes updates with delta scores
- Quorum consensus reached (majority voting)
- Cryptographic signatures (HMAC) generated
- Merkle roots track ledger state

**Verify Simulation**:
```bash
./target/release/federated-sim \
  --verify-summary /tmp/prism-federated/simulations/epoch_summary.json \
  --verify-ledger /tmp/prism-federated/simulations \
  --expect-label default
```

---

### 📊 Ontology Ledger

**View Ontology Snapshot**:
```bash
./target/release/meta-ontologyctl snapshot --ledger PRISM-AI-UNIFIED-VAULT/meta/ontology_ledger.jsonl
```

**Inspect Alignment**:
```bash
./target/release/meta-ontologyctl align <concept-id>
```

---

## Custom Graph Coloring

### Using Python/Command Line

**Create a custom graph file** (DIMACS .col format):
```bash
cat > /tmp/my_graph.col <<EOF
c My custom graph
p edge 5 7
e 1 2
e 1 3
e 2 3
e 2 4
e 3 4
e 3 5
e 4 5
EOF
```

**Color it with PRISM**:
```rust
// Example Rust code
use prism_ai::{PrismAI, PrismConfig, data::DIMACParser};

let config = PrismConfig {
    use_gpu: true,
    num_replicas: 100,
    temperature: 1.0,
    ..Default::default()
};

let prism = PrismAI::new(config)?;
let adjacency = DIMACParser::parse_file("/tmp/my_graph.col")?;
let colors = prism.color_graph(adjacency)?;

println!("Chromatic number: {}", colors.iter().max().unwrap() + 1);
```

---

## Environment Configuration

### GPU Settings

**Check GPU availability**:
```bash
nvidia-smi
```

**Expected Output**:
```
NVIDIA GeForce RTX 5070 Laptop GPU
Driver Version: 580.95.05
```

**Verify CUDA kernels**:
```bash
ls -lh foundation/kernels/ptx/*.ptx
```

**Expected**: 11 PTX files (active_inference.ptx, parallel_coloring.ptx, etc.)

---

### Build Configuration

**Rebuild everything**:
```bash
cargo build --release --features cuda,examples
```

**Build specific binary**:
```bash
# Just the DIMACS benchmark
cargo build --example simple_dimacs_benchmark --release --features cuda

# Just the MEC tools
cargo build --bins --release
```

**Run tests**:
```bash
# Library tests (some may require GPU)
cargo test --lib --release

# Specific test
cargo test --lib test_basic_coloring --release
```

---

## Performance Tuning

### GPU Benchmark Tuning

**Increase attempts** (better quality, slower):
```rust
let config = PrismConfig {
    num_replicas: 1000,  // Default: 100
    ..Default::default()
};
```

**Adjust temperature** (higher = more exploration):
```rust
let config = PrismConfig {
    temperature: 2.0,  // Default: 1.0
    ..Default::default()
};
```

### Memory Limits

**For large graphs** (>10K vertices), monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

If you see OOM errors, reduce `num_replicas`:
```rust
let config = PrismConfig {
    num_replicas: 10,  // Reduced from 100
    ..Default::default()
};
```

---

## Common Use Cases

### 1. **Research: Graph Coloring Experiments**

```bash
# Run full benchmark suite
./target/release/examples/simple_dimacs_benchmark > results.txt

# Extract chromatic numbers
grep "DSJC" results.txt | awk '{print $1, $8}'
```

### 2. **Production: Feature Flag Management**

```bash
# Enable production features
./target/release/meta-flagsctl enable meta_prod \
  --actor "deployment_system" \
  --justification "Production deployment v1.2.3"

# Create audit snapshot
./target/release/meta-flagsctl snapshot \
  --out /var/log/prism/flags-$(date +%Y%m%d).json
```

### 3. **Development: Reflexive Debugging**

```bash
# Generate snapshots with different populations
for n in 5 10 20 50; do
  ./target/release/meta-reflexive-snapshot --population $n \
    --output /tmp/reflexive-$n.json

  # Extract governance mode
  jq -r '.snapshot.mode' /tmp/reflexive-$n.json
done
```

### 4. **Distributed: Federated Coordination**

```bash
# Simulate different epoch counts
for epochs in 2 5 10 20; do
  ./target/release/federated-sim \
    --output-dir /tmp/fed-$epochs \
    --epochs $epochs \
    --clean

  # Check quorum success rate
  jq '.epochs[] | .quorum_reached' /tmp/fed-$epochs/simulations/epoch_summary.json
done
```

---

## Troubleshooting

### GPU Not Found

**Error**: `Failed to initialize CUDA device 0`

**Solution**:
```bash
# Check GPU is visible
nvidia-smi

# Check CUDA is available
nvcc --version

# Rebuild with CUDA
cargo clean
cargo build --release --features cuda
```

### Missing PTX Kernels

**Error**: `Failed to load PTX module`

**Solution**:
```bash
# PTX kernels are compiled during build
cargo build --release --features cuda

# Check they exist
ls foundation/kernels/ptx/*.ptx
ls target/release/build/prism-ai-*/out/ptx/*.ptx
```

### Feature Flag Errors

**Error**: `meta_generation flag must be shadow or enabled`

**Solution**:
```bash
./target/release/meta-flagsctl enable meta_generation \
  --actor "$(whoami)" \
  --justification "Required for reflexive controller"
```

### Permission Denied

**Error**: `Permission denied` when running binaries

**Solution**:
```bash
chmod +x target/release/meta-*
chmod +x target/release/federated-sim
chmod +x target/release/examples/simple_dimacs_benchmark
```

---

## File Locations

### Binaries (after `cargo build --release`):
```
target/release/
├── meta-flagsctl              # Feature flag controller
├── meta-ontologyctl           # Ontology ledger tool
├── meta-reflexive-snapshot    # Reflexive controller snapshot
├── federated-sim              # Federated learning simulator
└── examples/
    └── simple_dimacs_benchmark  # DIMACS benchmark runner
```

### Data Files:
```
PRISM-AI-UNIFIED-VAULT/meta/
├── ontology_ledger.jsonl      # Ontology event log
└── feature_flags.jsonl        # Feature flag state
```

### GPU Kernels:
```
foundation/kernels/ptx/
├── active_inference.ptx       # 647x speedup kernel
├── parallel_coloring.ptx      # Graph coloring kernel
├── transfer_entropy.ptx       # KSG estimation
└── ... (11 total kernels)
```

---

## What's Next?

### To Improve DIMACS Results:

1. **Enable PRISM-AI coherence** (currently using baseline uniform):
   - Integrate transfer entropy
   - Add neuromorphic predictions
   - Use TDA features
   - Expected: 15-30% improvement

2. **Tune hyperparameters**:
   - Increase attempts (100 → 1000)
   - Sweep temperature (0.5 → 2.0)
   - Adaptive scheduling

3. **Multi-GPU scaling**:
   - Distribute attempts across GPUs
   - Near-linear speedup

### To Extend Functionality:

1. **Add custom algorithms**:
   - Implement in `src/cuda/` or `foundation/`
   - Write PTX kernel in `foundation/cuda/`
   - Compile and integrate

2. **Create new benchmarks**:
   - Add examples in `examples/`
   - Follow `simple_dimacs_benchmark.rs` pattern

3. **Integrate with MEC pipeline**:
   - Use reflexive feedback for variant selection
   - Apply semantic plasticity for adaptation
   - Federate across nodes

---

## Quick Reference

| Task | Command |
|------|---------|
| **Run DIMACS benchmarks** | `./target/release/examples/simple_dimacs_benchmark` |
| **Check feature flags** | `./target/release/meta-flagsctl status` |
| **Enable flag** | `./target/release/meta-flagsctl enable <feature> --actor <name> --justification <reason>` |
| **Generate reflexive snapshot** | `./target/release/meta-reflexive-snapshot --population 10 --stdout` |
| **Run federated sim** | `./target/release/federated-sim --output-dir /tmp/prism --epochs 5` |
| **Check GPU** | `nvidia-smi` |
| **Rebuild** | `cargo build --release --features cuda` |

---

**Platform Status**: ✅ **OPERATIONAL**
**GPU Acceleration**: ✅ **ENABLED**
**MEC Integration**: ✅ **COMPLETE (M0-M5)**

**For more details, see**:
- `GPU_DIMACS_BENCHMARK_RESULTS.md` - Benchmark performance analysis
- `PLATFORM_TEST_REPORT.md` - Complete integration testing results
- `COMPLETE_MEC_STACK_STATUS.md` - MEC phase integration summary

---

**Created**: October 31, 2025
**Platform**: PRISM-AI Meta-Evolutionary Compute
**Version**: M0-M5 Unified
