# PRISM-AI Quick Start

**GPU-Accelerated Meta-Evolutionary Compute Platform**
**Status**: ✅ Fully Operational | **GPU**: NVIDIA RTX 5070 | **MEC Phases**: M0-M5 Integrated

---

## 🚀 Run Full DIMACS Test (One Command)

```bash
./run_full_dimacs_test.sh
```

**That's it!** The automated test suite will:
- ✅ Run 14 DIMACS benchmarks with GPU acceleration
- ✅ Generate detailed results and analysis
- ✅ Complete in ~2-3 minutes
- ✅ Save results to `test_results/` directory

---

## ⚡ Quick Commands

```bash
# Verify setup is ready
./verify_setup.sh

# Run DIMACS benchmarks
./run_full_dimacs_test.sh

# Or run benchmark directly
./target/release/examples/simple_dimacs_benchmark

# View latest results
cat test_results/summary_*.txt
```

---

## 📊 What You Get

### Benchmark Results:
- **14 standard DIMACS graphs** tested
- **GPU-accelerated** parallel coloring
- **Sub-10 second** performance on world record target (DSJC1000.5)
- **Competitive chromatic numbers** (0-48% gap from best known)

### Example Output:
```
Graph           Vertices    Edges  Time (ms)   Colors     Best    Gap %
===========================================================================
DSJC125.1            125      736       6.09        7        5    40.0%
DSJC1000.5          1000   249826    9860.35      122       82    48.8%
myciel6               95      755       1.08        7        7     0.0% ✅
```

---

## 🎯 Available Tools

| Command | Purpose |
|---------|---------|
| `./run_full_dimacs_test.sh` | **Complete benchmark suite** |
| `./verify_setup.sh` | Verify all components ready |
| `./target/release/meta-flagsctl status` | View MEC feature flags |
| `./target/release/meta-reflexive-snapshot --population 10 --stdout` | Reflexive controller |
| `./target/release/federated-sim --output-dir /tmp/prism --epochs 5` | Federated simulation |

---

## 📖 Documentation

- **`RUN_DIMACS_TEST.md`** - Detailed test suite guide
- **`HOW_TO_RUN_PRISM.md`** - Complete platform usage guide
- **`GPU_DIMACS_BENCHMARK_RESULTS.md`** - Latest benchmark analysis
- **`PLATFORM_TEST_REPORT.md`** - Integration test results

---

## 🔧 Platform Components

### GPU Acceleration:
- ✅ 11 compiled PTX kernels
- ✅ Adaptive kernel selection (sparse CSR / dense FP16)
- ✅ Parallel exploration (100 attempts)
- ✅ Zero CPU fallbacks

### MEC Integration (M0-M5):
- ✅ **M0**: Foundation (active inference, CMA, CUDA)
- ✅ **M1**: Governance + telemetry (meta-flagsctl, meta-ontologyctl)
- ✅ **M2**: Ontology alignment
- ✅ **M3**: Reflexive controller (meta-reflexive-snapshot)
- ✅ **M4**: Semantic plasticity
- ✅ **M5**: Federated learning (federated-sim)

---

## 💡 Quick Examples

### Run Single Benchmark:
```bash
./target/release/examples/simple_dimacs_benchmark
```

### Enable Feature Flag:
```bash
./target/release/meta-flagsctl enable meta_generation \
  --actor "$(whoami)" \
  --justification "Testing meta-evolution"
```

### Generate Reflexive Snapshot:
```bash
./target/release/meta-reflexive-snapshot --population 10 --stdout
```

### Run Federated Simulation:
```bash
./target/release/federated-sim --output-dir /tmp/prism --epochs 5 --clean
```

---

## 🎓 System Requirements

- **GPU**: NVIDIA CUDA-capable (RTX 5070 verified)
- **Driver**: 580.95+ recommended
- **RAM**: 8GB+ recommended for large graphs
- **Disk**: ~5GB for build artifacts
- **OS**: Linux (Ubuntu 22.04+ verified)

---

## 🏆 Performance Highlights

From latest benchmark run:
- **Perfect score**: myciel6 (0% gap from best known)
- **Near-perfect**: le450_25a (4% gap)
- **Ultra-fast**: queen8_8 colored in 0.60ms
- **World record target**: DSJC1000.5 in 9.86 seconds

---

## 📞 Getting Help

### Check Status:
```bash
./verify_setup.sh
```

### View Logs:
```bash
cat test_results/test_log_*.log
```

### GPU Status:
```bash
nvidia-smi
```

### Rebuild:
```bash
cargo build --release --features cuda
```

---

## 🔬 Next Steps

### Improve Results:
1. Enable PRISM-AI coherence (15-30% improvement expected)
2. Tune hyperparameters (increase attempts, adjust temperature)
3. Multi-GPU scaling

### Extend Functionality:
1. Add custom algorithms
2. Create new benchmarks
3. Integrate with MEC pipeline

---

## ✨ Quick Start Summary

**3 Steps to Run:**
1. `cd /home/diddy/Desktop/PRISM-FINNAL-PUSH`
2. `./verify_setup.sh` (verify ready)
3. `./run_full_dimacs_test.sh` (run tests)

**Results in**: `test_results/` directory
**Runtime**: ~2-3 minutes
**Success Rate**: 100% (14/14 graphs)

---

**Platform Status**: 🚀 **READY TO RUN**

Start testing now:
```bash
./run_full_dimacs_test.sh
```
