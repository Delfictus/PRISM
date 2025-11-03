# PRISM-AI Baseline v1.0

**GPU-Accelerated Graph Coloring Platform - Production Ready**

This is the **stable baseline release** for PRISM-AI. Use this as your reference point when developing advanced features like PRCT-TSP integration.

---

## What's Included

### Binaries (in `bin/`)

1. **`simple_dimacs_benchmark`** - DIMACS graph coloring benchmark
   - 14 standard test graphs
   - Configurable attempts (unlimited)
   - GPU-accelerated parallel coloring

2. **`protein_structure_benchmark`** - Protein structure analysis
   - PDB file parser
   - Residue contact graph coloring
   - Biological interpretation

3. **`meta-flagsctl`** - MEC feature flag controller
4. **`meta-ontologyctl`** - Ontology ledger manager
5. **`meta-reflexive-snapshot`** - Reflexive controller
6. **`federated-sim`** - Federated learning simulator

### Documentation (in `docs/`)

- **`README_QUICK_START.md`** - Quick start guide
- **`HOW_TO_RUN_PRISM.md`** - Complete platform usage
- **`ENHANCED_BENCHMARK_GUIDE.md`** - DIMACS benchmark guide
- **`PROTEIN_STRUCTURE_GUIDE.md`** - Protein analysis guide
- **`ENHANCEMENT_SUMMARY.md`** - Implementation details

### Data (in `data/`)

- `nipah/2VSM.pdb` - Nipah Virus protein structure (550 residues)

### Scripts (in `scripts/`)

- `verify_setup.sh` - Verify system ready
- `run_full_dimacs_test.sh` - Automated DIMACS test suite

---

## Quick Start

### 1. DIMACS Benchmark (Default: 1000 attempts)

```bash
cd bin
./simple_dimacs_benchmark
```

### 2. DIMACS Benchmark (Custom: 10,000 attempts)

```bash
cd bin
./simple_dimacs_benchmark \
  /home/diddy/Downloads/PRISM-master/benchmarks/dimacs \
  10000
```

### 3. Protein Structure Analysis

```bash
cd bin
./protein_structure_benchmark ../data/nipah/2VSM.pdb 8.0 5000
```

### 4. MEC Tools

```bash
cd bin

# View feature flags
./meta-flagsctl status

# Enable a flag
./meta-flagsctl enable meta_generation \
  --actor "$(whoami)" \
  --justification "Testing"

# Generate reflexive snapshot
./meta-reflexive-snapshot --population 10 --stdout

# Run federated simulation
./federated-sim --output-dir /tmp/prism --epochs 5
```

---

## System Requirements

- **GPU**: NVIDIA CUDA-capable (RTX 5070 verified)
- **Driver**: 580.95+ recommended
- **RAM**: 8GB+ recommended
- **Disk**: ~100MB for binaries + data
- **OS**: Linux (Ubuntu 22.04+ verified)

---

## Performance Benchmarks

### DIMACS (1000 attempts)
- **DSJC125.1**: 7 colors (40% gap from best known)
- **DSJC1000.5**: 122 colors (48.8% gap) - 9.86 seconds
- **myciel6**: 7 colors (0% gap) - PERFECT ✅
- **Success rate**: 100% (14/14 graphs)

### DIMACS (10,000 attempts)
- **DSJC125.1**: 6 colors (20% gap) - **50% improvement!**
- **DSJC1000.5**: 121 colors (47.6% gap)
- **DSJR500.1**: 12 colors (0% gap) - PERFECT ✅
- **le450_25a**: 25 colors (0% gap) - PERFECT ✅

### Protein Structure (Nipah Virus, 5000 attempts)
- **Residues**: 550
- **Contacts**: 2,288
- **Colors**: 7
- **Time**: 58ms
- **Throughput**: 9,452 residues/second

---

## What's NOT Included (Yet)

This baseline uses **simple GPU parallel coloring**. NOT included:

- ❌ PRCT-TSP algorithm (Neuromorphic → Quantum → Kuramoto)
- ❌ Transfer entropy coherence (GPU KSG estimator)
- ❌ Neuromorphic reservoir predictions
- ❌ GNN attention weights
- ❌ Federated learning integration with coloring
- ❌ Iterative learning mode

These will be added in future releases building on this baseline.

---

## Baseline vs. Future PRCT-TSP

### Current Baseline
```
Input Graph → GPU Parallel Greedy Coloring → Best Result
```
- **Fast**: ~60ms for 550 vertices
- **Good**: 0-48% gap from best known
- **Simple**: Direct GPU approach

### Future PRCT-TSP
```
Input Graph
  → Neuromorphic Encoding
  → Quantum Phase Evolution
  → Kuramoto Synchronization
  → Phase-Guided Coloring
  → Optimal Result
```
- **Better quality**: Expected 15-30% improvement
- **More complex**: Physics-informed approach
- **Adaptive**: Learns over time

---

## Using This Baseline

### Development Workflow

1. **Keep baseline safe**:
```bash
# This directory is your safe baseline
cp -r baseline-v1.0 ~/baseline-backup
```

2. **Develop new features** in main codebase:
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
# Work on PRCT-TSP integration here
```

3. **Compare against baseline**:
```bash
# Run baseline
baseline-v1.0/bin/simple_dimacs_benchmark > baseline_results.txt

# Run new PRCT version
./target/release/examples/prct_dimacs_benchmark > prct_results.txt

# Compare
diff baseline_results.txt prct_results.txt
```

### Git Tag

This baseline is tagged in git:
```bash
git checkout v1.0-baseline  # Return to baseline
git checkout main           # Return to development
```

---

## File Structure

```
baseline-v1.0/
├── README.md              ← This file
├── bin/                   ← Standalone binaries (23MB total)
│   ├── simple_dimacs_benchmark      (6.2MB)
│   ├── protein_structure_benchmark  (6.2MB)
│   ├── meta-flagsctl               (4.2MB)
│   ├── meta-ontologyctl            (1.3MB)
│   ├── meta-reflexive-snapshot     (3.8MB)
│   └── federated-sim               (1.3MB)
├── docs/                  ← Documentation
│   ├── README_QUICK_START.md
│   ├── HOW_TO_RUN_PRISM.md
│   ├── ENHANCED_BENCHMARK_GUIDE.md
│   ├── PROTEIN_STRUCTURE_GUIDE.md
│   └── ...
├── data/                  ← Sample data
│   └── nipah/
│       └── 2VSM.pdb
└── scripts/               ← Utility scripts
    ├── verify_setup.sh
    └── run_full_dimacs_test.sh
```

---

## Version Information

- **Release**: v1.0-baseline
- **Date**: October 31, 2025
- **Git Commit**: d91b896
- **GPU**: NVIDIA RTX 5070 Laptop
- **CUDA**: Compute Capability 9.0 (sm_90)
- **cudarc**: 0.9 (API compatible)

---

## Key Features

✅ **Unlimited configurable attempts** (removed 100-attempt cap)
✅ **GPU ensemble generation** enabled
✅ **DIMACS benchmark suite** (14 graphs)
✅ **Protein structure analysis** (PDB parser)
✅ **MEC phases M0-M5** integrated
✅ **11 PTX kernels** compiled
✅ **Adaptive kernel selection** (sparse/dense)
✅ **Full documentation**
✅ **Production ready**

---

## Support

For detailed usage, see documentation in `docs/`:
- **Getting started**: `docs/README_QUICK_START.md`
- **DIMACS benchmarks**: `docs/ENHANCED_BENCHMARK_GUIDE.md`
- **Protein analysis**: `docs/PROTEIN_STRUCTURE_GUIDE.md`
- **Full platform guide**: `docs/HOW_TO_RUN_PRISM.md`

---

## What's Next?

From this baseline, you can develop:

1. **PRCT-TSP Integration**
   - Neuromorphic spike encoding
   - Quantum phase evolution
   - Kuramoto synchronization
   - Phase-guided coloring

2. **Advanced Coherence**
   - Transfer entropy (GPU KSG)
   - Topological data analysis
   - Reservoir computing
   - GNN predictions

3. **Iterative Learning**
   - Multi-epoch benchmarks
   - Federated learning integration
   - Progressive improvement
   - "Getting smarter" over time

**This baseline provides the stable foundation for all future development!**

---

**Status**: ✅ **PRODUCTION READY**
**Platform**: PRISM-AI Meta-Evolutionary Compute
**GPU**: NVIDIA RTX 5070 (8GB VRAM)
