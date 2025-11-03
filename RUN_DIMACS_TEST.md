# How to Run Full DIMACS Test Suite

## Quick Start (1 Command)

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
./run_full_dimacs_test.sh
```

That's it! The script will:
1. ✅ Check GPU availability
2. ✅ Verify all 14 DIMACS benchmark files
3. ✅ Build the benchmark binary (if needed)
4. ✅ Configure feature flags
5. ✅ Run complete GPU-accelerated benchmark suite
6. ✅ Generate detailed results and analysis

---

## What You'll See

### Pre-flight Checks:
```
[1/6] Running pre-flight checks...
  Checking GPU... ✓ NVIDIA GeForce RTX 5070 Laptop GPU
  Checking benchmark binary... ✓
  Found 14 DIMACS benchmark files
```

### System Information Collection:
```
[2/6] Setting up output directory...
  Output dir: /home/diddy/Desktop/PRISM-FINNAL-PUSH/test_results
  Results file: test_results/dimacs_results_20251031_223045.txt
```

### Benchmark Execution:
```
[5/6] Running DIMACS benchmark suite...
  This may take several minutes for large graphs...

=== PRISM-AI DIMACS Benchmark Runner ===

Graph           Vertices    Edges  Time (ms)   Colors     Best    Gap %
===========================================================================
DSJC125.1            125      736       6.09        7        5    40.0%
DSJC1000.5          1000   249826    9860.35      122       82    48.8%
myciel6               95      755       1.08        7        7     0.0% ✅
...
```

### Final Summary:
```
╔════════════════════════════════════════════════════════════════╗
║              DIMACS Benchmark Summary Report                   ║
╚════════════════════════════════════════════════════════════════╝

Test Run: 20251031_223045
Duration: 125s (00:02:05)

Results:
--------
Total Graphs Tested: 14
Perfect Scores (0% gap): 1
Average Gap to Best Known: 28.5%

GPU Performance:
----------------
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Total GPU Time: 12.34 seconds
Fastest Graph: queen8_8 - 0.60ms
Slowest Graph: DSJC1000.5 - 9860.35ms

Files Generated:
----------------
Results: test_results/dimacs_results_20251031_223045.txt
Summary: test_results/summary_20251031_223045.txt
JSON: test_results/dimacs_results_20251031_223045.json
```

---

## Output Files

After running, you'll find these files in `test_results/`:

### 1. **Full Results** (`dimacs_results_TIMESTAMP.txt`)
Complete benchmark output with all graph results, timing, and GPU kernel info.

**View it**:
```bash
cat test_results/dimacs_results_*.txt | less
```

### 2. **Summary Report** (`summary_TIMESTAMP.txt`)
High-level statistics, best results, performance analysis.

**View it**:
```bash
cat test_results/summary_*.txt
```

### 3. **JSON Data** (`dimacs_results_TIMESTAMP.json`)
Programmatic access to results.

**View it**:
```bash
cat test_results/dimacs_results_*.json | jq '.'
```

Example:
```json
{
  "timestamp": "20251031_223045",
  "duration_seconds": 125,
  "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
  "total_graphs": 14,
  "perfect_scores": 1,
  "average_gap_percent": 28.5
}
```

### 4. **System Info** (`system_info_TIMESTAMP.txt`)
GPU specs, CPU info, memory, PRISM version.

### 5. **Full Log** (`test_log_TIMESTAMP.log`)
Complete verbose output including build logs, flag updates, etc.

---

## Expected Runtime

| Graph Size | Approximate Time |
|------------|-----------------|
| Small (64-125 vertices) | 1-10ms per graph |
| Medium (250-500 vertices) | 30-200ms per graph |
| Large (1000 vertices) | 5-10 seconds per graph |
| **Total Suite** | **~2-3 minutes** |

---

## Benchmarks Tested

The script tests all 14 standard DIMACS graphs:

| Graph | Vertices | Edges | Type | Best Known χ |
|-------|----------|-------|------|--------------|
| DSJC125.1 | 125 | 736 | Random sparse | 5 |
| DSJC125.5 | 125 | 3,891 | Random medium | 17 |
| DSJC125.9 | 125 | 6,961 | Random dense | 44 |
| DSJC250.5 | 250 | 15,668 | Random medium | 28 |
| DSJC500.5 | 500 | 62,624 | Random medium | 48 |
| **DSJC1000.5** | **1,000** | **249,826** | **Random (WR target)** | **82** |
| DSJR500.1 | 500 | 3,555 | Random sparse | 12 |
| DSJR500.5 | 500 | 62,335 | Random dense | 122 |
| queen8_8 | 64 | 1,456 | Geometric | 9 |
| queen11_11 | 121 | 3,960 | Geometric | 11 |
| myciel5 | 47 | 236 | Triangle-free | 6 |
| myciel6 | 95 | 755 | Triangle-free | 7 |
| le450_15a | 450 | 8,168 | Adversarial | 15 |
| le450_25a | 450 | 8,260 | Adversarial | 25 |

---

## Customization

### Run with Different Benchmark Directory

```bash
# Edit the script
nano run_full_dimacs_test.sh

# Change this line:
BENCHMARK_DIR="/your/custom/path"
```

### Run Single Graph

```bash
./target/release/examples/simple_dimacs_benchmark /path/to/single/graph.col
```

### Adjust GPU Parameters

Edit `examples/simple_dimacs_benchmark.rs` and rebuild:
```rust
let mut config = PrismConfig::default();
config.num_replicas = 1000;  // More attempts (slower, better quality)
config.temperature = 2.0;    // More exploration
```

Then rebuild:
```bash
cargo build --example simple_dimacs_benchmark --release --features cuda
```

---

## Troubleshooting

### GPU Not Found
```bash
# Check GPU
nvidia-smi

# If not found, install drivers:
sudo ubuntu-drivers autoinstall
```

### Build Errors
```bash
# Clean rebuild
cargo clean
cargo build --example simple_dimacs_benchmark --release --features cuda
```

### Permission Denied
```bash
# Make script executable
chmod +x run_full_dimacs_test.sh
```

### Missing Benchmarks
```bash
# Download DIMACS benchmarks
git clone https://github.com/Delfictus/PRISM
# Benchmarks are in: PRISM-master/benchmarks/dimacs/
```

---

## Compare Results

### Run Multiple Times
```bash
# Run 3 test iterations
for i in 1 2 3; do
  echo "Run $i"
  ./run_full_dimacs_test.sh
  sleep 5
done
```

### Compare Performance
```bash
# Extract chromatic numbers from all runs
grep "DSJC1000.5" test_results/dimacs_results_*.txt | awk '{print $8}'

# Compare timing
grep "Total GPU Time" test_results/summary_*.txt
```

---

## Next Steps After Testing

### 1. **Analyze Results**
```bash
# View best performing graphs
grep "0.0%" test_results/dimacs_results_*.txt

# View worst performing graphs
grep "DSJC" test_results/dimacs_results_*.txt | sort -k10 -rn | head -5
```

### 2. **Tune Parameters**
Based on results, adjust:
- `num_replicas`: Higher = better quality, slower
- `temperature`: Higher = more exploration
- Kernel selection threshold

### 3. **Enable PRISM-AI Features**
Currently using baseline (uniform coherence). To improve:
- Enable transfer entropy coherence
- Add neuromorphic predictions
- Integrate TDA features
- Expected: 15-30% improvement

### 4. **Share Results**
```bash
# Create shareable report
cat test_results/summary_*.txt test_results/system_info_*.txt > my_dimacs_results.txt
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./run_full_dimacs_test.sh` | **Run complete test suite** |
| `cat test_results/summary_*.txt` | View summary |
| `cat test_results/dimacs_results_*.txt` | View full results |
| `ls -lt test_results/` | List all test runs |
| `nvidia-smi` | Check GPU status |

---

**Ready to run!** Just execute:
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
./run_full_dimacs_test.sh
```

The test suite is fully automated and will take ~2-3 minutes to complete.
