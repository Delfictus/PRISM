# 🚀 Optimized PRISM Configuration Guide

## Configuration File Created

**File**: `foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml`

This configuration implements all recommended parameter optimizations for maximum color reduction on DSJC1000.5.

---

## 🎯 Key Optimizations Applied

### **1. Transfer Entropy (TE) - More Aggressive**
```toml
[transfer_entropy]
te_vs_kuramoto_weight = 0.85  # Was 0.70 → Push TE harder
geodesic_weight = 0.35         # Was 0.20 → Stronger geodesic influence
```
**Effect**: Lower initial chromatic number from TE phase

### **2. DSATUR Tie-Breaking - Lean on Active Inference**
```toml
[dsatur]
ai_weight = 0.55          # Was 0.40 → Leverage GPU Active Inference
reservoir_weight = 0.25   # Was 0.30 → Rebalance
geodesic_weight = 0.20    # Was 0.30 → Rebalance
```
**Effect**: Better vertex selection using GPU-computed expected free energy

### **3. Thermodynamic - Maximum Intensity**
```toml
[thermo]
replicas = 56         # Maximum VRAM-safe (was 48)
num_temps = 56        # Maximum VRAM-safe (was 48)
temperature_max = 15.0     # Was 10.0 → More exploration
temperature_min = 0.0005   # Was 0.001 → Finer refinement
steps_per_temp = 8000      # Was 5000 → More thorough
```
**Effect**: Dramatic escape from local minima

### **4. Quantum - Deeper Search**
```toml
[quantum]
iterations = 45           # Was 30
beta = 0.95              # Was 0.9
temperature = 0.7        # Was 1.0 → Less random

[orchestrator]
dsatur_target_offset = 2  # Was 3 → More aggressive target
```
**Effect**: Phase 3 pushes 2 colors below best (not 3), quantum searches deeper

### **5. Memetic - Aggressive Polishing**
```toml
[memetic]
population_size = 320      # Was 256
generations = 1200         # Was 900
mutation_rate = 0.09       # Was 0.05 → More exploration
local_search_depth = 20000 # Was 10000 → Intensive polishing
use_tsp_guidance = true    # Newly enabled
tsp_weight = 0.25          # Was 0.0
```
**Effect**: Better late-stage refinement

### **6. GPU Optimization**
```toml
[gpu]
streams = 3  # Was 1 → Enable TE/AI/Thermo overlap
enable_active_inference_gpu = true  # NEW
```
**Effect**: Better GPU utilization with concurrent execution

### **7. Clean Configuration**
```toml
use_pimc = false           # Not implemented, disable
use_gnn_screening = false  # Not implemented, disable
```
**Effect**: No time wasted on unimplemented features

---

## 📋 How to Use This Configuration

### **Quick Test Run** (to verify it works):
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Build if needed
cargo build --release --features cuda --example world_record_dsjc1000

# Run with optimized config
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml \
    2>&1 | tee results/optimized_test_$(date +%Y%m%d_%H%M%S).log
```

### **With GPU Monitoring**:
```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run PRISM
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml
```

### **Full 48-Hour World Record Attempt**:
```bash
mkdir -p results/wr_optimized

# Monitor GPU and save logs
nvidia-smi dmon -s pucvmet -c 200000 > results/wr_optimized/gpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &
GPU_PID=$!

# Run world record attempt
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml \
    2>&1 | tee results/wr_optimized/run_$(date +%Y%m%d_%H%M%S).log

kill $GPU_PID

# Check results
tail -100 results/wr_optimized/run_*.log | grep -E "Best|colors|GPU-STATUS"
```

---

## 🔧 Incremental Tuning Strategy

### **Phase 1: Test with Deterministic Mode** (as configured)
```bash
# Run 3 deterministic tests to establish baseline
for i in 1 2 3; do
    ./target/release/examples/world_record_dsjc1000 \
        foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml \
        2>&1 | tee results/deterministic_run_$i.log
done

# Compare results (should be identical)
grep "Best.*colors" results/deterministic_run_*.log
```

### **Phase 2: Switch to Exploration Mode**
Once you confirm the config works and consistently drops below 112 colors:

```bash
# Use prism-config to switch to non-deterministic
./target/release/prism-config set deterministic false \
    --config foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml

# Run exploration mode
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml
```

### **Phase 3: A/B Test Individual Parameters**
```bash
# Create variant configs
cp foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml variant_a.toml
cp foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml variant_b.toml

# Test different TE weights
./target/release/prism-config set transfer_entropy.te_vs_kuramoto_weight 0.85 --config variant_a.toml
./target/release/prism-config set transfer_entropy.te_vs_kuramoto_weight 0.90 --config variant_b.toml

# Run both
./target/release/examples/world_record_dsjc1000 variant_a.toml &
./target/release/examples/world_record_dsjc1000 variant_b.toml &
```

---

## 📊 Expected Performance

### **Conservative Estimates** (vs baseline 115 colors):
- Phase 1 (TE stronger): 121 → 118 colors (-3)
- Phase 2 (Thermo max): 118 → 114 colors (-4)
- Phase 3 (Quantum deeper): 114 → 111 colors (-3)
- Phase 4 (Memetic polish): 111 → 109 colors (-2)
- **Target**: **~109-112 colors** (vs world record 83)

### **Aggressive Estimates** (if everything clicks):
- Multi-pass iteration (3 passes)
- Each pass: 2-5 color reduction
- **Potential**: **~105-110 colors** range

---

## 🎯 Parameter Breakdown by Impact

### **High Impact** (expect 5-10 color reduction):
- Thermo max settings (replicas=56, steps=8000, wider temp range)
- TE aggressive weights (te_vs_kuramoto=0.85)
- Quantum deeper search (iterations=45, beta=0.95)

### **Medium Impact** (expect 2-5 color reduction):
- Memetic polish (pop=320, depth=20000)
- DSATUR AI weight (ai_weight=0.55)
- Iterative multi-pass (3 passes)

### **Low Impact** (expect 0-2 color reduction):
- GPU streams=3 (better utilization, not quality)
- TSP guidance (memetic local search)

---

## ⚠️ Important Notes

### **VRAM Safety**:
- replicas=56, num_temps=56 is at VRAM limit for 8GB GPUs
- If you get OOM errors, reduce both to 48
- Monitor with: `watch -n 1 nvidia-smi`

### **Runtime**:
- With these settings, expect ~20-30 hours for full 48h config
- Quick test (1 hour): Reduce `steps_per_temp = 2000`, `num_temps = 16`

### **Deterministic Mode**:
- Good for tuning and comparing configs
- Once satisfied, set `deterministic = false` for extra exploration
- Non-deterministic may find better solutions via randomness

---

## 🔥 Quick Start Commands

### **1. Verify Config**:
```bash
./target/release/prism-config validate \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml \
    --gpu --deep
```

### **2. Run Quick Test** (5-10 minutes):
```bash
# Create quick test version
cp foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml quick_opt_test.toml

# Reduce workload for testing
./target/release/prism-config set thermo.num_temps 8 --config quick_opt_test.toml
./target/release/prism-config set thermo.steps_per_temp 2000 --config quick_opt_test.toml
./target/release/prism-config set quantum.iterations 20 --config quick_opt_test.toml
./target/release/prism-config set memetic.generations 300 --config quick_opt_test.toml

# Run test
./target/release/examples/world_record_dsjc1000 quick_opt_test.toml
```

### **3. Run Full Optimized Attempt**:
```bash
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_optimized_aggressive.v1.1.toml \
    2>&1 | tee results/wr_optimized_$(date +%Y%m%d_%H%M%S).log
```

---

## 📈 Monitoring & Analysis

### **During Run**:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Watch progress
tail -f results/wr_optimized_*.log | grep -E "PHASE|colors|Best"
```

### **After Run**:
```bash
# Extract best result
grep "Best.*colors\|Final.*colors" results/wr_optimized_*.log | tail -5

# Check GPU usage
cat results/wr_optimized/gpu_*.log | awk 'BEGIN{sum=0; count=0} {sum+=$2; count++} END{print "Avg GPU: "sum/count"%"}'

# Check phase execution
grep "PHASE.*✅" results/wr_optimized_*.log
```

---

## 🎯 Next Steps

1. **Test the configuration** with quick settings first
2. **Verify it breaks 112 colors** consistently
3. **Enable non-deterministic** mode (`deterministic = false`)
4. **Run full 48-hour** world record attempt
5. **Report results**!

---

**Configuration ready to use!** Just run the commands above. 🚀
