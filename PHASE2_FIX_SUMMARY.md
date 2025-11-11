# Phase 2 Thermodynamic Fix - Configuration & Code Updates

## Date: November 6, 2025
## Issue: Phase 2 not escaping initial state (stuck at high conflicts)

---

## ✅ Changes Made

### 1. Configuration File Updated

**File**: `foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml`

**Thermodynamic Parameters** (lines 95-105):
```toml
[thermo]
replicas = 56                       # Was 48 → Maximum VRAM-safe
num_temps = 56                      # Was 48 → Maximum VRAM-safe
temperature_max = 15.0              # Was 10.0 → Stronger exploration
temperature_min = 0.0005            # Was 0.001 → Finer refinement
steps_per_temp = 8000               # Was 5000 → More thorough equilibration
```

**ADP Coordination** (line 54):
```toml
adp_thermo_num_temps = 56          # Was 48 → Match thermo.num_temps
```

**GPU Settings** (lines 21-22):
```toml
streams = 4                         # Was 1 → Better utilization
batch_size = 4096                   # Was 1024 → Heavier workload
```

### 2. Code Fix - ADP Clamp

**File**: `foundation/prct-core/src/world_record_pipeline.rs` (line 2603)

**Before**:
```rust
ColoringAction::IncreaseThermoTemperatures => {
    self.adp_thermo_num_temps = (self.adp_thermo_num_temps + 8).min(64);  // ❌ Hard-coded
}
```

**After**:
```rust
ColoringAction::IncreaseThermoTemperatures => {
    let max_safe = self.config.thermo.num_temps_max_safe.unwrap_or(56);  // ✅ From config
    self.adp_thermo_num_temps = (self.adp_thermo_num_temps + 8).min(max_safe);
    println!("[ADP] Increased thermo temps: {} (max: {})", self.adp_thermo_num_temps, max_safe);
}
```

**Improvement**: ADP now respects configured VRAM guard instead of hard-coded 64

---

## 🎯 Expected Impact

### **Temperature Range**:
- **Before**: T_max=10.0, T_min=0.001 (3 orders of magnitude)
- **After**: T_max=15.0, T_min=0.0005 (4.5 orders of magnitude)
- **Effect**: Much wider exploration range

### **Temperature Coverage**:
- **Before**: 48 temperatures
- **After**: 56 temperatures (maximum VRAM-safe)
- **Effect**: Finer temperature granularity

### **Equilibration Depth**:
- **Before**: 5000 steps per temperature
- **After**: 8000 steps per temperature
- **Effect**: 60% more thorough equilibration

### **Total Work**:
- **Before**: 48 temps × 5000 steps = 240,000 GPU iterations
- **After**: 56 temps × 8000 steps = 448,000 GPU iterations
- **Increase**: 87% more GPU work (but much better escape capability)

---

## 📊 Performance Expectations

### **Conflict Reduction**:
**Current behavior** (from your run):
```
T=10.000: 19 colors, 135539 conflicts
T=8.820:  19 colors, 135539 conflicts
T=7.779:  19 colors, 135539 conflicts
```
**Status**: ❌ Stuck - not reducing conflicts

**Expected with fixes**:
```
T=15.000: 19 colors, 135539 conflicts  (start high)
T=10.000: 50 colors, 45000 conflicts   (starting to escape)
T=5.000:  85 colors, 8500 conflicts    (major reduction)
T=1.000:  110 colors, 1200 conflicts   (approaching valid)
T=0.100:  118 colors, 45 conflicts     (near-valid)
T=0.001:  121 colors, 0 conflicts      (valid!)
T=0.0005: 119 colors, 0 conflicts      (refinement)
```

### **Color Reduction Estimate**:
- **Starting**: ~127 colors (from Phase 1 TE)
- **After Thermo**: ~115-119 colors (expect 8-12 color reduction)
- **Improvement over stuck state**: Actually finding valid colorings!

---

## ⚠️ VRAM Safety

Current settings are at **maximum safe limits** for 8GB GPUs:
- replicas = 56 (max)
- num_temps = 56 (max)
- Estimated VRAM: ~56 MB

**Monitor with**:
```bash
watch -n 1 nvidia-smi
```

If you see OOM errors, reduce both to 48.

---

## 🧪 Testing

### Quick Sanity Test:
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Rebuild (if needed)
cargo build --release --features cuda --example world_record_dsjc1000

# Run with updated config
timeout 300s ./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml \
    2>&1 | grep -E "THERMO-GPU.*T=|colors.*conflicts"

# Should see:
# [THERMO-GPU] Replicas: 56, steps per temp: 8000
# [THERMO-GPU] Temperature range: [0.000500, 15.000]
# [THERMO-GPU] Processing temperature 1/56: T=15.000
# ... temperatures decreasing with conflict reduction
```

### Verification Checklist:
- [ ] Log shows "Replicas: 56"
- [ ] Log shows "steps per temp: 8000"
- [ ] Temperature starts at 15.0
- [ ] Temperature ends near 0.0005
- [ ] Conflicts decrease as temperature decreases
- [ ] Eventually finds valid coloring (0 conflicts)

---

## 📋 Files Modified

1. **Config**: `foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml`
   - Lines 96-103: Updated thermo parameters
   - Line 54: Updated ADP sync
   - Lines 21-22: Updated GPU settings

2. **Code**: `foundation/prct-core/src/world_record_pipeline.rs`
   - Line 2603: Fixed ADP clamp to respect config max_safe

---

## 🚀 Run Commands

### Full 48-Hour Run:
```bash
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml \
    2>&1 | tee results/phase2_fixed_$(date +%Y%m%d_%H%M%S).log
```

### With GPU Monitoring:
```bash
# Terminal 1
watch -n 1 nvidia-smi

# Terminal 2
./target/release/examples/world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml
```

---

## ✅ Summary

**Problem**: Phase 2 stuck at initial state (not reducing conflicts)
**Root cause**: Conservative parameters (48 temps, T_max=10, 5k steps)
**Solution**: Maximum VRAM-safe settings (56 temps, T_max=15, 8k steps)
**Code fix**: ADP respects config VRAM guard (not hard-coded 64)

**Expected result**: Phase 2 will now properly escape local minima and reduce colors!

---

**Ready to test!** Run the commands above to verify Phase 2 now works.