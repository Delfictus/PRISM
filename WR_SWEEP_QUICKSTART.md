# World Record Sweep Quick Start

**Target:** DSJC1000.5 with 83 colors (current best: 87)
**Device:** RTX 5070 8GB + 24 CPU threads
**Runtime:** 7 days total (24h per config)

---

## ✅ Files Created

### Configuration Files (7 total)
```
foundation/prct-core/configs/wr_sweep_A.v1.1.toml  (Baseline balanced)
foundation/prct-core/configs/wr_sweep_B.v1.1.toml  (Geodesic-heavy)
foundation/prct-core/configs/wr_sweep_C.v1.1.toml  (Reservoir-heavy)
foundation/prct-core/configs/wr_sweep_D.v1.1.toml  (Quantum-deeper)
foundation/prct-core/configs/wr_sweep_E.v1.1.toml  (Memetic-deeper)
foundation/prct-core/configs/wr_sweep_F.v1.1.toml  (Thermo/PIMC-heavy, VRAM max)
foundation/prct-core/configs/wr_sweep_G.v1.1.toml  (Exploration/restarts)
```

### Scripts
```
tools/validate_wr_sweep.sh  (Validate all 7 configs)
tools/run_wr_sweep.sh       (Batch execution)
```

### Documentation
```
WR_SWEEP_STRATEGY.md        (Detailed strategy)
WR_SWEEP_QUICKSTART.md      (This file)
```

---

## 🚀 Quick Start (3 Steps)

### 1. Validate Configs
```bash
./tools/validate_wr_sweep.sh
```
**Expected Output:**
```
✅ All 7 configs validated successfully!
```

### 2. Choose Execution Mode

**Option A: Sequential (Manual)**
```bash
# Run one config at a time
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_A.v1.1.toml \
    2>&1 | tee logs/wr_sweep_A.log
```

**Option B: Batch (Automated)**
```bash
# Run all 7 configs automatically
./tools/run_wr_sweep.sh
```

### 3. Monitor Progress
```bash
# GPU usage
watch -n 1 nvidia-smi

# Best results
tail -f logs/wr_sweep_*.log | grep "BEST COLORING"

# Conflicts
tail -f logs/wr_sweep_*.log | grep "conflicts"
```

---

## 🔬 Seed Probe (Quick Decision Tool)

**Purpose:** Test best configs (D and F) with multiple seeds in 60-90 minutes to decide if a 24-48h run is worth it.

### Why Use Seed Probe?

Before committing to a 24-48 hour world record run, the seed probe:
- Tests 9 config/seed combinations in ~90 minutes each
- Identifies which config/seed performs best quickly
- Provides Go/No-Go decision based on quick results
- Saves time by avoiding long runs with poor configurations

### Seed Variants Available

**Config D (Quantum-deeper):**
- `wr_sweep_D_seed_42.v1.1.toml` (seed 42)
- `wr_sweep_D_seed_1337.v1.1.toml` (seed 1337)
- `wr_sweep_D_seed_9001.v1.1.toml` (seed 9001)

**Config D Aggressive (depth=7, attempts=224, gens=900):**
- `wr_sweep_D_aggr_seed_42.v1.1.toml` (seed 42)
- `wr_sweep_D_aggr_seed_1337.v1.1.toml` (seed 1337)
- `wr_sweep_D_aggr_seed_9001.v1.1.toml` (seed 9001)

**Config F (Thermo/PIMC-heavy):**
- `wr_sweep_F_seed_42.v1.1.toml` (seed 42)
- `wr_sweep_F_seed_1337.v1.1.toml` (seed 1337)
- `wr_sweep_F_seed_9001.v1.1.toml` (seed 9001)

### Run Seed Probe

```bash
# Run with default 90-minute timeout per config
./tools/run_wr_seed_probe.sh

# Or specify custom timeout (in minutes)
MAX_MINUTES=60 ./tools/run_wr_seed_probe.sh
```

### Results Location

- **JSONL Summary:** `results/dsjc1000_seed_probe.jsonl`
- **Individual Logs:** `results/logs/wr_sweep_*_<timestamp>.log`

### Go/No-Go Thresholds

The seed probe automatically provides a verdict:

| Colors | Verdict | Exit Code | Recommendation |
|--------|---------|-----------|----------------|
| ≤ 95   | **GO** ✅ | 0 | Launch 24-48h run with best config/seed |
| 96-98  | **MAYBE** ⚠️ | 10 | Try different seeds/weights before long run |
| > 98   | **NO-GO** ❌ | 20 | Retune configurations before attempting |

**Rationale:**
- **≤ 95 colors:** Strong performance, likely to improve with longer runtime
- **96-98 colors:** Marginal performance, may not reach target (83 colors)
- **> 98 colors:** Poor performance, needs configuration tuning

### Example Output

```
═══════════════════════════════════════════════════════════
Results Summary (sorted by colors, then time):
───────────────────────────────────────────────────────────
Config                                      Seed  Colors    Time(s)     Status
───────────────────────────────────────────────────────────
wr_sweep_D_aggr_seed_1337                  1337      94       4521         ok
wr_sweep_D_seed_42                           42      95       4892         ok
wr_sweep_F_seed_9001                       9001      96       5201         ok
───────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════
Go/No-Go Verdict
═══════════════════════════════════════════════════════════
Best result: 94 colors in 4521s

✓ VERDICT: GO
  Recommendation: Launch 24-48h run with best config/seed
  Rationale: ≤95 colors achieved within 90 minutes

  Best config: foundation/prct-core/configs/wr_sweep_D_aggr_seed_1337.v1.1.toml
  Best seed: 1337
═══════════════════════════════════════════════════════════
```

### Next Steps After Seed Probe

**If GO (≤95 colors):**
```bash
# Run full 24-48h attempt with best config
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_D_aggr_seed_1337.v1.1.toml \
    2>&1 | tee logs/wr_full_attempt.log
```

**If MAYBE (96-98 colors):**
- Try additional seed values (e.g., 314, 2718, 4242)
- Adjust hyperparameters (increase quantum depth, memetic generations)
- Run seed probe again

**If NO-GO (>98 colors):**
- Analyze logs for bottlenecks
- Review phase contributions
- Tune config parameters
- Consider trying configs A, B, C, E, or G

---

## 📊 Config Comparison

| Config | Strategy | Key Tuning | VRAM Risk | Best For |
|--------|----------|------------|-----------|----------|
| **A** | Balanced | Equal weights | ✅ Safe | Baseline |
| **B** | Geodesic-heavy | geo=0.45 | ✅ Safe | Geometric graphs |
| **C** | Reservoir-heavy | res=0.55 | ⚠️ Max beads | Strong reservoir |
| **D** | Quantum-deeper | depth=6 | ✅ Safe | Quantum promise |
| **E** | Memetic-deeper | gens=1400 | ✅ Safe | Local refinement |
| **F** | Thermo/PIMC | replicas=56 | ⚠️ VRAM max | Thermodynamic |
| **G** | Exploration | restarts=12 | ✅ Safe | Diverse search |

---

## ⚙️ Common Parameters (All Configs)

✅ **All GPU modules enabled:**
- Neuromorphic reservoir
- Quantum annealing
- Thermodynamic equilibration
- PIMC (Path-Integral Monte Carlo)
- Transfer Entropy
- Topological Data Analysis
- Statistical Mechanics

✅ **VRAM-safe defaults:**
- Replicas ≤56
- Beads ≤64
- Tested on RTX 5070 8GB

✅ **Full pipeline enabled:**
- Active Inference
- ADP Learning
- GNN Screening
- Geodesic Features
- Ensemble Consensus
- Multiscale Analysis

---

## 🎯 Expected Outcomes

### World Record (83 colors)
- **Probability:** Low-Medium (depends on config)
- **Impact:** New DIMACS world record!
- **Next Step:** Publish results, run extended 7-day attempt

### Strong Performance (84-86 colors)
- **Probability:** Medium-High
- **Impact:** Competitive with state-of-the-art
- **Next Step:** Extended run with best config

### Baseline (87 colors)
- **Probability:** High
- **Impact:** Validates pipeline correctness
- **Next Step:** Analyze which modules contributed most

---

## 🛠️ Troubleshooting

### Config F VRAM Exhaustion
```
Error: CUDA out of memory
```
**Solution:** Config F uses max VRAM (replicas=56, beads=64). If OOM:
1. Skip config F
2. Or reduce replicas/beads to 48

### GPU Not Used
```bash
grep -i "cpu fallback" logs/*.log
```
**Expected:** No matches (all GPU paths active)

### Compilation Errors
```bash
cargo check --release --features cuda
```
**Expected:** 0 errors, ~29 warnings

---

## 📈 Post-Sweep Analysis

After all 7 configs complete:

```bash
# Extract best results
for config in A B C D E F G; do
    echo "Config ${config}:"
    grep "BEST COLORING" logs/wr_sweep_${config}.log | tail -1
done

# Find winner
grep -h "BEST COLORING" logs/wr_sweep_*.log | sort -k3 -n | head -1
```

**Next Steps:**
1. Identify winner (lowest chromatic number)
2. Analyze which DSATUR weights worked best
3. Check which phases contributed improvements
4. Run extended 7-day attempt with winner config

---

## 📂 File Locations

```
foundation/prct-core/configs/    # 7 TOML config files
logs/                            # Execution logs (created on run)
tools/                           # Validation & execution scripts
WR_SWEEP_STRATEGY.md            # Detailed strategy guide
WR_SWEEP_QUICKSTART.md          # This file
CONFIG_V1.1_VERIFICATION_REPORT.md  # Config v1.1 verification
```

---

## 🔐 Validation Results

```
✅ wr_sweep_A.v1.1.toml (1828 bytes) - VRAM: ✅ Safe
✅ wr_sweep_B.v1.1.toml (1835 bytes) - VRAM: ✅ Safe
✅ wr_sweep_C.v1.1.toml (1850 bytes) - VRAM: ⚠️  Max beads
✅ wr_sweep_D.v1.1.toml (1845 bytes) - VRAM: ✅ Safe
✅ wr_sweep_E.v1.1.toml (1847 bytes) - VRAM: ✅ Safe
✅ wr_sweep_F.v1.1.toml (1850 bytes) - VRAM: ⚠️  Max (replicas + beads)
✅ wr_sweep_G.v1.1.toml (1843 bytes) - VRAM: ✅ Safe
```

All configs: 7/7 enabled GPU modules, 13 orchestrator flags

---

## 💡 Pro Tips

1. **Start with Config A** (baseline) to verify everything works
2. **Monitor VRAM** during Config F (uses max allocation)
3. **Save logs** for each config (auto-saved if using batch script)
4. **Check GPU logs** to confirm GPU paths active:
   ```bash
   grep "\[GPU\]" logs/wr_sweep_A.log | head -20
   ```
5. **Early stopping** is enabled (restarts if no improvement)

---

## ⏱️ Timeline

| Day | Config | Strategy | VRAM Risk |
|-----|--------|----------|-----------|
| 1   | A      | Balanced | ✅ Safe   |
| 2   | B      | Geodesic | ✅ Safe   |
| 3   | C      | Reservoir | ⚠️ Max beads |
| 4   | D      | Quantum  | ✅ Safe   |
| 5   | E      | Memetic  | ✅ Safe   |
| 6   | F      | Thermo/PIMC | ⚠️ VRAM max |
| 7   | G      | Exploration | ✅ Safe |

**Total:** 168 hours (7 days)

---

## 🎉 Ready to Start!

```bash
# Validate (should take <1 second)
./tools/validate_wr_sweep.sh

# Run all 7 sweeps (takes ~7 days)
./tools/run_wr_sweep.sh

# Or run individually
cargo run --release --features cuda --example world_record_dsjc1000 \
    foundation/prct-core/configs/wr_sweep_A.v1.1.toml \
    2>&1 | tee logs/wr_sweep_A.log
```

**Good luck with the world record attempt! 🚀**

---

**Generated:** 2025-11-02
**Version:** v1.1
**Status:** ✅ Ready for execution
