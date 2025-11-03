# ✅ Complete Profiling Setup - Ready to Use

**Status:** All scripts created and ready for execution
**Location:** `/home/diddy/Desktop/PRISM-FINNAL-PUSH/`
**Docker Image:** `delfictus/prism-ai-world-record:latest` (already pulled)

---

## 📁 What Was Created

### Profiling Scripts (in `scripts/`)

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| **profile_all_pipelines.sh** | Profile all 6 pipelines | 15-30 min | Complete baseline |
| **profile_single_pipeline.sh** | Profile one pipeline | 3-5 min | Quick iteration |
| **profile_docker.sh** | Profile Docker workload | 5-10 min | World record baseline |
| **analyze_profiles.sh** | Analyze collected data | 1-2 min | Fusion opportunities |
| **compare_profiles.sh** | Compare baseline vs fused | 1 min | Speedup validation |

### Documentation

| File | Purpose |
|------|---------|
| **PROFILING_QUICK_START.md** | Quick reference guide |
| **PROFILING_SETUP_GUIDE.md** | Detailed setup instructions |
| **PROFILING_COMPLETE_SETUP.md** | This file (summary) |

### Directory Structure

```
/home/diddy/Desktop/PRISM-FINNAL-PUSH/
├── scripts/                           ✅ Created
│   ├── profile_all_pipelines.sh      ✅ Executable
│   ├── profile_single_pipeline.sh    ✅ Executable
│   ├── profile_docker.sh             ✅ Executable
│   ├── analyze_profiles.sh           ✅ Executable
│   └── compare_profiles.sh           ✅ Executable
├── reports/                           ✅ Created
│   ├── nsys/                         ✅ Ready
│   ├── ncu/                          ✅ Ready
│   ├── csv/                          ✅ Ready
│   └── analysis/                     ✅ Ready
└── Documentation
    ├── PROFILING_QUICK_START.md      ✅ Created
    ├── PROFILING_SETUP_GUIDE.md      ✅ Created
    ├── CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md  ✅ Existing
    └── DOCKER_WORLD_RECORD_ANALYSIS.md              ✅ Existing
```

---

## 🚀 Your Next Steps

### Step 1: Run Baseline Profiling (Today)

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Option A: Profile everything (recommended for first time)
./scripts/profile_all_pipelines.sh

# Option B: Quick test with single pipeline
./scripts/profile_single_pipeline.sh graph-coloring

# Option C: Profile Docker world record workload
./scripts/profile_docker.sh
```

**Expected time:** 15-30 minutes for full profiling

### Step 2: Analyze Results (5 minutes later)

```bash
# Generate analysis reports
./scripts/analyze_profiles.sh

# View fusion opportunities
cat reports/analysis/fusion_opportunities.csv

# View baseline metrics
cat reports/analysis/BASELINE_METRICS.md
```

**Expected output:**
```csv
pipeline,kernel,duration,occupancy,dram,fusion_score,recommendation
active-inference,belief_update_kernel,150,45,78,4,HIGH
active-inference,kl_divergence_kernel,80,42,72,4,HIGH
transfer-entropy,build_histogram_3d_kernel,200,38,82,5,HIGH
graph-coloring,parallel_greedy_coloring_kernel,2400,52,85,4,HIGH
```

### Step 3: Implement Fusion (Week 1-7)

Follow the implementation guide:
```bash
# Read the guide
cat CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md

# Implement fused kernels for HIGH priority candidates
# Start with active-inference (highest impact)
```

### Step 4: Validate Speedup (Week 8)

```bash
# Build with fused kernels
cargo build --release --features cuda,fused

# Re-profile
./scripts/profile_all_pipelines.sh

# Compare baseline vs fused
./scripts/compare_profiles.sh reports_baseline reports

# View speedup report
cat reports/comparison/SPEEDUP_REPORT.md
```

**Expected speedup:** 3-4x overall platform performance

---

## 🎯 Profiling Commands Reference

### Quick Commands

```bash
# Profile everything
./scripts/profile_all_pipelines.sh

# Profile just one pipeline
./scripts/profile_single_pipeline.sh active-inference

# Analyze results
./scripts/analyze_profiles.sh

# Compare two profiling runs
./scripts/compare_profiles.sh reports_baseline reports_fused
```

### Manual Nsight Systems Commands

```bash
# Basic timeline
nsys profile -o timeline ./target/release/prism-ai --pipeline active-inference

# With statistics
nsys profile --stats=true -o timeline ./target/release/prism-ai --pipeline active-inference

# Export to CSV
nsys stats timeline.nsys-rep --format csv --output analysis
```

### Manual Nsight Compute Commands

```bash
# Full metrics
ncu --set full --csv ./target/release/prism-ai --pipeline active-inference

# Memory-focused
ncu --set memory --csv ./target/release/prism-ai --pipeline active-inference

# Specific kernels only
ncu --kernel-name "belief_update_kernel" --set full ./target/release/prism-ai
```

---

## 📊 Understanding the Output

### Key Files Generated

**After profiling:**
```
reports/
├── nsys/                              # Timeline data
│   └── active-inference/
│       ├── active-inference_timeline.nsys-rep   # Open in nsys-ui
│       └── nsys.log                             # Text output
├── ncu/                               # Kernel metrics
│   └── active-inference/
│       ├── ncu_full.csv              # All metrics
│       ├── ncu_memory.csv            # Memory focus
│       └── ncu_occupancy.csv         # Occupancy focus
├── csv/                               # Exported data
│   └── active-inference_nsys_cuda_gpu_kern_sum.csv
└── analysis/                          # Generated reports
    ├── BASELINE_METRICS.md           # Summary
    ├── fusion_opportunities.csv      # Top candidates
    └── active-inference_ncu_metrics.csv
```

### Fusion Opportunity Interpretation

| Fusion Score | Priority | Action |
|--------------|----------|--------|
| **5** | CRITICAL | Implement immediately |
| **4** | HIGH | Implement in Week 1-2 |
| **3** | MEDIUM | Implement in Week 3-4 |
| **2** | LOW | Implement if time permits |
| **0-1** | SKIP | Already optimized |

**Scoring factors:**
- High DRAM usage (>70%) = +2 points
- Low occupancy (<50%) = +2 points
- Short duration (<100μs) = +1 point

---

## 🐛 Common Issues

### Issue: "nsys: command not found"

**Solution:**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

### Issue: "Binary not found"

**Solution:**
```bash
# Extract from Docker
docker pull delfictus/prism-ai-world-record:latest
docker create --name tmp delfictus/prism-ai-world-record:latest
docker cp tmp:/usr/local/bin/world_record ./target/release/prism-ai
docker rm tmp
chmod +x ./target/release/prism-ai
```

### Issue: "Permission denied"

**Solution:**
```bash
chmod +x scripts/*.sh
```

### Issue: Profiling is very slow

**Solution:**
```bash
# Use lighter metric sets
ncu --set memory  # Instead of --set full

# Or reduce workload
export ATTEMPTS_PER_GPU=100  # Instead of default
```

---

## 📤 Sharing Results

After profiling, you can share these files for analysis:

```bash
# Most important (small files)
reports/analysis/fusion_opportunities.csv
reports/analysis/BASELINE_METRICS.md

# CSV exports (moderate size)
reports/csv/*.csv
reports/ncu/*/*.csv

# Timeline files (large, for GUI viewing)
reports/nsys/*/*.nsys-rep
```

**How to share:**
- Small files: Paste contents directly or upload to GitHub gist
- Large files: Upload to Google Drive/Dropbox and share link
- Or: `tar czf profiling_results.tar.gz reports/` and upload

---

## 🎓 Learning Resources

### View Results in GUI

```bash
# Nsight Systems (timeline visualization)
nsys-ui reports/nsys/active-inference/active-inference_timeline.nsys-rep

# Nsight Compute (kernel metrics)
ncu-ui  # Then open .ncu-rep files
```

### Key Metrics to Watch

**In Nsight Systems timeline:**
1. Kernel duration bars (longer = more time)
2. Gaps between kernels (white space = CPU overhead)
3. Memory transfer bars (H2D/D2H)
4. GPU utilization percentage

**In Nsight Compute:**
1. **SM %** - GPU occupancy (target: >50%)
2. **DRAM %** - Memory bandwidth (target: <70% = compute-bound)
3. **Duration** - Kernel execution time
4. **Tensor Core Active** - Tensor Core usage

---

## ✅ Validation Checklist

Before implementing fusion:

- [ ] Baseline profiling complete (`./scripts/profile_all_pipelines.sh`)
- [ ] Analysis generated (`./scripts/analyze_profiles.sh`)
- [ ] Fusion opportunities identified (`fusion_opportunities.csv`)
- [ ] Top 3 HIGH priority kernels selected
- [ ] Implementation guide reviewed (`CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md`)

After implementing fusion:

- [ ] Fused version profiled (same commands)
- [ ] Comparison report generated (`./scripts/compare_profiles.sh`)
- [ ] Speedup validated (≥3x target)
- [ ] Correctness tests passing
- [ ] Documentation updated

---

## 🎯 Success Criteria

**Week 1 Goal:**
- ✅ Baseline profiling complete
- ✅ Fusion opportunities identified
- ✅ Implementation plan created

**Week 8 Goal:**
- ✅ 3-4x overall speedup achieved
- ✅ All HIGH priority fusions implemented
- ✅ Validation tests passing
- ✅ Production-ready code

---

## 🚀 Ready to Start!

**Your profiling setup is complete and ready to use.**

Run this now:
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
./scripts/profile_all_pipelines.sh
```

Then follow the on-screen instructions to analyze results and begin fusion implementation.

**Expected deliverable:** Complete baseline profile in 30 minutes, ready to start fusion implementation.

---

**Questions?** See:
- Quick reference: `PROFILING_QUICK_START.md`
- Detailed guide: `PROFILING_SETUP_GUIDE.md`
- Implementation: `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md`
