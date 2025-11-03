# PRISM-AI Profiling Quick Start

**Complete profiling setup for baseline measurements before implementing fusion**

---

## 🚀 Quick Start (5 Minutes to First Profile)

### Option 1: Profile Everything (Recommended)

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Run complete profiling workflow
./scripts/profile_all_pipelines.sh

# Wait ~15-30 minutes for completion

# Analyze results
./scripts/analyze_profiles.sh

# View report
cat reports/analysis/BASELINE_METRICS.md
```

### Option 2: Profile Single Pipeline (Fast)

```bash
# Profile just active inference
./scripts/profile_single_pipeline.sh active-inference

# View timeline
nsys-ui reports/nsys/active-inference/active-inference_timeline.nsys-rep
```

### Option 3: Profile Docker World Record

```bash
# Profile the graph coloring workload
./scripts/profile_docker.sh

# View results
nsys-ui reports/nsys_graph_coloring.nsys-rep
```

---

## 📁 Directory Structure

```
/home/diddy/Desktop/PRISM-FINNAL-PUSH/
├── scripts/
│   ├── profile_all_pipelines.sh      # Profile all 6 pipelines
│   ├── profile_single_pipeline.sh    # Profile one pipeline
│   ├── profile_docker.sh             # Profile Docker image
│   └── analyze_profiles.sh           # Analyze results
├── reports/
│   ├── nsys/                         # Nsight Systems timelines
│   │   ├── active-inference/
│   │   ├── transfer-entropy/
│   │   ├── linear-algebra/
│   │   ├── quantum/
│   │   ├── neuromorphic/
│   │   └── graph-coloring/
│   ├── ncu/                          # Nsight Compute metrics
│   │   └── [same structure]
│   ├── csv/                          # Exported CSV data
│   └── analysis/                     # Generated reports
│       ├── BASELINE_METRICS.md       # Summary report
│       ├── fusion_opportunities.csv  # High-priority fusions
│       └── *_ncu_metrics.csv         # Per-pipeline metrics
└── target/release/prism-ai           # Binary to profile
```

---

## 🔧 Prerequisites

### 1. Check CUDA Toolkit (Includes Nsight Tools)

```bash
# Check if nsys and ncu are installed
which nsys
which ncu

# If not found, install CUDA Toolkit 12.0+
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run --toolkit --silent
```

### 2. Verify GPU Access

```bash
nvidia-smi

# Should show your RTX 5070
```

### 3. Get Binary

**Option A: Extract from Docker**
```bash
docker pull delfictus/prism-ai-world-record:latest
docker create --name tmp delfictus/prism-ai-world-record:latest
docker cp tmp:/usr/local/bin/world_record ./target/release/prism-ai
docker rm tmp
chmod +x ./target/release/prism-ai
```

**Option B: Build Locally**
```bash
cargo build --release --features cuda,profiling
```

---

## 📊 Understanding the Output

### Nsight Systems Timeline

**Key metrics to look for:**

1. **Kernel Duration** - Total time in each kernel
2. **Launch Gaps** - Time between kernel launches (CPU overhead)
3. **Memory Transfers** - H2D/D2H bandwidth usage
4. **Occupancy** - GPU utilization percentage

**Red flags (fusion opportunities):**
- ⚠️ Many short kernels (<100μs) - Launch overhead dominates
- ⚠️ Large gaps between kernels - CPU bottleneck
- ⚠️ Low GPU utilization (<50%) - Underutilized

**Example timeline interpretation:**
```
Timeline view shows:
├─ kl_divergence_kernel     [150μs]  Gap: 50μs
├─ sum_reduction_kernel     [80μs]   Gap: 45μs
├─ prediction_error_kernel  [120μs]  Gap: 40μs
├─ belief_update_kernel     [200μs]  Gap: 30μs
└─ Total: 715μs (550μs compute + 165μs overhead)

Fusion opportunity: Combine into single kernel = ~200μs total
Expected speedup: 3.6x
```

### Nsight Compute Metrics

**Critical columns in CSV:**

| Column | What It Means | Target |
|--------|---------------|--------|
| **Duration** | Kernel execution time | Shorter after fusion |
| **SM %** (Occupancy) | % of GPU utilized | >50% ideal |
| **DRAM %** | Memory bandwidth used | <70% ideal (compute-bound) |
| **Tensor Core Active** | Tensor Core usage | >0% for eligible ops |

**Fusion scoring:**
```
High Priority (Score 4-5):
  - DRAM > 70% (memory-bound)
  - Occupancy < 50% (underutilized)
  - Duration < 100μs (launch overhead)

Medium Priority (Score 2-3):
  - DRAM 50-70%
  - Occupancy 50-70%

Low Priority (Score 0-1):
  - Already optimized
  - Compute-bound with good occupancy
```

---

## 📈 Example Workflow

### Day 1: Baseline Profiling

```bash
# 1. Profile all pipelines
./scripts/profile_all_pipelines.sh

# 2. Analyze results
./scripts/analyze_profiles.sh

# 3. Review fusion opportunities
cat reports/analysis/fusion_opportunities.csv

# Expected output:
# pipeline,kernel,duration,occupancy,dram,fusion_score,recommendation
# active-inference,belief_update_kernel,150,45,78,4,HIGH
# active-inference,kl_divergence_kernel,80,42,72,4,HIGH
# transfer-entropy,build_histogram_3d_kernel,200,38,82,5,HIGH
```

### Day 2-7: Implement Fusion

Follow `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md` to implement fused kernels for HIGH priority candidates.

### Day 8: Re-Profile Fused Version

```bash
# Build with fused kernels
cargo build --release --features cuda,fused

# Profile again
./scripts/profile_all_pipelines.sh

# Compare
./scripts/compare_profiles.sh baseline_reports fused_reports
```

### Day 9: Validate Speedup

**Expected results:**
```
Pipeline: active-inference
  Baseline:    715μs  (4 kernel launches)
  Fused:       195μs  (1 kernel launch)
  Speedup:     3.67x  ✓

Pipeline: transfer-entropy
  Baseline:    1842μs (5 kernel launches)
  Fused:       256μs  (1 kernel launch)
  Speedup:     7.19x  ✓

Overall platform speedup: 3.4x ✓
```

---

## 🔍 Advanced: Manual Profiling

### Profile Specific Kernels

```bash
# Focus on specific kernels only
ncu \
  --kernel-name "belief_update_kernel,kl_divergence_kernel" \
  --set full \
  --csv \
  ./target/release/prism-ai --pipeline active-inference
```

### Profile with Tensor Core Metrics

```bash
# Check if Tensor Cores are being used
ncu \
  --metrics \
    tensor_precision_fu_utilization,\
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
  --csv \
  ./target/release/prism-ai --pipeline linear-algebra
```

### Profile Memory Access Patterns

```bash
# Detailed memory analysis
ncu \
  --set memory \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Chart \
  --section MemoryWorkloadAnalysis_Tables \
  ./target/release/prism-ai --pipeline transfer-entropy
```

### Profile with Source Correlation

```bash
# Correlate performance with CUDA source
ncu \
  --set full \
  --source-folding \
  --nvtx \
  -o reports/detailed_profile \
  ./target/release/prism-ai --pipeline quantum
```

---

## 🐛 Troubleshooting

### Problem: "nsys: command not found"

```bash
# Add CUDA toolkit to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Verify
which nsys
```

### Problem: "Permission denied: ./scripts/profile_all_pipelines.sh"

```bash
chmod +x scripts/*.sh
```

### Problem: Binary not found

```bash
# Extract from Docker
docker pull delfictus/prism-ai-world-record:latest
docker create --name tmp delfictus/prism-ai-world-record:latest
docker cp tmp:/usr/local/bin/world_record ./target/release/prism-ai
docker rm tmp
chmod +x ./target/release/prism-ai
```

### Problem: GPU not accessible

```bash
# Check nvidia-smi
nvidia-smi

# If fails, check driver
sudo dmesg | grep -i nvidia

# Reinstall driver if needed
sudo apt install nvidia-driver-535
```

### Problem: Profiling extremely slow

```bash
# Use lighter metric set
ncu --set memory  # Instead of --set full

# Or profile fewer iterations
export ATTEMPTS_PER_GPU=100  # Instead of 100000
```

---

## 📤 Sharing Results for Analysis

After profiling, share these files:

```bash
# 1. Fusion opportunities (most important)
reports/analysis/fusion_opportunities.csv

# 2. Baseline metrics summary
reports/analysis/BASELINE_METRICS.md

# 3. CSV exports from Nsight Systems
reports/csv/*_nsys_cuda_gpu_kern_sum.csv

# 4. Detailed kernel metrics from Nsight Compute
reports/ncu/*/ncu_full.csv
reports/ncu/*/ncu_memory.csv
reports/ncu/*/ncu_occupancy.csv

# 5. Timeline files (for GUI viewing)
reports/nsys/*/*.nsys-rep
```

**Upload to:**
- GitHub issue/discussion
- Google Drive/Dropbox for large files
- Or paste CSV contents directly

---

## 🎯 Expected Timeline

| Day | Task | Output |
|-----|------|--------|
| **1** | Run baseline profiling | All CSV reports generated |
| **2** | Analyze fusion opportunities | Prioritized kernel list |
| **3-7** | Implement fused kernels | Fused CUDA code |
| **8** | Re-profile fused version | Comparison reports |
| **9** | Validate speedup | Performance graphs |

**Total time:** 9 days from profiling to validated speedup

---

## 📚 Key Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `profile_all_pipelines.sh` | Profile entire system | Initial baseline, final validation |
| `profile_single_pipeline.sh` | Profile one domain | Quick iterations during development |
| `profile_docker.sh` | Profile Docker workload | World record optimization |
| `analyze_profiles.sh` | Generate reports | After any profiling run |
| `fusion_opportunities.csv` | Top fusion candidates | Implementation prioritization |
| `BASELINE_METRICS.md` | Summary report | Sharing results |

---

## 🚀 Next Steps

1. **Today:** Run `./scripts/profile_all_pipelines.sh`
2. **Review:** `reports/analysis/fusion_opportunities.csv`
3. **Implement:** Follow `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md`
4. **Validate:** Re-profile and measure speedup

**Goal:** 3-4x speedup across PRISM-AI platform in 8 weeks

---

## 💡 Pro Tips

1. **Always profile in release mode** - Debug builds are 10-100x slower
2. **Run profiling 3 times** - Take median to account for variance
3. **Cool down GPU between runs** - Avoid thermal throttling
4. **Focus on hot paths first** - 80/20 rule: 20% of kernels take 80% of time
5. **Validate correctness first** - Speed is useless if output is wrong

---

**Questions?** See `PROFILING_SETUP_GUIDE.md` for detailed explanations of each tool and metric.
