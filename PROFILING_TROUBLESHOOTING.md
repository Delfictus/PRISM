## Profiling Troubleshooting Guide

**Issue:** Empty CSV reports after running profiling scripts

### Root Cause Analysis

The profiling scripts were designed to work with a binary that accepts `--pipeline` arguments, but your actual binaries work differently:

1. **`./target/release/prism-ai`** - Main PRISM-AI binary
2. **`./target/release/meta_bootstrap`** - Meta bootstrap binary
3. **Docker `world_record` binary** - Graph coloring workload

None of these accept `--pipeline active-inference` style arguments, so the profiling commands failed silently.

---

## ✅ Solution: Three Working Approaches

### Approach 1: Test Profiling Setup (Verify Tools Work)

```bash
# This creates a simple CUDA test and profiles it
./scripts/test_profiling.sh
```

**What it does:**
- Compiles a simple CUDA vector add program
- Profiles it with nsys and ncu
- Verifies your profiling tools work correctly
- Creates sample reports

**Expected output:**
```
✓ nsys found
✓ ncu found
✓ GPU detected: RTX 5070
✓ Test program compiled
✓ Nsight Systems profile created
✓ CSV export successful
✓ Nsight Compute metrics collected
```

**Results:** `./reports/test/`

---

### Approach 2: Profile Your Actual Binary

```bash
# Profile whatever prism-ai does when executed
./scripts/profile_prism_binary.sh
```

**What it does:**
- Runs your actual `prism-ai` binary
- Captures whatever CUDA kernels it launches
- Profiles for 30 seconds max
- Exports timeline and metrics

**Note:** If your binary needs specific arguments or input files, edit this script to add them.

---

### Approach 3: Profile Specific Workload (Manual)

If you know exactly how to run your binary with a specific workload:

```bash
# Example: If your binary needs specific args
nsys profile \
    --stats=true \
    --force-overwrite=true \
    -o ./reports/my_workload \
    ./target/release/prism-ai [YOUR ARGS HERE]

# Then export stats
nsys stats ./reports/my_workload.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output ./reports/my_workload
```

---

## 🔧 Understanding Your Binaries

### Binary 1: `prism-ai` (Main binary)

```bash
# Check what it does
./target/release/prism-ai

# If it needs input, try:
./target/release/prism-ai --help
./target/release/prism-ai < input_file
```

**To profile:**
```bash
nsys profile -o reports/prism_main ./target/release/prism-ai [args]
```

### Binary 2: `meta_bootstrap`

```bash
# Check what it does
./target/release/meta_bootstrap

# Profile it
nsys profile -o reports/meta_bootstrap ./target/release/meta_bootstrap
```

### Binary 3: Docker `world_record` (Graph Coloring)

This is the graph coloring workload. To profile it:

```bash
# Extract and profile
docker run --rm --gpus all \
    -v $(pwd)/reports:/reports \
    --entrypoint="" \
    delfictus/prism-ai-world-record:latest \
    bash -c "nsys profile -o /reports/world_record /usr/local/bin/world_record"
```

Or use nvidia-docker with profiling:

```bash
docker run --rm --gpus all \
    -v $(pwd)/reports:/output \
    -e NUM_GPUS=1 \
    -e ATTEMPTS_PER_GPU=1000 \
    delfictus/prism-ai-world-record:latest
```

---

## 📊 How to Get Non-Empty Reports

### Step 1: Identify Your Workload

What do you want to profile?

**Option A: Graph coloring (world record attempt)**
- Use Docker container
- Profiles: `parallel_greedy_coloring_kernel`, `validate_coloring`, etc.

**Option B: Active inference**
- Find how to run active inference workload
- May be in tests or examples
- Look for: `cargo test --release active_inference` or similar

**Option C: Full PRISM-AI system**
- Profile the main binary with real input
- Need to know what input it expects

### Step 2: Create Workload Runner

Create a script that runs your workload:

```bash
# workload_runner.sh
#!/bin/bash

# Example: Run graph coloring
./target/release/prism-ai --mode graph-coloring --input data/DSJC1000-5.col

# Or: Run test suite
cargo test --release --no-fail-fast test_active_inference

# Or: Run benchmark
cargo bench --bench active_inference_bench
```

### Step 3: Profile the Workload

```bash
nsys profile -o reports/workload ./workload_runner.sh
```

---

## 🎯 Quick Wins: Profile Existing Tests

If you have tests that use GPU:

```bash
# List tests
cargo test --release --lib -- --list | grep -i gpu

# Profile a specific test
nsys profile -o reports/test_gpu \
    cargo test --release test_active_inference_gpu -- --nocapture

# Profile all GPU tests
nsys profile -o reports/all_gpu_tests \
    cargo test --release --lib -- --nocapture gpu
```

---

## 🔍 Debug: Check What Kernels Are Available

### Method 1: Search source code

```bash
# Find all CUDA kernel definitions
grep -r "__global__" foundation/ src/ | grep "void"

# Find PTX loading
grep -r "load_ptx" foundation/ src/
```

### Method 2: Check PTX files

```bash
# List all compiled PTX files
find . -name "*.ptx" -type f

# Check kernels in PTX
grep "\.visible \.entry" foundation/kernels/ptx/*.ptx
```

### Method 3: Run with CUDA profiler API

Add to your Rust code:

```rust
use std::ffi::CString;

// At start of GPU code
unsafe {
    let name = CString::new("MyWorkload").unwrap();
    cuda_sys::cudaProfilerStart();
}

// Your GPU work here

// At end
unsafe {
    cuda_sys::cudaProfilerStop();
}
```

---

## 🛠️ Fixing the Original Scripts

To make the original scripts work, you need to modify your binary to accept pipeline arguments.

Add to `src/main.rs` (or appropriate binary):

```rust
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pipeline: Option<String>,

    #[arg(long)]
    profile_mode: bool,
}

fn main() {
    let args = Args::parse();

    match args.pipeline.as_deref() {
        Some("active-inference") => run_active_inference_pipeline(),
        Some("transfer-entropy") => run_transfer_entropy_pipeline(),
        Some("linear-algebra") => run_linear_algebra_pipeline(),
        Some("quantum") => run_quantum_pipeline(),
        Some("neuromorphic") => run_neuromorphic_pipeline(),
        Some("graph-coloring") => run_graph_coloring_pipeline(),
        _ => run_default(),
    }
}
```

Then rebuild and the original scripts will work.

---

## 📋 Checklist: Getting Working Profiles

- [ ] Verify tools work: `./scripts/test_profiling.sh`
- [ ] Identify your workload: What do you want to profile?
- [ ] Find how to run it: Command line args? Input files? Tests?
- [ ] Create runner script: Encapsulate the workload
- [ ] Profile the runner: `nsys profile -o reports/my_workload ./runner.sh`
- [ ] Verify results: Check `reports/my_workload.nsys-rep` exists and is not empty
- [ ] Export CSV: `nsys stats` to get CSV data
- [ ] Analyze: Open in `nsys-ui` or parse CSV

---

## 🚀 Recommended Next Steps

### For Quick Results (Today):

1. **Run test profiling:**
   ```bash
   ./scripts/test_profiling.sh
   ```
   This proves your profiling tools work.

2. **Profile Docker workload:**
   ```bash
   # The Docker container runs graph coloring
   docker run --rm --gpus all \
       -e NUM_GPUS=1 \
       -e ATTEMPTS_PER_GPU=100 \
       delfictus/prism-ai-world-record:latest
   ```

   To profile it, you need to run nsys/ncu inside the container or extract the binary.

3. **Find your actual workload:**
   ```bash
   # Check if there are examples
   ls examples/

   # Check if there are benchmarks
   ls benches/

   # Check tests
   cargo test --lib -- --list
   ```

### For Complete Profiling (This Week):

1. Identify all GPU workloads in your code
2. Create runner scripts for each
3. Profile each with nsys and ncu
4. Analyze results with provided analysis scripts

---

## 💡 Key Insight

The empty reports happened because:
```bash
./target/release/prism-ai --pipeline active-inference --profile-mode
```

This command **doesn't exist** in your binary. It needs to be:
```bash
./target/release/prism-ai [whatever args it actually takes]
```

Once you know the correct invocation, profiling will work perfectly.

---

## 🆘 Still Stuck?

Share:
1. Output of `./scripts/test_profiling.sh` (to verify tools work)
2. Output of `./target/release/prism-ai` (to see what it does)
3. Output of `cargo test --list | grep gpu` (to see GPU tests)
4. Contents of `examples/` or `benches/` if they exist

Then I can create exact profiling commands for your specific workload.
