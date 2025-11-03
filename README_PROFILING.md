# PRISM-AI Profiling - READ THIS FIRST

## ⚠️ Important: Empty Reports Issue Resolved

The initial profiling scripts created empty reports because they assumed your binary accepts `--pipeline` arguments. Your actual binaries work differently.

---

## ✅ Working Solutions (Choose One)

### Option 1: Quick Test (Verify Setup Works) ⭐ START HERE

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
./scripts/test_profiling.sh
```

**What this does:**
- Verifies nsys and ncu are installed and working
- Creates a simple CUDA program
- Profiles it successfully
- Generates actual (non-empty) reports

**Time:** 2 minutes

**Output:** `./reports/test/` with real profiling data

---

### Option 2: Profile Your Binary Directly

```bash
./scripts/profile_prism_binary.sh
```

**What this does:**
- Runs your actual `prism-ai` binary
- Captures whatever it does (up to 30 seconds)
- Profiles all CUDA kernel launches

**Note:** If your binary needs specific arguments, edit the script first.

---

### Option 3: Profile Docker World Record Workload

The Docker container runs graph coloring on DSJC1000-5. This is a real workload with real kernels.

```bash
# Option 3a: Run Docker container (it will execute the workload)
docker run --rm --gpus all \
    -e NUM_GPUS=1 \
    -e ATTEMPTS_PER_GPU=1000 \
    delfictus/prism-ai-world-record:latest

# To profile it, you'll need to run nsys inside the container
# See PROFILING_TROUBLESHOOTING.md for details
```

---

## 📁 Files You Have

### Working Scripts
- ✅ `test_profiling.sh` - Test that profiling works (USE THIS FIRST)
- ✅ `profile_prism_binary.sh` - Profile your actual binary
- ⚠️ `profile_all_pipelines.sh` - Needs binary modifications to work
- ⚠️ `profile_single_pipeline.sh` - Needs binary modifications to work

### Documentation
- 📖 `PROFILING_TROUBLESHOOTING.md` - Detailed solutions for empty reports
- 📖 `PROFILING_QUICK_START.md` - Original quick start guide
- 📖 `PROFILING_COMPLETE_SETUP.md` - Full setup documentation
- 📖 `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md` - Implementation guide

---

## 🎯 Recommended Workflow

### Day 1: Verify Setup ✅

```bash
# Step 1: Test profiling tools
./scripts/test_profiling.sh

# Step 2: Check output
ls -lh reports/test/

# Step 3: View timeline (if nsys-ui available)
nsys-ui reports/test/test_timeline.nsys-rep

# Step 4: Check CSV data
cat reports/test/test_cuda_gpu_kern_sum.csv
```

**Expected:** You should see real kernel data (vector_add, vector_multiply)

### Day 2: Profile Real Workload ✅

Once test profiling works, identify your actual workload:

**Option A: You have tests**
```bash
cargo test --release --lib -- --list | grep -i gpu
nsys profile -o reports/gpu_tests cargo test --release test_name
```

**Option B: You have benchmarks**
```bash
cargo bench --bench benchmark_name
nsys profile -o reports/bench cargo bench --bench benchmark_name
```

**Option C: You know the command**
```bash
nsys profile -o reports/workload ./target/release/prism-ai [your args]
```

### Day 3-7: Implement Fusion

Once you have real profiling data, follow:
- `CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md`

---

## 🐛 Why Were Reports Empty?

The original scripts tried to run:
```bash
./target/release/prism-ai --pipeline active-inference --profile-mode
```

But your binary doesn't accept these arguments. It either:
1. Runs a default workload when executed
2. Needs different arguments
3. Needs input files
4. Is meant to be used differently

**Solution:** Find the correct way to invoke your binary, then profile that.

---

## 🔍 Finding Your Workload

### Check what the binary does:

```bash
# Try running it
./target/release/prism-ai

# Check if it has help
./target/release/prism-ai --help
./target/release/prism-ai -h

# Check source for main function
grep -A 20 "fn main" src/main.rs
grep -A 20 "fn main" src/bin/*.rs
```

### Check for examples:

```bash
ls examples/
cargo run --release --example [name]
```

### Check for tests:

```bash
cargo test --lib -- --list
cargo test --release --lib -- --nocapture [test_name]
```

---

## ✅ Validation: You Know It's Working When...

**Good signs:**
- Timeline file `.nsys-rep` is created and is > 100KB
- CSV files have multiple rows of data
- You see kernel names like `vector_add`, `matmul`, `coloring`, etc.
- `nsys-ui` shows timeline with kernel launches

**Bad signs:**
- Empty CSV files (0 bytes or only headers)
- Timeline file doesn't exist
- Error messages about "no CUDA"
- Binary exits immediately

---

## 🚀 Your Action Plan

1. **Right now:** Run `./scripts/test_profiling.sh` to verify setup ✅

2. **Next:** Figure out how to run your actual workload
   - Read `PROFILING_TROUBLESHOOTING.md` for guidance
   - Try different invocations of your binary
   - Check tests, examples, benchmarks

3. **Then:** Profile the real workload
   ```bash
   nsys profile -o reports/my_workload [your command]
   ```

4. **Finally:** Implement fusion using the guide

---

## 💬 Need Help?

Share these:
1. Output of `./scripts/test_profiling.sh` (proves tools work)
2. Output of `./target/release/prism-ai` (shows what binary does)
3. Output of `cargo test --list` (shows available tests)
4. Any error messages you're seeing

---

## 📚 Documentation Index

| File | When to Read |
|------|-------------|
| **README_PROFILING.md** | NOW (you're here) |
| **PROFILING_TROUBLESHOOTING.md** | If you have empty reports or errors |
| **PROFILING_QUICK_START.md** | When you have working profiles |
| **CUSTOM_FUSED_KERNEL_IMPLEMENTATION_GUIDE.md** | When implementing fusion |

---

**Bottom line:** Start with `./scripts/test_profiling.sh` to prove your setup works, then we'll tackle profiling your actual workload.
