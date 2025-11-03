# Fix Compilation Errors Before Docker Build

## The Problem

Your code has compilation errors that need to be fixed before building a Docker image:

1. **Missing dependency**: `rand_chacha` is used but not in Cargo.toml
2. **Missing dependency**: `cudarc` is optional but code requires it
3. **Missing types**: `CausalManifold`, `Solution`, `Ensemble` not found in `crate::cma`

## ✅ Solution: Fix Locally First, Then Build

### Step 1: Fix Dependencies

Add missing dependencies to `Cargo.toml`:

```bash
cd ~/Desktop/PRISM-FINNAL-PUSH

# Edit Cargo.toml and add these lines to [dependencies]:
nano Cargo.toml
```

Add:
```toml
rand_chacha = "0.3"
```

And change `cudarc` from optional to required:
```toml
# Change this:
cudarc = { version = "0.9", optional = true }

# To this:
cudarc = { version = "0.9", features = ["std"] }
```

### Step 2: Fix Missing Types

The errors show missing types in `src/cma/mod.rs`. You need to:

1. Define the missing types:
   - `CausalManifold`
   - `Solution`
   - `Ensemble`

OR

2. Comment out the problematic neural modules temporarily

**Quick fix** (comment out broken modules):

```bash
# Edit src/cma/neural/mod.rs
nano src/cma/neural/mod.rs

# Comment out the functions that use undefined types
# Or add stub implementations
```

### Step 3: Test Build Locally

```bash
# Try building locally first
cargo build --release

# If it fails, check errors:
cargo check 2>&1 | less

# Fix errors one by one
```

### Step 4: Once Local Build Works

```bash
# Now build Docker image
docker build -t my-prism:gpu .

# Run with GPU
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000
```

---

## Alternative: Use Pre-Built Binary

Since you have `target/release/prism-ai` already built, you can:

### Option A: Run Locally (No Docker)

```bash
# Just run your existing binary directly
./target/release/prism-ai --input data/nipah/2VSM.mtx --attempts 1000
```

This works if your local build is fine.

### Option B: Docker with Pre-Built Binary

Create a simpler Dockerfile:

```dockerfile
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /prism-ai

# Copy pre-built binary
COPY target/release/prism-ai /usr/local/bin/prism-ai
COPY target/ptx /prism-ai/target/ptx

# Copy data
COPY data /prism-ai/data

RUN chmod +x /usr/local/bin/prism-ai

ENTRYPOINT ["/usr/local/bin/prism-ai"]
CMD ["--help"]
```

Build:
```bash
docker build -f Dockerfile.prebuilt -t my-prism:gpu .
```

---

## Quick Decision Tree

**Q: Does `cargo build --release` work locally?**

**YES** → Build Docker image normally
```bash
./build-and-run-gpu.sh
```

**NO** → Use one of these:

1. **Fix the code** (recommended):
   - Add missing dependencies
   - Fix type errors
   - Test locally until `cargo build` succeeds

2. **Use pre-built binary**:
   ```bash
   # Just run it locally
   ./target/release/prism-ai --input data/nipah/2VSM.mtx --attempts 1000
   ```

3. **Skip Docker entirely** (simplest):
   Your binary already exists and can run with GPU if you have CUDA installed locally.

---

## Check CUDA Availability Locally

```bash
# Do you have CUDA?
nvidia-smi

# Can your binary see CUDA?
ldd ./target/release/prism-ai | grep cuda
```

If yes, just run locally:
```bash
./target/release/prism-ai --input data/nipah/2VSM.mtx --attempts 1000
```

---

## Recommended Path Forward

### 1. Try Local Run First (Fastest)

```bash
cd ~/Desktop/PRISM-FINNAL-PUSH
./target/release/prism-ai --input data/nipah/2VSM.mtx --attempts 1000
```

**If this works**: You don't need Docker! Your GPU binary already works.

### 2. If Local Run Needs CUDA Libs

Install CUDA locally:
```bash
sudo apt install nvidia-cuda-toolkit
```

Then run again.

### 3. If You Really Want Docker

Fix the code errors first:

```bash
# Add dependencies
echo 'rand_chacha = "0.3"' >> Cargo.toml

# Try building
cargo build --release

# Fix errors shown
# Repeat until it builds

# Then Docker
docker build -t my-prism:gpu .
```

---

## Summary

**Problem**: Code won't compile in Docker due to missing deps and type errors

**Solutions**:
1. **Easiest**: Run `./target/release/prism-ai` directly (if you have CUDA)
2. **Quick**: Use pre-built binary in simpler Docker image
3. **Proper**: Fix code errors, then build Docker image

**Recommendation**: Try option #1 first - your binary is already built!

```bash
./target/release/prism-ai --input data/nipah/2VSM.mtx --attempts 1000
```
