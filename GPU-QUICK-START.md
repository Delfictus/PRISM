# PRISM GPU Quick Start Guide

## 🚀 Run Your PRISM Project with Full GPU Acceleration

### Step 1: Build GPU Docker Image (One-Time)

```bash
cd ~/Desktop/PRISM-FINNAL-PUSH
./build-and-run-gpu.sh
```

This will:
- Build a Docker image with CUDA 12.6.2
- Compile your Rust code with GPU support
- Tag it as `my-prism:gpu`

**Takes**: 5-10 minutes on first build

---

### Step 2: Run with GPU

```bash
# Quick run
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000

# More attempts
./run-prism-gpu.sh data/nipah/2VSM.mtx 10000

# Even more
./run-prism-gpu.sh data/nipah/2VSM.mtx 100000
```

**That's it!** Your code runs with full GPU acceleration on your RTX 5070.

---

## 📋 Manual Commands

If you prefer manual control:

```bash
# Build
docker build -t my-prism:gpu .

# Run
docker run --rm --runtime=nvidia --gpus all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  my-prism:gpu \
  --input /data/nipah/2VSM.mtx --attempts 1000
```

---

## 🔧 How It Works

The scripts use the **same GPU solution** we implemented for the other container:

1. **`--runtime=nvidia`** - Uses NVIDIA container runtime
2. **`--gpus all`** - Enables GPU access
3. **Data mounting** - Your local files accessible in container
4. **Output mounting** - Results saved to `./output/`

---

## 📊 GPU Verification

Check if GPU is working:

```bash
# Test GPU access
docker run --rm --runtime=nvidia --gpus all \
  my-prism:gpu \
  bash -c "nvidia-smi"
```

Should show your RTX 5070.

---

## 🎯 Different Input Files

```bash
# Protein structure
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000

# Other MTX files (if you have them)
./run-prism-gpu.sh data/other/file.mtx 5000

# Benchmark files
./run-prism-gpu.sh benchmarks/DSJC1000-5.mtx 100000
```

---

## 💡 Multi-GPU Support

When you add more GPUs:

```bash
# Use specific GPU
docker run --rm --runtime=nvidia --gpus '"device=0"' \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  my-prism:gpu --input /data/nipah/2VSM.mtx --attempts 1000

# Use multiple GPUs
docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  my-prism:gpu --input /data/nipah/2VSM.mtx --attempts 1000
```

See `MULTI-GPU-GUIDE.md` in `~/Desktop/test/` for advanced patterns.

---

## 🔄 Rebuilding After Code Changes

When you modify your Rust code:

```bash
# Rebuild the image
./build-and-run-gpu.sh

# Run with new code
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000
```

---

## 📁 Output Location

Results are saved to `./output/` directory in your project.

```bash
ls -lh output/
```

---

## ⚡ Performance Tips

### More Attempts for Better Results
```bash
./run-prism-gpu.sh data/nipah/2VSM.mtx 100000   # 100k attempts
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000000  # 1M attempts
```

### Monitor GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Parallel Runs (Multiple Files)
```bash
# Run different files simultaneously
./run-prism-gpu.sh data/file1.mtx 1000 &
./run-prism-gpu.sh data/file2.mtx 1000 &
./run-prism-gpu.sh data/file3.mtx 1000 &
wait
```

---

## 🐛 Troubleshooting

### "Image not found"
```bash
./build-and-run-gpu.sh
```

### "GPU not accessible"
```bash
# Check GPU setup
nvidia-smi

# Verify Docker GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

### "File not found"
Make sure file path is relative to `data/`:
```bash
# Correct
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000

# Wrong
./run-prism-gpu.sh /home/diddy/Desktop/PRISM-FINNAL-PUSH/data/nipah/2VSM.mtx 1000
```

---

## 🎓 Summary

**Files Created**:
- `Dockerfile` - GPU-enabled container definition
- `build-and-run-gpu.sh` - One-command build script
- `run-prism-gpu.sh` - Quick runner for your GPU binary

**Usage**:
```bash
# Build once
./build-and-run-gpu.sh

# Run anytime
./run-prism-gpu.sh data/nipah/2VSM.mtx 1000
```

**Features**:
- ✅ Full GPU acceleration
- ✅ Same proven GPU solution
- ✅ Easy to use scripts
- ✅ Hot-swappable configurations
- ✅ Multi-GPU ready

Your PRISM project now has the same GPU capabilities as the world-record container! 🚀
