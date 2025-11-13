# How to Access PRISM on RunPod

## ❌ NOT via Web Browser
The container doesn't run a web service. The URL `https://h6qc6lng7skotv-8888.proxy.runpod.net/` won't work.

## ✅ Access via SSH Terminal

### Method 1: RunPod Web Terminal (Easiest)

1. Go to https://www.runpod.io/console/pods
2. Find your running pod
3. Click **"Connect"** button
4. Select **"Start Web Terminal"** or **"Connect via SSH"**
5. You'll get a terminal directly in your browser

### Method 2: SSH from Your Local Machine

1. In RunPod console, click your pod
2. Copy the SSH connection string (looks like):
   ```
   ssh root@X.X.X.X -p XXXXX -i ~/.ssh/runpod_key
   ```
3. Run that command from your local terminal

### Method 3: RunPod CLI

```bash
# List your pods
runpod get pods

# SSH into specific pod
runpod ssh <pod-id>
```

---

## Once Connected via Terminal

You should see the PRISM welcome screen:

```
════════════════════════════════════════════════════════════
  PRISM AI - World Record Pipeline (8x B200)
════════════════════════════════════════════════════════════

→ GPU Check:
index, name, memory.total [MiB]
0, NVIDIA B200, 180000
1, NVIDIA B200, 180000
...
✓ Found 8 GPU(s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PRISM Quick Start Commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🚀 Quick Test (5-10 min):
     prism-quick

  🎯 World Record (72h max):
     prism-wr

  🧠 Adaptive RL (with persistence):
     prism-adaptive

  📊 View Results:
     prism-results

  💾 View RL Cache:
     prism-cache

  🎮 GPU Status:
     gpus
     gpu-watch
```

### Run Your First Test

```bash
# Quick 5-10 minute test
prism-quick

# Or start world record attempt (72h)
prism-wr

# Or adaptive RL with persistence (48h)
prism-adaptive
```

---

## Monitoring Progress

### Watch GPU Utilization
```bash
# Continuous monitoring
gpu-watch

# One-time check
gpus
```

### Monitor Live Telemetry
```bash
# In another SSH session (or screen/tmux)
tail -f /app/results/*.jsonl
```

### Check Results
```bash
prism-results    # List all result files
prism-cache      # Check Q-table files
```

---

## Troubleshooting

### Issue: "Cannot connect via SSH"
**Solution**: 
- Make sure pod is in "Running" state (not "Stopped" or "Starting")
- Check SSH port in RunPod console (changes for each pod)
- Ensure you have the correct SSH key

### Issue: "Container not responding"
**Solution**:
```bash
# Check if container is running
docker ps

# If not running, start it
docker run --gpus all -it \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest
```

### Issue: "GPUs not detected"
**Solution**:
```bash
# Verify GPU access
nvidia-smi

# Check CUDA environment
echo $CUDA_VISIBLE_DEVICES  # Should be: 0,1,2,3,4,5,6,7
```

---

## Download Results After Training

### While Still Connected via SSH:

```bash
# Package everything
cd /app
tar -czf trained-model.tar.gz fluxnet_cache/ results/

# Copy to accessible location
cp trained-model.tar.gz /workspace/
```

### From Your Local Machine:

```bash
# Use scp to download
scp -P <port> root@<runpod-ip>:/workspace/trained-model.tar.gz ./

# Or use RunPod web interface:
# 1. Go to pod page
# 2. Click "Storage" or "Files"
# 3. Download from /workspace/trained-model.tar.gz
```

---

## Running Long Training Sessions

### Use Screen or Tmux (Recommended)

```bash
# Start screen session
screen -S prism-training

# Run your command
prism-wr

# Detach: Press Ctrl+A, then D

# Later, reattach
screen -r prism-training

# Kill session when done
screen -X -S prism-training quit
```

### Or Run in Background with Nohup

```bash
nohup /app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml > /app/logs/run.log 2>&1 &

# Monitor progress
tail -f /app/logs/run.log

# Check process
ps aux | grep world_record
```

---

## Summary

**DON'T**: Try to access via web browser at `https://xxx.proxy.runpod.net/`  
**DO**: Use SSH terminal to access the container and run commands  

**Quick Steps**:
1. RunPod Console → Your Pod → "Connect" → "Web Terminal"
2. Wait for PRISM welcome screen
3. Run: `prism-quick` (test) or `prism-wr` (full run)
4. Monitor: `gpu-watch` or `tail -f /app/results/*.jsonl`
5. Download results when done via scp or RunPod web interface

Good luck! 🚀
