# RunPod Access Troubleshooting

## Issue: Web Terminal Toggle Keeps Unchecking

This is a known RunPod UI bug. Here are working alternatives:

---

## ✅ Solution 1: Direct SSH Connection

### Step 1: Get SSH Command from RunPod

1. Go to your pod page in RunPod console
2. Look for **"Connection Options"** or **"SSH"** section
3. You'll see something like:
   ```
   ssh root@123.45.67.89 -p 22334
   ```
4. Copy that entire command

### Step 2: Connect from Your Terminal

```bash
# Paste and run the SSH command
ssh root@123.45.67.89 -p 22334

# If it asks about authenticity, type: yes
```

### Step 3: Once Connected

```bash
# Check if Docker container is running
docker ps

# If you see the PRISM container, attach to it:
docker exec -it $(docker ps -q) bash

# Or start it if not running:
docker run --gpus all -it \
  -v /workspace:/workspace \
  delfictus/prism-ai-world-record:latest
```

---

## ✅ Solution 2: RunPod CLI (Fastest)

### Install RunPod CLI
```bash
pip install runpod
```

### Configure and Connect
```bash
# Set up API key (get from RunPod account settings)
runpod config

# List your pods
runpod get pods

# SSH directly into your pod
runpod ssh <pod-id>
```

---

## ✅ Solution 3: Use RunPod's Jupyter Terminal

If your pod has Jupyter enabled:

1. In RunPod console, click **"Connect"** → **"Jupyter Lab"**
2. Once Jupyter opens, click **"File"** → **"New"** → **"Terminal"**
3. Now you have a terminal in the browser

```bash
# In Jupyter terminal
docker ps
docker exec -it $(docker ps -q) bash
```

---

## ✅ Solution 4: Auto-Start Container (Set and Forget)

Edit your pod configuration to auto-start the container:

1. In RunPod console, go to pod settings
2. Under **"Docker Command"**, set:
   ```bash
   docker run --gpus all -d \
     -v /workspace:/workspace \
     --name prism \
     delfictus/prism-ai-world-record:latest \
     tail -f /dev/null
   ```
3. Save and restart pod

Now when you SSH in, just run:
```bash
docker exec -it prism bash
```

---

## Troubleshooting Specific Issues

### Issue: "Connection Refused"
**Cause**: Pod might not be fully started
**Solution**: 
- Wait 1-2 minutes after pod shows "Running"
- Check pod logs in RunPod console for errors

### Issue: "No Docker Container Running"
**Solution**:
```bash
# List all containers (including stopped)
docker ps -a

# Start the PRISM container manually
docker run --gpus all -it \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest
```

### Issue: "Permission Denied (publickey)"
**Solution**:
```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096

# Add your public key to RunPod:
# 1. Go to RunPod Settings → SSH Keys
# 2. Add your ~/.ssh/id_rsa.pub content
# 3. Restart pod
```

### Issue: "Port Already in Use"
**Solution**:
```bash
# Kill existing container
docker stop $(docker ps -q)
docker rm $(docker ps -aq)

# Start fresh
docker run --gpus all -it \
  -v /workspace:/workspace \
  delfictus/prism-ai-world-record:latest
```

---

## Quick Verification Steps

Once you get SSH access, verify everything:

```bash
# 1. Check you're on the RunPod host
nvidia-smi  # Should show 8 GPUs

# 2. Check Docker is running
docker --version

# 3. Pull the image if needed
docker pull delfictus/prism-ai-world-record:latest

# 4. Start container
docker run --gpus all -it \
  -v /workspace/qtables:/app/fluxnet_cache \
  -v /workspace/results:/app/results \
  delfictus/prism-ai-world-record:latest

# 5. You should see the PRISM welcome screen
```

---

## If Nothing Works: Contact RunPod Support

Sometimes pods have configuration issues. Contact RunPod:
- Discord: https://discord.gg/runpod
- Email: support@runpod.io
- Live chat on runpod.io website

Provide them with:
- Your pod ID
- Error message
- Screenshot of the toggle issue

---

## Workaround: Use Community Cloud

If Secure Cloud is having issues, try Community Cloud:
1. Go to https://www.runpod.io/console/gpu-cloud
2. Select **"Community Cloud"** tab
3. Deploy with same image: `delfictus/prism-ai-world-record:latest`
4. Community Cloud terminals usually work more reliably

---

## Last Resort: Download Image and Run Locally

If RunPod continues to have issues:

```bash
# On your local machine with GPU
docker pull delfictus/prism-ai-world-record:latest

docker run --gpus all -it \
  -v $(pwd)/qtables:/app/fluxnet_cache \
  -v $(pwd)/results:/app/results \
  delfictus/prism-ai-world-record:latest

# Then run
prism-quick  # Test run
```

---

## Summary of Access Methods (In Order of Reliability)

1. ✅ **Direct SSH** - Most reliable, works 99% of the time
2. ✅ **RunPod CLI** - Fast and scriptable
3. ✅ **Jupyter Terminal** - Good alternative if available
4. ⚠️ **Web Terminal** - Buggy, avoid if possible
5. ✅ **Community Cloud** - Try if Secure Cloud has issues

**Recommended**: Use direct SSH. It's the most stable way to access RunPod pods.
