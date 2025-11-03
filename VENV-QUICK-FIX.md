# Quick Fix: Virtual Environment for Biopython

## The Problem

Ubuntu 24.04+ won't let you install Python packages system-wide:
```
error: externally-managed-environment
```

## ✅ Quick Solution (2 Steps)

### Step 1: Run Setup Script
```bash
cd ~/Desktop/PRISM-FINNAL-PUSH
./setup-venv.sh
```

This creates a virtual environment and installs biopython automatically.

### Step 2: Use Your Scripts
```bash
# Activate the environment (if not already active)
source venv/bin/activate

# Run your Python scripts
python your_script.py

# Deactivate when done
deactivate
```

---

## Manual Method (If You Prefer)

```bash
cd ~/Desktop/PRISM-FINNAL-PUSH

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install biopython
pip install biopython

# Install any other packages you need
pip install numpy pandas matplotlib  # etc.

# Run your code
python your_script.py

# Deactivate
deactivate
```

---

## Every Time You Work on This Project

```bash
cd ~/Desktop/PRISM-FINNAL-PUSH
source venv/bin/activate
python your_script.py
deactivate  # when done
```

---

## Quick Commands

| Task | Command |
|------|---------|
| Setup (once) | `./setup-venv.sh` |
| Activate | `source venv/bin/activate` |
| Run Python | `python script.py` |
| Install package | `pip install package-name` |
| Deactivate | `deactivate` |
| Delete venv | `rm -rf venv` |

---

## Why This Happens

**Ubuntu 24.04+** protects system Python:
- Prevents breaking OS tools
- Forces use of virtual environments
- Industry best practice

**Virtual Environment**:
- Isolated Python for your project
- Safe to experiment
- Each project has its own packages

---

## For All Your Python Projects

Use this pattern for **any** Python project:

```bash
# In any project directory
python3 -m venv venv
source venv/bin/activate
pip install your-packages
python your_script.py
deactivate
```

Or just copy the `setup-venv.sh` script to any project!

---

## Troubleshooting

### "No such file or directory: setup-venv.sh"
Make it executable:
```bash
chmod +x setup-venv.sh
./setup-venv.sh
```

### "No module named 'xyz'"
You forgot to activate:
```bash
source venv/bin/activate
```

### Want to start fresh?
Delete and recreate:
```bash
rm -rf venv
./setup-venv.sh
```

---

## System-Wide Alternative (Not Recommended)

If you really want biopython system-wide:

```bash
sudo apt install python3-biopython
```

But:
- ⚠️ Not always latest version
- ⚠️ Can't easily update
- ⚠️ Virtual env is better practice

---

## Summary

**Problem**: Can't install biopython with pip
**Solution**: Use virtual environment
**Quick Fix**: `./setup-venv.sh` then `source venv/bin/activate`
**Result**: Clean isolated Python environment for your Prism project! ✅
