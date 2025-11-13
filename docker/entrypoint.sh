#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# PRISM RunPod Entrypoint - Terminal-Ready with Quick Start
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PRISM AI - World Record Pipeline (8x B200)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Check GPU availability
echo -e "\n${BLUE}→ GPU Check:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head -10

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}\n"

# Show environment
echo -e "${BLUE}→ Environment:${NC}"
echo "  Working directory: /app"
echo "  Binaries: /app/bin/"
echo "  Configs: /app/configs/"
echo "  Results: /app/results/"
echo "  Cache: /app/fluxnet_cache/"
echo ""

# Create quick-start alias helper
cat > /root/.bashrc << 'BASHRC_EOF'
# PRISM Quick Start Aliases
alias prism-quick='cd /app && /app/bin/world_record_dsjc1000 /app/configs/quick.v1.1.toml'
alias prism-wr='cd /app && /app/bin/world_record_dsjc1000 /app/configs/runpod_8gpu.v1.1.toml'
alias prism-adaptive='cd /app && /app/bin/world_record_dsjc1000 /app/configs/wr_adaptive_rl.v1.1.toml'
alias prism-results='ls -lh /app/results/'
alias prism-cache='ls -lh /app/fluxnet_cache/'
alias gpus='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'

export PS1='\[\033[01;32m\]PRISM@RunPod\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRISM Quick Start Commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🚀 Quick Test (5-10 min):"
echo "     prism-quick"
echo ""
echo "  🎯 World Record (72h max):"
echo "     prism-wr"
echo ""
echo "  🧠 Adaptive RL (with persistence):"
echo "     prism-adaptive"
echo ""
echo "  📊 View Results:"
echo "     prism-results"
echo ""
echo "  💾 View RL Cache:"
echo "     prism-cache"
echo ""
echo "  🎮 GPU Status:"
echo "     gpus"
echo "     gpu-watch"
echo ""
echo "  📁 All configs in: /app/configs/"
echo "  🔧 Custom run:"
echo "     /app/bin/world_record_dsjc1000 /app/configs/YOUR_CONFIG.toml"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
BASHRC_EOF

# If command is provided, run it (for automated runs)
if [ $# -gt 0 ]; then
    echo -e "${BLUE}→ Executing: $@${NC}\n"
    exec "$@"
else
    # No command = interactive terminal
    echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  🖥️  TERMINAL MODE${NC}"
    echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
    echo -e "\n${GREEN}✓ Ready! Run 'prism-quick' to start a quick test.${NC}\n"

    # Start bash with custom prompt
    exec /bin/bash --rcfile /root/.bashrc
fi
