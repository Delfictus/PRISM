#!/bin/bash
# PRISM World Record Runner Script
# Usage: ./run_wr.sh [config_file] [timeout_minutes]

set -e  # Exit on error

# Configuration
CONFIG="${1:-foundation/prct-core/configs/wr_sweep_D_aggr_seed_42.v1.1.toml}"
TIMEOUT="${2:-90}"  # Default 90 minutes
THREADS="${RAYON_NUM_THREADS:-24}"

# Create log directory
mkdir -p results/logs

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_NAME=$(basename "$CONFIG" .toml)
LOGFILE="results/logs/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          PRISM World Record Runner                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Config:   $CONFIG"
echo "Timeout:  ${TIMEOUT}m"
echo "Threads:  $THREADS"
echo "Log:      $LOGFILE"
echo ""
echo "Starting run..."
echo ""

# Run the example
RAYON_NUM_THREADS=$THREADS timeout ${TIMEOUT}m \
    cargo run --release --features cuda \
    --example world_record_dsjc1000 \
    "$CONFIG" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Run Complete                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Log saved to: $LOGFILE"
echo ""

# Extract result if available
COLORS=$(grep "^FINAL RESULT: colors=" "$LOGFILE" 2>/dev/null | grep -oE 'colors=[0-9]+' | grep -oE '[0-9]+' | head -1)
if [ -n "$COLORS" ]; then
    echo "✅ FINAL RESULT: $COLORS colors"
else
    echo "⚠️  Run did not complete (timeout or error)"
fi
