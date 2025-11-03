#!/bin/bash
# Batch execution script for all 7 WR sweep configs

set -e

echo "========================================="
echo "World Record Sweep Batch Execution"
echo "========================================="
echo ""
echo "Runtime: ~7 days (24h per config)"
echo "Configs: A, B, C, D, E, F, G"
echo "Target: 83 colors (current best: 87)"
echo ""

# Create logs directory
mkdir -p logs

# Configs to run
CONFIGS=("A" "B" "C" "D" "E" "F" "G")

# Start timestamp
START_TIME=$(date +%s)

for config in "${CONFIGS[@]}"; do
    echo "========================================="
    echo "🚀 Starting WR Sweep ${config}"
    echo "========================================="
    echo "Config: foundation/prct-core/configs/wr_sweep_${config}.v1.1.toml"
    echo "Log: logs/wr_sweep_${config}.log"
    echo "Started: $(date)"
    echo ""

    # Run the world record attempt
    cargo run --release --features cuda --example world_record_dsjc1000 \
        foundation/prct-core/configs/wr_sweep_${config}.v1.1.toml \
        2>&1 | tee logs/wr_sweep_${config}.log

    # Extract best result
    BEST=$(grep "BEST COLORING" logs/wr_sweep_${config}.log | tail -1 || echo "N/A")

    echo ""
    echo "========================================="
    echo "✅ Completed WR Sweep ${config}"
    echo "========================================="
    echo "Result: ${BEST}"
    echo "Finished: $(date)"
    echo ""

    # Brief pause between configs
    sleep 5
done

# End timestamp
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo "========================================="
echo "🎉 All 7 WR Sweeps Complete!"
echo "========================================="
echo "Total Runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "📊 Results Summary:"
echo "----------------------------------------"

for config in "${CONFIGS[@]}"; do
    if [ -f "logs/wr_sweep_${config}.log" ]; then
        BEST=$(grep "BEST COLORING" logs/wr_sweep_${config}.log | tail -1 | awk '{print $3}' || echo "N/A")
        CONFLICTS=$(grep "conflicts" logs/wr_sweep_${config}.log | tail -1 | grep -oP '\d+ conflicts' || echo "N/A")
        echo "Config ${config}: ${BEST} colors (${CONFLICTS})"
    else
        echo "Config ${config}: Log not found"
    fi
done

echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Analyze results to find winner"
echo "  2. Run extended 7-day attempt with best config"
echo "  3. Check logs for GPU utilization"
echo ""
echo "Logs location: logs/"
echo "========================================="
