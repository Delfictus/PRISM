#!/bin/bash
# Validate all 7 WR sweep TOML configs for v1.1

set -e

echo "========================================="
echo "WR Sweep v1.1 Configuration Validation"
echo "========================================="
echo ""

CONFIG_DIR="foundation/prct-core/configs"
CONFIGS=("A" "B" "C" "D" "E" "F" "G")
ERRORS=0

# ANSI colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "📋 Validating 7 WR sweep configurations..."
echo ""

for config in "${CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/wr_sweep_${config}.v1.1.toml"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ ${CONFIG_FILE}: NOT FOUND${NC}"
        ((ERRORS++))
        continue
    fi

    # Check file size (should be >1KB)
    SIZE=$(stat -f%z "$CONFIG_FILE" 2>/dev/null || stat -c%s "$CONFIG_FILE" 2>/dev/null)
    if [ "$SIZE" -lt 1000 ]; then
        echo -e "${RED}❌ ${CONFIG_FILE}: TOO SMALL (${SIZE} bytes)${NC}"
        ((ERRORS++))
        continue
    fi

    # Check VRAM limits
    THERMO_REPLICAS=$(grep "^replicas = " "$CONFIG_FILE" | head -1 | awk '{print $3}')
    PIMC_REPLICAS=$(grep "^replicas = " "$CONFIG_FILE" | tail -1 | awk '{print $3}')
    PIMC_BEADS=$(grep "^beads = " "$CONFIG_FILE" | awk '{print $3}')

    VRAM_STATUS="✅ Safe"
    if [ "$THERMO_REPLICAS" -gt 56 ] || [ "$PIMC_REPLICAS" -gt 56 ] || [ "$PIMC_BEADS" -gt 64 ]; then
        VRAM_STATUS="${RED}❌ VRAM EXCEEDED${NC}"
        ((ERRORS++))
    elif [ "$THERMO_REPLICAS" -eq 56 ] || [ "$PIMC_REPLICAS" -eq 56 ] || [ "$PIMC_BEADS" -eq 64 ]; then
        VRAM_STATUS="${YELLOW}⚠️  VRAM Max${NC}"
    fi

    # Check GPU flags
    GPU_FLAGS=$(grep -c "enable_.*_gpu = true" "$CONFIG_FILE" || echo 0)

    # Check orchestrator flags
    ORCH_FLAGS=$(grep -c "use_.* = true" "$CONFIG_FILE" || echo 0)

    # Display results
    if [ "$ERRORS" -eq 0 ]; then
        echo -e "${GREEN}✅ wr_sweep_${config}.v1.1.toml${NC}"
    else
        echo -e "${RED}❌ wr_sweep_${config}.v1.1.toml${NC}"
    fi

    echo "   Size: ${SIZE} bytes"
    echo "   GPU Modules: ${GPU_FLAGS}/7 enabled"
    echo "   Orchestrator: ${ORCH_FLAGS} modules enabled"
    echo "   Thermo Replicas: ${THERMO_REPLICAS}"
    echo "   PIMC: replicas=${PIMC_REPLICAS}, beads=${PIMC_BEADS}"
    echo -e "   VRAM: ${VRAM_STATUS}"
    echo ""
done

echo "========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All 7 configs validated successfully!${NC}"
    echo ""
    echo "Ready to run:"
    echo "  1. Sequential: Run each config for 24h"
    echo "  2. Batch: ./run_wr_sweep.sh"
    echo ""
    echo "Monitor GPU:"
    echo "  watch -n 1 nvidia-smi"
else
    echo -e "${RED}❌ Validation failed with ${ERRORS} error(s)${NC}"
    exit 1
fi
echo "========================================="
