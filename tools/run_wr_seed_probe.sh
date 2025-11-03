#!/bin/bash
# WR Seed Probe Runner
# Runs seed variants of best configs (D and F) with time limits to determine Go/No-Go for 24-48h runs

set -euo pipefail

# Configuration
MAX_MINUTES=${MAX_MINUTES:-90}
TIMEOUT_SECONDS=$((MAX_MINUTES * 60))
RESULTS_DIR="results"
LOGS_DIR="${RESULTS_DIR}/logs"
JSONL_FILE="${RESULTS_DIR}/dsjc1000_seed_probe.jsonl"

# Create results directories
mkdir -p "${LOGS_DIR}"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Config list
CONFIGS=(
    "foundation/prct-core/configs/wr_sweep_D_seed_42.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_D_seed_1337.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_D_seed_9001.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_D_aggr_seed_42.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_D_aggr_seed_1337.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_F_seed_42.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_F_seed_1337.v1.1.toml"
    "foundation/prct-core/configs/wr_sweep_F_seed_9001.v1.1.toml"
)

# Initialize JSONL file with header comment
echo "# DSJC1000.5 Seed Probe Results - $(date -Iseconds)" > "${JSONL_FILE}"

echo "═══════════════════════════════════════════════════════════"
echo -e "${BLUE}WR Seed Probe Runner${NC}"
echo "═══════════════════════════════════════════════════════════"
echo "Max runtime per config: ${MAX_MINUTES} minutes"
echo "Results directory: ${RESULTS_DIR}/"
echo "JSONL output: ${JSONL_FILE}"
echo "Total configs: ${#CONFIGS[@]}"
echo ""

# Function to parse colors from output
parse_colors() {
    local log_file="$1"

    # ONLY look for "FINAL RESULT: colors=X" format (unambiguous)
    local colors=$(grep "^FINAL RESULT: colors=" "$log_file" 2>/dev/null | \
                   grep -oE 'colors=[0-9]+' | \
                   grep -oE '[0-9]+' | head -1)

    # Fallback: look for JSON telemetry
    if [ -z "$colors" ]; then
        colors=$(grep '"event":"final_result"' "$log_file" 2>/dev/null | \
                 grep -oE '"colors":[0-9]+' | \
                 grep -oE '[0-9]+' | head -1)
    fi

    # If still empty, return sentinel (don't guess!)
    echo "${colors:-999}"
}

# Function to parse time from output
parse_time() {
    local log_file="$1"
    # Look for "Total Time: X seconds" or "Total Time: HH:MM:SS"
    local time_line=$(grep -E "Total Time|Runtime" "$log_file" 2>/dev/null | tail -1)

    if echo "$time_line" | grep -qE "[0-9]+ seconds"; then
        # Extract seconds directly
        echo "$time_line" | grep -oE '[0-9]+' | head -1
    elif echo "$time_line" | grep -qE "[0-9]+:[0-9]+:[0-9]+"; then
        # Convert HH:MM:SS to seconds
        local hms=$(echo "$time_line" | grep -oE '[0-9]+:[0-9]+:[0-9]+')
        local hours=$(echo "$hms" | cut -d: -f1)
        local mins=$(echo "$hms" | cut -d: -f2)
        local secs=$(echo "$hms" | cut -d: -f3)
        echo $((hours * 3600 + mins * 60 + secs))
    else
        echo "0"
    fi
}

# Function to extract seed from config path
extract_seed() {
    local config="$1"
    if echo "$config" | grep -qE "seed_[0-9]+"; then
        echo "$config" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+'
    else
        echo "0"
    fi
}

# Run each config
TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    ((CURRENT++))

    if [ ! -f "$config" ]; then
        echo -e "${RED}✗ Config not found: $config${NC}"
        continue
    fi

    # Extract metadata
    SEED=$(extract_seed "$config")
    CONFIG_NAME=$(basename "$config" .v1.1.toml)
    TIMESTAMP=$(date -Iseconds)
    LOG_FILE="${LOGS_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

    echo "═══════════════════════════════════════════════════════════"
    echo -e "${BLUE}[$CURRENT/$TOTAL] Running: $CONFIG_NAME${NC}"
    echo "Config: $config"
    echo "Seed: $SEED"
    echo "Timeout: ${MAX_MINUTES} minutes"
    echo "Log: $LOG_FILE"
    echo ""

    # Run with timeout
    START_TIME=$(date +%s)
    STATUS="ok"

    if timeout "${TIMEOUT_SECONDS}s" \
        cargo run --release --features cuda --example world_record_dsjc1000 \
        "$config" > "$LOG_FILE" 2>&1; then
        STATUS="ok"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            STATUS="timeout"
            echo -e "${YELLOW}⏱  Timeout after ${MAX_MINUTES} minutes${NC}"
        else
            STATUS="error"
            echo -e "${RED}✗ Error (exit code: $EXIT_CODE)${NC}"
        fi
    fi

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Parse results
    COLORS=$(parse_colors "$LOG_FILE")
    TIME_S=$(parse_time "$LOG_FILE")

    # Use elapsed time if parsing failed
    if [ "$TIME_S" -eq 0 ]; then
        TIME_S=$ELAPSED
    fi

    # Write JSONL entry
    echo "{\"config\":\"$config\",\"seed\":$SEED,\"colors\":$COLORS,\"time_s\":$TIME_S,\"status\":\"$STATUS\",\"ts\":\"$TIMESTAMP\"}" >> "${JSONL_FILE}"

    # Display result
    if [ "$COLORS" -le 95 ]; then
        echo -e "${GREEN}✓ Colors: $COLORS (EXCELLENT - ${TIME_S}s)${NC}"
    elif [ "$COLORS" -le 98 ]; then
        echo -e "${YELLOW}△ Colors: $COLORS (GOOD - ${TIME_S}s)${NC}"
    else
        echo -e "${RED}✗ Colors: $COLORS (${TIME_S}s)${NC}"
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo -e "${BLUE}Seed Probe Complete${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Parse JSONL results
echo "Results Summary (sorted by colors, then time):"
echo "───────────────────────────────────────────────────────────"
printf "%-40s %6s %8s %10s %10s\n" "Config" "Seed" "Colors" "Time(s)" "Status"
echo "───────────────────────────────────────────────────────────"

# Skip comment line and sort by colors, then time
grep -v "^#" "${JSONL_FILE}" | \
    jq -r '. | "\(.config)|\(.seed)|\(.colors)|\(.time_s)|\(.status)"' 2>/dev/null | \
    sort -t'|' -k3,3n -k4,4n | \
    while IFS='|' read -r cfg seed colors time_s status; do
        cfg_short=$(basename "$cfg" .v1.1.toml)
        printf "%-40s %6s %8s %10s %10s\n" "$cfg_short" "$seed" "$colors" "$time_s" "$status"
    done

echo "───────────────────────────────────────────────────────────"
echo ""

# Determine Go/No-Go verdict
BEST_COLORS=$(grep -v "^#" "${JSONL_FILE}" | jq -r '.colors' 2>/dev/null | sort -n | head -1)
BEST_TIME=$(grep -v "^#" "${JSONL_FILE}" | jq -r 'select(.colors == '"$BEST_COLORS"') | .time_s' 2>/dev/null | sort -n | head -1)

echo "═══════════════════════════════════════════════════════════"
echo -e "${BLUE}Go/No-Go Verdict${NC}"
echo "═══════════════════════════════════════════════════════════"
echo "Best result: $BEST_COLORS colors in ${BEST_TIME}s"
echo ""

VERDICT=""
EXIT_CODE=0

if [ "$BEST_COLORS" -le 95 ]; then
    VERDICT="GO"
    EXIT_CODE=0
    echo -e "${GREEN}✓ VERDICT: GO${NC}"
    echo "  Recommendation: Launch 24-48h run with best config/seed"
    echo "  Rationale: ≤95 colors achieved within ${MAX_MINUTES} minutes"

    # Find best config
    BEST_CONFIG=$(grep -v "^#" "${JSONL_FILE}" | jq -r "select(.colors == $BEST_COLORS) | .config" 2>/dev/null | head -1)
    BEST_SEED=$(grep -v "^#" "${JSONL_FILE}" | jq -r "select(.colors == $BEST_COLORS) | .seed" 2>/dev/null | head -1)
    echo ""
    echo "  Best config: $BEST_CONFIG"
    echo "  Best seed: $BEST_SEED"

elif [ "$BEST_COLORS" -le 98 ]; then
    VERDICT="MAYBE"
    EXIT_CODE=10
    echo -e "${YELLOW}△ VERDICT: MAYBE${NC}"
    echo "  Recommendation: Try different seeds/weights before long run"
    echo "  Rationale: 96-98 colors achieved - close but not optimal"
    echo "  Suggestion: Run more seed variants or adjust hyperparameters"

else
    VERDICT="NO-GO"
    EXIT_CODE=20
    echo -e "${RED}✗ VERDICT: NO-GO${NC}"
    echo "  Recommendation: Retune before launching long run"
    echo "  Rationale: >98 colors - not ready for world record attempt"
    echo "  Suggestion: Analyze logs, adjust configs, retry probe"
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Full results: $JSONL_FILE"
echo "Logs: $LOGS_DIR/"
echo ""

exit $EXIT_CODE
