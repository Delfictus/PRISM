#!/bin/bash
# Live Telemetry Viewer for PRISM

echo "🔍 PRISM Live Telemetry Monitor"
echo "================================"
echo ""

# Find latest telemetry file
find_latest() {
    find target/run_artifacts -name "live_metrics_*.jsonl" -type f -mmin -60 2>/dev/null | head -1
}

TELEMETRY_FILE=$(find_latest)

if [ -z "$TELEMETRY_FILE" ]; then
    echo "⏳ Waiting for telemetry file to appear..."
    echo "   (Will be created when pipeline starts)"
    echo ""

    # Wait up to 60 seconds for file to appear
    for i in {1..60}; do
        sleep 1
        TELEMETRY_FILE=$(find_latest)
        if [ -n "$TELEMETRY_FILE" ]; then
            break
        fi
        echo -ne "\r   Waiting... ${i}s"
    done
    echo ""
fi

if [ -z "$TELEMETRY_FILE" ]; then
    echo "❌ No telemetry file found after 60 seconds"
    echo "   Make sure pipeline is running with telemetry enabled"
    echo ""
    echo "Start pipeline with:"
    echo "  cargo run --release --features cuda --example world_record_dsjc1000 \\"
    echo "    foundation/prct-core/configs/wr_sweep_D.v1.1.toml"
    exit 1
fi

echo "✅ Found telemetry: $TELEMETRY_FILE"
echo ""
echo "┌──────────┬──────────┬─────────────────────────┬────────┬──────────┬──────────┐"
echo "│   Time   │  Phase   │         Step            │ Colors │ Conflict │  Time(ms)│"
echo "├──────────┼──────────┼─────────────────────────┼────────┼──────────┼──────────┤"

# Tail and format
tail -f "$TELEMETRY_FILE" | while IFS= read -r line; do
    # Skip empty lines or summary markers
    if [[ -z "$line" ]] || [[ "$line" == "---"* ]]; then
        continue
    fi

    # Parse JSON and format
    if command -v jq &> /dev/null; then
        # Using jq (better formatting)
        echo "$line" | jq -r '[
            .timestamp[11:19],
            .phase,
            .step,
            .chromatic_number,
            .conflicts,
            (.duration_ms | tostring)
        ] | @tsv' 2>/dev/null | while IFS=$'\t' read -r time phase step colors conflicts duration; do
            printf "│ %8s │ %8s │ %-23s │ %6s │ %8s │ %8s │\n" \
                "$time" "$phase" "${step:0:23}" "$colors" "$conflicts" "$duration"
        done
    else
        # Fallback without jq (basic parsing)
        TIME=$(echo "$line" | grep -o '"timestamp":"[^"]*"' | cut -d'"' -f4 | cut -c12-19)
        PHASE=$(echo "$line" | grep -o '"phase":"[^"]*"' | cut -d'"' -f4)
        STEP=$(echo "$line" | grep -o '"step":"[^"]*"' | cut -d'"' -f4)
        COLORS=$(echo "$line" | grep -o '"chromatic_number":[0-9]*' | cut -d':' -f2)
        CONFLICTS=$(echo "$line" | grep -o '"conflicts":[0-9]*' | cut -d':' -f2)
        DURATION=$(echo "$line" | grep -o '"duration_ms":[0-9.]*' | cut -d':' -f2)

        if [ -n "$TIME" ]; then
            printf "│ %8s │ %8s │ %-23s │ %6s │ %8s │ %8s │\n" \
                "$TIME" "$PHASE" "${STEP:0:23}" "$COLORS" "$CONFLICTS" "$DURATION"
        fi
    fi
done
