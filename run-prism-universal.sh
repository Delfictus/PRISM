#!/bin/bash

# Universal PRISM runner - accepts ALL data types

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input-file> [attempts] [--algorithm greedy|prct] [--verbose]"
    echo ""
    echo "Supported formats:"
    echo "  • MTX (Matrix Market) - Graph/matrix data"
    echo "  • PDB/CIF - Protein structure files"
    echo "  • DIMACS (.col) - Graph coloring benchmarks"
    echo "  • CSV/TSV - Tabular data"
    echo ""
    echo "Algorithms:"
    echo "  • greedy - Fast greedy coloring (default)"
    echo "  • prct   - Custom PRCT algorithm with probabilistic selection"
    echo ""
    echo "Examples:"
    echo "  $0 data/nipah/2VSM.mtx 1000"
    echo "  $0 data/nipah/2VSM.mtx 1000 --algorithm prct"
    echo "  $0 data/protein.pdb 5000 --verbose"
    echo "  $0 benchmarks/dimacs/DSJC1000.5.col 10000 --algorithm prct"
    echo ""
    exit 1
fi

INPUT_FILE="$1"
ATTEMPTS="${2:-1000}"
VERBOSE=""
ALGORITHM="greedy"

# Parse optional flags from all arguments
for ((i=3; i<=$#; i++)); do
    arg="${!i}"
    if [[ "$arg" == "--verbose" ]]; then
        VERBOSE="--verbose"
    elif [[ "$arg" == "--algorithm" ]]; then
        next_i=$((i+1))
        ALGORITHM="${!next_i}"
    fi
done

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Input file not found: $INPUT_FILE"
    exit 1
fi

# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PRISM Universal Platform - Running with GPU              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Run the universal binary
./target/release/prism_universal \
    --input "$INPUT_FILE" \
    --attempts "$ATTEMPTS" \
    --output "./output" \
    --algorithm "$ALGORITHM" \
    --gpu \
    $VERBOSE

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Success!"
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
