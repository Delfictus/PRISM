#!/bin/bash
# Validation script for aggressive fixes

echo "=== AGGRESSIVE FIXES VALIDATION ==="
echo ""

echo "1. Config Changes:"
echo "   - force_start_temp:"
grep "force_start_temp" foundation/prct-core/configs/wr_sweep_D.v1.1.toml || echo "   ❌ NOT FOUND"
echo "   - force_full_strength_temp:"
grep "force_full_strength_temp" foundation/prct-core/configs/wr_sweep_D.v1.1.toml || echo "   ❌ NOT FOUND"
echo "   - num_temps:"
grep "^num_temps" foundation/prct-core/configs/wr_sweep_D.v1.1.toml || echo "   ❌ NOT FOUND"
echo "   - steps_per_temp:"
grep "^steps_per_temp" foundation/prct-core/configs/wr_sweep_D.v1.1.toml || echo "   ❌ NOT FOUND"
echo ""

echo "2. Kernel Changes:"
echo "   - Band-aware force gains:"
grep -c "band_gain" foundation/kernels/thermodynamic.cu
echo "   - Coupling redistribution:"
grep -c "coupling_gain" foundation/kernels/thermodynamic.cu
echo "   - Uncertainty parameter:"
grep -c "const float\* uncertainty" foundation/kernels/thermodynamic.cu
echo ""

echo "3. Rust Changes:"
echo "   - Immediate slack expansion:"
grep -c "MOVE-3.*SLACK-EXPAND" foundation/prct-core/src/gpu_thermodynamic.rs
echo "   - Escalated shake (100 vertices):"
grep -c "take(100)" foundation/prct-core/src/gpu_thermodynamic.rs
echo "   - Snapshot re-seeding:"
grep -c "MOVE-4.*SNAPSHOT-RESET" foundation/prct-core/src/gpu_thermodynamic.rs
echo "   - Slack decay logic:"
grep -c "stable_temps" foundation/prct-core/src/gpu_thermodynamic.rs
echo ""

echo "4. Build Status:"
if cargo check --features cuda --quiet 2>&1 | grep -q "error"; then
    echo "   ❌ BUILD FAILED"
    cargo check --features cuda 2>&1 | grep "error" | head -5
else
    echo "   ✅ BUILD PASSED"
fi
echo ""

echo "5. Binary Verification:"
if [ -f "target/release/prism-ai" ]; then
    echo "   ✅ Binary exists ($(ls -lh target/release/prism-ai | awk '{print $5}'))"
else
    echo "   ❌ Binary not found"
fi
echo ""

echo "6. PTX Kernel Verification:"
if [ -f "target/ptx/thermodynamic.ptx" ]; then
    echo "   ✅ PTX kernel exists ($(ls -lh target/ptx/thermodynamic.ptx | awk '{print $5}'))"
else
    echo "   ❌ PTX kernel not found"
fi
echo ""

echo "=== VALIDATION COMPLETE ==="
echo ""
echo "Next: Run pipeline with:"
echo "  ./run_wr.sh foundation/prct-core/configs/wr_sweep_D.v1.1.toml"
echo ""
echo "Monitor for:"
echo "  - Force blend ≥ 0.5 at T=4.0"
echo "  - No collapse in T=3-8 range"
echo "  - Chromatic ≥ 70 at all temps after T=8"
echo "  - Final chromatic 95-100 (not 115)"
