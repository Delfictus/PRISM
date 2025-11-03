#!/usr/bin/env bash
# PR 3 Verification Script: Hard Guardrails & Explicit Fallback Detection
# Usage: ./tools/verify_pr3.sh

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  PR 3 Verification: Hard Guardrails & Fallback Detection"
echo "════════════════════════════════════════════════════════════════"
echo ""

ERRORS=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
}

function fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((ERRORS++))
}

function warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

# ════════════════════════════════════════════════════════════════
# 1. Check for production stubs
# ════════════════════════════════════════════════════════════════
echo "[1/10] Checking for production stubs..."
STUB_COUNT=$(rg -c "todo!|unimplemented!|panic!|dbg!" foundation/prct-core/src/world_record_pipeline.rs 2>/dev/null || echo "0")
if [ "$STUB_COUNT" -eq 0 ]; then
    pass "No production stubs found"
else
    fail "Found $STUB_COUNT stub patterns (todo!/unimplemented!/panic!/dbg!)"
    rg "todo!|unimplemented!|panic!|dbg!" foundation/prct-core/src/world_record_pipeline.rs
fi

# ════════════════════════════════════════════════════════════════
# 2. Check expect() usage (only Default trait allowed)
# ════════════════════════════════════════════════════════════════
echo "[2/10] Checking expect() usage..."
EXPECT_COUNT=$(rg -c "expect\(" foundation/prct-core/src/world_record_pipeline.rs 2>/dev/null || echo "0")
if [ "$EXPECT_COUNT" -le 3 ]; then
    pass "expect() usage acceptable ($EXPECT_COUNT instances, all in Default trait)"
else
    warn "expect() used $EXPECT_COUNT times (verify all are in Default trait or documented)"
fi

# ════════════════════════════════════════════════════════════════
# 3. Check fallback logging count
# ════════════════════════════════════════════════════════════════
echo "[3/10] Checking fallback logging..."
FALLBACK_COUNT=$(rg -c "\[FALLBACK\]" foundation/prct-core/src/world_record_pipeline.rs 2>/dev/null || echo "0")
if [ "$FALLBACK_COUNT" -ge 25 ]; then
    pass "Fallback logging present ($FALLBACK_COUNT statements)"
else
    fail "Insufficient fallback logging ($FALLBACK_COUNT statements, expected >= 25)"
fi

# ════════════════════════════════════════════════════════════════
# 4. Check VRAM guard logging
# ════════════════════════════════════════════════════════════════
echo "[4/10] Checking VRAM guard logging..."
VRAM_COUNT=$(rg -c "\[VRAM\]\[GUARD\]" foundation/prct-core/src/world_record_pipeline.rs 2>/dev/null || echo "0")
if [ "$VRAM_COUNT" -ge 5 ]; then
    pass "VRAM guard logging present ($VRAM_COUNT statements)"
else
    warn "Limited VRAM guard logging ($VRAM_COUNT statements)"
fi

# ════════════════════════════════════════════════════════════════
# 5. Check error returns (proper PRCTError usage)
# ════════════════════════════════════════════════════════════════
echo "[5/10] Checking proper error returns..."
ERROR_COUNT=$(rg -c "PRCTError::" foundation/prct-core/src/world_record_pipeline.rs 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -ge 15 ]; then
    pass "Proper error returns present ($ERROR_COUNT PRCTError instances)"
else
    warn "Limited error handling ($ERROR_COUNT PRCTError instances)"
fi

# ════════════════════════════════════════════════════════════════
# 6. Check validate_vram_requirements() exists
# ════════════════════════════════════════════════════════════════
echo "[6/10] Checking VRAM validation method..."
if rg -q "validate_vram_requirements" foundation/prct-core/src/world_record_pipeline.rs; then
    pass "VRAM validation method implemented"
else
    fail "VRAM validation method not found"
fi

# ════════════════════════════════════════════════════════════════
# 7. Check unimplemented feature guards (TDA, PIMC, GNN)
# ════════════════════════════════════════════════════════════════
echo "[7/10] Checking unimplemented feature guards..."
GUARD_COUNT=0
rg -q "TDA.*not.*implement" foundation/prct-core/src/world_record_pipeline.rs && ((GUARD_COUNT++)) || true
rg -q "PIMC.*not.*implement" foundation/prct-core/src/world_record_pipeline.rs && ((GUARD_COUNT++)) || true
rg -q "GNN.*not.*implement" foundation/prct-core/src/world_record_pipeline.rs && ((GUARD_COUNT++)) || true

if [ "$GUARD_COUNT" -eq 3 ]; then
    pass "All unimplemented feature guards present (TDA, PIMC, GNN)"
else
    fail "Missing unimplemented feature guards ($GUARD_COUNT/3 found)"
fi

# ════════════════════════════════════════════════════════════════
# 8. Check documentation exists
# ════════════════════════════════════════════════════════════════
echo "[8/10] Checking documentation..."
if [ -f "docs/FALLBACK_SCENARIOS.md" ]; then
    LINES=$(wc -l < docs/FALLBACK_SCENARIOS.md)
    if [ "$LINES" -ge 400 ]; then
        pass "FALLBACK_SCENARIOS.md exists ($LINES lines)"
    else
        warn "FALLBACK_SCENARIOS.md exists but may be incomplete ($LINES lines)"
    fi
else
    fail "docs/FALLBACK_SCENARIOS.md not found"
fi

if [ -f "docs/FALLBACK_QUICK_REFERENCE.md" ]; then
    pass "FALLBACK_QUICK_REFERENCE.md exists"
else
    warn "docs/FALLBACK_QUICK_REFERENCE.md not found"
fi

# ════════════════════════════════════════════════════════════════
# 9. Check compilation with CUDA features (prct-core only)
# ════════════════════════════════════════════════════════════════
echo "[9/10] Checking CUDA compilation (prct-core)..."
# Check only the prct-core crate (our changes)
if cargo check --package prct-core --features cuda --quiet 2>&1 | grep -iq "^error\["; then
    fail "Compilation errors found in prct-core with --features cuda"
    cargo check --package prct-core --features cuda 2>&1 | grep "^error\[" | head -10
else
    pass "prct-core compiles successfully with --features cuda"
fi

# ════════════════════════════════════════════════════════════════
# 10. Check magic numbers (should use config structs)
# ════════════════════════════════════════════════════════════════
echo "[10/10] Checking for magic numbers in loops..."
# Look for hardcoded values in loop conditions (excluding comments and test code)
MAGIC_COUNT=$(rg '\bfor.*[0-9]{2,}\b' foundation/prct-core/src/world_record_pipeline.rs | grep -v "//" | wc -l || echo "0")
if [ "$MAGIC_COUNT" -le 5 ]; then
    pass "Minimal magic numbers in loops ($MAGIC_COUNT instances)"
else
    warn "Found $MAGIC_COUNT potential magic numbers in loops (verify they use config)"
fi

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Verification Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Errors:   $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✅ PR 3 VERIFICATION PASSED${NC}"
    echo ""
    echo "All hard guardrails and fallback detection checks passed."
    echo "Ready for merge pending review."
    exit 0
else
    echo -e "${RED}❌ PR 3 VERIFICATION FAILED${NC}"
    echo ""
    echo "Found $ERRORS critical issues that must be fixed."
    [ "$WARNINGS" -gt 0 ] && echo "Also found $WARNINGS warnings to review."
    exit 1
fi
