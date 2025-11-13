#!/bin/bash
# FluxNet RL + GPU Implementation Validation Script
# Checks all three parts (A, B, C) for correctness

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  FluxNet RL + GPU Implementation Validation                ║"
echo "║  Branch: claude/fluxnet-rl-gpu-implementation-011CUzcMsiXXs║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    local cmd="$2"
    local expected="$3"

    echo -n "Checking: $name ... "

    if eval "$cmd" | grep -q "$expected"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Expected to find: $expected"
        ((FAIL++))
    fi
}

warn_check() {
    local name="$1"
    local cmd="$2"
    local expected="$3"

    echo -n "Checking: $name ... "

    if eval "$cmd" | grep -q "$expected"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠ WARN${NC} (optional)"
        ((WARN++))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part A: FluxNet Q-table Persistence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check "fluxnet_cache_dir field exists" \
    "grep -r 'fluxnet_cache_dir' foundation/prct-core/src/world_record_pipeline.rs" \
    "fluxnet_cache_dir"

check "Load pretrained Q-table logic" \
    "grep -A5 'load_pretrained' foundation/prct-core/src/world_record_pipeline.rs" \
    "load_qtable"

check "Phase 2 checkpoint saving" \
    "grep -r 'qtable_checkpoint_phase2.bin' foundation/prct-core/src/world_record_pipeline.rs" \
    "checkpoint_phase2"

check "Final Q-table save" \
    "grep -r 'qtable_final.bin' foundation/prct-core/src/world_record_pipeline.rs" \
    "qtable_final"

check "MemoryTier::Compact updated to 4096" \
    "grep -A2 'MemoryTier::Compact =>' foundation/prct-core/src/fluxnet/config.rs" \
    "4096"

check "PersistenceConfig in config file" \
    "grep -A5 '\[fluxnet.persistence\]' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "cache_dir"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part B: Aggressive Phase 2 Thermodynamic Hardening"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check "aggressive_midband field in struct" \
    "grep 'aggressive_midband' foundation/prct-core/src/world_record_pipeline.rs" \
    "aggressive_midband"

check "Config: num_temps = 58" \
    "grep 'num_temps.*58' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "58"

check "Config: steps_per_temp = 20000" \
    "grep 'steps_per_temp.*20000' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "20000"

check "Config: force_start_temp = 9.0" \
    "grep 'force_start_temp.*9' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "9.0"

check "Config: force_full_strength_temp = 3.0" \
    "grep 'force_full_strength_temp.*3' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "3.0"

check "Config: aggressive_midband = true" \
    "grep 'aggressive_midband.*true' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "true"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part C: TDA GPU Integration (Phase 6)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check "use_tda config field exists" \
    "grep 'pub use_tda:' foundation/prct-core/src/world_record_pipeline.rs" \
    "use_tda"

check "enable_tda_gpu config field exists" \
    "grep 'pub enable_tda_gpu:' foundation/prct-core/src/world_record_pipeline.rs" \
    "enable_tda_gpu"

check "Phase 6 TDA implementation exists" \
    "grep -A10 'PHASE 6: Topological Data Analysis' foundation/prct-core/src/world_record_pipeline.rs" \
    "PHASE 6"

check "Config: use_tda = true" \
    "grep '^use_tda.*true' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "true"

check "Config: enable_tda_gpu = true" \
    "grep 'enable_tda_gpu.*true' foundation/prct-core/configs/wr_sweep_D_aggr_seed_9001.v1.1.toml" \
    "true"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking: prct-core library compiles ... "
cd foundation/prct-core
if cargo check --features cuda --lib 2>&1 | grep -q "Finished"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAIL++))
fi
cd ../..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Policy Compliance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking: No unwrap/expect in modified code ... "
if ! grep -r "\.unwrap()\|\.expect(" \
    foundation/prct-core/src/world_record_pipeline.rs \
    foundation/prct-core/src/fluxnet/config.rs 2>/dev/null | grep -q "fluxnet\|aggressive_midband\|cache_dir"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAIL++))
fi

echo -n "Checking: No todo!/panic!/dbg! in modified code ... "
if ! grep -r "todo!\|panic!\|dbg!" \
    foundation/prct-core/src/world_record_pipeline.rs \
    foundation/prct-core/src/fluxnet/config.rs 2>/dev/null | grep -q "fluxnet\|aggressive_midband\|cache_dir"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAIL++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Optional Runtime Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

warn_check "FluxNet cache directory exists" \
    "ls -d target/fluxnet_cache 2>/dev/null" \
    "fluxnet_cache"

warn_check "Pretrained Q-table exists" \
    "ls target/fluxnet_cache/qtable_pretrained.bin 2>/dev/null" \
    "qtable_pretrained.bin"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  VALIDATION RESULTS                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "  ${GREEN}✓ Passed:${NC} $PASS"
echo -e "  ${YELLOW}⚠ Warnings:${NC} $WARN (optional checks)"
echo -e "  ${RED}✗ Failed:${NC} $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ ALL VALIDATION CHECKS PASSED                          ║${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}║  Implementation is READY FOR TESTING                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  VALIDATION FAILED                                     ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  Please review failed checks above                         ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
