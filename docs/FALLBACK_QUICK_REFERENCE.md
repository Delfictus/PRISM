# Fallback Quick Reference Card

**PRISM World Record Pipeline - Developer Cheat Sheet**

---

## Quick Diagnostics

### Check if running on GPU
```bash
cargo run --features cuda 2>&1 | grep "\[PIPELINE\]\[INIT\]"
# Expected: "CUDA device available (GPU acceleration enabled)"
```

### Count fallbacks in run
```bash
cargo run --features cuda 2>&1 | grep -c "\[FALLBACK\]"
# Expected: 0 (optimal run)
```

### Check VRAM usage
```bash
watch -n 1 nvidia-smi
# Monitor during pipeline execution
```

---

## Common Fallback Scenarios

| Symptom | Cause | Fix | Impact |
|---------|-------|-----|--------|
| "CUDA not compiled" | Missing `--features cuda` | Recompile with CUDA | 50-80% slower |
| "GPU reservoir failed" | VRAM OOM | Reduce graph or replicas | 10-50x slower |
| "Quantum solver failed" | Kuramoto invalid | Check phase values | 20-30% slower |
| "No valid colorings" | All solutions have conflicts | Increase generations | Approximate solution |
| Phase says "(CPU)" | GPU disabled in config | Set `enable_*_gpu=true` | Varies by phase |

---

## Performance Impact Cheat Sheet

```
[FALLBACK] Message                        Impact          Severity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUDA not compiled                         50-80% slower   🔴 Critical
GPU reservoir failed/disabled             10-50x slower   🔴 High
Thermodynamic GPU disabled                5x slower       🟡 Medium
Quantum solver failed                     20-30% slower   🟡 Medium
Transfer Entropy GPU disabled             2-3x slower     🟢 Low
PIMC/TDA/GNN requested                    None (skipped)  ⚪ Info
```

---

## VRAM Limits (8GB Devices)

```
Parameter               Limit   Current   Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
thermo.replicas         <= 56   56        ✅
thermo.num_temps        <= 56   56        ✅
reservoir.size          <= 2000 1000-2000 ✅
quantum.vertices^2      < 2GB   Varies    ✅
```

---

## Fallback Log Format

```
[PHASE X][FALLBACK] <What happened>
[PHASE X][FALLBACK] <Alternative action taken>
[PHASE X][FALLBACK] Performance impact: <quantitative estimate>
```

**Example**:
```
[PHASE 0][FALLBACK] GPU reservoir failed: VRAM allocation error
[PHASE 0][FALLBACK] Using CPU reservoir fallback
[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (loses GPU acceleration)
```

---

## Pre-Flight Checklist

Before running world record attempt:

```bash
# 1. Check CUDA compilation
cargo check --features cuda 2>&1 | grep -i error
# Expected: no output

# 2. Verify GPU available
nvidia-smi
# Expected: GPU 0 visible, VRAM > 4GB free

# 3. Check config VRAM limits
cat config.toml | grep -E "replicas|num_temps"
# Expected: Both <= 56

# 4. Verify stubs removed
rg "todo!|unimplemented!" foundation/prct-core/src/world_record_pipeline.rs
# Expected: no matches

# 5. Test run (dry run)
cargo run --features cuda --release -- --config test_config.toml 2>&1 | tee test.log
grep "\[FALLBACK\]" test.log
# Expected: 0 fallbacks
```

---

## Emergency Recovery

### If pipeline crashes mid-run:

1. **Check last log lines**:
   ```bash
   tail -20 pipeline.log
   ```

2. **Look for VRAM errors**:
   ```bash
   dmesg | grep -i "out of memory"
   ```

3. **Reduce VRAM usage**:
   ```toml
   [thermo]
   replicas = 32  # Was 56
   num_temps = 32 # Was 56
   ```

4. **Restart from checkpoint** (if enabled):
   ```bash
   cargo run --features cuda --resume checkpoint_15min.json
   ```

---

## Contact

- **Issues**: GitHub Issues tab
- **Docs**: `docs/FALLBACK_SCENARIOS.md` (full reference)
- **Logs**: Save logs with `2>&1 | tee pipeline.log`

---

**Last Updated**: 2025-11-02
**Version**: 1.0.0
