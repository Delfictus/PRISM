# Quick Start: Merged MEC Binaries

## Build All Binaries

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH

# Build all 4 merged binaries
cargo build --release --bin meta-flagsctl
cargo build --release --bin meta-ontologyctl
cargo build --release --bin meta-reflexive-snapshot
cargo build --release --bin federated-sim
```

## Binary Usage

### 1. meta-flagsctl - Feature Flag Management

**Check current status**:
```bash
cargo run --bin meta-flagsctl -- status
cargo run --bin meta-flagsctl -- status --json > flags.json
```

**Enable a feature**:
```bash
cargo run --bin meta-flagsctl -- enable \
  --feature reflexive_controller \
  --actor "admin" \
  --justification "M3 production deployment" \
  --evidence "QA-APPROVED"
```

**Shadow mode (testing)**:
```bash
cargo run --bin meta-flagsctl -- shadow \
  --feature federated_learning \
  --actor "devops" \
  --rationale "Testing M5 integration" \
  --planned "2025-11-15T00:00:00Z"
```

**Gradual rollout**:
```bash
cargo run --bin meta-flagsctl -- gradual \
  --feature reflexive_controller \
  --actor "sre" \
  --current-pct 10 \
  --target-pct 50 \
  --eta "2025-11-30T23:59:59Z" \
  --rationale "Progressive M3 rollout"
```

**Disable a feature**:
```bash
cargo run --bin meta-flagsctl -- disable \
  --feature legacy_module \
  --actor "admin" \
  --rationale "Deprecated in favor of M3"
```

**Generate manifest snapshot**:
```bash
cargo run --bin meta-flagsctl -- snapshot \
  --out artifacts/flag_manifest.json
```

---

### 2. meta-ontologyctl - Ontology Management

**Generate ontology snapshot**:
```bash
cargo run --bin meta-ontologyctl -- snapshot \
  --out artifacts/ontology_snapshot.json
```

**Align a concept** (with explainability):
```bash
cargo run --bin meta-ontologyctl -- align \
  --concept reflexive_controller \
  --explain
```

**View current ontology digest**:
```bash
cargo run --bin meta-ontologyctl -- snapshot | jq '.digest'
```

---

### 3. meta-reflexive-snapshot - Reflexive State Capture

**Capture current reflexive state**:
```bash
cargo run --bin meta-reflexive-snapshot -- \
  --json > reflexive_state.json
```

**View governance mode**:
```bash
cargo run --bin meta-reflexive-snapshot -- --json | jq '.mode'
```

**View 16x16 free-energy lattice**:
```bash
cargo run --bin meta-reflexive-snapshot -- --json | jq '.lattice'
```

**Check alerts**:
```bash
cargo run --bin meta-reflexive-snapshot -- --json | jq '.alerts'
```

**Monitor energy trends**:
```bash
cargo run --bin meta-reflexive-snapshot -- --json | jq '.energy_trend'
```

---

### 4. federated-sim - Federated Learning Simulation

**Run federated simulation**:
```bash
cargo run --bin federated-sim -- \
  --nodes 5 \
  --rounds 100 \
  --output results/federated_run.json
```

**Test multi-node coordination**:
```bash
cargo run --bin federated-sim -- \
  --nodes 10 \
  --rounds 50 \
  --trust-threshold 0.8
```

**View help**:
```bash
cargo run --bin federated-sim -- --help
```

---

## Environment Variables

Set these for production deployment:

```bash
# Telemetry output path
export PRISM_META_TELEMETRY_PATH="telemetry/production_meta.jsonl"

# Selection report path
export PRISM_SELECTION_REPORT_PATH="artifacts/mec/selection_report.json"

# Ontology ledger path
export PRISM_ONTOLOGY_LEDGER_PATH="ledger/ontology.jsonl"

# Vault root for artifacts
export PRISM_VAULT_ROOT="PRISM-AI-UNIFIED-VAULT"
```

---

## Integration Example

**Complete MEC workflow**:

```bash
# 1. Bootstrap ontology
cargo run --bin meta-ontologyctl -- snapshot \
  --out bootstrap/ontology.json

# 2. Enable reflexive controller
cargo run --bin meta-flagsctl -- enable \
  --feature reflexive_controller \
  --actor "bootstrap" \
  --justification "Initial M3 activation"

# 3. Capture initial reflexive state
cargo run --bin meta-reflexive-snapshot -- --json \
  > state/initial_reflexive.json

# 4. Run federated simulation
cargo run --bin federated-sim -- \
  --nodes 3 \
  --rounds 10 \
  --output results/federation_test.json

# 5. Check feature flag status
cargo run --bin meta-flagsctl -- status --json
```

---

## Monitoring Commands

**Watch reflexive state in real-time**:
```bash
watch -n 1 'cargo run --bin meta-reflexive-snapshot -- --json | jq ".mode, .energy_mean, .alerts"'
```

**Monitor feature flags**:
```bash
watch -n 5 'cargo run --bin meta-flagsctl -- status'
```

**Tail telemetry logs**:
```bash
tail -f telemetry/meta_meta.jsonl | jq '.'
```

---

## Verification Tests

**Test all binaries**:
```bash
# 1. Check compilation
cargo check --bin meta-flagsctl \
             --bin meta-ontologyctl \
             --bin meta-reflexive-snapshot \
             --bin federated-sim

# 2. Run help commands
cargo run --bin meta-flagsctl -- --help
cargo run --bin meta-ontologyctl -- --help
cargo run --bin meta-reflexive-snapshot -- --help
cargo run --bin federated-sim -- --help

# 3. Test basic operations
cargo run --bin meta-flagsctl -- status
cargo run --bin meta-ontologyctl -- snapshot
cargo run --bin meta-reflexive-snapshot -- --json > /tmp/test.json
```

---

## Output Locations

| Binary | Output Path | Format |
|--------|-------------|--------|
| meta-flagsctl | `PRISM-AI-UNIFIED-VAULT/artifacts/meta_flags.json` | JSON |
| meta-ontologyctl | `ledger/ontology.jsonl` | JSONL |
| meta-reflexive-snapshot | stdout | JSON |
| federated-sim | stdout or `--output` | JSON |
| telemetry | `telemetry/meta_meta.jsonl` | JSONL |
| selection reports | `PRISM-AI-UNIFIED-VAULT/artifacts/mec/M1/selection_report.json` | JSON |

---

## Common Workflows

### Development Workflow
```bash
# 1. Enable shadow mode for testing
cargo run --bin meta-flagsctl -- shadow \
  --feature new_feature --actor dev --rationale "dev testing"

# 2. Monitor reflexive feedback
cargo run --bin meta-reflexive-snapshot -- --json

# 3. Check ontology alignment
cargo run --bin meta-ontologyctl -- align --concept new_feature
```

### Production Rollout
```bash
# 1. Start with 10% rollout
cargo run --bin meta-flagsctl -- gradual \
  --feature new_module --actor sre \
  --current-pct 0 --target-pct 10 \
  --rationale "Initial production rollout"

# 2. Monitor for issues
watch -n 30 'cargo run --bin meta-reflexive-snapshot -- --json | jq .alerts'

# 3. Increase to 50% if stable
cargo run --bin meta-flagsctl -- gradual \
  --feature new_module --actor sre \
  --current-pct 10 --target-pct 50 \
  --rationale "Stable, increasing rollout"

# 4. Full enable when ready
cargo run --bin meta-flagsctl -- enable \
  --feature new_module --actor sre \
  --justification "Production rollout complete"
```

### Debugging Workflow
```bash
# 1. Capture current state
cargo run --bin meta-reflexive-snapshot -- --json > debug/state.json

# 2. Check feature flags
cargo run --bin meta-flagsctl -- status --json > debug/flags.json

# 3. Export ontology
cargo run --bin meta-ontologyctl -- snapshot --out debug/ontology.json

# 4. Check telemetry
tail -100 telemetry/meta_meta.jsonl > debug/recent_telemetry.jsonl
```

---

## Performance Tips

1. **Use release builds** for production:
   ```bash
   cargo build --release --bin meta-flagsctl
   ./target/release/meta-flagsctl status
   ```

2. **Batch operations** when possible:
   ```bash
   # Good: Single snapshot
   cargo run --release --bin meta-reflexive-snapshot -- --json

   # Avoid: Repeated calls in tight loops
   ```

3. **Cache binary paths**:
   ```bash
   FLAGSCTL="./target/release/meta-flagsctl"
   $FLAGSCTL status
   ```

---

## Troubleshooting

**Binary not found**:
```bash
# Check if built
ls -lh target/release/meta-*

# Rebuild if needed
cargo build --release --bin meta-flagsctl
```

**Permission denied**:
```bash
# Check telemetry directory exists and is writable
mkdir -p telemetry
chmod 755 telemetry
```

**Missing environment variable**:
```bash
# Set defaults
export PRISM_META_TELEMETRY_PATH="${PRISM_META_TELEMETRY_PATH:-telemetry/meta_meta.jsonl}"
export PRISM_VAULT_ROOT="${PRISM_VAULT_ROOT:-PRISM-AI-UNIFIED-VAULT}"
```

---

## Next Steps

1. ✅ Build all binaries in release mode
2. ✅ Set environment variables
3. ✅ Bootstrap ontology with meta-ontologyctl
4. ✅ Initialize feature flags with meta-flagsctl
5. ✅ Verify reflexive state with meta-reflexive-snapshot
6. ✅ Test federated coordination with federated-sim

**Documentation**: See `MEC_UNIFIED_MERGE_COMPLETE.md` for complete details.

---

**Quick Start Date**: October 31, 2025
**Status**: ✅ Ready for Production
