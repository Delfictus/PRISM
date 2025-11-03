# PRISM Result Parsing Guide

## Critical Context: The "83 Colors" Bug

We discovered a parsing bug where the seed probe script incorrectly extracted "83 colors" from:

```
[DSATUR] Max colors: 83 (upper bound)
```

**This is the TARGET/UPPER BOUND, not the actual result!** The actual best result was 113 colors.

This document explains how to correctly parse PRISM pipeline results and avoid similar mistakes.

---

## Parser-Safe Output Format

### 1. Final Result (Plain Text)

**FORMAT:** `FINAL RESULT: colors=X conflicts=Y time=Z.ZZs`

**EXAMPLE:**
```
╔═══════════════════════════════════════════════════════════╗
║                     FINAL RESULT                          ║
╚═══════════════════════════════════════════════════════════╝
FINAL RESULT: colors=95 conflicts=0 time=4532.45s
```

**PARSING:**
```bash
# Bash example
colors=$(grep "^FINAL RESULT: colors=" log.txt | grep -oE 'colors=[0-9]+' | grep -oE '[0-9]+' | head -1)

# Python example
import re
match = re.search(r'^FINAL RESULT: colors=(\d+)', log, re.MULTILINE)
if match:
    colors = int(match.group(1))
```

---

### 2. JSON Telemetry (Machine-Parseable)

**FORMAT:** `{"event":"final_result","colors":X,"conflicts":Y,"time_s":Z.ZZ,...}`

**EXAMPLE:**
```json
{"event":"final_result","colors":95,"conflicts":0,"time_s":4532.452,"quality_score":0.987654,"graph":{"vertices":1000,"edges":249826,"density":0.500000}}
```

**PARSING:**
```bash
# Bash/jq example
colors=$(grep '"event":"final_result"' log.txt | jq -r '.colors')

# Python example
import json
for line in log:
    if '"event":"final_result"' in line:
        data = json.loads(line)
        colors = data['colors']
        break
```

---

## Lines That Should NEVER Be Parsed

These are intermediate results, targets, or reference values:

### DO NOT PARSE:

1. **Target/Goal Lines:**
   ```
   [WR-PIPELINE] Target: 83 colors (World Record - GOAL)
   ```
   This is the TARGET we're trying to beat, not the result!

2. **Reference Lines:**
   ```
   Best known: 83 colors (world record)
   ```
   This is the world record reference, not our result!

3. **Upper Bound Lines:**
   ```
   [DSATUR] Max colors: 83 (upper bound)
   ```
   This is a theoretical upper bound, not the actual coloring!

4. **Intermediate Phase Results:**
   ```
   [PHASE 1] ✅ TE-guided coloring: 145 colors
   [PHASE 2] 🎯 Thermodynamic improvement: 130 → 120 colors
   [PHASE 3] 🎯 Quantum-Classical breakthrough: 120 → 105 colors
   [PHASE 4] 🎯 Memetic+ADP improved: 105 → 95 colors
   [PHASE 5] 🎯 Ensemble consensus: 95 → 95 colors
   ```
   These are progress updates, not final results!

5. **Summary Statistics:**
   ```
   🏆 Summary: 95 colors achieved
   ```
   This appears BEFORE the parser-safe result line.

---

## Safe Parsing Strategy

### Sequential Approach (Recommended)

1. **First:** Look for `FINAL RESULT: colors=X` (plain text, unambiguous)
2. **Fallback:** Look for `{"event":"final_result","colors":X}` (JSON)
3. **If neither found:** Return sentinel value (e.g., 999) or error

### Example Implementation (Bash)

```bash
parse_colors() {
    local log_file="$1"

    # ONLY look for "FINAL RESULT: colors=X" format
    local colors=$(grep "^FINAL RESULT: colors=" "$log_file" 2>/dev/null | \
                   grep -oE 'colors=[0-9]+' | \
                   grep -oE '[0-9]+' | head -1)

    # Fallback: JSON telemetry
    if [ -z "$colors" ]; then
        colors=$(grep '"event":"final_result"' "$log_file" 2>/dev/null | \
                 grep -oE '"colors":[0-9]+' | \
                 grep -oE '[0-9]+' | head -1)
    fi

    # Return sentinel if not found
    echo "${colors:-999}"
}
```

### Example Implementation (Python)

```python
import re
import json

def parse_colors(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()

    # First: Look for plain text FINAL RESULT
    match = re.search(r'^FINAL RESULT: colors=(\d+)', log_content, re.MULTILINE)
    if match:
        return int(match.group(1))

    # Fallback: Look for JSON telemetry
    for line in log_content.splitlines():
        if '"event":"final_result"' in line:
            try:
                data = json.loads(line)
                return data['colors']
            except (json.JSONDecodeError, KeyError):
                pass

    # Not found
    return 999  # Sentinel value
```

---

## JSON Telemetry Events Reference

### Per-Phase Events

```json
{"event":"phase_start","phase":"1","name":"transfer_entropy"}
{"event":"phase_end","phase":"1","name":"transfer_entropy","time_s":123.456,"colors":145}
```

Phases:
- `0A`: Geodesic features
- `0B`: Reservoir conflict prediction
- `1`: Transfer entropy
- `2`: Thermodynamic equilibration
- `3`: Quantum-classical hybrid
- `4`: Memetic algorithm
- `5`: Ensemble consensus

### Quantum Retry Events

```json
{"event":"quantum_retry","attempt":1,"max_attempts":5,"target_colors":83}
{"event":"quantum_success","attempt":2,"colors":95,"conflicts":0}
{"event":"quantum_failed","attempt":1,"error":"Failed to find valid coloring"}
```

### Final Result Event

```json
{
  "event": "final_result",
  "colors": 95,
  "conflicts": 0,
  "time_s": 4532.452,
  "quality_score": 0.987654,
  "graph": {
    "vertices": 1000,
    "edges": 249826,
    "density": 0.500000
  }
}
```

---

## Common Parsing Mistakes

### Mistake 1: Parsing the Target

**WRONG:**
```bash
# This captures the GOAL, not the result!
grep "Target: [0-9]+ colors" log.txt
```

**RIGHT:**
```bash
# Only parse FINAL RESULT
grep "^FINAL RESULT: colors=" log.txt
```

---

### Mistake 2: Parsing Intermediate Results

**WRONG:**
```bash
# This might capture an intermediate phase result
grep "colors" log.txt | tail -1
```

**RIGHT:**
```bash
# Be explicit about what you're looking for
grep "^FINAL RESULT: colors=" log.txt
```

---

### Mistake 3: Ambiguous Patterns

**WRONG:**
```bash
# This will match MANY lines (target, phases, summary, etc.)
grep "[0-9]+ colors" log.txt | head -1
```

**RIGHT:**
```bash
# Use the exact, unambiguous format
grep "^FINAL RESULT: colors=" log.txt
```

---

### Mistake 4: No Fallback

**WRONG:**
```bash
# If the line doesn't exist, this returns empty
colors=$(grep "FINAL RESULT" log.txt | awk '{print $3}')
```

**RIGHT:**
```bash
# Always have a fallback and sentinel value
colors=$(grep "^FINAL RESULT: colors=" log.txt | grep -oE '[0-9]+' | head -1)
echo "${colors:-999}"  # Sentinel if not found
```

---

## Testing Your Parser

### Test Case 1: Complete Run

```bash
# Create a test log with both intermediate and final results
cat > test.log <<EOF
[WR-PIPELINE] Target: 83 colors (World Record - GOAL)
[PHASE 1] ✅ TE-guided coloring: 145 colors
[PHASE 2] 🎯 Thermodynamic improvement: 130 → 120 colors
FINAL RESULT: colors=95 conflicts=0 time=4532.45s
{"event":"final_result","colors":95,"conflicts":0,"time_s":4532.452}
EOF

# Your parser should return 95, not 83 or 145
parse_colors test.log
# Expected: 95
```

### Test Case 2: No Final Result (Timeout/Crash)

```bash
cat > test2.log <<EOF
[WR-PIPELINE] Target: 83 colors (World Record - GOAL)
[PHASE 1] ✅ TE-guided coloring: 145 colors
# Pipeline crashed before completion
EOF

# Your parser should return sentinel value
parse_colors test2.log
# Expected: 999
```

---

## Verification Checklist

Before deploying your parser:

- [ ] Does it correctly extract 95 from `FINAL RESULT: colors=95`?
- [ ] Does it ignore `Target: 83 colors`?
- [ ] Does it ignore `[PHASE X] ... colors`?
- [ ] Does it fall back to JSON telemetry if plain text not found?
- [ ] Does it return a sentinel value (999) if neither format found?
- [ ] Have you tested with real logs from PRISM runs?

---

## Summary

**ALWAYS PARSE:**
- `FINAL RESULT: colors=X` (plain text)
- `{"event":"final_result","colors":X}` (JSON)

**NEVER PARSE:**
- Target/goal lines
- Upper bound lines
- Intermediate phase results
- Summary statistics

**ALWAYS:**
- Use explicit patterns (e.g., `^FINAL RESULT:`)
- Have a fallback mechanism
- Return sentinel values on failure
- Test with real logs

---

## Contact

For questions or issues with result parsing, see:
- `/tools/run_wr_seed_probe.sh` - Reference implementation
- `/foundation/prct-core/src/world_record_pipeline.rs` - Output generation

---

**Document Version:** 1.0
**Last Updated:** 2025-11-02
**Related Bug:** "83 colors" parsing error in seed probe v0.1
