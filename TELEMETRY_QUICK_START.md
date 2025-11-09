# HYPER-DETAILED TELEMETRY - QUICK START GUIDE

## TL;DR

Telemetry is **automatically enabled** in all world-record pipeline runs. You now get **100-200+ detailed metrics** instead of 12-14.

---

## What Changed

| Phase | Old Metrics | New Metrics | What You Get |
|-------|-------------|-------------|--------------|
| **Thermodynamic** | 2 (start/end) | **48-64** | Temperature-by-temperature progress with exact tuning recommendations |
| **Quantum** | 0 | **8-12** | Annealing convergence tracking with energy/tunneling stats |
| **Memetic** | 0 | **20-40** | Generation-by-generation evolution with diversity metrics |

**New Feature**: Every metric includes `optimization_guidance` with:
- Status: `on_track`, `excellent`, `need_tuning`, or `critical`
- Specific recommendations (e.g., "Increase steps_per_temp to 20000+")
- Gap to world record (83 colors for DSJC1000.5)
- Estimated final chromatic number

---

## View Telemetry Output

### During Run (Terminal)

The pipeline already prints progress. Look for new detailed checkpoints:

```
[THERMO-GPU] T=5.234: 112 colors, 3 conflicts
[QUANTUM] Step 500/2000: E=-2345.67, temp=3.162
[MEMETIC] Gen 40: best=89, diversity=0.087
```

### After Run (JSONL Files)

Telemetry is written to `telemetry/run_XXXXXX.jsonl`:

```bash
# View all telemetry
jq . telemetry/run_XXXXXX.jsonl

# Extract recommendations only
jq -r 'select(.optimization_guidance.status == "need_tuning" or .optimization_guidance.status == "critical") | "\(.step): \(.optimization_guidance.recommendations[])"' telemetry/run_XXXXXX.jsonl

# Track chromatic progress
jq -r '"\(.step) | \(.chromatic_number) colors | gap=\(.optimization_guidance.gap_to_world_record // "N/A")"' telemetry/run_XXXXXX.jsonl
```

---

## Example Output

### Thermodynamic Temperature Checkpoint

```json
{
  "step": "temp_24/48",
  "chromatic_number": 108,
  "conflicts": 0,
  "parameters": {
    "temperature": 1.234,
    "effectiveness": 0.5,
    "cumulative_improvement": 12
  },
  "optimization_guidance": {
    "status": "on_track",
    "recommendations": ["On track - steady progress"],
    "gap_to_world_record": 25
  }
}
```

### Quantum Annealing Checkpoint

```json
{
  "step": "anneal_step_1000/2000",
  "chromatic_number": 98,
  "parameters": {
    "energy": -2567.89,
    "temperature": 1.000,
    "progress_pct": 50.0
  },
  "optimization_guidance": {
    "status": "on_track",
    "gap_to_world_record": 15
  }
}
```

### Memetic Generation Checkpoint

```json
{
  "step": "generation_60/100",
  "chromatic_number": 89,
  "parameters": {
    "diversity": 0.087,
    "stagnation_count": 12,
    "mutation_rate": 0.15
  },
  "optimization_guidance": {
    "status": "on_track",
    "gap_to_world_record": 6
  }
}
```

---

## Hypertuning Workflow

### 1. Run Pipeline

```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
./run_wr.sh  # Your existing script
```

### 2. Check for Issues

```bash
# Find critical issues
jq -r 'select(.optimization_guidance.status == "critical") | "\(.step): \(.optimization_guidance.recommendations[])"' telemetry/run_latest.jsonl
```

Example output:
```
temp_45/48: CRITICAL: 245 conflicts at temp 0.001 - increase steps_per_temp from 10000 to 20000+
```

### 3. Apply Recommendations

Edit your config file (e.g., `foundation/prct-core/configs/wr_sweep_D.v1.1.toml`):

```toml
[thermo]
steps_per_temp = 20000  # Was 10000 (from recommendation)
```

### 4. Re-run and Compare

```bash
./run_wr.sh

# Compare final chromatic numbers
jq -r 'select(.step | contains("final")) | .chromatic_number' telemetry/run_*.jsonl
```

---

## Common Recommendations

### Thermodynamic Phase

**Issue**: High conflicts at final temperature

**Recommendation**:
```
"CRITICAL: 245 conflicts at temp 0.001 - increase steps_per_temp from 10000 to 20000+"
```

**Fix**:
```toml
[thermo]
steps_per_temp = 20000
```

---

**Issue**: Limited chromatic improvement

**Recommendation**:
```
"Limited progress: 118 colors (started at 120) - increase num_temps to 64+"
"Or increase t_max from 10.0 to 15.0"
```

**Fix**:
```toml
[thermo]
num_temps = 64     # Was 48
t_max = 15.0       # Was 10.0
```

---

### Quantum Phase

**Issue**: Energy stagnation

**Recommendation**:
```
"Energy stagnant at -1234.56 - consider increasing num_steps from 2000 to 4000"
```

**Fix**: Edit `foundation/prct-core/src/quantum_coloring.rs` line 713:
```rust
let num_steps = 4000;  // Was 2000
```

(Or expose as config parameter)

---

### Memetic Phase

**Issue**: Diversity collapse

**Recommendation**:
```
"CRITICAL: Diversity collapsed to 0.0032 - increase mutation_rate from 0.15 to 0.23"
```

**Fix**:
```toml
[memetic]
mutation_rate = 0.23   # Was 0.15
population_size = 64   # Was 48 (for more diversity)
```

---

## Real-Time Monitoring

Watch chromatic number in real-time:

```bash
tail -f telemetry/run_latest.jsonl | jq -r 'select(.chromatic_number != null) | "\(.step) | \(.chromatic_number) colors | gap=\(.optimization_guidance.gap_to_world_record // "N/A")"'
```

Output:
```
temp_1/48 | 562 colors | gap=479
temp_10/48 | 234 colors | gap=151
temp_20/48 | 145 colors | gap=62
temp_48/48 | 108 colors | gap=25
anneal_step_500/2000 | 102 colors | gap=19
generation_40/100 | 89 colors | gap=6
```

---

## Goal: 83 Colors on DSJC1000.5

**Track your progress**:

```bash
# Show gap to world record over time
jq -r 'select(.optimization_guidance.gap_to_world_record != null) | "\(.step): gap=\(.optimization_guidance.gap_to_world_record)"' telemetry/run_latest.jsonl
```

**When gap reaches 0**, you've achieved the world record!

---

## Troubleshooting

### No telemetry files?

Check that the pipeline created the telemetry directory:
```bash
ls -la telemetry/
```

### Empty telemetry?

Ensure you're running with `--features cuda` and GPU phases are enabled in config:
```toml
use_thermodynamic_equilibration = true
use_quantum_classical_hybrid = true
use_memetic_coloring = true
```

### Too much output?

Filter by status:
```bash
# Only show issues
jq -r 'select(.optimization_guidance.status == "critical" or .optimization_guidance.status == "need_tuning")' telemetry/run_latest.jsonl
```

---

## What's Next?

The telemetry system is complete and operational. Future enhancements could include:

1. **Live dashboard** - Web UI showing real-time progress
2. **Auto-tuning** - ML model that applies recommendations automatically
3. **Historical comparison** - Compare runs to identify best configs
4. **Alert system** - Email/Slack notifications when critical issues detected

For now, the telemetry provides all the data you need to manually tune toward 83 colors.

---

**Full Documentation**: See `HYPER_TELEMETRY_IMPLEMENTATION_REPORT.md`

**Build Status**: ✅ PASSING (0 errors)

**Ready to use**: Yes! Just run your pipeline as normal.
