# Phase M5 · Federated Readiness Artifacts

This directory houses the governance evidence generated while Phase M5 is in
flight. The key deliverables are:

- `federated_plan.md`: execution roadmap and protocol design summary.
- `simulations/`: synthetic outputs that demonstrate hybrid orchestration.
- `ledger/`: Merkle anchors, signatures, and federation manifests.
- `scenarios/`: JSON descriptors that feed the federated simulator CLI.

Artifacts created here are referenced by the compliance validator and must be
kept under version control to satisfy governance checkpoints.

## Generating Synthetic Artifacts

Use the `federated-sim` helper to refresh simulation outputs:

```bash
cargo run --bin federated_sim -- \
  --scenario PRISM-AI-UNIFIED-VAULT/artifacts/mec/M5/scenarios/baseline.json \
  --epochs 5 \
  --clean
```

By default the tool writes baseline artifacts to `simulations/epoch_summary.json`
and `ledger/epoch_XXX.json`, satisfying the compliance validator. Specify
`--label <name>` to generate additional scenario outputs side-by-side (e.g.
`--label edge_failover` writes to `simulations/epoch_summary_edge_failover.json`
and `ledger/edge_failover/`). The `--output-dir` flag can redirect all artifacts
to a different base directory.

The `--clean` flag clears existing outputs in the resolved base directory before
generating new data. Avoid combining `--clean` with alternate labels unless you
intend to wipe the baseline outputs as well.
