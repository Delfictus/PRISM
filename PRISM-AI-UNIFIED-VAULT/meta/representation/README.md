# Meta Representation Workspace

Experimental artifacts for semantic plasticity live here. Phase M4 populates this directory with encoder weights, latent space diagnostics, and explainability reports.

## Getting Started

- Run `python3 ../../03-AUTOMATION/master_executor.py --use-sample-metrics --skip-build --skip-tests --skip-benchmarks` to generate a seeded explainability report at `artifacts/mec/M4/explainability_report.md`.
- The report and manifest are derived from the Rust `RepresentationAdapter` and `SemanticDriftDetector` implementations under `src/meta/plasticity/`. Update those modules before re-generating production reports.
- Derive live datasets from telemetry with `python3 pipelines/extract_telemetry.py --input ../../telemetry/semantic_plasticity.jsonl --output dataset.json`; edit the dataset only if you need to override raw metrics.
- To run the full governance gate in one step, execute `../../scripts/run_representation_gate.sh` (uses sample metrics for non-M4 artifacts by default).
- Edit `dataset.json` to seed ontology embeddings and observation trails; these feed both the manifest and explainability table.
- Attach supplementary embeddings or latent diagnostics in this directory under a new `snapshots/` subfolder so governance tooling can link the evidence.
- Compliance treats `drift_status == "warning"` as a warning and `drift_status == "drifted"` as a blocker; investigate anomalies before rerunning the executor.
