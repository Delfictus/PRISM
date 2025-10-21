# Meta Representation Workspace

Experimental artifacts for semantic plasticity live here. Phase M4 populates this directory with encoder weights, latent space diagnostics, and explainability reports.

## Getting Started

- Run `python3 ../../03-AUTOMATION/master_executor.py --use-sample-metrics --skip-build --skip-tests --skip-benchmarks` to generate a seeded explainability report at `artifacts/mec/M4/explainability_report.md`.
- The report is derived from the Rust `RepresentationAdapter` and `SemanticDriftDetector` implementations under `src/meta/plasticity/`. Update those modules before re-generating production reports.
- Attach supplementary embeddings or latent diagnostics in this directory under a new `snapshots/` subfolder so governance tooling can link the evidence.
