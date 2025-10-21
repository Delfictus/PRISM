# MEC M3 Artifacts

Phase M3 introduces the reflexive feedback loop that governs exploration
pressure using free-energy lattice snapshots. This directory now contains the
baseline snapshot and associated metadata consumed by CI (`ci-lattice`):

- `lattice_report.json` – serialized `ReflexiveSnapshot` emitted by the master
  executor when sample metrics are enabled. Includes entropy, divergence,
  effective temperature, lattice grid, and SHA-256 fingerprint.

Additional snapshots captured by governed runs should be appended alongside the
baseline file and referenced from `determinism/meta/lattice_manifest.json`.
