//! Semantic plasticity regression tests (Phase M4).

use std::collections::{BTreeMap, BTreeSet};

use prism_ai::meta::ontology::ConceptAnchor;
use prism_ai::meta::plasticity::{
    explainability_report, AdapterMode, DriftStatus, RepresentationAdapter, RepresentationManifest,
    SemanticDriftDetector,
};
use serde_json;

fn build_adapter() -> RepresentationAdapter {
    let mut prototypes = BTreeMap::new();
    prototypes.insert("concept://coloring".into(), vec![0.2, 0.5, 0.3, 0.4]);
    prototypes.insert("concept://ontology".into(), vec![0.1, 0.1, 0.1, 0.1]);

    RepresentationAdapter::new("semantic_plasticity_m4", 4, prototypes).unwrap()
}

#[test]
fn adapter_records_history_and_transitions_to_stable_mode() {
    let mut adapter = build_adapter().with_adaptation_rate(0.5).with_history_cap(8);

    let coloring_anchor = sample_anchor("concept://coloring");
    let ontology_anchor = sample_anchor("concept://ontology");
    adapter.register_anchor(&coloring_anchor);
    adapter.register_anchor(&ontology_anchor);

    for idx in 0..12 {
        let scale = 1.0 + idx as f32 * 0.01;
        let embedding = vec![0.18 * scale, 0.47 * scale, 0.34 * scale, 0.39 * scale];
        let event = adapter.adapt("concept://coloring", &embedding).unwrap();
        assert!(
            event.drift.metrics.cosine_similarity <= 1.0,
            "cosine similarity should be normalized"
        );
    }

    // Generate a single update for the ontology concept to populate manifests.
    adapter
        .adapt("concept://ontology", &[0.12, 0.12, 0.12, 0.12])
        .unwrap();

    let snapshot = adapter.snapshot();
    assert_eq!(snapshot.mode, AdapterMode::Stable);
    assert_eq!(snapshot.recent_events.len(), 8, "history should respect cap");
    assert_eq!(
        snapshot.tracked_concepts, 2,
        "adapter should retain existing prototypes"
    );

    let manifest = adapter.manifest();
    let coloring_entry = manifest
        .concepts
        .iter()
        .find(|c| c.concept_id == "concept://coloring")
        .expect("coloring concept present");
    assert_eq!(
        coloring_entry.anchor_hash.as_deref(),
        Some(coloring_anchor.canonical_fingerprint().as_str())
    );
    assert_eq!(coloring_entry.observation_count, 12);
}
}

#[test]
fn drift_detector_flags_drift_when_threshold_exceeded() {
    let detector = SemanticDriftDetector::default();
    let baseline = vec![0.4, 0.4, 0.4, 0.4];
    let candidate = vec![0.9, 0.1, 0.1, 0.1];

    let evaluation = detector.evaluate(&baseline, &candidate).unwrap();
    assert_eq!(evaluation.status, DriftStatus::Drifted);
    assert!(
        evaluation.metrics.cosine_similarity < 0.85,
        "cosine similarity should breach drift threshold"
    );
}

#[test]
fn explainability_report_contains_operational_sections() {
    let mut adapter = build_adapter();
    let coloring_anchor = sample_anchor("concept://coloring");
    adapter.register_anchor(&coloring_anchor);
    adapter
        .adapt("concept://coloring", &[0.19, 0.51, 0.33, 0.41])
        .unwrap();
    let snapshot = adapter.snapshot();
    let report = explainability_report(&snapshot);

    assert!(
        report.contains("# Semantic Plasticity Explainability Report"),
        "report should contain heading"
    );
    assert!(
        report.contains("## Adapter Overview"),
        "overview section missing"
    );
    assert!(
        report.contains("## Recent Adaptation Events"),
        "drift section missing"
    );
    assert!(report.contains("stable") || report.contains("warning"));
}

#[test]
fn manifest_serialization_roundtrip() {
    let mut adapter = build_adapter();
    let coloring_anchor = sample_anchor("concept://coloring");
    adapter.register_anchor(&coloring_anchor);
    adapter
        .adapt("concept://coloring", &[0.25, 0.45, 0.32, 0.41])
        .unwrap();

    let manifest = adapter.manifest();
    let json = serde_json::to_string(&manifest).unwrap();
    let decoded: RepresentationManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.concepts.len(), manifest.concepts.len());
    assert_eq!(decoded.mode, manifest.mode);
}

fn sample_anchor(id: &str) -> ConceptAnchor {
    ConceptAnchor {
        id: id.to_owned(),
        description: format!("Anchor for {id}"),
        attributes: BTreeMap::from([("domain".into(), "meta".into())]),
        related: BTreeSet::from(["meta_generation".into()]),
    }
}
