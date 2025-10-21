//! Semantic plasticity regression tests (Phase M4).

use std::collections::BTreeMap;

use prism_ai::meta::plasticity::{
    explainability_report, AdapterMode, DriftStatus, RepresentationAdapter, SemanticDriftDetector,
};

fn build_adapter() -> RepresentationAdapter {
    let mut prototypes = BTreeMap::new();
    prototypes.insert("concept://coloring".into(), vec![0.2, 0.5, 0.3, 0.4]);
    prototypes.insert("concept://ontology".into(), vec![0.1, 0.1, 0.1, 0.1]);

    RepresentationAdapter::new("semantic_plasticity_m4", 4, prototypes).unwrap()
}

#[test]
fn adapter_records_history_and_transitions_to_stable_mode() {
    let mut adapter = build_adapter().with_adaptation_rate(0.5).with_history_cap(8);

    for idx in 0..12 {
        let scale = 1.0 + idx as f32 * 0.01;
        let embedding = vec![0.18 * scale, 0.47 * scale, 0.34 * scale, 0.39 * scale];
        let event = adapter.adapt("concept://coloring", &embedding).unwrap();
        assert!(
            event.drift.metrics.cosine_similarity <= 1.0,
            "cosine similarity should be normalized"
        );
    }

    let snapshot = adapter.snapshot();
    assert_eq!(snapshot.mode, AdapterMode::Stable);
    assert_eq!(snapshot.recent_events.len(), 8, "history should respect cap");
    assert_eq!(
        snapshot.tracked_concepts, 2,
        "adapter should retain existing prototypes"
    );
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
    let adapter = build_adapter();
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
}
