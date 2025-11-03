pub mod adapters;
pub mod drift;

pub use adapters::{
    AdaptationEvent, AdaptationMetadata, AdapterError, AdapterMode, ConceptManifest,
    RepresentationAdapter, RepresentationDataset, RepresentationManifest, RepresentationSnapshot,
};
pub use drift::{DriftError, DriftEvaluation, DriftMetrics, DriftStatus, SemanticDriftDetector};

/// Render the explainability report for the provided adapter snapshot.
pub fn explainability_report(snapshot: &RepresentationSnapshot) -> String {
    snapshot.render_markdown()
}

/// Placeholder module root for semantic plasticity work.
pub fn initialized() -> bool {
    true
}
