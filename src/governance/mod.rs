//! Governance utilities for determinism proofs and benchmark gating.

pub mod benchmark;
pub mod determinism;

pub use benchmark::{
    BenchmarkArtifact, BenchmarkManifest, BenchmarkResult, GateDecision, PerformanceGate,
    PerformanceMetrics, PerformanceThresholds, Severity, ValidationConfig,
};
pub use determinism::{
    DeterminismAuditTrail, DeterminismProof, DeterminismRecorder, EnvironmentFingerprint,
};
