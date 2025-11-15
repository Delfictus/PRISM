//! Integration Module
//!
//! Constitution: Phase 3 - Integration Architecture
//!
//! Couples neuromorphic and quantum domains through information-theoretic
//! principles without physical simulation. Implements:
//!
//! 1. **Mutual Information Maximization**: I(X;Y) maximization for coupling
//! 2. **Information Bottleneck**: Compress while preserving task-relevant info
//! 3. **Causal Consistency**: Maintain cause-effect relationships
//! 4. **Phase Synchronization**: Coordinate oscillatory dynamics
//!
//! Mathematical Foundation:
//! ```text
//! L = I(X;Y) - β·I(X;Z)  // Information bottleneck
//! I(X;Y) = H(X) + H(Y) - H(X,Y)  // Mutual information
//! ρ = |⟨e^{iθ}⟩|  // Kuramoto order parameter (phase coherence)
//! ```

pub mod adapters;
pub mod cross_domain_bridge;
pub mod information_channel;
pub mod ports;
#[cfg(feature = "quantum_mlir_support")]
pub mod quantum_mlir_integration;
pub mod synchronization;
pub mod unified_platform;

pub use cross_domain_bridge::{BridgeMetrics, CouplingStrength, CrossDomainBridge, DomainState};

pub use information_channel::{ChannelState, InformationChannel, TransferResult};

pub use synchronization::{CoherenceLevel, PhaseSynchronizer, SynchronizationMetrics};

pub use unified_platform::{
    PerformanceMetrics, PlatformInput, PlatformOutput, ProcessingPhase, UnifiedPlatform,
};

#[cfg(feature = "quantum_mlir_support")]
pub use quantum_mlir_integration::{QuantumGate, QuantumMlirIntegration};

pub use ports::{
    ActiveInferencePort, InformationFlowPort, NeuromorphicPort, QuantumPort, ThermodynamicPort,
};

pub use adapters::{
    ActiveInferenceAdapter, InformationFlowAdapter, NeuromorphicAdapter, QuantumAdapter,
    ThermodynamicAdapter,
};
pub mod multi_modal_reasoner;
