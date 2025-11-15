// Active Inference Module
// Constitution: Phase 2 - Active Inference Implementation
//
// Implements hierarchical generative models for adaptive optics
// based on variational free energy minimization.

pub mod controller;
pub mod generative_model;
pub mod gpu_optimization;
pub mod hierarchical_model;
pub mod observation_model;
pub mod policy_selection;
pub mod recognition_model;
pub mod transition_model;
pub mod variational_inference;

#[cfg(feature = "cuda")]
pub mod gpu_inference;

// GPU-accelerated variational inference
#[cfg(feature = "cuda")]
pub mod gpu;

// GPU-accelerated policy evaluation
#[cfg(feature = "cuda")]
pub mod gpu_policy_eval;

#[cfg(feature = "cuda")]
pub use gpu::ActiveInferenceGpu;

#[cfg(feature = "cuda")]
pub use gpu_policy_eval::GpuPolicyEvaluator;

pub use generative_model::{GenerativeModel, PerformanceMetrics};
pub use gpu_optimization::ActiveInferenceGpuExt;
pub use hierarchical_model::{
    GaussianBelief, GeneralizedCoordinates, HierarchicalModel, StateSpaceLevel,
};
pub use observation_model::{MeasurementPattern, ObservationModel};
pub use policy_selection::{ActiveInferenceController, Policy, PolicySelector, SensingStrategy};
pub use transition_model::{ControlAction, TransitionModel};
pub use variational_inference::{FreeEnergyComponents, VariationalInference};
