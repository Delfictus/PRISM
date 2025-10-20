// Active Inference Module
// Constitution: Phase 2 - Active Inference Implementation
//
// Implements hierarchical generative models for adaptive optics
// based on variational free energy minimization.

pub mod generative_model;
pub mod hierarchical_model;
pub mod observation_model;
pub mod transition_model;
pub mod variational_inference;
pub mod recognition_model;
pub mod policy_selection;
pub mod controller;
pub mod gpu_optimization;

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
pub use hierarchical_model::{HierarchicalModel, StateSpaceLevel, GaussianBelief, GeneralizedCoordinates};
pub use observation_model::{ObservationModel, MeasurementPattern};
pub use transition_model::{TransitionModel, ControlAction};
pub use variational_inference::{VariationalInference, FreeEnergyComponents};
pub use policy_selection::{PolicySelector, ActiveInferenceController, SensingStrategy, Policy};
pub use gpu_optimization::ActiveInferenceGpuExt;
