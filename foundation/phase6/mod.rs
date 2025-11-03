//! Phase 6: Adaptive Problem-Space Modeling
//!
//! Constitutional Amendment implementing world-record performance through:
//! 1. Topological Data Analysis (TDA) - Mathematical structure discovery
//! 2. Graph Neural Networks (GNN) - Learned solution priors
//! 3. Predictive Neuromorphic Computing - Active inference with dendritic processing
//!
//! These form a meta-learning feedback loop that dynamically modulates the quantum
//! Hamiltonian based on problem structure, enabling escape from local minima.

pub mod tda;
pub mod predictive_neuro;
pub mod meta_learning;
pub mod integration;

// Core exports
pub use tda::{TdaAdapter, TdaPort, PersistenceBarcode};
pub use predictive_neuro::{PredictiveNeuromorphic, PredictionError, DendriticModel};
pub use meta_learning::{MetaLearningCoordinator, ModulatedHamiltonian};
pub use integration::{Phase6Integration, AdaptiveSolver, AdaptiveSolution};

// Re-export GNN from CMA module (already implemented)
pub use crate::cma::neural::gnn_integration::E3EquivariantGNN;pub mod gpu_tda;
pub use gpu_tda::GpuTDA;
