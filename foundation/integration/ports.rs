//! Port Definitions (Hexagonal Architecture)
//!
//! Domain interfaces that adapters must implement.
//! Platform depends on these ports, not concrete implementations.

use anyhow::Result;
use ndarray::{Array1, Array2};
use shared_types::{KuramotoState, PhaseField};

use crate::statistical_mechanics::ThermodynamicState;

/// Neuromorphic spike encoding port
pub trait NeuromorphicPort: Send + Sync {
    /// Encode input data as spike train
    fn encode_spikes(&mut self, input: &Array1<f64>) -> Result<Array1<bool>>;

    /// Get spike history for temporal processing
    fn get_spike_history(&self) -> &[Array1<bool>];
}

/// Information flow analysis port
pub trait InformationFlowPort: Send + Sync {
    /// Compute transfer entropy between spike trains
    fn compute_transfer_entropy(
        &mut self,
        source: &Array1<bool>,
        target: &Array1<bool>,
    ) -> Result<f64>;

    /// Build full coupling matrix from spike history
    fn compute_coupling_matrix(&mut self, spike_history: &[Array1<bool>]) -> Result<Array2<f64>>;
}

/// Thermodynamic evolution port
pub trait ThermodynamicPort: Send + Sync {
    /// Evolve thermodynamic state under coupling
    fn evolve(&mut self, coupling: &Array2<f64>, dt: f64) -> Result<ThermodynamicState>;

    /// Get current entropy production rate
    fn entropy_production(&self) -> f64;

    /// Get Kuramoto synchronization state
    fn get_kuramoto_state(&self) -> Option<KuramotoState>;
}

/// Quantum processing port
pub trait QuantumPort: Send + Sync {
    /// Apply quantum processing to thermodynamic state
    fn quantum_process(&mut self, thermo_state: &ThermodynamicState) -> Result<Array1<f64>>;

    /// Get quantum observables
    fn get_observables(&self) -> Array1<f64>;

    /// Get phase field state
    fn get_phase_field(&self) -> Option<PhaseField>;
}

/// Active inference port
pub trait ActiveInferencePort: Send + Sync {
    /// Update beliefs and compute free energy
    fn infer(&mut self, observations: &Array1<f64>, quantum_obs: &Array1<f64>) -> Result<f64>;

    /// Select control action
    fn select_action(&mut self, targets: &Array1<f64>) -> Result<Array1<f64>>;
}
