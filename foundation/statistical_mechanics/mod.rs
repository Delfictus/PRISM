//! Statistical Mechanics Module
//!
//! Constitution: Phase 1, Task 1.3 - Thermodynamically Consistent Oscillator Network
//!
//! This module implements a network of coupled oscillators that rigorously respects
//! the laws of statistical mechanics:
//!
//! 1. **Second Law of Thermodynamics**: dS/dt ≥ 0 (entropy never decreases)
//! 2. **Fluctuation-Dissipation Theorem**: <F(t)F(t')> = 2γk_BT δ(t-t')
//! 3. **Boltzmann Distribution**: P(E) ∝ exp(-E/k_BT) at equilibrium
//! 4. **Information-Theoretic Coupling**: Uses transfer entropy to gate interactions
//!
//! Mathematical Foundation:
//! ```text
//! dθ_i/dt = ω_i + Σ_j C_ij sin(θ_j - θ_i) - γ ∂S/∂θ_i + √(2γk_BT) η(t)
//! ```
//!
//! where:
//! - θ_i: Phase of oscillator i
//! - ω_i: Natural frequency of oscillator i
//! - C_ij: Coupling matrix (information-gated)
//! - γ: Damping coefficient
//! - S: System entropy
//! - k_B: Boltzmann constant
//! - T: Temperature
//! - η(t): White noise (zero mean, unit variance)

pub mod gpu_bindings;
pub mod gpu_integration;
pub mod thermodynamic_network;

// GPU-accelerated thermodynamic network
#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "cuda")]
pub use gpu::ThermodynamicGpu;

pub use thermodynamic_network::{
    EvolutionResult, NetworkConfig, ThermodynamicMetrics, ThermodynamicNetwork, ThermodynamicState,
};

pub use gpu_bindings::GpuThermodynamicNetwork;
pub use gpu_integration::ThermodynamicNetworkGpuExt;
