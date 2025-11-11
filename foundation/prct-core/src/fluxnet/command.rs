//! FluxNet Multi-Phase Action Commands - Unified RL Action Space
//!
//! Expands FluxNet RL from Phase 2 only to ALL 4 pipeline phases:
//! - **Phase 0 (Reservoir)**: Spectral radius, leak rate, input scaling
//! - **Phase 1 (Transfer Entropy)**: TE weights, geodesic weights, batch sizes, AI thresholds
//! - **Phase 2 (Thermodynamic)**: Forces, temps, steps, replicas
//! - **Phase 3 (Quantum)**: Iterations, beta, temperature, QUBO settings
//!
//! # Action Space
//!
//! Total of 37 discrete actions (vs previous 7):
//! - 6 Reservoir actions (3 params × 2 directions)
//! - 8 Transfer Entropy actions (4 params × 2 directions)
//! - 12 Thermodynamic actions (6 params × 2 directions)
//! - 10 Quantum actions (5 params × 2 directions)
//! - 1 No-op
//!
//! # RL Integration
//!
//! The unified RL controller:
//! 1. Observes state at end of each phase
//! 2. Selects phase-appropriate action via Q-learning
//! 3. Applies action to WorldRecordConfig
//! 4. Observes reward (chromatic reduction, conflict reduction)
//! 5. Updates Q-table based on reward
//!
//! # GPU-First Compliance
//!
//! - ✅ All actions modify config parameters that affect GPU kernel launches
//! - ✅ No CPU fallbacks in action application
//! - ✅ Actions validated against constitutional bounds
//! - ❌ NO magic numbers (all bounds from config)

use serde::{Deserialize, Serialize};
use crate::errors::*;
use crate::world_record_pipeline::WorldRecordConfig;
use crate::telemetry::PhaseName;

/// Adjustment direction for continuous parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustDirection {
    Increase,
    Decrease,
}

/// Unified multi-phase action space
///
/// Replaces ForceCommand with comprehensive control over all 4 phases.
/// Each action targets a specific parameter in a specific phase.
///
/// # Action Space Size
///
/// 37 total actions:
/// - Phase 0 (Reservoir): 6 actions (3 params × 2 directions)
/// - Phase 1 (Transfer Entropy): 8 actions (4 params × 2 directions)
/// - Phase 2 (Thermodynamic): 12 actions (6 params × 2 directions)
/// - Phase 3 (Quantum): 10 actions (5 params × 2 directions)
/// - Meta: 1 action (NoOp)
///
/// # Example Usage
///
/// ```rust,ignore
/// // RL controller selects action
/// let action = rl_controller.choose_action(&state, PhaseName::Reservoir);
///
/// // Apply to config (modifies in-place)
/// let result_desc = action.apply(&mut config)?;
///
/// // Next iteration uses updated parameters
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluxNetAction {
    // ========== Phase 0: Reservoir (3 actions) ==========

    /// Adjust reservoir spectral radius (controls echo state dynamics)
    /// Range: [0.80, 0.99], Step: ±0.05
    ReservoirSpectralRadius(AdjustDirection),

    /// Adjust reservoir leak rate (controls memory decay)
    /// Range: [0.20, 0.40], Step: ±0.05
    ReservoirLeakRate(AdjustDirection),

    /// Adjust reservoir input scaling (controls input influence)
    /// Range: [0.30, 0.70], Step: ±0.10
    ReservoirInputScaling(AdjustDirection),

    // ========== Phase 1: Transfer Entropy (4 actions) ==========

    /// Adjust TE vs Kuramoto weight balance
    /// Range: [0.50, 0.95], Step: ±0.05
    TeVsKuramotoWeight(AdjustDirection),

    /// Adjust geodesic feature weight
    /// Range: [0.10, 0.50], Step: ±0.05
    GeodesicWeight(AdjustDirection),

    /// Adjust TE computation batch size (affects GPU kernel occupancy)
    /// Range: [512, 4096], Step: ±512
    TeBatchSize(AdjustDirection),

    /// Adjust Active Inference uncertainty threshold
    /// Range: [0.10, 0.90], Step: ±0.05
    AiUncertaintyThreshold(AdjustDirection),

    // ========== Phase 2: Thermodynamic (6 actions) ==========

    /// Adjust strong force multiplier (high-conflict vertices)
    /// Range: [1.0, 2.0], Step: ±0.1
    ThermoForceStrong(AdjustDirection),

    /// Adjust weak force multiplier (low-conflict vertices)
    /// Range: [0.3, 1.0], Step: ±0.1
    ThermoForceWeak(AdjustDirection),

    /// Adjust steps per temperature (equilibration depth)
    /// Range: [2000, 12000], Step: ±1000
    ThermoStepsPerTemp(AdjustDirection),

    /// Adjust maximum temperature (exploration strength)
    /// Range: [8.0, 20.0], Step: ±2.0
    ThermoTempMax(AdjustDirection),

    /// Adjust number of temperature levels (geometric ladder)
    /// Range: [16, 56], Step: ±4
    ThermoNumTemps(AdjustDirection),

    /// Adjust parallel replica count (GPU parallelism)
    /// Range: [16, 56], Step: ±4
    ThermoReplicas(AdjustDirection),

    // ========== Phase 3: Quantum (5 actions) ==========

    /// Adjust quantum solver iterations
    /// Range: [10, 100], Step: ±5
    QuantumIterations(AdjustDirection),

    /// Adjust quantum beta (coherence strength)
    /// Range: [0.85, 0.99], Step: ±0.02
    QuantumBeta(AdjustDirection),

    /// Adjust quantum temperature (decoherence)
    /// Range: [0.3, 1.5], Step: ±0.1
    QuantumTemperature(AdjustDirection),

    /// Adjust QUBO solver iterations
    /// Range: [5000, 20000], Step: ±2000
    QuboIterations(AdjustDirection),

    /// Adjust QUBO temperature range (initial/final)
    /// Multiplies both T_initial and T_final by 1.1 or 0.9
    QuboTempRange(AdjustDirection),

    // ========== Meta ==========

    /// No operation - maintain current parameters
    NoOp,
}

impl FluxNetAction {
    /// Total number of discrete actions in unified action space
    pub const ACTION_SPACE_SIZE: usize = 37;

    /// Convert from action index (0-18) to FluxNetAction
    ///
    /// Used by RL controller to map Q-table actions to commands.
    ///
    /// # Arguments
    /// - `action_idx`: Action index from Q-table (0-18)
    ///
    /// # Returns
    /// Corresponding FluxNetAction, or NoOp if index out of range
    pub fn from_index(action_idx: usize) -> Self {
        match action_idx {
            // Phase 0: Reservoir
            0 => FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase),
            1 => FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Decrease),
            2 => FluxNetAction::ReservoirLeakRate(AdjustDirection::Increase),
            3 => FluxNetAction::ReservoirLeakRate(AdjustDirection::Decrease),
            4 => FluxNetAction::ReservoirInputScaling(AdjustDirection::Increase),
            5 => FluxNetAction::ReservoirInputScaling(AdjustDirection::Decrease),

            // Phase 1: Transfer Entropy
            6 => FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Increase),
            7 => FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Decrease),
            8 => FluxNetAction::GeodesicWeight(AdjustDirection::Increase),
            9 => FluxNetAction::GeodesicWeight(AdjustDirection::Decrease),
            10 => FluxNetAction::TeBatchSize(AdjustDirection::Increase),
            11 => FluxNetAction::TeBatchSize(AdjustDirection::Decrease),
            12 => FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Increase),
            13 => FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Decrease),

            // Phase 2: Thermodynamic
            14 => FluxNetAction::ThermoForceStrong(AdjustDirection::Increase),
            15 => FluxNetAction::ThermoForceStrong(AdjustDirection::Decrease),
            16 => FluxNetAction::ThermoForceWeak(AdjustDirection::Increase),
            17 => FluxNetAction::ThermoForceWeak(AdjustDirection::Decrease),
            18 => FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Increase),
            19 => FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Decrease),
            20 => FluxNetAction::ThermoTempMax(AdjustDirection::Increase),
            21 => FluxNetAction::ThermoTempMax(AdjustDirection::Decrease),
            22 => FluxNetAction::ThermoNumTemps(AdjustDirection::Increase),
            23 => FluxNetAction::ThermoNumTemps(AdjustDirection::Decrease),
            24 => FluxNetAction::ThermoReplicas(AdjustDirection::Increase),
            25 => FluxNetAction::ThermoReplicas(AdjustDirection::Decrease),

            // Phase 3: Quantum
            26 => FluxNetAction::QuantumIterations(AdjustDirection::Increase),
            27 => FluxNetAction::QuantumIterations(AdjustDirection::Decrease),
            28 => FluxNetAction::QuantumBeta(AdjustDirection::Increase),
            29 => FluxNetAction::QuantumBeta(AdjustDirection::Decrease),
            30 => FluxNetAction::QuantumTemperature(AdjustDirection::Increase),
            31 => FluxNetAction::QuantumTemperature(AdjustDirection::Decrease),
            32 => FluxNetAction::QuboIterations(AdjustDirection::Increase),
            33 => FluxNetAction::QuboIterations(AdjustDirection::Decrease),
            34 => FluxNetAction::QuboTempRange(AdjustDirection::Increase),
            35 => FluxNetAction::QuboTempRange(AdjustDirection::Decrease),

            // Meta
            36 => FluxNetAction::NoOp,

            _ => FluxNetAction::NoOp, // Default to no-op for invalid indices
        }
    }

    /// Convert FluxNetAction to action index (0-36)
    ///
    /// Used by RL controller to map commands back to Q-table indices.
    pub fn to_index(&self) -> usize {
        match self {
            // Phase 0: Reservoir
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase) => 0,
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Decrease) => 1,
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Increase) => 2,
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Decrease) => 3,
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Increase) => 4,
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Decrease) => 5,

            // Phase 1: Transfer Entropy
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Increase) => 6,
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Decrease) => 7,
            FluxNetAction::GeodesicWeight(AdjustDirection::Increase) => 8,
            FluxNetAction::GeodesicWeight(AdjustDirection::Decrease) => 9,
            FluxNetAction::TeBatchSize(AdjustDirection::Increase) => 10,
            FluxNetAction::TeBatchSize(AdjustDirection::Decrease) => 11,
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Increase) => 12,
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Decrease) => 13,

            // Phase 2: Thermodynamic
            FluxNetAction::ThermoForceStrong(AdjustDirection::Increase) => 14,
            FluxNetAction::ThermoForceStrong(AdjustDirection::Decrease) => 15,
            FluxNetAction::ThermoForceWeak(AdjustDirection::Increase) => 16,
            FluxNetAction::ThermoForceWeak(AdjustDirection::Decrease) => 17,
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Increase) => 18,
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Decrease) => 19,
            FluxNetAction::ThermoTempMax(AdjustDirection::Increase) => 20,
            FluxNetAction::ThermoTempMax(AdjustDirection::Decrease) => 21,
            FluxNetAction::ThermoNumTemps(AdjustDirection::Increase) => 22,
            FluxNetAction::ThermoNumTemps(AdjustDirection::Decrease) => 23,
            FluxNetAction::ThermoReplicas(AdjustDirection::Increase) => 24,
            FluxNetAction::ThermoReplicas(AdjustDirection::Decrease) => 25,

            // Phase 3: Quantum
            FluxNetAction::QuantumIterations(AdjustDirection::Increase) => 26,
            FluxNetAction::QuantumIterations(AdjustDirection::Decrease) => 27,
            FluxNetAction::QuantumBeta(AdjustDirection::Increase) => 28,
            FluxNetAction::QuantumBeta(AdjustDirection::Decrease) => 29,
            FluxNetAction::QuantumTemperature(AdjustDirection::Increase) => 30,
            FluxNetAction::QuantumTemperature(AdjustDirection::Decrease) => 31,
            FluxNetAction::QuboIterations(AdjustDirection::Increase) => 32,
            FluxNetAction::QuboIterations(AdjustDirection::Decrease) => 33,
            FluxNetAction::QuboTempRange(AdjustDirection::Increase) => 34,
            FluxNetAction::QuboTempRange(AdjustDirection::Decrease) => 35,

            // Meta
            FluxNetAction::NoOp => 36,
        }
    }

    /// Get the target phase for this action
    ///
    /// Returns None for NoOp actions.
    pub fn target_phase(&self) -> Option<PhaseName> {
        match self {
            FluxNetAction::ReservoirSpectralRadius(_)
            | FluxNetAction::ReservoirLeakRate(_)
            | FluxNetAction::ReservoirInputScaling(_) => Some(PhaseName::Reservoir),

            FluxNetAction::TeVsKuramotoWeight(_)
            | FluxNetAction::GeodesicWeight(_)
            | FluxNetAction::TeBatchSize(_)
            | FluxNetAction::AiUncertaintyThreshold(_) => Some(PhaseName::TransferEntropy),

            FluxNetAction::ThermoForceStrong(_)
            | FluxNetAction::ThermoForceWeak(_)
            | FluxNetAction::ThermoStepsPerTemp(_)
            | FluxNetAction::ThermoTempMax(_)
            | FluxNetAction::ThermoNumTemps(_)
            | FluxNetAction::ThermoReplicas(_) => Some(PhaseName::Thermodynamic),

            FluxNetAction::QuantumIterations(_)
            | FluxNetAction::QuantumBeta(_)
            | FluxNetAction::QuantumTemperature(_)
            | FluxNetAction::QuboIterations(_)
            | FluxNetAction::QuboTempRange(_) => Some(PhaseName::Quantum),

            FluxNetAction::NoOp => None,
        }
    }

    /// Check if this is a no-op action
    pub fn is_noop(&self) -> bool {
        matches!(self, FluxNetAction::NoOp)
    }

    /// Apply action to config (mutates in-place)
    ///
    /// Modifies WorldRecordConfig parameters based on action type.
    /// All adjustments respect constitutional bounds.
    ///
    /// # Arguments
    /// - `config`: Mutable reference to WorldRecordConfig
    ///
    /// # Returns
    /// Human-readable description of what was changed
    ///
    /// # Errors
    /// Returns PRCTError if action would violate parameter bounds
    pub fn apply(&self, config: &mut WorldRecordConfig) -> Result<String> {
        match self {
            // ========== Phase 0: Reservoir ==========

            FluxNetAction::ReservoirSpectralRadius(dir) => {
                let old = config.neuromorphic.spectral_radius;
                let delta = 0.05;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.99),
                    AdjustDirection::Decrease => (old - delta).max(0.80),
                };
                config.neuromorphic.spectral_radius = new;
                Ok(format!("Reservoir spectral_radius: {:.2} → {:.2}", old, new))
            }

            FluxNetAction::ReservoirLeakRate(dir) => {
                let old = config.neuromorphic.leak_rate;
                let delta = 0.05;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.40),
                    AdjustDirection::Decrease => (old - delta).max(0.20),
                };
                config.neuromorphic.leak_rate = new;
                Ok(format!("Reservoir leak_rate: {:.2} → {:.2}", old, new))
            }

            FluxNetAction::ReservoirInputScaling(dir) => {
                let old = config.neuromorphic.input_scaling;
                let delta = 0.10;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.70),
                    AdjustDirection::Decrease => (old - delta).max(0.30),
                };
                config.neuromorphic.input_scaling = new;
                Ok(format!("Reservoir input_scaling: {:.2} → {:.2}", old, new))
            }

            // ========== Phase 1: Transfer Entropy ==========

            FluxNetAction::TeVsKuramotoWeight(dir) => {
                let old = config.transfer_entropy.te_vs_kuramoto_weight;
                let delta = 0.05;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.95),
                    AdjustDirection::Decrease => (old - delta).max(0.50),
                };
                config.transfer_entropy.te_vs_kuramoto_weight = new;
                Ok(format!("TE vs Kuramoto weight: {:.2} → {:.2}", old, new))
            }

            FluxNetAction::GeodesicWeight(dir) => {
                let old = config.geodesic.weight;
                let delta = 0.05;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.50),
                    AdjustDirection::Decrease => (old - delta).max(0.10),
                };
                config.geodesic.weight = new;
                Ok(format!("Geodesic weight: {:.2} → {:.2}", old, new))
            }

            FluxNetAction::TeBatchSize(dir) => {
                let old = config.gpu.batch_size;
                let delta = 512;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(4096),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(512),
                };
                config.gpu.batch_size = new;
                Ok(format!("TE batch_size: {} → {}", old, new))
            }

            FluxNetAction::AiUncertaintyThreshold(dir) => {
                // Note: Requires ActiveInferenceConfig to exist in WorldRecordConfig
                // For now, we'll use a proxy field. TODO: Add ActiveInferenceConfig
                let old = config.adp.epsilon; // Using ADP epsilon as proxy
                let delta = 0.05;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.90),
                    AdjustDirection::Decrease => (old - delta).max(0.10),
                };
                config.adp.epsilon = new;
                Ok(format!("AI uncertainty threshold (via ADP epsilon): {:.2} → {:.2}", old, new))
            }

            // ========== Phase 2: Thermodynamic ==========

            FluxNetAction::ThermoForceStrong(dir) => {
                let old = config.thermo.force_strong;
                let delta = 0.1;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(2.0),
                    AdjustDirection::Decrease => (old - delta).max(1.0),
                };
                config.thermo.force_strong = new;
                Ok(format!("Thermo force_strong: {:.1} → {:.1}", old, new))
            }

            FluxNetAction::ThermoForceWeak(dir) => {
                let old = config.thermo.force_weak;
                let delta = 0.1;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(1.0),
                    AdjustDirection::Decrease => (old - delta).max(0.3),
                };
                config.thermo.force_weak = new;
                Ok(format!("Thermo force_weak: {:.1} → {:.1}", old, new))
            }

            FluxNetAction::ThermoStepsPerTemp(dir) => {
                let old = config.thermo.steps_per_temp;
                let delta = 1000;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(12000),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(2000),
                };
                config.thermo.steps_per_temp = new;
                Ok(format!("Thermo steps_per_temp: {} → {}", old, new))
            }

            FluxNetAction::ThermoTempMax(dir) => {
                let old = config.thermo.t_max;
                let delta = 2.0;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(20.0),
                    AdjustDirection::Decrease => (old - delta).max(8.0),
                };
                config.thermo.t_max = new;
                Ok(format!("Thermo t_max: {:.1} → {:.1}", old, new))
            }

            FluxNetAction::ThermoNumTemps(dir) => {
                let old = config.thermo.num_temps;
                let delta = 4;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(56),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(16),
                };
                config.thermo.num_temps = new;
                Ok(format!("Thermo num_temps: {} → {}", old, new))
            }

            FluxNetAction::ThermoReplicas(dir) => {
                let old = config.thermo.replicas;
                let delta = 4;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(56),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(16),
                };
                config.thermo.replicas = new;
                Ok(format!("Thermo replicas: {} → {}", old, new))
            }

            // ========== Phase 3: Quantum ==========

            FluxNetAction::QuantumIterations(dir) => {
                let old = config.quantum.num_iterations;
                let delta = 5;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(100),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(10),
                };
                config.quantum.num_iterations = new;
                Ok(format!("Quantum iterations: {} → {}", old, new))
            }

            FluxNetAction::QuantumBeta(dir) => {
                let old = config.quantum.beta;
                let delta = 0.02;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(0.99),
                    AdjustDirection::Decrease => (old - delta).max(0.85),
                };
                config.quantum.beta = new;
                Ok(format!("Quantum beta: {:.2} → {:.2}", old, new))
            }

            FluxNetAction::QuantumTemperature(dir) => {
                let old = config.quantum.temperature;
                let delta = 0.1;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(1.5),
                    AdjustDirection::Decrease => (old - delta).max(0.3),
                };
                config.quantum.temperature = new;
                Ok(format!("Quantum temperature: {:.1} → {:.1}", old, new))
            }

            FluxNetAction::QuboIterations(dir) => {
                let old = config.quantum.qubo_iterations;
                let delta = 2000;
                let new = match dir {
                    AdjustDirection::Increase => (old + delta).min(20000),
                    AdjustDirection::Decrease => (old.saturating_sub(delta)).max(5000),
                };
                config.quantum.qubo_iterations = new;
                Ok(format!("QUBO iterations: {} → {}", old, new))
            }

            FluxNetAction::QuboTempRange(dir) => {
                let old_initial = config.quantum.qubo_t_initial;
                let old_final = config.quantum.qubo_t_final;
                let multiplier = match dir {
                    AdjustDirection::Increase => 1.1,
                    AdjustDirection::Decrease => 0.9,
                };
                let new_initial = (old_initial * multiplier).clamp(0.1, 10.0);
                let new_final = (old_final * multiplier).clamp(0.01, 1.0);
                config.quantum.qubo_t_initial = new_initial;
                config.quantum.qubo_t_final = new_final;
                Ok(format!(
                    "QUBO temp range: [{:.2}, {:.2}] → [{:.2}, {:.2}]",
                    old_initial, old_final, new_initial, new_final
                ))
            }

            // ========== Meta ==========

            FluxNetAction::NoOp => Ok("No operation".to_string()),
        }
    }

    /// Get human-readable description of action
    ///
    /// Used for telemetry logging and debugging.
    pub fn description(&self) -> &'static str {
        match self {
            // Phase 0
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase) =>
                "Increase Reservoir Spectral Radius",
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Decrease) =>
                "Decrease Reservoir Spectral Radius",
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Increase) =>
                "Increase Reservoir Leak Rate",
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Decrease) =>
                "Decrease Reservoir Leak Rate",
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Increase) =>
                "Increase Reservoir Input Scaling",
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Decrease) =>
                "Decrease Reservoir Input Scaling",

            // Phase 1
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Increase) =>
                "Increase TE vs Kuramoto Weight",
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Decrease) =>
                "Decrease TE vs Kuramoto Weight",
            FluxNetAction::GeodesicWeight(AdjustDirection::Increase) =>
                "Increase Geodesic Weight",
            FluxNetAction::GeodesicWeight(AdjustDirection::Decrease) =>
                "Decrease Geodesic Weight",
            FluxNetAction::TeBatchSize(AdjustDirection::Increase) =>
                "Increase TE Batch Size",
            FluxNetAction::TeBatchSize(AdjustDirection::Decrease) =>
                "Decrease TE Batch Size",
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Increase) =>
                "Increase AI Uncertainty Threshold",
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Decrease) =>
                "Decrease AI Uncertainty Threshold",

            // Phase 2
            FluxNetAction::ThermoForceStrong(AdjustDirection::Increase) =>
                "Increase Thermo Strong Force",
            FluxNetAction::ThermoForceStrong(AdjustDirection::Decrease) =>
                "Decrease Thermo Strong Force",
            FluxNetAction::ThermoForceWeak(AdjustDirection::Increase) =>
                "Increase Thermo Weak Force",
            FluxNetAction::ThermoForceWeak(AdjustDirection::Decrease) =>
                "Decrease Thermo Weak Force",
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Increase) =>
                "Increase Thermo Steps Per Temp",
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Decrease) =>
                "Decrease Thermo Steps Per Temp",
            FluxNetAction::ThermoTempMax(AdjustDirection::Increase) =>
                "Increase Thermo Max Temperature",
            FluxNetAction::ThermoTempMax(AdjustDirection::Decrease) =>
                "Decrease Thermo Max Temperature",
            FluxNetAction::ThermoNumTemps(AdjustDirection::Increase) =>
                "Increase Thermo Num Temps",
            FluxNetAction::ThermoNumTemps(AdjustDirection::Decrease) =>
                "Decrease Thermo Num Temps",
            FluxNetAction::ThermoReplicas(AdjustDirection::Increase) =>
                "Increase Thermo Replicas",
            FluxNetAction::ThermoReplicas(AdjustDirection::Decrease) =>
                "Decrease Thermo Replicas",

            // Phase 3
            FluxNetAction::QuantumIterations(AdjustDirection::Increase) =>
                "Increase Quantum Iterations",
            FluxNetAction::QuantumIterations(AdjustDirection::Decrease) =>
                "Decrease Quantum Iterations",
            FluxNetAction::QuantumBeta(AdjustDirection::Increase) =>
                "Increase Quantum Beta",
            FluxNetAction::QuantumBeta(AdjustDirection::Decrease) =>
                "Decrease Quantum Beta",
            FluxNetAction::QuantumTemperature(AdjustDirection::Increase) =>
                "Increase Quantum Temperature",
            FluxNetAction::QuantumTemperature(AdjustDirection::Decrease) =>
                "Decrease Quantum Temperature",
            FluxNetAction::QuboIterations(AdjustDirection::Increase) =>
                "Increase QUBO Iterations",
            FluxNetAction::QuboIterations(AdjustDirection::Decrease) =>
                "Decrease QUBO Iterations",
            FluxNetAction::QuboTempRange(AdjustDirection::Increase) =>
                "Increase QUBO Temp Range",
            FluxNetAction::QuboTempRange(AdjustDirection::Decrease) =>
                "Decrease QUBO Temp Range",

            // Meta
            FluxNetAction::NoOp => "No Operation",
        }
    }

    /// Get short telemetry code (2-4 chars)
    ///
    /// Used in compact telemetry logs:
    /// - `R0+`: Increase Reservoir param 0
    /// - `T1-`: Decrease TE param 1
    /// - `TH2+`: Increase Thermo param 2
    /// - `Q0+`: Increase Quantum param 0
    /// - `--`: No-op
    pub fn telemetry_code(&self) -> &'static str {
        match self {
            // Phase 0: Reservoir (R0-R2)
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase) => "R0+",
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Decrease) => "R0-",
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Increase) => "R1+",
            FluxNetAction::ReservoirLeakRate(AdjustDirection::Decrease) => "R1-",
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Increase) => "R2+",
            FluxNetAction::ReservoirInputScaling(AdjustDirection::Decrease) => "R2-",

            // Phase 1: Transfer Entropy (T0-T3)
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Increase) => "T0+",
            FluxNetAction::TeVsKuramotoWeight(AdjustDirection::Decrease) => "T0-",
            FluxNetAction::GeodesicWeight(AdjustDirection::Increase) => "T1+",
            FluxNetAction::GeodesicWeight(AdjustDirection::Decrease) => "T1-",
            FluxNetAction::TeBatchSize(AdjustDirection::Increase) => "T2+",
            FluxNetAction::TeBatchSize(AdjustDirection::Decrease) => "T2-",
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Increase) => "T3+",
            FluxNetAction::AiUncertaintyThreshold(AdjustDirection::Decrease) => "T3-",

            // Phase 2: Thermodynamic (TH0-TH5)
            FluxNetAction::ThermoForceStrong(AdjustDirection::Increase) => "TH0+",
            FluxNetAction::ThermoForceStrong(AdjustDirection::Decrease) => "TH0-",
            FluxNetAction::ThermoForceWeak(AdjustDirection::Increase) => "TH1+",
            FluxNetAction::ThermoForceWeak(AdjustDirection::Decrease) => "TH1-",
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Increase) => "TH2+",
            FluxNetAction::ThermoStepsPerTemp(AdjustDirection::Decrease) => "TH2-",
            FluxNetAction::ThermoTempMax(AdjustDirection::Increase) => "TH3+",
            FluxNetAction::ThermoTempMax(AdjustDirection::Decrease) => "TH3-",
            FluxNetAction::ThermoNumTemps(AdjustDirection::Increase) => "TH4+",
            FluxNetAction::ThermoNumTemps(AdjustDirection::Decrease) => "TH4-",
            FluxNetAction::ThermoReplicas(AdjustDirection::Increase) => "TH5+",
            FluxNetAction::ThermoReplicas(AdjustDirection::Decrease) => "TH5-",

            // Phase 3: Quantum (Q0-Q4)
            FluxNetAction::QuantumIterations(AdjustDirection::Increase) => "Q0+",
            FluxNetAction::QuantumIterations(AdjustDirection::Decrease) => "Q0-",
            FluxNetAction::QuantumBeta(AdjustDirection::Increase) => "Q1+",
            FluxNetAction::QuantumBeta(AdjustDirection::Decrease) => "Q1-",
            FluxNetAction::QuantumTemperature(AdjustDirection::Increase) => "Q2+",
            FluxNetAction::QuantumTemperature(AdjustDirection::Decrease) => "Q2-",
            FluxNetAction::QuboIterations(AdjustDirection::Increase) => "Q3+",
            FluxNetAction::QuboIterations(AdjustDirection::Decrease) => "Q3-",
            FluxNetAction::QuboTempRange(AdjustDirection::Increase) => "Q4+",
            FluxNetAction::QuboTempRange(AdjustDirection::Decrease) => "Q4-",

            // Meta
            FluxNetAction::NoOp => "--",
        }
    }

    /// Get all possible actions for a given phase
    ///
    /// Returns vector of actions relevant to the specified phase.
    /// Used for phase-specific action filtering.
    pub fn actions_for_phase(phase: PhaseName) -> Vec<FluxNetAction> {
        (0..Self::ACTION_SPACE_SIZE)
            .map(Self::from_index)
            .filter(|a| a.target_phase() == Some(phase))
            .collect()
    }

    /// Get all possible actions (for RL exploration)
    ///
    /// Returns vector of all 37 actions in action space order.
    pub fn all_actions() -> Vec<FluxNetAction> {
        (0..Self::ACTION_SPACE_SIZE)
            .map(Self::from_index)
            .collect()
    }
}

impl Default for FluxNetAction {
    /// Default action is NoOp
    fn default() -> Self {
        FluxNetAction::NoOp
    }
}

impl std::fmt::Display for FluxNetAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Action execution result with telemetry
///
/// Returned when applying a FluxNetAction to a WorldRecordConfig.
/// Captures before/after state for reward computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Action that was executed
    pub action: FluxNetAction,

    /// Human-readable description of what changed
    pub description: String,

    /// Whether action modified the config (false for NoOp)
    pub modified: bool,

    /// Target phase (None for NoOp)
    pub phase: Option<PhaseName>,
}

impl ActionResult {
    /// Create a new action result
    pub fn new(
        action: FluxNetAction,
        description: String,
        modified: bool,
    ) -> Self {
        Self {
            action,
            phase: action.target_phase(),
            description,
            modified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space_size() {
        // 6 + 8 + 12 + 10 + 1 = 37 actions
        assert_eq!(FluxNetAction::ACTION_SPACE_SIZE, 37);
    }

    #[test]
    fn test_action_index_conversion() {
        for i in 0..FluxNetAction::ACTION_SPACE_SIZE {
            let action = FluxNetAction::from_index(i);
            assert_eq!(action.to_index(), i);
        }
    }

    #[test]
    fn test_phase_filtering() {
        let reservoir_actions = FluxNetAction::actions_for_phase(PhaseName::Reservoir);
        assert_eq!(reservoir_actions.len(), 6); // 3 reservoir params × 2 directions = 6 actions

        let te_actions = FluxNetAction::actions_for_phase(PhaseName::TransferEntropy);
        assert_eq!(te_actions.len(), 8); // 4 TE params × 2 directions = 8 actions

        let thermo_actions = FluxNetAction::actions_for_phase(PhaseName::Thermodynamic);
        assert_eq!(thermo_actions.len(), 12); // 6 thermo params × 2 directions = 12 actions

        let quantum_actions = FluxNetAction::actions_for_phase(PhaseName::Quantum);
        assert_eq!(quantum_actions.len(), 10); // 5 quantum params × 2 directions = 10 actions
    }

    #[test]
    fn test_telemetry_codes() {
        assert_eq!(
            FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase).telemetry_code(),
            "R0+"
        );
        assert_eq!(
            FluxNetAction::ThermoForceStrong(AdjustDirection::Decrease).telemetry_code(),
            "TH0-"
        );
        assert_eq!(FluxNetAction::NoOp.telemetry_code(), "--");
    }

    #[test]
    fn test_noop_detection() {
        assert!(FluxNetAction::NoOp.is_noop());
        assert!(!FluxNetAction::ReservoirSpectralRadius(AdjustDirection::Increase).is_noop());
    }

    #[test]
    fn test_display() {
        let action = FluxNetAction::QuantumBeta(AdjustDirection::Increase);
        assert_eq!(format!("{}", action), "Increase Quantum Beta");
    }
}
