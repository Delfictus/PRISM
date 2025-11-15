// Generative Model: Complete Active Inference System
// Constitution: Phase 2, Task 2.1 - Generative Model Architecture
//
// Integrates all components:
// 1. Hierarchical state space (3 levels with timescale separation)
// 2. Observation model p(o|x) with wavefront sensing
// 3. Transition model p(x_{t+1}|x_t,u_t) with physics-based dynamics
// 4. Variational inference (minimize free energy)
// 5. Policy selection (minimize expected free energy)
// 6. Online learning (adapt parameters from data)
//
// Mathematical Foundation:
// F = E_q[ln q(x) - ln p(o,x)] → minimize via message passing
// G(π) = E_q[ln q(o|π) - ln p(o|C)] → minimize for action selection

use ndarray::Array1;
use std::collections::VecDeque;

use super::hierarchical_model::HierarchicalModel;
use super::observation_model::ObservationModel;
use super::policy_selection::{ActiveInferenceController, PolicySelector, SensingStrategy};
use super::transition_model::{ControlAction, TransitionModel};
use super::variational_inference::{FreeEnergyComponents, VariationalInference};

/// Complete generative model for active inference
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    /// Hierarchical state space model
    pub model: HierarchicalModel,
    /// Observation model
    pub observation: ObservationModel,
    /// Transition model
    pub transition: TransitionModel,
    /// Variational inference engine
    pub inference: VariationalInference,
    /// Policy selector
    pub controller: ActiveInferenceController,
    /// History of free energies (for monitoring convergence)
    pub free_energy_history: VecDeque<f64>,
    /// History length to keep
    pub history_length: usize,
}

impl GenerativeModel {
    /// Create new generative model with default parameters
    pub fn new() -> Self {
        let model = HierarchicalModel::new();

        // Observation model: 100 measurements, magnitude 8 star, 10ms integration
        let observation = ObservationModel::new(100, 900, 8.0, 0.01);

        // Transition model with hierarchical timescales
        let transition = TransitionModel::default_timescales();

        // Variational inference
        let inference = VariationalInference::new(observation.clone(), transition.clone(), &model);

        // Policy selector with 3-step horizon
        let preferred_obs = Array1::ones(100); // Flat wavefront
        let selector = PolicySelector::new(
            3,  // horizon
            10, // n_policies
            preferred_obs,
            inference.clone(),
            transition.clone(),
        );

        let controller = ActiveInferenceController::new(selector, SensingStrategy::Adaptive);

        Self {
            model,
            observation,
            transition,
            inference,
            controller,
            free_energy_history: VecDeque::with_capacity(1000),
            history_length: 1000,
        }
    }

    /// Process new observation: infer state and update beliefs
    ///
    /// This is the main inference loop:
    /// 1. Receive observation o_t
    /// 2. Infer hidden state x_t (minimize F)
    /// 3. Select action u_t (minimize G)
    /// 4. Apply action
    /// 5. Predict next state x_{t+1}
    pub fn step(&mut self, observation: &Array1<f64>) -> ControlAction {
        // 1. Variational inference: minimize free energy
        let fe = self.inference.infer(&mut self.model, observation);

        // Track free energy
        self.free_energy_history.push_back(fe.total);
        if self.free_energy_history.len() > self.history_length {
            self.free_energy_history.pop_front();
        }

        // 2. Policy selection: minimize expected free energy
        let action = self.controller.control(&self.model);

        // 3. Apply action and predict forward
        self.transition.predict(&mut self.model, &action);

        action
    }

    /// Run active inference loop for multiple steps
    pub fn run(&mut self, observations: &[Array1<f64>]) -> Vec<ControlAction> {
        observations.iter().map(|obs| self.step(obs)).collect()
    }

    /// Compute current free energy
    pub fn free_energy(&self, observation: &Array1<f64>) -> FreeEnergyComponents {
        self.inference.compute_free_energy(&self.model, observation)
    }

    /// Check if free energy is decreasing (learning is working)
    pub fn is_learning(&self) -> bool {
        if self.free_energy_history.len() < 10 {
            return true; // Need more data
        }

        // Compare recent average to older average
        let recent: f64 = self.free_energy_history.iter().rev().take(5).sum::<f64>() / 5.0;

        let older: f64 = self
            .free_energy_history
            .iter()
            .skip(self.free_energy_history.len() - 10)
            .take(5)
            .sum::<f64>()
            / 5.0;

        recent < older // Free energy should decrease
    }

    /// Update model parameters from data (online learning)
    pub fn learn_parameters(&mut self, observations: &[Array1<f64>], states: &[Array1<f64>]) {
        self.inference.learn_parameters(observations, states);
    }

    /// Set goal state (preferred observations)
    pub fn set_goal(&mut self, preferred_observations: Array1<f64>) {
        self.controller.set_goal(preferred_observations);
    }

    /// Get current state estimate (posterior mean)
    pub fn state_estimate(&self) -> &Array1<f64> {
        &self.model.level1.belief.mean
    }

    /// Get current state uncertainty (posterior variance)
    pub fn state_uncertainty(&self) -> &Array1<f64> {
        &self.model.level1.belief.variance
    }

    /// Predict observations under current belief
    pub fn predict_observations(&self) -> Array1<f64> {
        self.inference.predict_observations(&self.model)
    }

    /// Compute prediction error (RMSE)
    pub fn prediction_rmse(&self, observation: &Array1<f64>) -> f64 {
        let predicted = self.predict_observations();
        let error = observation - &predicted;
        let mse = (&error * &error).sum() / error.len() as f64;
        mse.sqrt()
    }

    /// Reset model to initial state
    pub fn reset(&mut self) {
        self.model = HierarchicalModel::new();
        self.free_energy_history.clear();
    }

    /// Get performance metrics
    pub fn metrics(&self, observations: &[Array1<f64>]) -> PerformanceMetrics {
        if observations.is_empty() {
            return PerformanceMetrics::default();
        }

        // Compute RMSE over all observations
        let total_rmse: f64 = observations
            .iter()
            .map(|obs| self.prediction_rmse(obs))
            .sum();
        let mean_rmse = total_rmse / observations.len() as f64;

        // Free energy trend (decreasing = good)
        let fe_decreasing = self.is_learning();

        // Average uncertainty
        let mean_uncertainty = self.state_uncertainty().mean().unwrap();

        // Free energy
        let current_fe = if !observations.is_empty() {
            self.free_energy(&observations[observations.len() - 1])
                .total
        } else {
            f64::NAN
        };

        PerformanceMetrics {
            rmse: mean_rmse,
            free_energy: current_fe,
            uncertainty: mean_uncertainty,
            learning: fe_decreasing,
        }
    }
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for validation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Root mean square error (target: < 5%)
    pub rmse: f64,
    /// Variational free energy (should decrease over time)
    pub free_energy: f64,
    /// Mean state uncertainty
    pub uncertainty: f64,
    /// Is system learning? (free energy decreasing)
    pub learning: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            rmse: f64::NAN,
            free_energy: f64::NAN,
            uncertainty: f64::NAN,
            learning: false,
        }
    }
}

impl PerformanceMetrics {
    /// Check if all validation criteria are met (from constitution)
    ///
    /// Task 2.1 Validation Criteria:
    /// - [ ] Predictions match observations (RMSE < 5%)
    /// - [ ] Parameters learn online
    /// - [ ] Uncertainty properly quantified
    /// - [ ] Free energy decreases over time
    pub fn satisfies_constitution(&self) -> bool {
        self.rmse < 0.05           // RMSE < 5%
            && self.uncertainty.is_finite()  // Uncertainty quantified
            && self.learning // Learning (FE decreasing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_model_creation() {
        let model = GenerativeModel::new();

        assert_eq!(model.model.level1.n_windows, 900);
        assert_eq!(model.observation.n_measurements, 100);
    }

    #[test]
    fn test_single_step_inference() {
        let mut model = GenerativeModel::new();
        let obs = Array1::ones(100);

        let action = model.step(&obs);

        assert_eq!(action.phase_correction.len(), 900);
        assert!(!action.measurement_pattern.is_empty());
    }

    #[test]
    fn test_free_energy_tracking() {
        let mut model = GenerativeModel::new();
        let obs = Array1::ones(100);

        for _ in 0..10 {
            model.step(&obs);
        }

        assert_eq!(model.free_energy_history.len(), 10);
    }

    #[test]
    fn test_prediction_rmse() {
        let model = GenerativeModel::new();
        let obs = model.predict_observations();

        let rmse = model.prediction_rmse(&obs);

        // Perfect prediction should have near-zero RMSE
        assert!(rmse < 1e-3);
    }

    #[test]
    fn test_state_estimation() {
        let model = GenerativeModel::new();

        let state = model.state_estimate();
        let uncertainty = model.state_uncertainty();

        assert_eq!(state.len(), 900);
        assert_eq!(uncertainty.len(), 900);
        assert!(uncertainty.iter().all(|&u| u > 0.0));
    }

    #[test]
    fn test_goal_setting() {
        let mut model = GenerativeModel::new();

        let goal = Array1::from_elem(100, 2.0);
        model.set_goal(goal.clone());

        assert_eq!(model.controller.selector.preferred_observations, goal);
    }

    #[test]
    fn test_multi_step_run() {
        let mut model = GenerativeModel::new();

        let observations = vec![Array1::ones(100), Array1::ones(100), Array1::ones(100)];

        let actions = model.run(&observations);

        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn test_reset() {
        let mut model = GenerativeModel::new();

        // Run some steps
        let obs = Array1::ones(100);
        for _ in 0..5 {
            model.step(&obs);
        }

        // Reset
        model.reset();

        assert_eq!(model.free_energy_history.len(), 0);
        assert_eq!(model.model.level1.belief.mean.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn test_performance_metrics() {
        let model = GenerativeModel::new();

        let observations = vec![Array1::ones(100), Array1::ones(100)];

        let metrics = model.metrics(&observations);

        assert!(metrics.rmse.is_finite());
        assert!(metrics.uncertainty.is_finite());
    }

    #[test]
    fn test_learning_detection() {
        let mut model = GenerativeModel::new();

        // Initially, not enough data
        assert!(model.is_learning());

        // Add decreasing free energies
        for i in (0..20).rev() {
            model.free_energy_history.push_back(i as f64);
        }

        // Should detect learning
        assert!(model.is_learning());
    }

    #[test]
    fn test_online_parameter_learning() {
        let mut model = GenerativeModel::new();

        let observations = vec![Array1::ones(100); 5];
        let states = vec![Array1::zeros(900); 5];

        // Should not crash
        model.learn_parameters(&observations, &states);
    }
}
