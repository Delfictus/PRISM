// Policy Selection via Expected Free Energy
// Constitution: Phase 2, Task 2.3 - Active Inference Controller
//
// Implements active inference policy selection:
// G(π) = E_q[ln q(o|π) - ln p(o|C)] + E_q[ln q(θ|π) - ln q(θ)]
//      = Pragmatic value + Epistemic value
//      = Risk + Ambiguity - Novelty
//
// For DARPA Narcissus: Select which 100 of 900 windows to measure

use ndarray::Array1;

use super::hierarchical_model::{GaussianBelief, HierarchicalModel};
use super::observation_model::MeasurementPattern;
use super::transition_model::{ControlAction, TransitionModel};
use super::variational_inference::VariationalInference;

/// Policy: sequence of actions over time horizon
#[derive(Debug, Clone)]
pub struct Policy {
    /// Actions over time horizon
    pub actions: Vec<ControlAction>,
    /// Expected free energy (lower is better)
    pub expected_free_energy: f64,
    /// Policy identifier
    pub id: usize,
}

impl Policy {
    /// Create new policy
    pub fn new(actions: Vec<ControlAction>, id: usize) -> Self {
        Self {
            actions,
            expected_free_energy: f64::INFINITY,
            id,
        }
    }

    /// Null policy (no action)
    pub fn null(horizon: usize, n_windows: usize, id: usize) -> Self {
        let actions = (0..horizon)
            .map(|_| ControlAction::null(n_windows))
            .collect();

        Self::new(actions, id)
    }
}

/// Expected free energy components
#[derive(Debug, Clone)]
pub struct ExpectedFreeEnergyComponents {
    /// Total expected free energy
    pub total: f64,
    /// Pragmatic value (goal achievement)
    pub pragmatic: f64,
    /// Epistemic value (information gain)
    pub epistemic: f64,
    /// Risk (expected surprise about observations)
    pub risk: f64,
    /// Ambiguity (uncertainty in observations)
    pub ambiguity: f64,
    /// Novelty (information gain about model)
    pub novelty: f64,
}

impl ExpectedFreeEnergyComponents {
    /// Create from components
    pub fn new(pragmatic: f64, epistemic: f64, risk: f64, ambiguity: f64, novelty: f64) -> Self {
        let total = pragmatic + epistemic;
        Self {
            total,
            pragmatic,
            epistemic,
            risk,
            ambiguity,
            novelty,
        }
    }
}

/// Policy selector for active inference
#[derive(Debug)]
pub struct PolicySelector {
    /// Planning horizon (number of future steps)
    pub horizon: usize,
    /// Number of policies to evaluate
    pub n_policies: usize,
    /// Preferred observations (goal state)
    pub preferred_observations: Array1<f64>,
    /// Variational inference engine
    pub inference: VariationalInference,
    /// Transition model for prediction
    pub transition: TransitionModel,

    /// GPU policy evaluator (if CUDA enabled)
    #[cfg(feature = "cuda")]
    pub gpu_evaluator:
        Option<std::sync::Arc<std::sync::Mutex<super::gpu_policy_eval::GpuPolicyEvaluator>>>,
}

// Manual Clone implementation because Mutex doesn't derive Clone
impl Clone for PolicySelector {
    fn clone(&self) -> Self {
        Self {
            horizon: self.horizon,
            n_policies: self.n_policies,
            preferred_observations: self.preferred_observations.clone(),
            inference: self.inference.clone(),
            transition: self.transition.clone(),
            #[cfg(feature = "cuda")]
            gpu_evaluator: self.gpu_evaluator.clone(),
        }
    }
}

impl PolicySelector {
    /// Create new policy selector with optimized policy count
    ///
    /// GPU OPTIMIZATION: Reduced from 10 to 5 policies for performance
    /// - Policy 1: Exploitation (follow gradient)
    /// - Policy 2-3: Exploration (orthogonal directions)
    /// - Policy 4: Safety (minimal perturbation)
    /// - Policy 5: Information gain (high uncertainty regions)
    pub fn new(
        horizon: usize,
        n_policies: usize,
        preferred_observations: Array1<f64>,
        inference: VariationalInference,
        transition: TransitionModel,
    ) -> Self {
        // Optimize policy count for GPU performance
        let optimized_policies = n_policies.min(5);
        Self {
            horizon,
            n_policies: optimized_policies,
            preferred_observations,
            inference,
            transition,
            #[cfg(feature = "cuda")]
            gpu_evaluator: None, // Will be set by set_gpu_evaluator()
        }
    }

    /// Set GPU policy evaluator (CUDA feature only)
    #[cfg(feature = "cuda")]
    pub fn set_gpu_evaluator(
        &mut self,
        evaluator: std::sync::Arc<std::sync::Mutex<super::gpu_policy_eval::GpuPolicyEvaluator>>,
    ) {
        self.gpu_evaluator = Some(evaluator);
        println!("[POLICY] GPU policy evaluator enabled");
    }

    /// Select optimal policy: π* = argmin_π G(π)
    ///
    /// Evaluates candidate policies and returns best one
    pub fn select_policy(&self, model: &HierarchicalModel) -> Policy {
        // Generate candidate policies
        let policies = self.generate_policies(model);

        // Try GPU evaluation first (if available)
        #[cfg(feature = "cuda")]
        {
            if let Some(ref gpu_eval) = self.gpu_evaluator {
                println!("[POLICY] Attempting GPU policy evaluation...");

                // Extract observation matrix from inference engine
                let obs_matrix = &self.inference.observation_model.jacobian;

                match gpu_eval.lock() {
                    Ok(mut evaluator) => {
                        match evaluator.evaluate_policies_gpu(
                            model,
                            &policies,
                            obs_matrix,
                            &self.preferred_observations,
                        ) {
                            Ok(efe_values) => {
                                println!("[POLICY] GPU evaluation SUCCESS!");

                                // Assign EFE values to policies
                                let evaluated: Vec<_> = policies
                                    .into_iter()
                                    .zip(efe_values.iter())
                                    .map(|(mut policy, &efe)| {
                                        policy.expected_free_energy = efe;
                                        policy
                                    })
                                    .collect();

                                // Select minimum
                                return evaluated
                                    .into_iter()
                                    .min_by(|a, b| {
                                        a.expected_free_energy
                                            .partial_cmp(&b.expected_free_energy)
                                            .unwrap()
                                    })
                                    .unwrap();
                            }
                            Err(e) => {
                                eprintln!("[POLICY] GPU evaluation failed: {}", e);
                                eprintln!("[POLICY] Falling back to CPU evaluation");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[POLICY] Failed to lock GPU evaluator: {}", e);
                        eprintln!("[POLICY] Falling back to CPU evaluation");
                    }
                }
            }
        }

        // CPU evaluation (fallback or default)
        println!("[POLICY] Using CPU policy evaluation");
        let evaluated: Vec<_> = policies
            .into_iter()
            .map(|mut policy| {
                policy.expected_free_energy = self.compute_expected_free_energy(model, &policy);
                policy
            })
            .collect();

        // Select policy with minimum expected free energy
        evaluated
            .into_iter()
            .min_by(|a, b| {
                a.expected_free_energy
                    .partial_cmp(&b.expected_free_energy)
                    .unwrap()
            })
            .unwrap()
    }

    /// Generate optimized candidate policies
    ///
    /// GPU OPTIMIZATION: Strategic policies instead of random exploration
    /// - Policy 0: Exploitation (adaptive sensing + gradient correction)
    /// - Policy 1: Conservative (uniform sensing + small correction)
    /// - Policy 2: Aggressive (dense sensing + large correction)
    /// - Policy 3: Exploratory (random sensing + varied correction)
    /// - Policy 4: Information-seeking (sparse adaptive + minimal correction)
    fn generate_policies(&self, model: &HierarchicalModel) -> Vec<Policy> {
        let mut policies = Vec::with_capacity(self.n_policies);
        let n_windows = model.level1.n_windows;

        // Only generate 5 strategic policies (not 10)
        for policy_id in 0..self.n_policies {
            let mut actions = Vec::with_capacity(self.horizon);

            for _ in 0..self.horizon {
                // Strategic measurement patterns and corrections
                // MORE AGGRESSIVE for hard problems
                let (measurement_pattern, correction_gain) = match policy_id {
                    0 => {
                        // Exploitation: adaptive sensing + STRONG correction
                        (
                            MeasurementPattern::adaptive(100, &model.level1.belief),
                            0.95,
                        )
                    }
                    1 => {
                        // Aggressive: uniform sensing + FULL correction
                        (MeasurementPattern::uniform(100, n_windows), 1.0)
                    }
                    2 => {
                        // Super Aggressive: dense sensing + OVERCORRECTION
                        (MeasurementPattern::uniform(150, n_windows), 1.2)
                    }
                    3 => {
                        // Smart exploration: adaptive + strong correction
                        (
                            MeasurementPattern::adaptive(120, &model.level1.belief),
                            0.85,
                        )
                    }
                    _ => {
                        // Focused: target highest uncertainty + aggressive
                        (MeasurementPattern::adaptive(80, &model.level1.belief), 1.1)
                    }
                };

                let phase_correction = &model.level1.belief.mean * (-correction_gain);

                actions.push(ControlAction {
                    phase_correction,
                    measurement_pattern: measurement_pattern.active_windows,
                });
            }

            policies.push(Policy::new(actions, policy_id));
        }

        policies
    }

    /// Compute expected free energy: G(π) = Risk + Ambiguity - Novelty
    ///
    /// G(π) = E_q[ln q(o|π) - ln p(o|C)] + E_q[ln q(θ|π) - ln q(θ)]
    pub fn compute_expected_free_energy(&self, model: &HierarchicalModel, policy: &Policy) -> f64 {
        let components = self.compute_efe_components(model, policy);
        components.total
    }

    /// Compute all expected free energy components
    pub fn compute_efe_components(
        &self,
        model: &HierarchicalModel,
        policy: &Policy,
    ) -> ExpectedFreeEnergyComponents {
        // Simulate policy execution
        let trajectory = self
            .transition
            .multi_step_prediction(model, &policy.actions);

        let mut total_risk = 0.0;
        let mut total_ambiguity = 0.0;
        let mut total_novelty = 0.0;

        // Accumulate over trajectory
        for future_model in trajectory.iter().skip(1) {
            // Predict observations under this future state
            let predicted_obs = self.inference.predict_observations(future_model);

            // Risk: deviation from preferred observations
            let observation_error = &predicted_obs - &self.preferred_observations;
            total_risk += (&observation_error * &observation_error).sum();

            // Ambiguity: uncertainty in observations
            let obs_variance = self
                .inference
                .observation_model
                .observation_variance(&future_model.level1.belief);
            total_ambiguity += obs_variance.sum();

            // Novelty: information gain about state
            // H(x_prior) - H(x_posterior)
            let prior_entropy = self.inference.prior_level1.entropy();
            let posterior_entropy = future_model.level1.belief.entropy();
            total_novelty += prior_entropy - posterior_entropy;
        }

        // Normalize by horizon
        let horizon = policy.actions.len() as f64;
        total_risk /= horizon;
        total_ambiguity /= horizon;
        total_novelty /= horizon;

        // Pragmatic value: minimize risk
        let pragmatic = total_risk;

        // Epistemic value: balance ambiguity and novelty
        let epistemic = total_ambiguity - total_novelty;

        ExpectedFreeEnergyComponents::new(
            pragmatic,
            epistemic,
            total_risk,
            total_ambiguity,
            total_novelty,
        )
    }

    /// Information gain from policy: I(x; o | π)
    ///
    /// Mutual information between hidden states and observations
    pub fn information_gain(&self, model: &HierarchicalModel, policy: &Policy) -> f64 {
        let trajectory = self
            .transition
            .multi_step_prediction(model, &policy.actions);

        let mut total_info_gain = 0.0;

        for future_model in trajectory.iter().skip(1) {
            // Prior entropy: H(x)
            let prior_entropy = self.inference.prior_level1.entropy();

            // Posterior entropy: H(x|o)
            let posterior_entropy = future_model.level1.belief.entropy();

            // Information gain: I(x;o) = H(x) - H(x|o)
            total_info_gain += prior_entropy - posterior_entropy;
        }

        total_info_gain
    }

    /// Select active measurement pattern for next observation
    ///
    /// Which windows to measure? Maximize information gain
    pub fn select_measurement_pattern(&self, model: &HierarchicalModel) -> MeasurementPattern {
        // Simple heuristic: measure windows with highest uncertainty
        MeasurementPattern::adaptive(100, &model.level1.belief)
    }

    /// Compute phase correction to minimize expected error
    ///
    /// Deformable mirror control: u = -K·x (negative feedback)
    pub fn compute_phase_correction(&self, model: &HierarchicalModel) -> Array1<f64> {
        // Proportional control: correct fraction of estimated error
        let gain = 0.7; // 70% correction per step
        &model.level1.belief.mean * (-gain)
    }
}

/// Active sensing strategy
#[derive(Debug, Clone, Copy)]
pub enum SensingStrategy {
    /// Uniform sampling (all windows equally)
    Uniform,
    /// Adaptive (focus on high-uncertainty regions)
    Adaptive,
    /// Random exploration
    Random,
    /// Information-maximizing (greedy novelty)
    InfoMax,
}

impl SensingStrategy {
    /// Generate measurement pattern according to strategy
    pub fn generate_pattern(
        &self,
        n_active: usize,
        belief: &GaussianBelief,
        n_windows: usize,
    ) -> MeasurementPattern {
        match self {
            SensingStrategy::Uniform => MeasurementPattern::uniform(n_active, n_windows),
            SensingStrategy::Adaptive => MeasurementPattern::adaptive(n_active, belief),
            SensingStrategy::Random => MeasurementPattern::random(n_active, n_windows),
            SensingStrategy::InfoMax => {
                // Info-maximizing: measure windows with highest variance
                MeasurementPattern::adaptive(n_active, belief)
            }
        }
    }
}

/// Controller: combines policy selection and control execution
#[derive(Debug, Clone)]
pub struct ActiveInferenceController {
    /// Policy selector
    pub selector: PolicySelector,
    /// Sensing strategy
    pub strategy: SensingStrategy,
}

impl ActiveInferenceController {
    /// Create new controller
    pub fn new(selector: PolicySelector, strategy: SensingStrategy) -> Self {
        Self { selector, strategy }
    }

    /// Compute control action for current time step
    ///
    /// Returns: (phase_correction, measurement_pattern)
    pub fn control(&self, model: &HierarchicalModel) -> ControlAction {
        // Select optimal policy
        let policy = self.selector.select_policy(model);

        // Extract first action from policy
        policy.actions[0].clone()
    }

    /// Update preferred observations (goal state)
    ///
    /// For adaptive optics: sharp image = flat wavefront
    pub fn set_goal(&mut self, preferred_observations: Array1<f64>) {
        self.selector.preferred_observations = preferred_observations;
    }
}

#[cfg(test)]
mod tests {
    use super::super::hierarchical_model::constants;
    use super::super::observation_model::ObservationModel;
    use super::*;

    fn create_test_controller() -> (ActiveInferenceController, HierarchicalModel) {
        let model = HierarchicalModel::new();
        let obs_model = ObservationModel::new(100, constants::N_WINDOWS, 8.0, 0.01);
        let trans_model = TransitionModel::default_timescales();
        let inference = VariationalInference::new(obs_model, trans_model.clone(), &model);

        let preferred_obs = Array1::ones(100); // Flat wavefront
        let selector = PolicySelector::new(3, 10, preferred_obs, inference, trans_model);

        let controller = ActiveInferenceController::new(selector, SensingStrategy::Adaptive);

        (controller, model)
    }

    #[test]
    fn test_policy_generation() {
        let (controller, model) = create_test_controller();

        let policies = controller.selector.generate_policies(&model);

        // PolicySelector limits n_policies to 5 for GPU optimization
        assert_eq!(policies.len(), 5);
        assert!(policies.iter().all(|p| p.actions.len() == 3));
    }

    #[test]
    fn test_efe_is_finite() {
        let (controller, model) = create_test_controller();

        let policy = Policy::null(3, constants::N_WINDOWS, 0);
        let efe = controller
            .selector
            .compute_expected_free_energy(&model, &policy);

        assert!(efe.is_finite());
    }

    #[test]
    fn test_efe_components() {
        let (controller, model) = create_test_controller();

        let policy = Policy::null(3, constants::N_WINDOWS, 0);
        let components = controller.selector.compute_efe_components(&model, &policy);

        assert!(components.total.is_finite());
        assert!(components.risk >= 0.0);
        assert!(components.ambiguity >= 0.0);
        assert!(components.novelty.is_finite());
    }

    #[test]
    fn test_policy_selection() {
        let (controller, model) = create_test_controller();

        let policy = controller.selector.select_policy(&model);

        assert!(policy.expected_free_energy.is_finite());
        assert!(policy.actions.len() == 3);
    }

    #[test]
    fn test_information_gain_is_nonnegative() {
        let (controller, model) = create_test_controller();

        let policy = Policy::null(3, constants::N_WINDOWS, 0);
        let info_gain = controller.selector.information_gain(&model, &policy);

        // Information gain should be non-negative (data processing inequality)
        assert!(info_gain >= -1e-6); // Allow small numerical error
    }

    #[test]
    fn test_phase_correction_opposes_aberration() {
        let (controller, mut model) = create_test_controller();

        // Introduce aberration
        model.level1.belief.mean.fill(0.5);

        let correction = controller.selector.compute_phase_correction(&model);

        // Correction should be negative (opposite sign)
        assert!(correction.iter().all(|&x| x < 0.0));
    }

    #[test]
    fn test_adaptive_measurement_prioritizes_uncertainty() {
        let (_, mut model) = create_test_controller();

        // Create high uncertainty at specific window
        model.level1.belief.variance[42] = 10.0;

        let pattern = MeasurementPattern::adaptive(10, &model.level1.belief);

        // Should include high-uncertainty window
        assert!(pattern.active_windows.contains(&42));
    }

    #[test]
    fn test_control_action_generation() {
        let (controller, model) = create_test_controller();

        let action = controller.control(&model);

        assert_eq!(action.phase_correction.len(), constants::N_WINDOWS);
        assert!(!action.measurement_pattern.is_empty());
    }

    #[test]
    fn test_sensing_strategies() {
        let belief = GaussianBelief::isotropic(900, 0.0, 1.0);

        let uniform = SensingStrategy::Uniform.generate_pattern(100, &belief, 900);
        let adaptive = SensingStrategy::Adaptive.generate_pattern(100, &belief, 900);
        let random = SensingStrategy::Random.generate_pattern(100, &belief, 900);

        assert_eq!(uniform.active_windows.len(), 100);
        assert_eq!(adaptive.active_windows.len(), 100);
        assert_eq!(random.active_windows.len(), 100);
    }
}
