// Transition Model: p(x_{t+1} | x_t, u_t)
// Constitution: Phase 2, Task 2.1 - Generative Model Architecture
//
// Implements hierarchical temporal dynamics with timescale separation:
// - Level 1 (10ms): Window phase evolution driven by atmosphere
// - Level 2 (1s): Atmospheric turbulence evolution (Taylor frozen flow)
// - Level 3 (60s): Satellite orbital dynamics (Keplerian)
//
// Mathematical Foundation:
// Euler-Maruyama discretization: x_{t+1} = x_t + f(x_t)·dt + √dt·ξ_t
// where f(x) is drift and ξ_t ~ N(0, Σ) is diffusion

use ndarray::Array1;
use std::f64::consts::PI;

use super::hierarchical_model::{
    AtmosphericLevel, GaussianBelief, HierarchicalModel, SatelliteLevel, WindowPhaseLevel,
};

/// Control action (for active inference)
#[derive(Debug, Clone)]
pub struct ControlAction {
    /// Corrective phase applied to windows (deformable mirror)
    pub phase_correction: Array1<f64>,
    /// Measurement pattern selection (which windows to sense)
    pub measurement_pattern: Vec<usize>,
}

impl ControlAction {
    /// Null action (no correction)
    pub fn null(n_windows: usize) -> Self {
        Self {
            phase_correction: Array1::zeros(n_windows),
            measurement_pattern: (0..n_windows).collect(),
        }
    }
}

/// Transition model for all hierarchical levels
#[derive(Debug, Clone)]
pub struct TransitionModel {
    /// Time step for Level 1 (window phases)
    pub dt_fast: f64,
    /// Time step for Level 2 (atmosphere)
    pub dt_medium: f64,
    /// Time step for Level 3 (satellite)
    pub dt_slow: f64,
    /// Number of substeps for hierarchical integration
    pub substeps: usize,
}

impl TransitionModel {
    /// Create new transition model with timescale hierarchy
    pub fn new(dt_fast: f64, dt_medium: f64, dt_slow: f64) -> Self {
        // Compute substeps to maintain timescale separation
        let substeps = (dt_medium / dt_fast).round() as usize;

        Self {
            dt_fast,
            dt_medium,
            dt_slow,
            substeps,
        }
    }

    /// Default timescales from constitution
    pub fn default_timescales() -> Self {
        Self::new(0.01, 1.0, 60.0) // 10ms, 1s, 60s
    }

    /// Predict next state: p(x_{t+1} | x_t, u_t)
    ///
    /// Hierarchical evolution:
    /// 1. Evolve Level 3 (satellite) - slowest
    /// 2. Evolve Level 2 (atmosphere) with satellite context
    /// 3. Evolve Level 1 (windows) with atmospheric driving
    pub fn predict(&self, model: &mut HierarchicalModel, action: &ControlAction) {
        // Level 3: Satellite orbital dynamics (slowest)
        self.evolve_satellite(&mut model.level3, self.dt_slow);

        // Level 2: Atmospheric turbulence (medium)
        self.evolve_atmosphere(&mut model.level2, self.dt_medium);

        // Level 1: Window phases (fastest, multiple substeps)
        for _ in 0..self.substeps {
            self.evolve_windows(&mut model.level1, &model.level2, action, self.dt_fast);
        }
    }

    /// Evolve satellite orbital state
    ///
    /// Keplerian dynamics: d²r/dt² = -μ·r/|r|³
    /// State: x = [r_x, r_y, r_z, v_x, v_y, v_z]
    fn evolve_satellite(&self, level: &mut SatelliteLevel, dt: f64) {
        let state = &level.belief.mean;

        // Extract position and velocity
        let position = [state[0], state[1], state[2]];
        let velocity = [state[3], state[4], state[5]];

        // Compute acceleration
        let acceleration = level.gravitational_acceleration(&position);

        // Verlet integration (symplectic, conserves energy)
        // r_{n+1} = r_n + v_n·dt + 0.5·a_n·dt²
        // v_{n+1} = v_n + 0.5·(a_n + a_{n+1})·dt

        let new_position = [
            position[0] + velocity[0] * dt + 0.5 * acceleration[0] * dt * dt,
            position[1] + velocity[1] * dt + 0.5 * acceleration[1] * dt * dt,
            position[2] + velocity[2] * dt + 0.5 * acceleration[2] * dt * dt,
        ];

        let new_acceleration = level.gravitational_acceleration(&new_position);

        let new_velocity = [
            velocity[0] + 0.5 * (acceleration[0] + new_acceleration[0]) * dt,
            velocity[1] + 0.5 * (acceleration[1] + new_acceleration[1]) * dt,
            velocity[2] + 0.5 * (acceleration[2] + new_acceleration[2]) * dt,
        ];

        // Update belief mean
        level.belief.mean[0] = new_position[0];
        level.belief.mean[1] = new_position[1];
        level.belief.mean[2] = new_position[2];
        level.belief.mean[3] = new_velocity[0];
        level.belief.mean[4] = new_velocity[1];
        level.belief.mean[5] = new_velocity[2];

        // Variance grows slightly due to orbital perturbations
        for i in 0..6 {
            level.belief.variance[i] *= 1.0 + 1e-6 * dt;
        }
    }

    /// Evolve atmospheric turbulence
    ///
    /// Taylor frozen turbulence: ∂φ/∂t + v·∇φ = ν·∇²φ + ξ
    ///
    /// Simplified: exponential decorrelation with wind advection
    fn evolve_atmosphere(&self, level: &mut AtmosphericLevel, dt: f64) {
        let decorrelation_rate = 1.0 / 10.0; // 10s coherence time

        // Wind advection (phase shift in spatial modes)
        // For simplicity: exponential decay with noise injection

        for i in 0..level.n_modes {
            // Decorrelation
            level.belief.mean[i] *= (-decorrelation_rate * dt).exp();

            // Noise injection (maintains stationary statistics)
            let noise_scale = (2.0 * decorrelation_rate * level.c_n_squared * dt).sqrt();
            level.belief.mean[i] += noise_scale * rand::random::<f64>().sin();

            // Variance reaches stationary value
            let target_variance = level.c_n_squared;
            level.belief.variance[i] += (target_variance - level.belief.variance[i]) * dt;
        }
    }

    /// Evolve window phase dynamics
    ///
    /// Langevin dynamics: dφ/dt = -γ·φ + C·sin(φ_atm) + √(2D)·η(t)
    ///
    /// This is the thermodynamic network from Phase 1!
    fn evolve_windows(
        &self,
        level: &mut WindowPhaseLevel,
        atmosphere: &AtmosphericLevel,
        action: &ControlAction,
        dt: f64,
    ) {
        let state = &level.belief.mean;

        // Atmospheric driving field (project turbulence onto windows)
        let atmospheric_drive = self.project_atmosphere_to_windows(atmosphere, level.n_windows);

        // Drift: f(x) = -γ·x + C·sin(x_atm)
        let drift = level.drift(state, &atmospheric_drive);

        // Diffusion: √(2D)·η
        let diffusion_noise = level.diffusion_noise();

        // Control action (corrective phase from active inference)
        let control_effect = &action.phase_correction * (-1.0); // Negate to cancel aberrations

        // Euler-Maruyama update: x_{n+1} = x_n + f(x_n)·dt + √dt·ξ_n + u_n
        level.belief.mean = state + &drift * dt + &diffusion_noise * dt.sqrt() + &control_effect;

        // Update generalized coordinates (position + velocity)
        level.generalized.velocity = drift.clone(); // velocity = dx/dt
        level.generalized.position = level.belief.mean.clone();

        // Variance evolution (Fokker-Planck for diagonal covariance)
        // dΣ/dt = -2γ·Σ + 2D (approaches steady-state Σ_∞ = D/γ)
        let steady_state_variance = level.diffusion / level.damping;

        for i in 0..level.n_windows {
            level.belief.variance[i] +=
                (-2.0 * level.damping * level.belief.variance[i] + 2.0 * level.diffusion) * dt;

            // Clamp to reasonable range
            level.belief.variance[i] = level.belief.variance[i]
                .max(1e-6)
                .min(steady_state_variance * 10.0);

            // Update precision
            level.belief.precision[i] = 1.0 / level.belief.variance[i];
        }
    }

    /// Project atmospheric turbulence onto window array
    ///
    /// Maps turbulence modes to window phases via spatial correlation
    pub fn project_atmosphere_to_windows(
        &self,
        atmosphere: &AtmosphericLevel,
        n_windows: usize,
    ) -> Array1<f64> {
        // For now: simple spatial interpolation
        // Full implementation: Fourier projection with Kolmogorov spectrum

        let mut window_phases = Array1::zeros(n_windows);

        // Map atmospheric modes to windows with spatial correlation
        let modes_per_window = atmosphere.n_modes as f64 / n_windows as f64;

        for i in 0..n_windows {
            let mode_idx = (i as f64 * modes_per_window) as usize % atmosphere.n_modes;
            window_phases[i] = atmosphere.belief.mean[mode_idx];

            // Add neighboring mode contributions (spatial correlation)
            if mode_idx > 0 {
                window_phases[i] += 0.3 * atmosphere.belief.mean[mode_idx - 1];
            }
            if mode_idx + 1 < atmosphere.n_modes {
                window_phases[i] += 0.3 * atmosphere.belief.mean[mode_idx + 1];
            }
        }

        window_phases
    }

    /// Predict multiple steps ahead
    ///
    /// For policy evaluation: what will happen if I take action u?
    pub fn multi_step_prediction(
        &self,
        model: &HierarchicalModel,
        actions: &[ControlAction],
    ) -> Vec<HierarchicalModel> {
        let mut trajectory = Vec::with_capacity(actions.len() + 1);
        trajectory.push(model.clone());

        let mut current = model.clone();
        for action in actions {
            self.predict(&mut current, action);
            trajectory.push(current.clone());
        }

        trajectory
    }

    /// Compute transition probability: p(x_{t+1} | x_t, u_t)
    ///
    /// For Gaussian processes: N(x_{t+1} | μ_pred, Σ_pred)
    pub fn transition_probability(
        &self,
        next_state: &Array1<f64>,
        current_belief: &GaussianBelief,
        action: &ControlAction,
    ) -> f64 {
        // Predict mean
        let predicted_mean = &current_belief.mean + &action.phase_correction;

        // Compute log-probability
        let residual = next_state - &predicted_mean;
        let mahalanobis = (&residual * &residual * &current_belief.precision).sum();
        let log_det = current_belief.variance.iter().map(|v| v.ln()).sum::<f64>();
        let n = next_state.len() as f64;

        let log_prob = -0.5 * (mahalanobis + n * (2.0 * PI).ln() + log_det);

        log_prob.exp()
    }

    /// Learn transition parameters online
    ///
    /// Update dynamics model from observed state transitions
    /// (x_t, u_t) → x_{t+1}
    pub fn update_from_data(
        &mut self,
        states: &[Array1<f64>],
        actions: &[ControlAction],
        next_states: &[Array1<f64>],
    ) {
        assert_eq!(states.len(), actions.len());
        assert_eq!(states.len(), next_states.len());

        // Parameter learning via recursive least squares
        // For now: placeholder (full implementation requires gradient descent)

        // Could estimate:
        // - Damping coefficient γ
        // - Diffusion coefficient D
        // - Coupling matrix C (from transfer entropy)
        // - Atmospheric decorrelation rate
    }
}

/// Prediction error for hierarchical levels
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// Error at Level 1 (windows)
    pub window_error: Array1<f64>,
    /// Error at Level 2 (atmosphere)
    pub atmosphere_error: Array1<f64>,
    /// Error at Level 3 (satellite)
    pub satellite_error: Array1<f64>,
}

impl PredictionError {
    /// Compute total squared error
    pub fn total_squared_error(&self) -> f64 {
        self.window_error.iter().map(|e| e * e).sum::<f64>()
            + self.atmosphere_error.iter().map(|e| e * e).sum::<f64>()
            + self.satellite_error.iter().map(|e| e * e).sum::<f64>()
    }

    /// RMS error
    pub fn rms_error(&self) -> f64 {
        let n = self.window_error.len() + self.atmosphere_error.len() + self.satellite_error.len();
        (self.total_squared_error() / n as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::super::hierarchical_model::constants;
    use super::*;

    #[test]
    fn test_transition_model_creation() {
        let model = TransitionModel::default_timescales();
        assert_eq!(model.dt_fast, 0.01);
        assert_eq!(model.dt_medium, 1.0);
        assert_eq!(model.dt_slow, 60.0);
        assert_eq!(model.substeps, 100);
    }

    #[test]
    fn test_satellite_evolution_conserves_energy() {
        let transition = TransitionModel::default_timescales();
        let mut level = SatelliteLevel::new(400.0);

        // Initial energy
        let r0 = (level.belief.mean[0].powi(2)
            + level.belief.mean[1].powi(2)
            + level.belief.mean[2].powi(2))
        .sqrt();
        let v0 = (level.belief.mean[3].powi(2)
            + level.belief.mean[4].powi(2)
            + level.belief.mean[5].powi(2))
        .sqrt();
        let e0 = 0.5 * v0 * v0 - level.mu / r0;

        // Evolve
        for _ in 0..100 {
            transition.evolve_satellite(&mut level, transition.dt_slow);
        }

        // Final energy
        let r1 = (level.belief.mean[0].powi(2)
            + level.belief.mean[1].powi(2)
            + level.belief.mean[2].powi(2))
        .sqrt();
        let v1 = (level.belief.mean[3].powi(2)
            + level.belief.mean[4].powi(2)
            + level.belief.mean[5].powi(2))
        .sqrt();
        let e1 = 0.5 * v1 * v1 - level.mu / r1;

        // Energy should be approximately conserved
        let energy_drift = (e1 - e0).abs() / e0.abs();
        assert!(energy_drift < 0.01); // <1% drift
    }

    #[test]
    fn test_atmospheric_evolution_stationarity() {
        let transition = TransitionModel::default_timescales();
        let mut level = AtmosphericLevel::new(100, 1e-14, 10.0);

        // Evolve for long time
        for _ in 0..1000 {
            transition.evolve_atmosphere(&mut level, transition.dt_medium);
        }

        // Variance should reach stationary value ≈ C_n²
        let mean_variance = level.belief.variance.mean().unwrap();
        assert!((mean_variance - level.c_n_squared).abs() / level.c_n_squared < 0.1);
    }

    #[test]
    fn test_window_evolution_with_damping() {
        let transition = TransitionModel::default_timescales();
        let mut level = WindowPhaseLevel::new(constants::N_WINDOWS, 300.0);
        let atmosphere = AtmosphericLevel::new(100, 1e-14, 10.0);

        // Set initial phase
        level.belief.mean[0] = 1.0;

        // Evolve without atmospheric driving or control
        let null_action = ControlAction::null(constants::N_WINDOWS);

        for _ in 0..100 {
            transition.evolve_windows(&mut level, &atmosphere, &null_action, transition.dt_fast);
        }

        // Phase should decay due to damping
        assert!(level.belief.mean[0].abs() < 0.5);
    }

    #[test]
    fn test_control_action_reduces_phase() {
        let transition = TransitionModel::default_timescales();
        let mut level = WindowPhaseLevel::new(constants::N_WINDOWS, 300.0);
        let atmosphere = AtmosphericLevel::new(100, 1e-14, 10.0);

        // Set aberrated phase
        level.belief.mean.fill(0.5);

        // Apply corrective action
        let mut action = ControlAction::null(constants::N_WINDOWS);
        action.phase_correction.fill(0.5); // Corrects aberration

        transition.evolve_windows(&mut level, &atmosphere, &action, transition.dt_fast);

        // Phase should be reduced (on average, accounting for drift + diffusion + numerical noise)
        let mean_abs_phase =
            level.belief.mean.iter().map(|p| p.abs()).sum::<f64>() / level.n_windows as f64;
        assert!(
            mean_abs_phase <= 0.50001,
            "Mean |phase| = {} should be ≤ 0.5 (with tolerance)",
            mean_abs_phase
        );
    }

    #[test]
    fn test_multi_step_prediction() {
        let transition = TransitionModel::default_timescales();
        let model = HierarchicalModel::new();

        let actions = vec![
            ControlAction::null(constants::N_WINDOWS),
            ControlAction::null(constants::N_WINDOWS),
            ControlAction::null(constants::N_WINDOWS),
        ];

        let trajectory = transition.multi_step_prediction(&model, &actions);

        assert_eq!(trajectory.len(), 4); // Initial + 3 steps
    }

    #[test]
    fn test_projection_atmosphere_to_windows() {
        let transition = TransitionModel::default_timescales();
        let mut atmosphere = AtmosphericLevel::new(100, 1e-14, 10.0);

        // Set non-zero turbulence
        atmosphere.belief.mean.fill(0.1);

        let window_phases = transition.project_atmosphere_to_windows(&atmosphere, 900);

        assert_eq!(window_phases.len(), 900);
        assert!(window_phases.iter().any(|&p| p.abs() > 0.01));
    }
}
