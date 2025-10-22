//! Reflexive feedback controller and free-energy lattice snapshots.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReflexiveMode {
    Strict,
    Explore,
    Recovery,
}

impl ReflexiveMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReflexiveMode::Strict => "strict",
            ReflexiveMode::Explore => "explore",
            ReflexiveMode::Recovery => "recovery",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexiveObservation {
    pub free_energy: f64,
    pub exploration: f64,
    pub exploitation: f64,
    pub divergence: f64,
    pub confidence: f64,
    pub temperature: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexiveState {
    pub free_energy: f64,
    pub exploration: f64,
    pub exploitation: f64,
    pub divergence: f64,
    pub confidence: f64,
    pub temperature: f64,
    pub mode: ReflexiveMode,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeSnapshot {
    pub state: ReflexiveState,
    pub gradient: [f64; 3],
    pub stability: f64,
    pub entropy: f64,
    pub hash: String,
}

impl LatticeSnapshot {
    pub fn hash_snapshot(state: &ReflexiveState, gradient: [f64; 3], entropy: f64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(state.free_energy.to_le_bytes());
        hasher.update(state.exploration.to_le_bytes());
        hasher.update(state.exploitation.to_le_bytes());
        hasher.update(state.divergence.to_le_bytes());
        hasher.update(state.confidence.to_le_bytes());
        hasher.update(state.temperature.to_le_bytes());
        hasher.update(
            state
                .timestamp
                .timestamp_nanos_opt()
                .unwrap_or_default()
                .to_le_bytes(),
        );
        hasher.update(entropy.clamp(0.0, 1.0).to_le_bytes());
        for value in gradient {
            hasher.update(value.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

#[derive(Debug)]
pub struct ReflexiveController {
    history: Vec<ReflexiveState>,
    window: usize,
    smoothing: f64,
}

impl Default for ReflexiveController {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            window: 24,
            smoothing: 0.15,
        }
    }
}

impl ReflexiveController {
    pub fn new(window: usize, smoothing: f64) -> Self {
        Self {
            history: Vec::with_capacity(window),
            window: window.max(4),
            smoothing: smoothing.clamp(0.05, 0.5),
        }
    }

    pub fn observe(&mut self, obs: ReflexiveObservation) -> LatticeSnapshot {
        let prev_state = self.history.last().cloned();
        let mode = self.select_mode(&obs, prev_state.as_ref());
        let filtered_confidence = self.smooth(
            prev_state
                .as_ref()
                .map(|s| s.confidence)
                .unwrap_or(obs.confidence),
            obs.confidence,
        );
        let filtered_free_energy = self.smooth(
            prev_state
                .as_ref()
                .map(|s| s.free_energy)
                .unwrap_or(obs.free_energy),
            obs.free_energy,
        );
        let state = ReflexiveState {
            free_energy: filtered_free_energy,
            exploration: obs.exploration.clamp(0.0, 1.0),
            exploitation: obs.exploitation.clamp(0.0, 1.0),
            divergence: obs.divergence.max(0.0),
            confidence: filtered_confidence.clamp(0.0, 1.0),
            temperature: obs.temperature,
            mode,
            timestamp: obs.timestamp,
        };

        let gradient = if let Some(prev) = prev_state.as_ref() {
            [
                state.free_energy - prev.free_energy,
                state.exploration - prev.exploration,
                state.exploitation - prev.exploitation,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        if self.history.len() == self.window {
            self.history.remove(0);
        }
        self.history.push(state.clone());

        let entropy = entropy_from_exploration(state.exploration);
        let stability = 1.0 - state.divergence.tanh().clamp(0.0, 1.0);
        let hash = LatticeSnapshot::hash_snapshot(&state, gradient, entropy);
        LatticeSnapshot {
            state,
            gradient,
            stability,
            entropy,
            hash,
        }
    }

    pub fn history(&self) -> &[ReflexiveState] {
        &self.history
    }

    fn smooth(&self, previous: f64, current: f64) -> f64 {
        (1.0 - self.smoothing) * previous + self.smoothing * current
    }

    fn select_mode(
        &self,
        obs: &ReflexiveObservation,
        prev: Option<&ReflexiveState>,
    ) -> ReflexiveMode {
        if obs.exploration > 0.65 {
            ReflexiveMode::Explore
        } else if let Some(prev_state) = prev {
            let delta = obs.free_energy - prev_state.free_energy;
            if delta > 0.08 && obs.confidence < 0.5 {
                ReflexiveMode::Recovery
            } else {
                ReflexiveMode::Strict
            }
        } else {
            ReflexiveMode::Strict
        }
    }
}

fn entropy_from_exploration(exploration: f64) -> f64 {
    exploration.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_hash_deterministic() {
        let mut controller = ReflexiveController::default();
        let obs = ReflexiveObservation {
            free_energy: 1.23,
            exploration: 0.4,
            exploitation: 0.6,
            divergence: 0.2,
            confidence: 0.7,
            temperature: 0.9,
            timestamp: Utc::now(),
        };
        let snapshot = controller.observe(obs.clone());
        let mut controller_again = ReflexiveController::default();
        let snapshot_again = controller_again.observe(obs);
        assert_eq!(snapshot.hash, snapshot_again.hash);
    }

    #[test]
    fn history_keeps_window() {
        let mut controller = ReflexiveController::new(5, 0.2);
        let base = ReflexiveObservation {
            free_energy: 0.5,
            exploration: 0.3,
            exploitation: 0.7,
            divergence: 0.1,
            confidence: 0.9,
            temperature: 0.4,
            timestamp: Utc::now(),
        };

        for idx in 0..10 {
            let mut obs = base.clone();
            obs.free_energy += idx as f64 * 0.01;
            obs.timestamp = base.timestamp + chrono::Duration::seconds(idx as i64);
            controller.observe(obs);
        }

        assert_eq!(controller.history().len(), 5);
    }
}
