//! Reflexive feedback controller for MEC Phase M3.
//!
//! The reflexive controller regulates exploration versus strict governance modes
//! using the meta evolutionary metrics emitted by the orchestrator. It produces
//! a lattice snapshot of weighted free-energy values which downstream tooling
//! persists for audit and determinism manifests.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

const EPS: f64 = 1e-12;

/// Operating mode selected by the reflexive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceMode {
    Strict,
    Recovery,
    Exploration,
}

impl GovernanceMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            GovernanceMode::Strict => "strict",
            GovernanceMode::Recovery => "recovery",
            GovernanceMode::Exploration => "exploration",
        }
    }
}

/// Configuration tuning for the reflexive controller.
#[derive(Debug, Clone)]
pub struct ReflexiveConfig {
    pub lattice_edge: usize,
    pub strict_entropy_floor: f64,
    pub strict_divergence_cap: f64,
    pub exploration_entropy_floor: f64,
    pub exploration_divergence_cap: f64,
    pub strict_temperature_floor: f64,
    pub strict_temperature_cap: f64,
    pub recovery_temperature_floor: f64,
    pub recovery_temperature_cap: f64,
    pub exploration_temperature_floor: f64,
    pub exploration_temperature_cap: f64,
    pub strict_penalty: f64,
    pub recovery_mix: f64,
    pub exploration_sharpness: f64,
    pub exploration_weight_floor: f64,
    pub trend_window: usize,
    pub energy_trend_ceiling: f64,
}

impl Default for ReflexiveConfig {
    fn default() -> Self {
        Self {
            lattice_edge: 16,
            strict_entropy_floor: 1.05,
            strict_divergence_cap: 0.18,
            exploration_entropy_floor: 1.45,
            exploration_divergence_cap: 0.12,
            strict_temperature_floor: 0.85,
            strict_temperature_cap: 1.10,
            recovery_temperature_floor: 0.95,
            recovery_temperature_cap: 1.30,
            exploration_temperature_floor: 1.05,
            exploration_temperature_cap: 1.55,
            strict_penalty: 0.25,
            recovery_mix: 0.12,
            exploration_sharpness: 0.85,
            exploration_weight_floor: 0.18,
            trend_window: 12,
            energy_trend_ceiling: 0.075,
        }
    }
}

/// Lightweight view of the evolutionary metrics used by the reflexive engine.
#[derive(Debug, Clone)]
pub struct ReflexiveMetric {
    pub energy: f64,
    pub chromatic_loss: f64,
    pub divergence: f64,
    pub fitness: f64,
}

impl ReflexiveMetric {
    #[must_use]
    pub fn free_energy(&self) -> f64 {
        self.energy + self.chromatic_loss + self.divergence
    }
}

/// Snapshot emitted after reflexive regulation is applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexiveSnapshot {
    pub timestamp: DateTime<Utc>,
    pub mode: GovernanceMode,
    pub entropy: f64,
    pub divergence: f64,
    pub energy_mean: f64,
    pub energy_variance: f64,
    pub energy_trend: f64,
    pub exploration_ratio: f64,
    pub effective_temperature: f64,
    pub lattice_edge: usize,
    pub lattice: Vec<Vec<f64>>,
    pub alerts: Vec<String>,
}

impl ReflexiveSnapshot {
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let payload =
            serde_json::to_vec(self).expect("reflexive snapshot serialization should succeed");
        let mut hasher = Sha256::new();
        hasher.update(payload);
        hex::encode(hasher.finalize())
    }
}

/// Decision returned by the reflexive controller.
#[derive(Debug, Clone)]
pub struct ReflexiveDecision {
    pub distribution: Vec<f64>,
    pub temperature: f64,
    pub snapshot: ReflexiveSnapshot,
}

/// Reflexive controller maintains free-energy history to regulate MEC exploration.
#[derive(Debug)]
pub struct ReflexiveController {
    config: ReflexiveConfig,
    energy_history: VecDeque<f64>,
}

impl ReflexiveController {
    pub fn new(config: ReflexiveConfig) -> Self {
        Self {
            energy_history: VecDeque::with_capacity(config.trend_window.max(2)),
            config,
        }
    }

    #[must_use]
    pub fn config(&self) -> &ReflexiveConfig {
        &self.config
    }

    pub fn evaluate(
        &mut self,
        metrics: &[ReflexiveMetric],
        distribution: &[f64],
        base_temperature: f64,
    ) -> ReflexiveDecision {
        if metrics.is_empty() || distribution.is_empty() {
            return ReflexiveDecision {
                distribution: vec![],
                temperature: base_temperature,
                snapshot: self.empty_snapshot(base_temperature),
            };
        }

        let weights = normalize_distribution(distribution);
        let entropy = shannon_entropy(&weights);
        let free_energy: Vec<f64> = metrics.iter().map(ReflexiveMetric::free_energy).collect();
        let divergence_values: Vec<f64> = metrics.iter().map(|m| m.divergence).collect();
        let divergence = weighted_mean(&divergence_values, &weights);
        let energy_mean = weighted_mean(&free_energy, &weights);
        let energy_variance = weighted_variance(&free_energy, energy_mean, &weights);
        let exploration_ratio = weights
            .iter()
            .filter(|w| **w >= self.config.exploration_weight_floor)
            .sum::<f64>()
            .clamp(0.0, 1.0);

        let previous_avg = if self.energy_history.is_empty() {
            energy_mean
        } else {
            self.energy_history.iter().copied().sum::<f64>() / self.energy_history.len() as f64
        };
        self.energy_history.push_back(energy_mean);
        if self.energy_history.len() > self.config.trend_window {
            self.energy_history.pop_front();
        }
        let energy_trend = energy_mean - previous_avg;

        let mut alerts = Vec::new();
        let mut mode = GovernanceMode::Recovery;

        if divergence > self.config.strict_divergence_cap {
            alerts.push(format!(
                "divergence {:.3} exceeded cap {:.3}",
                divergence, self.config.strict_divergence_cap
            ));
            mode = GovernanceMode::Strict;
        }
        if entropy < self.config.strict_entropy_floor {
            alerts.push(format!(
                "entropy {:.3} below floor {:.3}",
                entropy, self.config.strict_entropy_floor
            ));
            mode = GovernanceMode::Strict;
        }
        if energy_trend > self.config.energy_trend_ceiling {
            alerts.push(format!(
                "energy trend {:.3} exceeded ceiling {:.3}",
                energy_trend, self.config.energy_trend_ceiling
            ));
            mode = GovernanceMode::Strict;
        }
        if mode != GovernanceMode::Strict
            && entropy >= self.config.exploration_entropy_floor
            && divergence <= self.config.exploration_divergence_cap
        {
            mode = GovernanceMode::Exploration;
        }

        let mut regulated = weights.clone();
        match mode {
            GovernanceMode::Strict => {
                for (idx, metric) in metrics.iter().enumerate() {
                    if metric.divergence > self.config.strict_divergence_cap {
                        regulated[idx] *= self.config.strict_penalty;
                    }
                }
                regulated = normalize_distribution(&regulated);
            }
            GovernanceMode::Recovery => {
                let mix = self.config.recovery_mix.clamp(0.0, 0.5);
                let uniform = 1.0 / regulated.len() as f64;
                for value in regulated.iter_mut() {
                    *value = (*value) * (1.0 - mix) + uniform * mix;
                }
                regulated = normalize_distribution(&regulated);
            }
            GovernanceMode::Exploration => {
                let gamma = self.config.exploration_sharpness.clamp(0.2, 2.0);
                let mut powered = Vec::with_capacity(regulated.len());
                for value in regulated.iter() {
                    powered.push(value.powf(gamma));
                }
                regulated = normalize_distribution(&powered);
            }
        }

        let temperature = match mode {
            GovernanceMode::Strict => base_temperature.clamp(
                self.config.strict_temperature_floor,
                self.config.strict_temperature_cap,
            ),
            GovernanceMode::Recovery => base_temperature.clamp(
                self.config.recovery_temperature_floor,
                self.config.recovery_temperature_cap,
            ),
            GovernanceMode::Exploration => base_temperature.clamp(
                self.config.exploration_temperature_floor,
                self.config.exploration_temperature_cap,
            ),
        };

        let lattice = build_lattice(&free_energy, &regulated, self.config.lattice_edge);

        let snapshot = ReflexiveSnapshot {
            timestamp: Utc::now(),
            mode,
            entropy,
            divergence,
            energy_mean,
            energy_variance,
            energy_trend,
            exploration_ratio,
            effective_temperature: temperature,
            lattice_edge: self.config.lattice_edge,
            lattice,
            alerts,
        };

        ReflexiveDecision {
            distribution: regulated,
            temperature,
            snapshot,
        }
    }

    fn empty_snapshot(&self, base_temperature: f64) -> ReflexiveSnapshot {
        ReflexiveSnapshot {
            timestamp: Utc::now(),
            mode: GovernanceMode::Strict,
            entropy: 0.0,
            divergence: 0.0,
            energy_mean: 0.0,
            energy_variance: 0.0,
            energy_trend: 0.0,
            exploration_ratio: 0.0,
            effective_temperature: base_temperature,
            lattice_edge: self.config.lattice_edge,
            lattice: vec![vec![0.0; self.config.lattice_edge]; self.config.lattice_edge],
            alerts: vec!["no_metrics_available".into()],
        }
    }
}

impl Default for ReflexiveController {
    fn default() -> Self {
        Self::new(ReflexiveConfig::default())
    }
}

fn normalize_distribution(values: &[f64]) -> Vec<f64> {
    let mut result = values.iter().map(|v| v.max(0.0)).collect::<Vec<f64>>();
    let sum: f64 = result.iter().sum();
    if sum > EPS {
        for value in result.iter_mut() {
            *value /= sum;
        }
    } else {
        let uniform = 1.0 / result.len().max(1) as f64;
        for value in result.iter_mut() {
            *value = uniform;
        }
    }
    result
}

fn shannon_entropy(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .map(|p| {
            let p = p.clamp(EPS, 1.0);
            -p * p.ln()
        })
        .sum()
}

fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut normalizer = 0.0;
    for (value, weight) in values.iter().zip(weights.iter()) {
        acc += value * weight;
        normalizer += weight;
    }
    if normalizer <= EPS {
        0.0
    } else {
        acc / normalizer
    }
}

fn weighted_variance(values: &[f64], mean: f64, weights: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut normalizer = 0.0;
    for (value, weight) in values.iter().zip(weights.iter()) {
        let delta = value - mean;
        acc += weight * delta * delta;
        normalizer += weight;
    }
    if normalizer <= EPS {
        0.0
    } else {
        acc / normalizer
    }
}

fn build_lattice(values: &[f64], weights: &[f64], edge: usize) -> Vec<Vec<f64>> {
    let mut lattice = vec![vec![0.0; edge]; edge];
    if edge == 0 {
        return lattice;
    }

    for (idx, (value, weight)) in values.iter().zip(weights.iter()).enumerate() {
        let x = idx % edge;
        let y = (idx / edge) % edge;
        lattice[y][x] += value * weight;
    }

    // Fill remaining empty cells with a mild decay to avoid zero planes.
    for row in lattice.iter_mut() {
        let mut last = 0.0;
        for cell in row.iter_mut() {
            if cell.abs() < EPS {
                *cell = last * 0.92;
            } else {
                last = *cell;
            }
        }
    }
    lattice
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_distribution_handles_zero_sum() {
        let normalized = normalize_distribution(&[0.0, 0.0, 0.0]);
        assert!((normalized.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_fingerprint_changes_with_values() {
        let mut ctrl = ReflexiveController::default();
        let metrics = vec![
            ReflexiveMetric {
                energy: 0.2,
                chromatic_loss: 0.1,
                divergence: 0.05,
                fitness: 0.9,
            },
            ReflexiveMetric {
                energy: 0.6,
                chromatic_loss: 0.2,
                divergence: 0.08,
                fitness: 0.5,
            },
        ];
        let decision = ctrl.evaluate(&metrics, &[0.6, 0.4], 1.2);
        let fingerprint_a = decision.snapshot.fingerprint();
        let decision_b = ctrl.evaluate(&metrics, &[0.7, 0.3], 1.2);
        let fingerprint_b = decision_b.snapshot.fingerprint();
        assert_ne!(fingerprint_a, fingerprint_b);
    }

    #[test]
    fn divergence_exceeds_cap_triggers_strict_mode() {
        let mut ctrl = ReflexiveController::default();
        let metrics = vec![
            ReflexiveMetric {
                energy: 0.1,
                chromatic_loss: 0.2,
                divergence: 0.25,
                fitness: 0.7,
            },
            ReflexiveMetric {
                energy: 0.15,
                chromatic_loss: 0.18,
                divergence: 0.24,
                fitness: 0.6,
            },
        ];
        let decision = ctrl.evaluate(&metrics, &[0.5, 0.5], 1.1);
        assert_eq!(decision.snapshot.mode, GovernanceMode::Strict);
        assert!(
            decision
                .snapshot
                .alerts
                .iter()
                .any(|alert| alert.contains("divergence")),
            "expected divergence alert, got {:?}",
            decision.snapshot.alerts
        );
    }

    #[test]
    fn low_entropy_distribution_enforces_strict_mode() {
        let mut ctrl = ReflexiveController::default();
        let metrics = vec![
            ReflexiveMetric {
                energy: 0.05,
                chromatic_loss: 0.05,
                divergence: 0.02,
                fitness: 0.8,
            },
            ReflexiveMetric {
                energy: 0.07,
                chromatic_loss: 0.04,
                divergence: 0.02,
                fitness: 0.7,
            },
        ];
        let decision = ctrl.evaluate(&metrics, &[0.99, 0.01], 1.1);
        assert_eq!(decision.snapshot.mode, GovernanceMode::Strict);
        assert!(
            decision
                .snapshot
                .alerts
                .iter()
                .any(|alert| alert.contains("entropy")),
            "expected entropy alert, got {:?}",
            decision.snapshot.alerts
        );
    }

    #[test]
    fn exploration_requires_entropy_and_low_divergence() {
        let mut config = ReflexiveConfig::default();
        config.strict_entropy_floor = 0.5;
        config.exploration_entropy_floor = 1.0;
        config.exploration_divergence_cap = 0.2;
        let mut ctrl = ReflexiveController::new(config);

        let metrics = vec![
            ReflexiveMetric {
                energy: 0.1,
                chromatic_loss: 0.15,
                divergence: 0.05,
                fitness: 0.8,
            },
            ReflexiveMetric {
                energy: 0.12,
                chromatic_loss: 0.12,
                divergence: 0.04,
                fitness: 0.78,
            },
            ReflexiveMetric {
                energy: 0.09,
                chromatic_loss: 0.11,
                divergence: 0.03,
                fitness: 0.76,
            },
        ];

        let decision = ctrl.evaluate(&metrics, &[0.34, 0.33, 0.33], 1.2);
        assert_eq!(decision.snapshot.mode, GovernanceMode::Exploration);
        assert!(
            decision
                .distribution
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "distribution must remain normalized, got {:?}",
            decision.distribution
        );
    }

    #[test]
    fn recovery_mix_softens_distribution() {
        let mut config = ReflexiveConfig::default();
        config.recovery_mix = 0.2;
        config.strict_divergence_cap = 0.5;
        config.strict_entropy_floor = 0.1;
        config.exploration_entropy_floor = 10.0; // force recovery instead of exploration
        let mut ctrl = ReflexiveController::new(config);

        let metrics = vec![
            ReflexiveMetric {
                energy: 0.1,
                chromatic_loss: 0.2,
                divergence: 0.05,
                fitness: 0.9,
            },
            ReflexiveMetric {
                energy: 0.08,
                chromatic_loss: 0.18,
                divergence: 0.04,
                fitness: 0.85,
            },
        ];

        let decision = ctrl.evaluate(&metrics, &[0.9, 0.1], 1.0);
        assert_eq!(decision.snapshot.mode, GovernanceMode::Recovery);
        let diff = (decision.distribution[0] - decision.distribution[1]).abs();
        assert!(
            diff < 0.7,
            "recovery mix should soften distribution gap, diff {diff}, distribution {:?}",
            decision.distribution
        );
    }
}
