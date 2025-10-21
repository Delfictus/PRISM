use crate::features::{registry, MetaFeatureId, MetaFeatureState};
use chrono::{DateTime, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::sync::Arc;
use thiserror::Error;

type Result<T> = std::result::Result<T, OrchestratorError>;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("meta_generation flag must be shadow or enabled before orchestrator can run")]
    MetaGenerationDisabled,

    #[error("population size must be > 0")]
    EmptyPopulation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantGenome {
    pub seed: u64,
    pub parameters: BTreeMap<String, VariantParameter>,
    pub feature_toggles: BTreeMap<String, bool>,
    pub hash: String,
}

impl VariantGenome {
    pub fn new(
        seed: u64,
        parameters: BTreeMap<String, VariantParameter>,
        feature_toggles: BTreeMap<String, bool>,
    ) -> Self {
        let hash = Self::compute_hash(seed, &parameters, &feature_toggles);
        Self {
            seed,
            parameters,
            feature_toggles,
            hash,
        }
    }

    fn compute_hash(
        seed: u64,
        parameters: &BTreeMap<String, VariantParameter>,
        toggles: &BTreeMap<String, bool>,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(seed.to_be_bytes());
        let param_bytes = serde_json::to_vec(parameters).expect("serialize parameters");
        hasher.update(param_bytes);
        let toggle_bytes = serde_json::to_vec(toggles).expect("serialize toggles");
        hasher.update(toggle_bytes);
        hex::encode(hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VariantParameter {
    Continuous { value: f64, min: f64, max: f64 },
    Discrete { value: i64, min: i64, max: i64 },
    Categorical { value: String, choices: Vec<String> },
}

impl Display for VariantParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariantParameter::Continuous { value, .. } => write!(f, "{:.6}", value),
            VariantParameter::Discrete { value, .. } => write!(f, "{}", value),
            VariantParameter::Categorical { value, .. } => f.write_str(value),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPlan {
    pub generated_at: DateTime<Utc>,
    pub base_seed: u64,
    pub genomes: Vec<VariantGenome>,
}

#[derive(Clone)]
pub struct MetaOrchestrator {
    rng: Arc<std::sync::Mutex<StdRng>>,
    schedule_dimension: usize,
}

impl MetaOrchestrator {
    pub fn new(seed: u64) -> Result<Self> {
        ensure_meta_generation_enabled()?;
        Ok(Self {
            rng: Arc::new(std::sync::Mutex::new(StdRng::seed_from_u64(seed))),
            schedule_dimension: 5,
        })
    }

    pub fn schedule_population(&self, base_seed: u64, population: usize) -> Result<EvolutionPlan> {
        if population == 0 {
            return Err(OrchestratorError::EmptyPopulation);
        }
        let mut genomes = Vec::with_capacity(population);
        for index in 0..population {
            let genome_seed = mix_seed(base_seed, index as u64);
            let parameters = self.generate_parameters(genome_seed, index as u64);
            let toggles = self.generate_feature_toggles(genome_seed);
            genomes.push(VariantGenome::new(genome_seed, parameters, toggles));
        }
        Ok(EvolutionPlan {
            generated_at: Utc::now(),
            base_seed,
            genomes,
        })
    }

    fn generate_parameters(
        &self,
        genome_seed: u64,
        ordinal: u64,
    ) -> BTreeMap<String, VariantParameter> {
        let mut map = BTreeMap::new();
        let coordinates = halton_point(ordinal + 1, self.schedule_dimension as u64);
        map.insert(
            "annealing.beta".into(),
            VariantParameter::Continuous {
                value: remap_unit_interval(coordinates[0], 0.25, 5.0),
                min: 0.25,
                max: 5.0,
            },
        );
        map.insert(
            "ensemble.replicas".into(),
            VariantParameter::Discrete {
                value: remap_unit_interval(coordinates[1], 64.0, 4096.0).round() as i64,
                min: 64,
                max: 4096,
            },
        );
        let categories = vec![
            "density_aware",
            "thermodynamic",
            "quantum_bias",
            "neuromorphic_phase",
        ];
        let categorical_index = (coordinates[2] * categories.len() as f64).floor() as usize;
        map.insert(
            "fusion.strategy".into(),
            VariantParameter::Categorical {
                value: categories[categorical_index.min(categories.len() - 1)].to_string(),
                choices: categories.iter().map(|s| s.to_string()).collect(),
            },
        );
        map.insert(
            "refinement.iterations".into(),
            VariantParameter::Discrete {
                value: remap_unit_interval(coordinates[3], 512.0, 8192.0).round() as i64,
                min: 512,
                max: 8192,
            },
        );
        map.insert(
            "mutation.strength".into(),
            VariantParameter::Continuous {
                value: remap_unit_interval(coordinates[4], 0.01, 0.45),
                min: 0.01,
                max: 0.45,
            },
        );

        // Deterministic jitter for diversity using RNG seeded per genome.
        let mut guard = self.rng.lock().expect("rng lock");
        let mut local_rng = StdRng::seed_from_u64(genome_seed ^ guard.gen_range(0..u64::MAX));
        drop(guard);
        map.insert(
            "mutation.temperature".into(),
            VariantParameter::Continuous {
                value: remap_unit_interval(local_rng.gen::<f64>(), 0.75, 1.75),
                min: 0.75,
                max: 1.75,
            },
        );
        map
    }

    fn generate_feature_toggles(&self, genome_seed: u64) -> BTreeMap<String, bool> {
        let mut toggles = BTreeMap::new();
        let mut guard = self.rng.lock().expect("rng lock");
        let mut local_rng = StdRng::seed_from_u64(genome_seed.rotate_left(13));
        toggles.insert(
            "use_quantum_bias".into(),
            local_rng.gen_bool(0.5) && guard.gen_bool(0.8),
        );
        toggles.insert(
            "enable_neuromorphic_feedback".into(),
            local_rng.gen_bool(0.6) || guard.gen_bool(0.2),
        );
        toggles.insert("tensor_core_prefetch".into(), guard.gen_bool(0.7));
        toggles
    }
}

fn ensure_meta_generation_enabled() -> Result<()> {
    let registry = registry();
    if registry.is_enabled(MetaFeatureId::MetaGeneration) {
        return Ok(());
    }
    // Allow shadow/gradual states to proceed during Phase M0.
    let snapshot = registry.snapshot();
    let meta_record = snapshot
        .records
        .iter()
        .find(|record| record.id == MetaFeatureId::MetaGeneration)
        .expect("meta_generation flag present");
    match &meta_record.state {
        MetaFeatureState::Shadow { .. }
        | MetaFeatureState::Gradual { .. }
        | MetaFeatureState::Enabled { .. } => Ok(()),
        MetaFeatureState::Disabled => Err(OrchestratorError::MetaGenerationDisabled),
    }
}

fn mix_seed(seed: u64, offset: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(offset.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn remap_unit_interval(x: f64, min: f64, max: f64) -> f64 {
    min + (max - min) * x.clamp(0.0, 0.999_999)
}

fn halton_point(index: u64, dimension: u64) -> Vec<f64> {
    const PRIMES: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    (0..dimension)
        .map(|d| radical_inverse(index, PRIMES[d as usize % PRIMES.len()]))
        .collect()
}

fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    while index > 0 {
        result += f * (index % base) as f64;
        index /= base;
        f /= base as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_sequence_is_low_discrepancy() {
        let pts: Vec<f64> = (1..5).map(|i| radical_inverse(i, 2)).collect();
        assert!(pts[0] < pts[1]);
        assert!(pts[2] < pts[3]);
    }
}
