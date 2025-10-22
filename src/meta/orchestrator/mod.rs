use crate::features::{registry, MetaFeatureId, MetaFeatureState};
use crate::governance::determinism::{DeterminismProof, DeterminismRecorder, MetaDeterminism};
use crate::meta::reflexive::{
    ReflexiveController, ReflexiveDecision, ReflexiveMetric, ReflexiveSnapshot,
};
use crate::telemetry::{
    ComponentId, EventData, EventLevel, Metrics, TelemetryEntry, TelemetrySink,
};
use chrono::{DateTime, Utc};
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fmt::{self, Display};
use std::sync::Arc;
use thiserror::Error;

const HALTON_PRIMES: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
const SOBOL_DIR_V: [u32; 32] = [
    0b10000000000000000000000000000000,
    0b11000000000000000000000000000000,
    0b10100000000000000000000000000000,
    0b10010000000000000000000000000000,
    0b10001000000000000000000000000000,
    0b10000100000000000000000000000000,
    0b10000010000000000000000000000000,
    0b10000001000000000000000000000000,
    0b10000000100000000000000000000000,
    0b10000000010000000000000000000000,
    0b10000000001000000000000000000000,
    0b10000000000100000000000000000000,
    0b10000000000010000000000000000000,
    0b10000000000001000000000000000000,
    0b10000000000000100000000000000000,
    0b10000000000000010000000000000000,
    0b10000000000000001000000000000000,
    0b10000000000000000100000000000000,
    0b10000000000000000010000000000000,
    0b10000000000000000001000000000000,
    0b10000000000000000000100000000000,
    0b10000000000000000000010000000000,
    0b10000000000000000000001000000000,
    0b10000000000000000000000100000000,
    0b10000000000000000000000010000000,
    0b10000000000000000000000001000000,
    0b10000000000000000000000000100000,
    0b10000000000000000000000000010000,
    0b10000000000000000000000000001000,
    0b10000000000000000000000000000100,
    0b10000000000000000000000000000010,
    0b10000000000000000000000000000001,
];

const BASELINE_VECTOR: Vector3<f64> = Vector3::new(0.6, 0.7, 0.1);
const REPLICATOR_STEPS: usize = 18;
const LATTICE_EDGE: usize = 16;

type Result<T> = std::result::Result<T, OrchestratorError>;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("meta_generation flag must be shadow or enabled before orchestrator can run")]
    MetaGenerationDisabled,
    #[error("population size must be > 0")]
    EmptyPopulation,
    #[error("missing parameter {0}")]
    MissingParameter(String),
    #[error("determinism recorder error: {0}")]
    Recorder(String),
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
            VariantParameter::Continuous { value, .. } => write!(f, "{value:.6}"),
            VariantParameter::Discrete { value, .. } => write!(f, "{value}"),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    pub energy: f64,
    pub chromatic_loss: f64,
    pub divergence: f64,
    pub weights: [f64; 3],
    pub vector: [f64; 3],
    pub scalar: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantEvaluation {
    pub genome: VariantGenome,
    pub metrics: EvolutionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionOutcome {
    pub plan: EvolutionPlan,
    pub evaluations: Vec<VariantEvaluation>,
    pub distribution: Vec<f64>,
    pub temperature: f64,
    pub reflexive: ReflexiveSnapshot,
    pub best_index: usize,
    pub determinism_proof: DeterminismProof,
    pub meta: MetaDeterminism,
    pub telemetry_entry: TelemetryEntry,
}

#[derive(Clone)]
pub struct MetaOrchestrator {
    rng: Arc<std::sync::Mutex<StdRng>>,
    schedule_dimension: usize,
    telemetry: TelemetrySink,
    reflexive: Arc<std::sync::Mutex<ReflexiveController>>,
}

impl MetaOrchestrator {
    pub fn new(seed: u64) -> Result<Self> {
        ensure_meta_generation_enabled()?;
        Ok(Self {
            rng: Arc::new(std::sync::Mutex::new(StdRng::seed_from_u64(seed))),
            schedule_dimension: 5,
            telemetry: TelemetrySink::new("meta_orchestrator"),
            reflexive: Arc::new(std::sync::Mutex::new(ReflexiveController::default())),
        })
    }

    pub fn run_generation(&self, base_seed: u64, population: usize) -> Result<EvolutionOutcome> {
        let plan = self.schedule_population(base_seed, population)?;
        let evaluations = self.evaluate_population(&plan)?;
        let (raw_distribution, raw_temperature) = replicator_dynamics(&evaluations);
        let metrics: Vec<ReflexiveMetric> = evaluations
            .iter()
            .map(|eval| ReflexiveMetric {
                energy: eval.metrics.energy,
                chromatic_loss: eval.metrics.chromatic_loss,
                divergence: eval.metrics.divergence,
                fitness: eval.metrics.scalar,
            })
            .collect();

        let ReflexiveDecision {
            distribution,
            temperature,
            snapshot: reflexive_snapshot,
        } = {
            let mut guard = self
                .reflexive
                .lock()
                .expect("reflexive controller lock poisoned");
            guard.evaluate(&metrics, &raw_distribution, raw_temperature)
        };

        let best_index = evaluations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.metrics.scalar.total_cmp(&b.1.metrics.scalar))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best = &evaluations[best_index];
        let manifest = registry().snapshot();
        let free_energy_hash = compute_free_energy_hash(&evaluations, temperature);
        let meta = MetaDeterminism {
            meta_genome_hash: best.genome.hash.clone(),
            meta_merkle_root: manifest.merkle_root.clone(),
            ontology_hash: None,
            free_energy_hash: Some(free_energy_hash.clone()),
            reflexive_mode: Some(reflexive_snapshot.mode.as_str().to_string()),
            lattice_fingerprint: Some(reflexive_snapshot.fingerprint()),
        };

        let mut recorder = DeterminismRecorder::new(plan.base_seed);
        recorder
            .record_input(&plan)
            .map_err(|err| OrchestratorError::Recorder(err.to_string()))?;
        recorder
            .record_intermediate("meta.evaluations", &evaluations)
            .map_err(|err| OrchestratorError::Recorder(err.to_string()))?;
        recorder
            .record_intermediate("meta.distribution", &distribution)
            .map_err(|err| OrchestratorError::Recorder(err.to_string()))?;
        recorder
            .record_output(&best.genome)
            .map_err(|err| OrchestratorError::Recorder(err.to_string()))?;
        recorder.attach_meta(meta.clone());
        let proof = recorder.finalize();

        let telemetry_entry = self.emit_telemetry(
            &plan,
            &evaluations,
            &distribution,
            temperature,
            &meta,
            &reflexive_snapshot,
        );

        Ok(EvolutionOutcome {
            plan,
            evaluations,
            distribution,
            temperature,
            reflexive: reflexive_snapshot,
            best_index,
            determinism_proof: proof,
            meta,
            telemetry_entry,
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

    fn evaluate_population(&self, plan: &EvolutionPlan) -> Result<Vec<VariantEvaluation>> {
        let mut evaluations = Vec::with_capacity(plan.genomes.len());
        for genome in &plan.genomes {
            let metrics = evaluate_genome(genome)?;
            evaluations.push(VariantEvaluation {
                genome: genome.clone(),
                metrics,
            });
        }
        Ok(evaluations)
    }

    fn generate_parameters(
        &self,
        genome_seed: u64,
        ordinal: u64,
    ) -> BTreeMap<String, VariantParameter> {
        let mut map = BTreeMap::new();
        let mut halton = halton_point(ordinal + 1, self.schedule_dimension as u64);
        refine_with_sobol(&mut halton, ordinal + 1);

        map.insert(
            "annealing.beta".into(),
            VariantParameter::Continuous {
                value: remap_unit_interval(halton[0], 0.25, 5.0),
                min: 0.25,
                max: 5.0,
            },
        );
        map.insert(
            "ensemble.replicas".into(),
            VariantParameter::Discrete {
                value: remap_unit_interval(halton[1], 64.0, 4096.0).round() as i64,
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
        let categorical_index = (halton[2] * categories.len() as f64).floor() as usize;
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
                value: remap_unit_interval(halton[3], 512.0, 8192.0).round() as i64,
                min: 512,
                max: 8192,
            },
        );
        map.insert(
            "mutation.strength".into(),
            VariantParameter::Continuous {
                value: remap_unit_interval(halton[4], 0.01, 0.45),
                min: 0.01,
                max: 0.45,
            },
        );

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

    fn emit_telemetry(
        &self,
        plan: &EvolutionPlan,
        evaluations: &[VariantEvaluation],
        distribution: &[f64],
        temperature: f64,
        meta: &MetaDeterminism,
        reflexive: &ReflexiveSnapshot,
    ) -> TelemetryEntry {
        let best = evaluations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.metrics.scalar.total_cmp(&b.1.metrics.scalar))
            .map(|(_, eval)| eval)
            .expect("at least one evaluation");

        let payload = serde_json::json!({
            "meta_variant": {
                "genome_hash": best.genome.hash,
                "determinism_manifest": meta.meta_merkle_root,
                "flags": best
                    .genome
                    .feature_toggles
                    .iter()
                    .filter(|(_, v)| **v)
                    .map(|(k, _)| k.clone())
                    .collect::<Vec<_>>(),
                "free_energy": {
                    "lattice_norm": best.metrics.energy,
                    "mode_confidence": best.metrics.chromatic_loss,
                    "divergence": best.metrics.divergence,
                },
                "distribution_entropy": shannon_entropy(distribution),
                "temperature": temperature,
                "reflexive": {
                    "mode": reflexive.mode.as_str(),
                    "entropy": reflexive.entropy,
                    "divergence": reflexive.divergence,
                    "trend": reflexive.energy_trend,
                    "alerts": reflexive.alerts,
                },
                "plan": {
                    "population": plan.genomes.len(),
                    "base_seed": plan.base_seed,
                }
            }
        });

        let entry = TelemetryEntry::new(
            ComponentId::Orchestrator,
            EventLevel::Info,
            EventData::Custom { payload },
            None,
            Some(Metrics::default()),
        );
        self.telemetry.log(&entry);
        entry
    }
}

fn ensure_meta_generation_enabled() -> Result<()> {
    if env::var("PRISM_ALLOW_META_DISABLED")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
    {
        return Ok(());
    }
    let registry = registry();
    if registry.is_enabled(MetaFeatureId::MetaGeneration) {
        return Ok(());
    }
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

fn evaluate_genome(genome: &VariantGenome) -> Result<EvolutionMetrics> {
    let beta = get_continuous(genome, "annealing.beta")?;
    let replicas = get_discrete(genome, "ensemble.replicas")? as f64;
    let iterations = get_discrete(genome, "refinement.iterations")? as f64;
    let mutation_strength = get_continuous(genome, "mutation.strength")?;
    let mutation_temperature = get_continuous(genome, "mutation.temperature")?;

    let energy = simulate_ising_energy(genome.hash.as_bytes(), beta);
    let chromatic = chromatic_surrogate(
        beta,
        replicas,
        iterations,
        mutation_strength,
        mutation_temperature,
    );
    let divergence = free_energy_divergence(beta, mutation_strength, mutation_temperature);

    let weights = adaptive_weights(beta, replicas, iterations);
    let vector = Vector3::new(-energy, -chromatic, -divergence);
    let scalar = weights.dot(&vector);

    Ok(EvolutionMetrics {
        energy,
        chromatic_loss: chromatic,
        divergence,
        weights: [weights[0], weights[1], weights[2]],
        vector: [vector[0], vector[1], vector[2]],
        scalar,
    })
}

fn simulate_ising_energy(hash_seed: &[u8], beta: f64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed_from_bytes(hash_seed, b"ising"));
    let mut lattice = vec![1i8; LATTICE_EDGE * LATTICE_EDGE];
    for sweep in 0..512 {
        for idx in 0..lattice.len() {
            let x = idx % LATTICE_EDGE;
            let y = idx / LATTICE_EDGE;
            let mut neighbor_sum = 0i32;
            let left = if x == 0 { LATTICE_EDGE - 1 } else { x - 1 };
            let right = (x + 1) % LATTICE_EDGE;
            let up = if y == 0 { LATTICE_EDGE - 1 } else { y - 1 };
            let down = (y + 1) % LATTICE_EDGE;
            neighbor_sum += lattice[y * LATTICE_EDGE + left] as i32;
            neighbor_sum += lattice[y * LATTICE_EDGE + right] as i32;
            neighbor_sum += lattice[up * LATTICE_EDGE + x] as i32;
            neighbor_sum += lattice[down * LATTICE_EDGE + x] as i32;
            let delta = 2.0 * lattice[idx] as f64 * neighbor_sum as f64;
            if delta <= 0.0 || rng.gen::<f64>() < (-beta * delta).exp() {
                lattice[idx] = -lattice[idx];
            }
        }
        if sweep % 64 == 0 {
            beta_step(0.0005 * sweep as f64);
        }
    }

    let mut energy = 0.0;
    for y in 0..LATTICE_EDGE {
        for x in 0..LATTICE_EDGE {
            let idx = y * LATTICE_EDGE + x;
            let spin = lattice[idx] as f64;
            let right = lattice[y * LATTICE_EDGE + ((x + 1) % LATTICE_EDGE)] as f64;
            let down = lattice[((y + 1) % LATTICE_EDGE) * LATTICE_EDGE + x] as f64;
            energy -= spin * (right + down);
        }
    }
    energy / (LATTICE_EDGE * LATTICE_EDGE) as f64
}

fn chromatic_surrogate(
    beta: f64,
    replicas: f64,
    iterations: f64,
    mutation_strength: f64,
    mutation_temperature: f64,
) -> f64 {
    let logits =
        0.35 * beta.ln() + 0.27 * (replicas / 2048.0 - 1.0) + 0.18 * (iterations / 4096.0 - 1.0)
            - 0.22 * mutation_strength
            + 0.09 * mutation_temperature;
    (1.0 + (-logits).exp()).ln()
}

fn free_energy_divergence(beta: f64, mutation_strength: f64, mutation_temperature: f64) -> f64 {
    let p = Vector3::new(
        clamp_probability(beta / 5.0),
        clamp_probability(mutation_strength / 0.45),
        clamp_probability(mutation_temperature / 1.75),
    );
    let q = BASELINE_VECTOR;
    kl_divergence(&p, &q)
}

fn adaptive_weights(beta: f64, replicas: f64, iterations: f64) -> Vector3<f64> {
    let entropy = ((beta + replicas / 4096.0 + iterations / 8192.0) / 3.0).clamp(0.2, 0.95);
    let w_energy = 0.5 + 0.2 * entropy;
    let w_chromatic = 0.35 + 0.15 * (1.0 - entropy);
    let w_divergence = 1.0 - w_energy - w_chromatic;
    Vector3::new(w_energy, w_chromatic, w_divergence.max(0.05))
}

fn replicator_dynamics(evaluations: &[VariantEvaluation]) -> (Vec<f64>, f64) {
    let n = evaluations.len();
    if n == 0 {
        return (vec![], 1.0);
    }
    let mut probs = vec![1.0 / n as f64; n];
    let mut temperature = 1.0;
    for step in 0..REPLICATOR_STEPS {
        let variance = score_variance(evaluations, &probs);
        let eta = (0.12 / (1.0 + variance)).clamp(0.01, 0.18);
        let mean = mean_score(evaluations, &probs);
        for (idx, prob) in probs.iter_mut().enumerate() {
            let delta = (evaluations[idx].metrics.scalar - mean) * eta;
            *prob *= delta.clamp(-40.0, 40.0).exp();
        }
        normalize(&mut probs);
        temperature = ((variance + 1e-6).sqrt() * 0.9 + 0.35).clamp(0.35, 1.75);
        if step % 5 == 0 {
            temperature *= 0.98;
        }
    }
    (probs, temperature)
}

fn mean_score(evaluations: &[VariantEvaluation], probs: &[f64]) -> f64 {
    evaluations
        .iter()
        .zip(probs.iter())
        .map(|(eval, p)| eval.metrics.scalar * p)
        .sum::<f64>()
}

fn score_variance(evaluations: &[VariantEvaluation], probs: &[f64]) -> f64 {
    let mean = mean_score(evaluations, probs);
    evaluations
        .iter()
        .zip(probs.iter())
        .map(|(eval, p)| p * (eval.metrics.scalar - mean).powi(2))
        .sum::<f64>()
}

fn compute_free_energy_hash(evals: &[VariantEvaluation], temperature: f64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"meta_free_energy");
    hasher.update(temperature.to_le_bytes());
    for eval in evals {
        hasher.update(eval.genome.hash.as_bytes());
        hasher.update(eval.metrics.energy.to_le_bytes());
        hasher.update(eval.metrics.chromatic_loss.to_le_bytes());
        hasher.update(eval.metrics.divergence.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>()
}

fn halton_point(index: u64, dimension: u64) -> Vec<f64> {
    (0..dimension)
        .map(|d| radical_inverse(index, HALTON_PRIMES[d as usize % HALTON_PRIMES.len()]))
        .collect()
}

fn refine_with_sobol(coords: &mut [f64], index: u64) {
    for (dim, value) in coords.iter_mut().enumerate() {
        let sobol = sobol_value(dim as u32, index);
        *value = ((*value + sobol) * 0.5).fract();
    }
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

fn sobol_value(dimension: u32, mut index: u64) -> f64 {
    let mut result: u32 = 0;
    let mut bit = 0;
    while index != 0 {
        if (index & 1) != 0 {
            result ^= SOBOL_DIR_V[(bit + dimension as usize) % SOBOL_DIR_V.len()];
        }
        index >>= 1;
        bit += 1;
    }
    (result as f64) / (u32::MAX as f64 + 1.0)
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

fn get_continuous(genome: &VariantGenome, key: &str) -> Result<f64> {
    match genome.parameters.get(key) {
        Some(VariantParameter::Continuous { value, .. }) => Ok(*value),
        _ => Err(OrchestratorError::MissingParameter(key.into())),
    }
}

fn get_discrete(genome: &VariantGenome, key: &str) -> Result<i64> {
    match genome.parameters.get(key) {
        Some(VariantParameter::Discrete { value, .. }) => Ok(*value),
        _ => Err(OrchestratorError::MissingParameter(key.into())),
    }
}

fn clamp_probability(x: f64) -> f64 {
    x.clamp(1e-6, 1.0 - 1e-6)
}

fn kl_divergence(p: &Vector3<f64>, q: &Vector3<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        let pi = clamp_probability(p[i]);
        let qi = clamp_probability(q[i]);
        sum += pi * (pi / qi).ln();
    }
    sum
}

fn seed_from_bytes(seed: &[u8], salt: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed);
    hasher.update(salt);
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn beta_step(_delta: f64) {}

fn normalize(probs: &mut [f64]) {
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for prob in probs.iter_mut() {
            *prob /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_sequence_low_discrepancy() {
        let pts: Vec<f64> = (1..16).map(|i| radical_inverse(i, 2)).collect();
        assert!(pts.iter().all(|&p| p > 0.0 && p < 1.0));

        let mut sorted = pts.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let max_gap = sorted
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f64, f64::max);

        assert!(max_gap < 0.4, "halton gaps too large: {}", max_gap);
    }

    #[test]
    fn replicate_dynamics_normalizes() {
        let genome = VariantGenome {
            seed: 1,
            parameters: BTreeMap::new(),
            feature_toggles: BTreeMap::new(),
            hash: "deadbeef".into(),
        };
        let eval = VariantEvaluation {
            genome,
            metrics: EvolutionMetrics {
                energy: 0.1,
                chromatic_loss: 0.2,
                divergence: 0.05,
                weights: [0.5, 0.35, 0.15],
                vector: [0.1, 0.2, 0.05],
                scalar: 0.1,
            },
        };
        let (probs, _) = replicator_dynamics(&[eval]);
        assert!((probs[0] - 1.0).abs() < 1e-9);
    }
}
