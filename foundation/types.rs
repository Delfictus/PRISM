//! Core types for the neuromorphic-quantum platform

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Universal input data for neuromorphic-quantum processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInput {
    /// Unique identifier for this input
    pub id: Uuid,
    /// Raw data values
    pub values: Vec<f64>,
    /// Timestamp of the data point
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Data source identifier
    pub source: String,
    /// Processing configuration
    pub config: ProcessingConfig,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Processing configuration for platform operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Enable neuromorphic processing
    pub neuromorphic_enabled: bool,
    /// Enable quantum optimization
    pub quantum_enabled: bool,
    /// Neuromorphic processing parameters
    pub neuromorphic_config: NeuromorphicConfig,
    /// Quantum processing parameters
    pub quantum_config: QuantumConfig,
}

/// Neuromorphic processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicConfig {
    /// Number of neurons for spike encoding
    pub neuron_count: usize,
    /// Time window for processing (ms)
    pub window_ms: f64,
    /// Spike encoding method
    pub encoding_method: String,
    /// Reservoir size for liquid state machines
    pub reservoir_size: usize,
    /// Pattern detection threshold
    pub detection_threshold: f64,
}

/// Quantum optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Number of qubits (simulated)
    pub qubit_count: usize,
    /// Time evolution step size
    pub time_step: f64,
    /// Target evolution time
    pub evolution_time: f64,
    /// Energy tolerance for convergence
    pub energy_tolerance: f64,
}

/// Platform processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformOutput {
    /// Input ID this output corresponds to
    pub input_id: Uuid,
    /// Neuromorphic processing results
    pub neuromorphic_results: Option<NeuromorphicResults>,
    /// Quantum optimization results
    pub quantum_results: Option<QuantumResults>,
    /// Combined platform prediction
    pub prediction: PlatformPrediction,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Results from neuromorphic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicResults {
    /// Detected patterns
    pub patterns: Vec<DetectedPattern>,
    /// Spike pattern analysis
    pub spike_analysis: SpikeAnalysis,
    /// Reservoir state
    pub reservoir_state: ReservoirState,
}

/// Detected pattern from neuromorphic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Confidence/strength (0.0 to 1.0)
    pub strength: f64,
    /// Spatial characteristics
    pub spatial_features: Vec<f64>,
    /// Temporal characteristics
    pub temporal_features: Vec<f64>,
}

/// Spike pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeAnalysis {
    /// Total spike count
    pub spike_count: usize,
    /// Average spike rate (Hz)
    pub spike_rate: f64,
    /// Pattern coherence
    pub coherence: f64,
    /// Temporal dynamics
    pub dynamics: Vec<f64>,
}

/// Reservoir computing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirState {
    /// Current activations
    pub activations: Vec<f64>,
    /// Average activation
    pub avg_activation: f64,
    /// Memory capacity
    pub memory_capacity: f64,
    /// Separation property
    pub separation: f64,
}

/// Results from quantum optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResults {
    /// Final energy state
    pub energy: f64,
    /// Phase coherence
    pub phase_coherence: f64,
    /// Convergence information
    pub convergence: ConvergenceInfo,
    /// Quantum state features
    pub state_features: Vec<f64>,
}

/// Convergence information for quantum optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final error/residual
    pub final_error: f64,
    /// Energy drift over time
    pub energy_drift: f64,
}

/// Combined platform prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPrediction {
    /// Prediction direction/classification
    pub direction: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted magnitude of change
    pub magnitude: Option<f64>,
    /// Time horizon for prediction
    pub time_horizon_ms: f64,
    /// Contributing factors
    pub factors: Vec<String>,
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Processing start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Processing end time
    pub end_time: chrono::DateTime<chrono::Utc>,
    /// Total processing duration (ms)
    pub duration_ms: f64,
    /// Neuromorphic processing time (ms)
    pub neuromorphic_time_ms: Option<f64>,
    /// Quantum processing time (ms)
    pub quantum_time_ms: Option<f64>,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Current memory usage (bytes)
    pub current_memory_bytes: usize,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            neuromorphic_enabled: true,
            quantum_enabled: true,
            neuromorphic_config: NeuromorphicConfig::default(),
            quantum_config: QuantumConfig::default(),
        }
    }
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            neuron_count: 1000,
            window_ms: 1000.0,
            encoding_method: "Rate".to_string(),
            reservoir_size: 500,
            detection_threshold: 0.7,
        }
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            qubit_count: 10,
            time_step: 0.001,
            evolution_time: 1.0,
            energy_tolerance: 1e-6,
        }
    }
}

impl PlatformInput {
    /// Create new platform input
    pub fn new(source: String, values: Vec<f64>) -> Self {
        Self {
            id: Uuid::new_v4(),
            values,
            timestamp: chrono::Utc::now(),
            source,
            config: ProcessingConfig::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to input
    pub fn with_metadata(mut self, key: String, value: f64) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set processing configuration
    pub fn with_config(mut self, config: ProcessingConfig) -> Self {
        self.config = config;
        self
    }
}
