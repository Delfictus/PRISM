//! Error types for Mission Charlie orchestration

use thiserror::Error;

/// Main error type for orchestration operations
#[derive(Error, Debug)]
pub enum OrchestrationError {
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("Singular matrix encountered: {matrix_name}")]
    SingularMatrix { matrix_name: String },

    #[error("Insufficient data: required {required}, available {available}")]
    InsufficientData { required: usize, available: usize },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Invalid parameter: {name} = {value}")]
    InvalidParameter { name: String, value: String },

    #[error("LLM error: {message}")]
    LLMError { message: String },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("External service error: {service} - {error}")]
    ExternalService { service: String, error: String },

    #[error("Consensus failed: {reason}")]
    ConsensusFailed { reason: String },

    #[error("Integration error: {component} - {message}")]
    IntegrationError { component: String, message: String },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timeout after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimitExceeded { limit: usize, window: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Numerical error: {operation} - {details}")]
    NumericalError { operation: String, details: String },

    #[error("State error: {expected} but got {actual}")]
    StateError { expected: String, actual: String },

    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid index: {0}")]
    InvalidIndex(String),

    #[error("Missing data: {0}")]
    MissingData(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("No solution: {0}")]
    NoSolution(String),

    #[error("Invalid matrix: {0}")]
    InvalidMatrix(String),
}

impl From<anyhow::Error> for OrchestrationError {
    fn from(error: anyhow::Error) -> Self {
        OrchestrationError::Unknown {
            message: error.to_string(),
        }
    }
}
