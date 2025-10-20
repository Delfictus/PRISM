//! Domain Errors

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PRCTError {
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Coloring failed: {0}")]
    ColoringFailed(String),

    #[error("TSP solver failed: {0}")]
    TSPFailed(String),

    #[error("Physics coupling failed: {0}")]
    CouplingFailed(String),

    #[error("Neuromorphic processing failed: {0}")]
    NeuromorphicFailed(String),

    #[error("Quantum processing failed: {0}")]
    QuantumFailed(String),

    #[error("Port operation failed: {0}")]
    PortError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, PRCTError>;
