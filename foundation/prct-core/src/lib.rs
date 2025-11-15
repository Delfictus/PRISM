//! PRCT Core Domain
//!
//! Pure domain logic for Phase Resonance Chromatic-TSP algorithm.
//! This crate contains ONLY business logic - no infrastructure dependencies.
//!
//! Architecture: Hexagonal (Ports & Adapters)
//! - Domain logic depends on port abstractions (traits)
//! - Infrastructure adapters implement ports
//! - Dependency arrows point INWARD to domain

pub mod ports;
pub mod algorithm;
pub mod drpp_algorithm;
pub mod coupling;
pub mod coloring;
pub mod simulated_annealing;
pub mod tsp;
pub mod errors;
pub mod dimacs_parser;
pub mod dsatur_backtracking;

// Re-export main types
pub use ports::*;
pub use algorithm::*;
pub use drpp_algorithm::*;
pub use coupling::*;
pub use coloring::*;
pub use simulated_annealing::*;
pub use errors::*;
pub use dimacs_parser::{parse_dimacs_file, parse_mtx_file, parse_graph_file};
pub use dsatur_backtracking::{ColoringSolution, DSaturSolver};

// Re-export shared types for convenience
pub use shared_types::*;
pub mod gpu_prct;
pub use gpu_prct::GpuPRCT;
