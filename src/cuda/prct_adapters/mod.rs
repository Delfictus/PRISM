//! PRCT Port Adapters
//!
//! Implements the port interfaces required by the core PRCT algorithm.
//! These adapters connect the domain logic to concrete implementations.

pub mod neuromorphic;
pub mod quantum;
pub mod coupling;

pub use neuromorphic::NeuromorphicAdapter;
pub use quantum::QuantumAdapter;
pub use coupling::PhysicsCouplingAdapter;
