//! PRCT Port Adapters
//!
//! Implements the port interfaces required by the core PRCT algorithm.
//! These adapters connect the domain logic to concrete implementations.

pub mod coupling;
pub mod neuromorphic;
pub mod quantum;

pub use coupling::PhysicsCouplingAdapter;
pub use neuromorphic::NeuromorphicAdapter;
pub use quantum::QuantumAdapter;
