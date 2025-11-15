//! Thermodynamic module

pub mod gpu_thermodynamic_consensus;
pub mod hamiltonian;
pub mod quantum_consensus;
pub mod thermodynamic_consensus;

pub use gpu_thermodynamic_consensus::{
    GpuThermodynamicConsensus, LLMModel, ThermodynamicState as GpuThermodynamicState,
};
pub use hamiltonian::InformationHamiltonian;
pub use quantum_consensus::QuantumConsensusOptimizer as ThermodynamicConsensus;
pub use quantum_consensus::{ConsensusState, QuantumConsensusOptimizer};
