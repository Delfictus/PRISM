//! Routing module
//!
//! Intelligent LLM routing algorithms

pub mod gpu_transfer_entropy_router;
pub mod thermodynamic_balancer;
pub mod transfer_entropy_router;

pub use gpu_transfer_entropy_router::{GpuTransferEntropyRouter, QueryDomain, RoutingDecision};
pub use thermodynamic_balancer::{QuantumVotingConsensus, ThermodynamicLoadBalancer};
pub use transfer_entropy_router::{PIDSynergyDetector, TransferEntropyPromptRouter};
