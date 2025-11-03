//! Routing module
//!
//! Intelligent LLM routing algorithms

pub mod thermodynamic_balancer;
pub mod transfer_entropy_router;
pub mod gpu_transfer_entropy_router;

pub use thermodynamic_balancer::{ThermodynamicLoadBalancer, QuantumVotingConsensus};
pub use transfer_entropy_router::{TransferEntropyPromptRouter, PIDSynergyDetector};
pub use gpu_transfer_entropy_router::{GpuTransferEntropyRouter, QueryDomain, RoutingDecision};
