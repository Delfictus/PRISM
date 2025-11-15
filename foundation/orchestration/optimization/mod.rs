//! Optimization module
//!
//! Cutting-edge optimization algorithms:
//! - MDL prompt optimization
//! - Information bottleneck
//! - Quantum prompt search
//! - Geometric manifold optimization

pub mod geometric_manifold;
pub mod mdl_prompt_optimizer;
pub mod quantum_prompt_search;

pub use mdl_prompt_optimizer::{MDLPromptOptimizer, OptimizedPrompt, QueryType};
pub use quantum_prompt_search::{InformationBottleneckCompressor, QuantumPromptSearch};
