//! Manifold optimization module

pub mod causal_manifold_optimizer;

pub use causal_manifold_optimizer::{
    CausalManifoldOptimizer, HybridOptimizer, NaturalGradientOptimizer,
};
