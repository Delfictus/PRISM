//! Feature flag registry for the Meta Evolutionary Compute stack.

pub mod meta_flags;

pub use meta_flags::{
    registry, MetaFeatureId, MetaFeatureRegistry, MetaFeatureState, MetaFlagError, MetaFlagManifest,
};
