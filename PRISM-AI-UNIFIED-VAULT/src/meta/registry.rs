//! Meta variant registry placeholder (Phase M1 target).
//!
//! This file is intentionally lightweight. Phase M0 will stub the data
//! structures and serialization contracts; later phases will flesh out the
//! orchestration logic. Keeping the module committed now ensures paths exist
//! for documentation cross-references and avoids missing-file errors when the
//! master executor scans for artifacts.

/// Represents a single meta variant candidate (stub).
#[derive(Debug, Clone)]
pub struct MetaVariant {
    /// Unique identifier (to be replaced with Merkle-derived ids).
    pub id: String,
}

impl MetaVariant {
    /// Creates a placeholder variant with the supplied id.
    pub fn placeholder(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

/// Stub registry storing meta variants in memory.
#[derive(Default)]
pub struct MetaRegistry {
    variants: Vec<MetaVariant>,
}

impl MetaRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a variant to the registry (no-op placeholder).
    pub fn register(&mut self, variant: MetaVariant) {
        self.variants.push(variant);
    }

    /// Returns registered variants (placeholder).
    pub fn variants(&self) -> &[MetaVariant] {
        &self.variants
    }
}
