//! Ontogenic Input/Output System
pub mod probes;
use anyhow::Result;

pub struct OntogenicIO {
    context: Vec<f32>,
}

impl OntogenicIO {
    pub fn new() -> Result<Self> {
        Ok(Self { context: Vec::new() })
    }
}
