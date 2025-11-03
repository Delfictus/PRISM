//! Sensory Probes
use anyhow::Result;

pub trait SensoryProbe {
    type Output;
    fn capture(&self) -> Result<Self::Output>;
}
