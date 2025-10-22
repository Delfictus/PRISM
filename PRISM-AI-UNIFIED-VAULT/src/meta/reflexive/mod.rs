//! Reflexive feedback controller (Phase M3 target).
//!
//! The production implementation lives in `src/meta/reflexive/mod.rs`. This copy
//! summarizes the critical structures used by governance when auditing Phase M3:
//!
//! ```rust
//! #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
//! pub enum GovernanceMode {
//!     Strict,
//!     Recovery,
//!     Exploration,
//! }
//!
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! pub struct ReflexiveSnapshot {
//!     pub entropy: f64,
//!     pub divergence: f64,
//!     pub energy_mean: f64,
//!     pub energy_variance: f64,
//!     pub energy_trend: f64,
//!     pub exploration_ratio: f64,
//!     pub effective_temperature: f64,
//!     pub lattice_edge: usize,
//!     pub lattice: Vec<Vec<f64>>,
//!     pub alerts: Vec<String>,
//! }
//! ```
//!
//! The controller regulates exploration temperature and normalizes population
//! weights when divergence or entropy violate safety thresholds. It emits a
//! free-energy lattice snapshot whose SHA-256 fingerprint is attached to the
//! determinism manifest for meta replay validation.
