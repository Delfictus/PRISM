//! Hypertuning Controller
//!
//! Monitors telemetry metrics and dynamically adjusts pipeline parameters
//! to improve performance and chromatic number reduction.

pub mod controller;
pub mod action;

pub use controller::HypertuneController;
pub use action::{TelemetryEvent, AdpControl};
