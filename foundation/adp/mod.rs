//! Adaptive Decision Processor (ADP)
//!
//! Implements reinforcement learning and adaptive decision making
//! for PRCT parameter optimization. Based on CSF's C-Logic ADP module.

pub mod decision_processor;
pub mod reinforcement;

pub use decision_processor::{AdaptiveDecisionProcessor, AdpStats, Decision};
pub use reinforcement::{Action, ReinforcementLearner, RlConfig, RlStats, State};
