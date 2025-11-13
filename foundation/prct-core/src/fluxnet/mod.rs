//! FluxNet RL Force Profile System
//!
//! GPU-accelerated reinforcement learning for adaptive Phase 2 thermodynamic equilibration.
//!
//! # Overview
//!
//! FluxNet extends the PRISM world-record pipeline with:
//! 1. **ForceProfile**: Band-based force multipliers (Strong/Neutral/Weak)
//! 2. **RL Controller**: Q-learning agent that adapts forces per temperature
//! 3. **GPU Integration**: Device buffers synced with Phase 2 thermodynamic kernel
//! 4. **Pre-training**: Warm-start Q-table on DSJC250.5
//!
//! # Architecture
//!
//! ```text
//! Phase 0: Reservoir → difficulty_scores ────┐
//!                                             │
//! Phase 1: AI Inference → ai_uncertainty ────┤
//!                                             ▼
//!                                    ForceProfile
//!                                             │
//!                                             ▼
//! Phase 2: Thermodynamic ◄─── RL Controller ──┘
//!          (per temperature)      (Q-learning)
//!
//! GPU Flow:
//! ForceProfile.to_device() → CUDA Kernel → Telemetry → RL Update
//! ```
//!
//! # GPU Mandate Compliance
//!
//! All FluxNet components follow PRISM GPU-FIRST standards:
//! - ✅ Arc<CudaDevice> for shared context (Article V)
//! - ✅ CudaSlice<T> device buffers for all GPU data
//! - ✅ Explicit host-device synchronization
//! - ❌ NO CPU fallbacks
//! - ❌ NO conditional compilation beyond `#[cfg(feature = "cuda")]`
//!
//! # Modules
//!
//! - `profile`: ForceProfile, ForceBand, ForceBandStats (Phase A.1)
//! - `command`: ForceCommand for RL actions (Phase A.2)
//! - `controller`: Q-learning RL controller (Phase D)
//! - `config`: FluxNetConfig for TOML configuration (Phase A.3)
//! - `telemetry`: FluxNet telemetry data structures (Phase E)

pub mod profile;
pub mod command;
pub mod config;
pub mod controller;
pub mod unified_state;
pub mod reward;
pub mod telemetry;

pub use profile::{ForceBand, ForceBandStats, ForceProfile};
pub use command::{FluxNetAction, AdjustDirection, ActionResult};
pub use config::{FluxNetConfig, MemoryTier, ForceProfileConfig, RLConfig, PersistenceConfig};
pub use controller::{MultiPhaseRLController, QTable, Experience, ReplayBuffer};
pub use unified_state::UnifiedRLState;
pub use reward::compute_reward;
pub use telemetry::{
    FluxNetTelemetry, ForceBandTelemetry, RLDecisionTelemetry, RLStateTelemetry,
    QUpdateTelemetry, FluxNetConfigSnapshot,
};
