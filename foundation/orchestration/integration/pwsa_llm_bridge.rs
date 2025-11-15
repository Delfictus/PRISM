//! PWSA-LLM Integration Bridge
//!
//! Mission Charlie: Task 3.4 (CRITICAL)
//!
//! Integrates Mission Bravo sensor fusion with Mission Charlie LLM intelligence

use anyhow::Result;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::foundation::pwsa::satellite_adapters::{
    MissionAwareness, PwsaFusionPlatform, ThreatDetection,
};
use crate::orchestration::llm_clients::LLMOrchestrator;

/// Complete Intelligence (Sensor + AI)
#[derive(Debug)]
pub struct CompleteIntelligence {
    pub sensor_assessment: MissionAwareness,
    pub ai_context: Option<String>,
    pub combined_confidence: f64,
}

/// PWSA-LLM Fusion Platform
///
/// Integrates Mission Bravo sensor fusion with Mission Charlie LLM intelligence
pub struct PwsaLLMFusionPlatform {
    pwsa_platform: Arc<Mutex<PwsaFusionPlatform>>,
    llm_orchestrator: Option<Arc<Mutex<LLMOrchestrator>>>,
}

impl PwsaLLMFusionPlatform {
    pub fn new(pwsa: Arc<Mutex<PwsaFusionPlatform>>) -> Self {
        Self {
            pwsa_platform: pwsa,
            llm_orchestrator: None,
        }
    }

    /// Enable LLM intelligence (optional)
    pub fn enable_llm_intelligence(&mut self, orchestrator: Arc<Mutex<LLMOrchestrator>>) {
        self.llm_orchestrator = Some(orchestrator);
    }

    /// Fuse sensor + AI intelligence
    pub async fn fuse_complete_intelligence(
        &self,
        sensor_data: &SensorInput,
    ) -> Result<CompleteIntelligence> {
        // Phase 1: Sensor fusion (Mission Bravo)
        let sensor_assessment = {
            let mut pwsa = self.pwsa_platform.lock();
            pwsa.fuse_mission_data(
                &sensor_data.transport,
                &sensor_data.tracking,
                &sensor_data.ground,
            )?
        };

        // Phase 2: AI intelligence (Mission Charlie - if enabled)
        let ai_context = if let Some(ref llm_orch) = self.llm_orchestrator {
            let prompt = self.create_intelligence_prompt(&sensor_assessment);
            let mut orch = llm_orch.lock();
            let response = orch.query_optimal(&prompt, 0.7).await?;
            Some(response.response.text)
        } else {
            None
        };

        // Combined confidence
        let combined = if ai_context.is_some() { 0.95 } else { 0.85 };

        Ok(CompleteIntelligence {
            sensor_assessment,
            ai_context,
            combined_confidence: combined,
        })
    }

    fn create_intelligence_prompt(&self, _mission_awareness: &MissionAwareness) -> String {
        "Analyze this threat detection and provide geopolitical context.".to_string()
    }
}

pub struct SensorInput {
    pub transport: crate::pwsa::satellite_adapters::OctTelemetry,
    pub tracking: crate::pwsa::satellite_adapters::IrSensorFrame,
    pub ground: crate::pwsa::satellite_adapters::GroundStationData,
}
