# Cursor IDE Prompts - Copy & Paste Ready
## Exact Prompts for Each Phase

**Instructions**: Copy these prompts directly into Cursor IDE and execute them in order.

---

## 📋 PHASE 0: FOUNDATION SETUP

### Prompt 0.1: Create Directory Structure (Composer: `Cmd/Ctrl + I`)

**Copy and paste this into Cursor Composer**:

```
Create the MEC system directory structure and initial stub files:

1. Create these directories:
   - src/mec/
   - src/ontogenic_io/
   - src/ontogenic_io/probes/
   - src/meta_learning/
   - foundation/orchestration/integration/bridges/

2. Create src/mec/mod.rs with:
   //! Meta Emergent Computation Engine
   //!
   //! Self-evolving algorithmic substrate for PRISM-AI

   use anyhow::Result;
   use std::collections::HashMap;

   pub struct MetaEmergentComputation {
       current_parameters: HashMap<String, f64>,
       performance_history: Vec<f64>,
       mutation_rate: f64,
   }

   impl MetaEmergentComputation {
       pub fn new() -> Result<Self> {
           Ok(Self {
               current_parameters: HashMap::new(),
               performance_history: Vec::new(),
               mutation_rate: 0.05,
           })
       }

       pub async fn evolve_cycle(&mut self) -> Result<()> {
           // Will implement in Phase 4
           Ok(())
       }
   }

3. Create src/ontogenic_io/mod.rs with:
   //! Ontogenic Input/Output System
   //!
   //! Continuous environmental awareness

   pub mod probes;

   use anyhow::Result;

   pub struct OntogenicIO {
       context: Vec<f32>,
   }

   impl OntogenicIO {
       pub fn new() -> Result<Self> {
           Ok(Self { context: Vec::new() })
       }

       pub async fn tick(&mut self) -> Result<()> {
           // Will implement in Phase 3
           Ok(())
       }
   }

4. Create src/ontogenic_io/probes/mod.rs with:
   //! Sensory Probes

   use anyhow::Result;

   pub trait SensoryProbe {
       type Output;
       fn capture(&self) -> Result<Self::Output>;
   }

5. Create src/meta_learning/mod.rs with:
   //! Meta-Learning Controller

   use anyhow::Result;

   pub struct MetaLearningController {}

   impl MetaLearningController {
       pub fn new() -> Result<Self> {
           Ok(Self {})
       }
   }

6. Create foundation/orchestration/integration/bridges/mod.rs with:
   //! Integration bridges for connecting orchestrator to applications

   pub mod llm_consensus_bridge;

7. Update src/lib.rs to add at the end:
   // MEC modules
   pub mod mec;
   pub mod ontogenic_io;
   pub mod meta_learning;

After creating all files, run cargo check to verify everything compiles.
```

**Verify**: Run `cargo check` in terminal - should compile

---

## 🤖 PHASE 1: LLM CONSENSUS

### Prompt 1.1: Create LLM Consensus Bridge (Composer: `Cmd/Ctrl + I`)

**Copy and paste this into Cursor Composer**:

```
I need to implement LLM consensus with quantum voting in PRISM-AI.

Context:
- The PrismAIOrchestrator already exists at foundation/orchestration/integration/prism_ai_integration.rs
- It has a charlie_integration field of type Arc<RwLock<MissionCharlieIntegration>>
- MissionCharlieIntegration has these working components:
  * quantum_voting: QuantumVotingConsensus
  * thermodynamic_consensus: ThermodynamicConsensus
  * transfer_entropy_router: TransferEntropyRouter
- LLM clients exist at foundation/orchestration/llm_clients/

Task:
Create foundation/orchestration/integration/bridges/llm_consensus_bridge.rs

This file should define these types:
```rust
pub struct ConsensusRequest {
    pub query: String,
    pub models: Vec<String>,
    pub temperature: f32,
}

pub struct ConsensusResponse {
    pub text: String,
    pub confidence: f64,
    pub agreement_score: f64,
    pub model_responses: Vec<ModelResponse>,
    pub algorithm_weights: Vec<(String, f64)>,
}

pub struct ModelResponse {
    pub model: String,
    pub text: String,
    pub tokens: usize,
    pub cost: f64,
}
```

Then add these methods to the PrismAIOrchestrator impl block in prism_ai_integration.rs:

1. pub async fn llm_consensus(&self, query: &str, models: &[&str]) -> Result<ConsensusResponse>
   - Call query_all_llms() to get responses from each model
   - Call apply_quantum_voting() using charlie_integration.quantum_voting
   - Call apply_thermodynamic_consensus() using charlie_integration.thermodynamic_consensus
   - Call apply_transfer_entropy_routing() using charlie_integration.transfer_entropy_router
   - Fuse all three results with weighted averaging
   - Return ConsensusResponse with combined results

2. async fn query_all_llms(&self, query: &str, models: &[&str]) -> Result<Vec<LLMResponse>>
   - For each model name, get the client from charlie_integration
   - Call client.generate(query, 0.7)
   - Collect all responses
   - Log each query with log::info!

3. async fn apply_quantum_voting(&self, responses: &[LLMResponse]) -> Result<QuantumVoteResult>
   - Get charlie_integration.read()
   - Call quantum_voting.vote(responses).await
   - Log result with log::debug!

4. async fn apply_thermodynamic_consensus(&self, responses: &[LLMResponse]) -> Result<ThermodynamicResult>
   - Get charlie_integration.read()
   - Call thermodynamic_consensus.converge(responses).await
   - Log result

5. async fn apply_transfer_entropy_routing(&self, responses: &[LLMResponse]) -> Result<RoutedResponses>
   - Get charlie_integration.read()
   - Call transfer_entropy_router.route(responses).await
   - Log result

6. fn fuse_consensus_results(&self, quantum: &QuantumVoteResult, thermo: &ThermodynamicResult, routed: &RoutedResponses) -> Result<ConsensusResponse>
   - Combine results with weights: quantum 40%, thermo 35%, routing 25%
   - Calculate overall confidence
   - Extract best response text
   - Return ConsensusResponse

Add comprehensive error handling and logging throughout.
Use anyhow::Result for error handling.
```

**Expected Time**: 30 minutes for Cursor to generate, 30 minutes for you to review

---

### Prompt 1.2: Create Main Executable (Composer: `Cmd/Ctrl + I`)

**Copy and paste**:

```
Create src/bin/prism_mec.rs as the main executable for MEC.

Use clap for CLI with these commands:

#[derive(Parser)]
#[command(name = "prism-mec")]
#[command(about = "PRISM-AI Meta Emergent Computation System")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get LLM consensus with quantum voting
    Consensus {
        /// Query to ask
        query: String,
        /// Models (comma-separated)
        #[arg(short, long, default_value = "gpt-4,claude,gemini")]
        models: String,
    },

    /// Run system diagnostics
    Diagnostics,
}

Implementation requirements:

1. #[tokio::main] async fn main()
   - Initialize env_logger
   - Print welcome banner
   - Parse CLI args
   - Initialize PrismAIOrchestrator with default config
   - Match on command and dispatch

2. async fn run_llm_consensus(orchestrator: &PrismAIOrchestrator, query: &str, models_str: &str)
   - Split models by comma
   - Call orchestrator.llm_consensus(query, &models).await
   - Print beautiful output:
     * Query
     * Models used
     * Consensus text (wrapped nicely)
     * Confidence as percentage
     * Agreement score
     * Algorithm contributions as table

3. async fn run_diagnostics(orchestrator: &PrismAIOrchestrator)
   - Print system status
   - Show available LLM clients
   - Show active algorithms
   - Confirm operational

Add these to Cargo.toml [bin] section:
[[bin]]
name = "prism-mec"
path = "src/bin/prism_mec.rs"

Use nice formatting. Add colors if using a crate like colored or owo-colors.
```

---

### Prompt 1.3: Test and Debug (Chat: `Cmd/Ctrl + L`)

**Copy and paste into Chat**:

```
I just implemented LLM consensus. Help me:

1. What API keys do I need to set up?
2. How do I test it locally?
3. What compilation errors should I expect?
4. How do I debug if the quantum voting doesn't get called?

Look at:
- foundation/orchestration/integration/bridges/llm_consensus_bridge.rs
- src/bin/prism_mec.rs

Tell me the exact commands to run to test this.
```

---

## 🔬 PHASE 2: MATERIALS DISCOVERY

### Prompt 2.1: Materials Bridge (Composer: `Cmd/Ctrl + I`)

```
Create foundation/orchestration/integration/bridges/materials_bridge.rs

This implements materials discovery using existing components.

Types to define:
```rust
pub struct MaterialDiscoveryRequest {
    pub target_properties: MaterialProperties,
    pub max_candidates: usize,
}

pub struct MaterialDiscoveryResult {
    pub candidates: Vec<MaterialCandidate>,
    pub computation_time_ms: f64,
    pub total_evaluated: usize,
}
```

Add to PrismAIOrchestrator in prism_ai_integration.rs:

1. pub async fn discover_materials(&self, target: MaterialProperties) -> Result<MaterialDiscoveryResult>

   Implementation steps:
   - Log start with target properties
   - Create CMA problem using materials_to_cma_problem()
   - Initialize CausalManifoldAnnealing with default config
   - Call cma_engine.solve(problem).await
   - Create MaterialsAdapter::new()
   - Call adapter.discover_material(&target, &solution)
   - Measure elapsed time
   - Return MaterialDiscoveryResult

2. fn materials_to_cma_problem(&self, target: &MaterialProperties) -> Result<Box<dyn Problem>>

   This converts MaterialProperties to a CMA optimization problem.
   For now, create a simple wrapper that treats each property as a dimension.
   The cost function should measure distance from target properties.

The MaterialsAdapter already exists at src/cma/applications/mod.rs
The CausalManifoldAnnealing exists at src/cma/mod.rs
The MaterialProperties type exists in src/cma/applications/mod.rs

Use those existing types and implementations.
Add comprehensive logging with log::info! and log::debug!
```

---

### Prompt 2.2: Add Materials Command (Inline Edit: `Cmd/Ctrl + K`)

**Navigate to `src/bin/prism_mec.rs`, select the Commands enum, press `Cmd/Ctrl + K`**:

```
Add Materials command to this enum:

/// Discover materials with target properties
Materials {
    /// Target bandgap in eV
    #[arg(long)]
    bandgap: f64,

    /// Target conductivity in S/m
    #[arg(long)]
    conductivity: f64,

    /// Target hardness in GPa
    #[arg(long, default_value = "5.0")]
    hardness: f64,
},

Then add the match case in main() that:
- Creates MaterialProperties from the arguments
- Calls orchestrator.discover_materials(props).await
- Prints results nicely
```

---

## 🎤 PHASE 3: TEXT TONE PROBE

### Prompt 3.1: Text Tone Probe (Composer: `Cmd/Ctrl + I`)

```
Create src/ontogenic_io/probes/text_tone.rs

This extracts emotional and stylistic features from text.

Types:
```rust
pub struct TextToneProbe {
    positive_words: Vec<String>,
    negative_words: Vec<String>,
}

pub struct ToneFeatures {
    pub caps_ratio: f32,
    pub punct_entropy: f32,
    pub valence: f32,
    pub arousal: f32,
    pub exclamation_count: usize,
    pub question_count: usize,
}
```

Implementation:

1. impl TextToneProbe
   - new() -> Self
     Create with simple positive/negative word lists:
     Positive: ["good", "great", "excellent", "happy", "love", "wonderful", "amazing"]
     Negative: ["bad", "terrible", "awful", "sad", "hate", "horrible", "disappointing"]

   - analyze(&self, text: &str) -> Result<ToneFeatures>
     Call all the feature extraction methods and return ToneFeatures

   - caps_ratio(&self, text: &str) -> f32
     Count uppercase letters / total letters

   - punct_entropy(&self, text: &str) -> f32
     Calculate Shannon entropy of punctuation character distribution
     H = -Σ p(x) log2(p(x))

   - lexicon_valence(&self, text: &str) -> f32
     Count positive words - negative words, normalize by total words
     Return value in [-1.0, 1.0]

   - lexicon_arousal(&self, text: &str) -> f32
     Count exclamations + questions as arousal indicators
     Normalize to [0.0, 1.0]

2. Add impl SensoryProbe for TextToneProbe
   - Implement the trait from probes/mod.rs

3. Add comprehensive unit tests:
   - test_positive_text()
   - test_negative_text()
   - test_neutral_text()
   - test_all_caps()
   - test_lots_of_punctuation()

Use only std library - no external crates needed.
```

---

## 🖱️ PHASE 3: CURSOR PROBE

### Prompt 3.2: Cursor Dynamics Probe (Composer: `Cmd/Ctrl + I`)

```
Create src/ontogenic_io/probes/cursor.rs

This tracks mouse cursor movements and extracts kinematic features.

Types:
```rust
pub struct CursorProbe {
    history: Vec<CursorPoint>,
    max_history: usize,
}

pub struct CursorPoint {
    pub x: f32,
    pub y: f32,
    pub timestamp_ms: u64,
}

pub struct CursorFeatures {
    pub velocity_mean: f32,
    pub velocity_std: f32,
    pub acceleration_mean: f32,
    pub jerk_mean: f32,
    pub hesitation_count: usize,
    pub path_efficiency: f32,
}
```

Implementation:

1. impl CursorProbe

   - new() -> Self
     Initialize with empty history and max_history = 100

   - add_point(&mut self, x: f32, y: f32)
     Create CursorPoint with current timestamp
     Add to history
     If history.len() > max_history, remove oldest

   - extract_features(&self) -> Result<CursorFeatures>
     Return default if < 3 points
     Otherwise:
     - Calculate velocities
     - Calculate accelerations from velocities
     - Calculate jerks from accelerations
     - Count hesitations (velocity < 10.0 threshold)
     - Calculate path efficiency
     Return CursorFeatures

   - calculate_velocities(&self) -> Vec<f32>
     For each pair of consecutive points:
     - dx = x1 - x0, dy = y1 - y0
     - dt = t1 - t0
     - v = sqrt(dx² + dy²) / dt
     Return vector of velocities

   - calculate_accelerations(&self, velocities: &[f32]) -> Vec<f32>
     For each pair of consecutive velocities:
     - a = (v1 - v0) / dt
     Return accelerations

   - calculate_jerks(&self, accelerations: &[f32]) -> Vec<f32>
     For each pair of consecutive accelerations:
     - j = (a1 - a0) / dt
     Return jerks

   - count_hesitations(&self, velocities: &[f32]) -> usize
     Count how many times velocity drops below 10.0

   - calculate_efficiency(&self) -> f32
     straight_line_distance / total_path_length
     Return ratio in [0.0, 1.0]

2. Helper functions:
   - fn mean(values: &[f32]) -> f32
   - fn std_dev(values: &[f32]) -> f32
   - fn current_time_ms() -> u64

3. Add unit tests

Use only std library.
```

---

## 🔗 PHASE 1: WIRE TO ORCHESTRATOR

### Prompt 1.4: Update Orchestrator (Composer: `Cmd/Ctrl + I`)

```
Update foundation/orchestration/integration/prism_ai_integration.rs

Add these methods to the impl PrismAIOrchestrator block:

1. pub async fn llm_consensus(
       &self,
       query: &str,
       models: &[&str]
   ) -> Result<ConsensusResponse>

   Implementation:
   - Convert models slice to Vec<String>
   - Log query and models
   - Call self.query_all_llms(query, models).await
   - Call self.apply_quantum_voting(&responses).await
   - Call self.apply_thermodynamic_consensus(&responses).await
   - Call self.apply_transfer_entropy_routing(&responses).await
   - Call self.fuse_consensus_results(quantum, thermo, routed)
   - Log final confidence and agreement
   - Return ConsensusResponse

2. async fn query_all_llms(
       &self,
       query: &str,
       models: &[&str]
   ) -> Result<Vec<LLMResponse>>

   - Get charlie_integration lock
   - For each model name:
     * Get LLM client (you'll need to add get_llm_client method to MissionCharlieIntegration)
     * Call client.generate(query, 0.7).await
     * Add to responses vector
   - Return responses

3. async fn apply_quantum_voting(&self, responses: &[LLMResponse]) -> Result<QuantumVoteResult>

   - Get charlie_integration.read()
   - Call quantum_voting.vote(responses).await
   - Return result

4. async fn apply_thermodynamic_consensus(&self, responses: &[LLMResponse]) -> Result<ThermodynamicResult>

   - Get charlie_integration.read()
   - Call thermodynamic_consensus.converge(responses).await
   - Return result

5. async fn apply_transfer_entropy_routing(&self, responses: &[LLMResponse]) -> Result<RoutedResponses>

   - Get charlie_integration.read()
   - Call transfer_entropy_router.route(responses).await
   - Return result

6. fn fuse_consensus_results(
       &self,
       quantum: &QuantumVoteResult,
       thermo: &ThermodynamicResult,
       routed: &RoutedResponses
   ) -> Result<ConsensusResponse>

   - Weight quantum vote at 0.4
   - Weight thermodynamic at 0.35
   - Weight routing at 0.25
   - Combine confidences with weights
   - Select best response text (use quantum's if highest weight)
   - Create algorithm_weights vec with the three contributions
   - Return ConsensusResponse

Import the bridge types:
use crate::foundation::orchestration::integration::bridges::*;

Add all necessary use statements for the types you're using.
```

---

### Prompt 1.5: Create CLI Binary (Composer: `Cmd/Ctrl + I`)

```
Create src/bin/prism_mec.rs

This is the main executable binary for the MEC system.

Requirements:

1. Use clap for argument parsing
2. Support these commands:
   - consensus <query> [--models gpt-4,claude,gemini]
   - diagnostics

3. Main function:
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize env_logger with info level
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    // Parse CLI
    let cli = Cli::parse();

    // Print banner
    println!("🧠 PRISM-AI Meta Emergent Computation System");
    println!("="​.repeat(70));

    // Initialize orchestrator
    let orchestrator = initialize_orchestrator().await?;

    // Dispatch command
    match cli.command {
        Commands::Consensus { query, models } => {
            run_llm_consensus(&orchestrator, &query, &models).await?;
        }
        Commands::Diagnostics => {
            run_diagnostics(&orchestrator).await?;
        }
    }

    Ok(())
}
```

4. async fn initialize_orchestrator() -> Result<PrismAIOrchestrator>
   - Create OrchestratorConfig with sensible defaults
   - Call PrismAIOrchestrator::new(config).await
   - Log initialization steps
   - Return orchestrator

5. async fn run_llm_consensus(orchestrator: &PrismAIOrchestrator, query: &str, models_str: &str)
   - Print query and models
   - Split models string by comma
   - Call orchestrator.llm_consensus(query, &models).await
   - Print results beautifully:
     * Separator line
     * Consensus text with wrapping
     * Separator line
     * Confidence percentage
     * Agreement score
     * Algorithm contributions table

6. async fn run_diagnostics(orchestrator: &PrismAIOrchestrator)
   - Print "System Diagnostics" header
   - Show LLM clients available
   - Show consensus algorithms active
   - Show system status

Add to Cargo.toml dependencies if needed:
clap = { version = "4.4", features = ["derive"] }
tokio = { version = "1.35", features = ["full"] }
env_logger = "0.10"
anyhow = "1.0"

Also add the binary target to Cargo.toml.
```

---

## 🧪 TESTING PROMPTS

### Test Prompt (Chat: `Cmd/Ctrl + L`)

**After implementing each phase**:

```
Help me test the [component] I just implemented:

1. What unit tests should I write?
2. How do I run integration tests?
3. What edge cases should I check?
4. Generate test code for me

Component location: [file path]
```

---

## 🐛 DEBUGGING PROMPTS

### Debug Prompt (Chat: `Cmd/Ctrl + L`)

**When you hit an error**:

```
I'm getting this compilation error:

[paste error]

The code is in [file path].

Help me:
1. Understand what's wrong
2. Fix the error
3. Explain why it happened
```

---

## 🎯 QUICK REFERENCE

### Cursor Shortcuts:
- `Cmd/Ctrl + I` - **Composer** (multi-file, big changes)
- `Cmd/Ctrl + K` - **Inline Edit** (single location, quick edit)
- `Cmd/Ctrl + L` - **Chat** (questions, debugging)
- `Cmd/Ctrl + J` - **Terminal** (run commands)
- `Tab` - **Accept autocomplete**

### When to Use What:
- **Composer**: Creating new files, implementing features
- **Inline Edit**: Small changes, adding methods
- **Chat**: Questions, debugging, architecture
- **Terminal**: Testing, running, building

---

## 📅 DAILY WORKFLOW

### Morning (9am-12pm):
1. Open Cursor
2. Pick component from priority guide
3. Use Composer to generate scaffold
4. Review and adjust
5. Test compilation

### Afternoon (1pm-5pm):
1. Use Inline Edit for tweaks
2. Use Chat for debugging
3. Write tests
4. Run tests
5. Commit when working

---

## ✅ COPY-PASTE WORKFLOW

### Day 1 (Phase 0):
1. Open Cursor → Open Composer
2. Copy Prompt 0.1 → Paste → Execute
3. Open Terminal → Run `cargo check`
4. ✅ Done!

### Days 2-5 (Phase 1):
1. Open Composer → Copy Prompt 1.1 → Execute
2. Review code → Accept or modify
3. Open Composer → Copy Prompt 1.2 → Execute
4. Test compilation
5. Copy Prompt 1.3 into Chat → Get test instructions
6. Run tests
7. ✅ Working LLM consensus!

### Days 6-10 (Phase 2):
1. Copy Prompt 2.1 into Composer
2. Copy Prompt 2.2 into Inline Edit
3. Test with synthetic data
4. ✅ Materials discovery working!

---

## 💡 PRO TIPS

### Tip 1: Build Context for Cursor
```
"Look at @codebase and tell me where the MaterialsAdapter is implemented"
"Look at @folder foundation/orchestration and show me the quantum voting implementation"
```

### Tip 2: Iterative Refinement
```
"The code you generated doesn't compile. Fix the import statements."
"Make this more idiomatic Rust"
"Add better error handling here"
```

### Tip 3: Test Generation
```
"Generate comprehensive tests for this file"
"Add a test that verifies [specific behavior]"
```

### Tip 4: Documentation
```
"Add detailed rustdoc comments to all public methods"
"Generate a README for this module"
```

---

## 🎬 YOUR LITERAL NEXT ACTIONS

### Action 1 (Right Now - 2 minutes):
1. Open Cursor IDE
2. Open folder: `/home/diddy/Desktop/PRISM-FINNAL-PUSH`

### Action 2 (Next 10 minutes):
1. Press `Cmd/Ctrl + I`
2. Copy Prompt 0.1 from above
3. Paste into Composer
4. Review results
5. Press Accept

### Action 3 (Next 1 hour):
1. Press `Cmd/Ctrl + I`
2. Copy Prompt 1.1
3. Paste and execute
4. Review the llm_consensus_bridge.rs it creates
5. Make any adjustments

### Action 4 (Today):
1. Copy Prompt 1.2
2. Create main executable
3. Test compilation
4. Set up API keys
5. **Run your first consensus query!** 🎉

---

## 📊 ESTIMATED TIME WITH CURSOR

| Phase | Manual Time | With Cursor | Savings |
|-------|-------------|-------------|---------|
| P0 Setup | 4 hours | **30 min** | 87% ⚡ |
| P1 LLM Consensus | 5 days | **2 days** | 60% ⚡ |
| P2 Materials | 5 days | **2 days** | 60% ⚡ |
| P3 Context | 7 days | **3 days** | 57% ⚡ |
| P4 MEC Loop | 10 days | **5 days** | 50% ⚡ |
| P5 Advanced | 15 days | **8 days** | 47% ⚡ |
| **TOTAL** | **43 days** | **20 days** | **53% faster!** |

**With Cursor IDE: Complete MEC in 4 weeks instead of 9 weeks!**

---

## 🚀 START NOW

1. **Open Cursor**
2. **Open this file** (CURSOR_PROMPTS_READY_TO_USE.md)
3. **Copy Prompt 0.1**
4. **Paste into Composer** (`Cmd/Ctrl + I`)
5. **Accept results**
6. **Move to Prompt 1.1**
7. **Keep going!**

**You're 70% done. With Cursor, the final 30% takes 4 weeks, not 3 months!** 🚀

---

*Cursor implementation guide created: October 25, 2024*
*Prompts ready to copy-paste*
*Start with Prompt 0.1 in Composer mode*
