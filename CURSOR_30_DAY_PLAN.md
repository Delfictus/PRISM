# 30-Day MEC Implementation Plan with Cursor IDE
## Day-by-Day Execution Guide

**Goal**: Working MEC system in 30 days using Cursor IDE
**Hours per day**: 4-6 hours
**Approach**: Copy-paste prompts, let Cursor generate, test, iterate

---

## 📅 WEEK 1: FOUNDATION + LLM CONSENSUS

### **DAY 1: Setup & Structure** [4 hours]

**Morning (9am-11am): Create Structure**

1. **Open Cursor**:
   ```bash
   cursor /home/diddy/Desktop/PRISM-FINNAL-PUSH
   ```

2. **Composer (`Cmd/Ctrl + I`)** - Copy this exact prompt:
   ```
   Create MEC system directory structure with stub files:

   Directories:
   - src/mec/
   - src/ontogenic_io/probes/
   - src/meta_learning/
   - foundation/orchestration/integration/bridges/

   Create these files:

   src/mec/mod.rs:
   ```rust
   //! Meta Emergent Computation Engine
   use anyhow::Result;
   use std::collections::HashMap;

   pub struct MetaEmergentComputation {
       current_parameters: HashMap<String, f64>,
       mutation_rate: f64,
   }

   impl MetaEmergentComputation {
       pub fn new() -> Result<Self> {
           Ok(Self {
               current_parameters: HashMap::new(),
               mutation_rate: 0.05,
           })
       }
   }
   ```

   src/ontogenic_io/mod.rs:
   ```rust
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
   ```

   src/ontogenic_io/probes/mod.rs:
   ```rust
   //! Sensory Probes
   use anyhow::Result;

   pub trait SensoryProbe {
       type Output;
       fn capture(&self) -> Result<Self::Output>;
   }
   ```

   src/meta_learning/mod.rs:
   ```rust
   //! Meta-Learning Controller
   use anyhow::Result;

   pub struct MetaLearningController {}
   impl MetaLearningController {
       pub fn new() -> Result<Self> { Ok(Self {}) }
   }
   ```

   foundation/orchestration/integration/bridges/mod.rs:
   ```rust
   //! Integration bridges
   ```

   Also update src/lib.rs to add at the end:
   pub mod mec;
   pub mod ontogenic_io;
   pub mod meta_learning;
   ```

3. **Terminal (`Cmd/Ctrl + J`)**:
   ```bash
   cargo check
   ```
   ✅ Should compile!

**Afternoon (1pm-5pm): Verify Existing Infrastructure**

4. **Chat (`Cmd/Ctrl + L`)**:
   ```
   @codebase Show me where the following exist:
   1. LLM clients (OpenAI, Claude, Gemini)
   2. Quantum voting consensus
   3. Thermodynamic consensus
   4. MissionCharlieIntegration
   5. PrismAIOrchestrator

   I need to understand what's already implemented before building the bridges.
   ```

5. **Read** the files Cursor shows you
6. **Document** what you learned

**End of Day 1**: ✅ Structure ready, existing code understood

---

### **DAY 2: LLM Consensus Bridge** [6 hours]

**Morning (9am-12pm): Create Bridge**

1. **Composer (`Cmd/Ctrl + I`)** - Copy this:
   ```
   I need to implement LLM consensus that uses existing infrastructure.

   Create foundation/orchestration/integration/bridges/llm_consensus_bridge.rs

   Define these types:
   ```rust
   use anyhow::Result;
   use serde::{Serialize, Deserialize};

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ConsensusRequest {
       pub query: String,
       pub models: Vec<String>,
       pub temperature: f32,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ConsensusResponse {
       pub text: String,
       pub confidence: f64,
       pub agreement_score: f64,
       pub model_responses: Vec<ModelResponse>,
       pub algorithm_weights: Vec<(String, f64)>,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ModelResponse {
       pub model: String,
       pub text: String,
       pub tokens: usize,
       pub cost: f64,
   }
   ```

   Then look at foundation/orchestration/integration/prism_ai_integration.rs

   Add this method to PrismAIOrchestrator:
   ```rust
   pub async fn llm_consensus(
       &self,
       query: &str,
       models: &[&str]
   ) -> Result<ConsensusResponse> {
       // Implementation steps:
       // 1. Query all LLM clients in parallel
       // 2. Get quantum voting result from charlie_integration.quantum_voting.vote()
       // 3. Get thermodynamic consensus from charlie_integration.thermodynamic_consensus.converge()
       // 4. Get routing from charlie_integration.transfer_entropy_router.route()
       // 5. Fuse results with 40% quantum, 35% thermo, 25% routing weights
       // 6. Return ConsensusResponse
   }
   ```

   The charlie_integration field is Arc<RwLock<MissionCharlieIntegration>>.
   MissionCharlieIntegration has quantum_voting, thermodynamic_consensus, and transfer_entropy_router fields.

   Add detailed logging with log::info! for each step.
   Handle errors properly.
   ```

2. **Review** the generated code
3. **Test** compilation

**Afternoon (1pm-5pm): Fix Issues**

4. **Terminal**:
   ```bash
   cargo check 2>&1 | head -50
   ```

5. **Chat (`Cmd/Ctrl + L`)** if errors:
   ```
   I'm getting these compilation errors:
   [paste errors]

   Looking at the code in:
   - foundation/orchestration/integration/prism_ai_integration.rs
   - foundation/orchestration/integration/bridges/llm_consensus_bridge.rs

   Help me fix them.
   ```

6. **Iterate** until it compiles

**End of Day 2**: ✅ LLM consensus bridge implemented

---

### **DAY 3: Main Executable** [6 hours]

**Morning (9am-12pm): Create Binary**

1. **Composer (`Cmd/Ctrl + I`)**:
   ```
   Create src/bin/prism_mec.rs as the main MEC executable.

   Requirements:
   1. Use clap v4 for CLI parsing
   2. Commands:
      - consensus <query> --models <models>
      - diagnostics

   3. Main function structure:
   ```rust
   use anyhow::Result;
   use clap::{Parser, Subcommand};
   use prism_ai::foundation::PrismAIOrchestrator;

   #[derive(Parser)]
   #[command(name = "prism-mec")]
   #[command(about = "PRISM-AI MEC System")]
   struct Cli {
       #[command(subcommand)]
       command: Commands,
   }

   #[derive(Subcommand)]
   enum Commands {
       Consensus {
           query: String,
           #[arg(short, long, default_value = "gpt-4,claude,gemini")]
           models: String,
       },
       Diagnostics,
   }

   #[tokio::main]
   async fn main() -> Result<()> {
       env_logger::init();
       let cli = Cli::parse();

       println!("🧠 PRISM-AI MEC System");
       println!("="​.repeat(70));

       let orchestrator = init_orchestrator().await?;

       match cli.command {
           Commands::Consensus { query, models } => {
               run_consensus(&orchestrator, &query, &models).await?;
           }
           Commands::Diagnostics => {
               run_diagnostics(&orchestrator).await?;
           }
       }
       Ok(())
   }

   async fn init_orchestrator() -> Result<PrismAIOrchestrator> {
       // Initialize with default config
       // Log each step
   }

   async fn run_consensus(orch: &PrismAIOrchestrator, query: &str, models: &str) -> Result<()> {
       // Parse models
       // Call orch.llm_consensus()
       // Print results beautifully
   }

   async fn run_diagnostics(orch: &PrismAIOrchestrator) -> Result<()> {
       // Print system status
   }
   ```

   Also update Cargo.toml to add the binary:
   [[bin]]
   name = "prism-mec"
   path = "src/bin/prism_mec.rs"
   ```

**Afternoon (1pm-5pm): Test and Polish**

2. **Terminal**:
   ```bash
   cargo build --bin prism-mec 2>&1 | head -100
   ```

3. **Fix** compilation errors with Chat

4. **Inline Edit** (`Cmd/Ctrl + K`) to improve output formatting

**End of Day 3**: ✅ Main executable compiles

---

### **DAY 4: Wire Everything Together** [6 hours]

**Morning (9am-12pm): Integration**

1. **Chat (`Cmd/Ctrl + L`)**:
   ```
   Look at @file foundation/orchestration/integration/mission_charlie_integration.rs

   I need to add a get_llm_client() method that returns the appropriate LLM client based on model name.

   The MissionCharlieIntegration struct should have LLM client fields. Show me how to:
   1. Add the LLM client fields if missing
   2. Implement get_llm_client(model_name: &str) -> Result<Arc<dyn LLMClient>>
   ```

2. **Apply** Cursor's suggestions

3. **Inline Edit** to add any missing methods

**Afternoon (1pm-5pm): First Test Run**

4. **Terminal**: Set up environment
   ```bash
   # Create .env file
   cat > .env << 'EOF'
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   RUST_LOG=info
   EOF

   # Source it
   source .env
   ```

5. **Terminal**: Build
   ```bash
   cargo build --release --bin prism-mec
   ```

6. **Terminal**: Test
   ```bash
   ./target/release/prism-mec diagnostics
   ```

7. **Debug** any issues

**End of Day 4**: ✅ Executable runs diagnostics

---

### **DAY 5: First Real Consensus!** [6 hours]

**Morning (9am-12pm): Debug Consensus**

1. **Terminal**: Try consensus
   ```bash
   ./target/release/prism-mec consensus "What is 2+2?"
   ```

2. **Chat** if errors:
   ```
   Getting error when running LLM consensus:
   [paste error]

   Debug this for me. Check:
   - Are the LLM clients initialized?
   - Is the quantum voting being called?
   - Are API keys loaded correctly?
   ```

3. **Fix** issues with Inline Edit

**Afternoon (1pm-5pm): Polish & Demo**

4. **Inline Edit**: Improve output
5. **Test** multiple queries
6. **Document** how to use it

**🎉 END OF WEEK 1: WORKING LLM CONSENSUS WITH QUANTUM VOTING!**

Demo query:
```bash
./target/release/prism-mec consensus "What is consciousness?" \
  --models gpt-4,claude,gemini
```

---

## 📅 WEEK 2: MATERIALS DISCOVERY

### **DAY 6: Materials Bridge Start** [6 hours]

**Composer Prompt**:
```
Create foundation/orchestration/integration/bridges/materials_bridge.rs

Look at src/cma/applications/mod.rs to see the existing MaterialsAdapter.
Look at src/cma/mod.rs to see CausalManifoldAnnealing.

Implement:
1. MaterialDiscoveryResult type
2. materials_to_cma_problem() method
3. discover_materials() method for PrismAIOrchestrator

The method should:
- Create CMA problem from MaterialProperties
- Run CausalManifoldAnnealing solver
- Apply MaterialsAdapter
- Return results with timing

Add to prism_ai_integration.rs impl block.
```

### **DAY 7: Wire CMA Solver** [6 hours]

**Chat**:
```
@codebase How do I create and run a CMA problem?

Show me examples of using CausalManifoldAnnealing.
I need to convert MaterialProperties to a Problem trait object.
```

**Apply** what you learn

### **DAY 8: Add CLI Command** [4 hours]

**Inline Edit in prism_mec.rs**:
```
Add Materials command with bandgap, conductivity, hardness args.
Implement handler that calls discover_materials().
```

### **DAY 9: Test Materials** [4 hours]

**Terminal**:
```bash
./target/release/prism-mec materials --bandgap 1.5 --conductivity 1e6
```

**Debug** and iterate

### **DAY 10: Polish** [4 hours]

**🎉 END OF WEEK 2: MATERIALS DISCOVERY WORKING!**

---

## 📅 WEEK 3: CONTEXT AWARENESS

### **DAY 11-12: Text Tone Probe** [12 hours total]

**Day 11 - Composer**:
```
Create src/ontogenic_io/probes/text_tone.rs

Full implementation of TextToneProbe with:
- ToneFeatures struct (caps_ratio, punct_entropy, valence, arousal, etc.)
- analyze() method
- Feature extraction methods (caps_ratio, punct_entropy, etc.)
- Simple positive/negative word lists
- Unit tests

Use only std library - no external dependencies.
```

**Day 12 - Test**:
```bash
cargo test text_tone
```

### **DAY 13-14: Cursor Probe** [12 hours]

**Day 13 - Composer**:
```
Create src/ontogenic_io/probes/cursor.rs

Implement CursorProbe that tracks mouse dynamics.
See CURSOR_PROMPTS_READY_TO_USE.md Prompt 3.2 for full spec.
```

**Day 14 - Test and integrate**

### **DAY 15-17: Fusion & Integration** [18 hours]

**Day 15 - Composer**:
```
Create src/ontogenic_io/fusion.rs

Implement multi-modal fusion that:
- Takes latent vectors from multiple probes
- Applies attention-based weighting
- Fuses into unified context vector
- Returns ContextState
```

**Day 16 - Integration**:
```
Update PrismAIOrchestrator to:
- Take OntogenicIO as optional parameter
- Use context when making decisions
- Show context influencing module selection
```

**Day 17 - Demo**

**🎉 END OF WEEK 3: CONTEXT-AWARE SYSTEM!**

---

## 📅 WEEK 4: BASIC SELF-EVOLUTION

### **DAY 18-20: MEC Evolution Engine** [18 hours]

**Day 18 - Composer**:
```
Implement the core evolution cycle in src/mec/mod.rs

Add these methods to MetaEmergentComputation:

1. pub async fn evolve_cycle(&mut self) -> Result<EvolutionReport>
   - Monitor current performance
   - Generate parameter mutation
   - Test mutation in sandbox
   - Apply if better
   - Record change

2. fn generate_mutation(&self) -> ParameterMutation
   - Select random parameter
   - Add Gaussian noise
   - Return mutation

3. async fn test_mutation(&self, mutation: &ParameterMutation) -> Result<f64>
   - Apply mutation temporarily
   - Run test task
   - Measure fitness
   - Restore original parameters

4. fn apply_mutation(&mut self, mutation: ParameterMutation)
   - Update current_parameters
   - Record in history

Define ParameterMutation and EvolutionReport types.
```

**Days 19-20**: Test and debug evolution

### **DAY 21-22: Evolution Command** [12 hours]

**Inline Edit in prism_mec.rs**:
```
Add Evolve command:
- Takes number of cycles
- Runs MEC evolution
- Prints progress
- Shows improvements
```

**Test**:
```bash
./target/release/prism-mec evolve --cycles 10
```

**🎉 END OF WEEK 4: SYSTEM CAN SELF-EVOLVE!**

---

## 📅 DAYS 23-30: ADVANCED FEATURES

### **DAY 23-25: Drug Discovery** [18 hours]

**Composer**:
```
Create drug discovery bridge similar to materials bridge.
Use BiomolecularAdapter from src/cma/applications/mod.rs
```

### **DAY 26-28: Audio Probe** [18 hours]

**Composer**:
```
Create src/ontogenic_io/probes/audio.rs
Extract prosody features using rustfft and basic DSP
```

### **DAY 29-30: Integration & Polish** [12 hours]

**Tasks**:
- Integration testing
- Documentation
- Bug fixes
- Performance tuning

**🎉 END OF DAY 30: COMPLETE MEC SYSTEM!**

---

## ⚡ SUPER QUICK VERSION (Aggressive Timeline)

### Week 1:
- **Day 1**: Setup (Prompt 0.1)
- **Days 2-3**: LLM Bridge (Prompts 1.1, 1.4)
- **Day 4**: Main executable (Prompt 1.5)
- **Day 5**: Test & Demo 🎉

### Week 2:
- **Days 6-8**: Materials (Prompt 2.1, 2.2)
- **Days 9-10**: Test & Demo 🎉

### Week 3-4:
- **Days 11-14**: Context probes
- **Days 15-17**: Integration
- **Days 18-22**: MEC evolution
- **Days 23-30**: Advanced features

---

## 📊 DAILY PROGRESS TRACKER

```
Day  Phase    Component               Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
01   P0       Structure               [ ]
02   P1       LLM Bridge              [ ]
03   P1       Wire Consensus          [ ]
04   P1       Main Executable         [ ]
05   P1       Test & Demo             [ ] 🎉
06   P2       Materials Bridge        [ ]
07   P2       CMA Integration         [ ]
08   P2       CLI Command             [ ]
09   P2       Test Materials          [ ]
10   P2       Demo                    [ ] 🎉
11   P3       Text Tone               [ ]
12   P3       Test Text               [ ]
13   P3       Cursor Probe            [ ]
14   P3       Test Cursor             [ ]
15   P3       Fusion                  [ ]
16   P3       Integration             [ ]
17   P3       Demo Context            [ ] 🎉
18   P4       MEC Engine              [ ]
19   P4       Evolution Methods       [ ]
20   P4       Test Evolution          [ ]
21   P4       Evolution Command       [ ]
22   P4       Demo Self-Evolve        [ ] 🎉
23   P5       Drug Discovery          [ ]
24   P5       Test Drugs              [ ]
25   P5       Audio Probe             [ ]
26   P5       Test Audio              [ ]
27   P5       Visual Probe            [ ]
28   P5       Full Integration        [ ]
29   P5       Testing                 [ ]
30   P5       COMPLETE!               [ ] 🚀
```

---

## 🎯 CURSOR TIPS FOR SPEED

### Use Tabs for Context
```
@codebase - Reference whole codebase
@file path/to/file.rs - Reference specific file
@folder src/mec - Reference folder
```

### Batch Similar Work
Instead of creating probes one-by-one, create multiple at once:
```
Create all probe files following the same pattern:
- text_tone.rs
- cursor.rs
- audio.rs
Each with [Probe]Probe struct and analyze() method
```

### Ask Cursor to Test
```
Generate comprehensive unit tests for @file src/ontogenic_io/probes/text_tone.rs
Cover normal cases, edge cases, and error conditions
```

### Let Cursor Debug
```
This code isn't working: @file src/mec/mod.rs

Error: [paste error]

Fix it.
```

---

## ✅ DAILY CHECKLIST TEMPLATE

**Copy this for each day**:

```
□ Open Cursor IDE
□ Review yesterday's progress
□ Pick today's component from 30-day plan
□ Copy Cursor prompt for that component
□ Paste into Composer (Cmd+I)
□ Review generated code
□ Make adjustments with Inline Edit (Cmd+K)
□ Test compilation (Cmd+J → cargo check)
□ Fix errors with Chat (Cmd+L)
□ Write/run tests
□ Commit when working (git commit)
□ Update progress tracker
□ Document learnings
```

---

## 🎬 START THIS SECOND

1. **Open Cursor** (right now!)
2. **Open PRISM-FINNAL-PUSH folder**
3. **Press `Cmd/Ctrl + I`**
4. **Copy Day 1 prompt** (from above)
5. **Paste and execute**
6. **Watch Cursor build your structure!** ✨

**In 30 days you'll have a complete self-evolving AI system!** 🚀

---

*30-Day plan created: October 25, 2024*
*Each day has exact prompts ready to copy*
*Just follow the plan and let Cursor do the work!*
