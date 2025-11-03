# MEC Implementation Guide for Cursor IDE
## Complete Workflow Using Cursor's AI Features

**Target**: Implement MEC system efficiently using Cursor IDE's AI capabilities
**Philosophy**: Let Cursor do the heavy lifting while you guide the architecture

---

## 🎯 CURSOR IDE STRATEGY

### Why Cursor is Perfect for This:
1. **Context Awareness** - Can see your entire codebase
2. **Composer Mode** - Multi-file edits for integration work
3. **Inline Edit** - Quick fixes and small changes
4. **AI Chat** - Architecture discussions and planning
5. **Auto-complete** - Speeds up boilerplate

### Workflow Split (60/40 Rule):
- **60% Composer** - For new features, multi-file integration
- **40% Inline Edit** - For tweaks, bug fixes, small additions

---

## 📂 PHASE 0: Initial Setup with Cursor

### Step 1: Open Project in Cursor

```bash
# Open Cursor IDE
cursor /home/diddy/Desktop/PRISM-FINNAL-PUSH
```

### Step 2: Create Directory Structure (Composer Mode)

**Press**: `Cmd/Ctrl + I` (open Composer)

**Prompt to Cursor**:
```
Create the following directory structure and stub files for the MEC system:

Directories needed:
- src/mec/
- src/ontogenic_io/probes/
- src/meta_learning/
- foundation/orchestration/integration/bridges/

Files to create with basic structure:
1. src/mec/mod.rs - Main MEC engine module with MetaEmergentComputation struct
2. src/ontogenic_io/mod.rs - Ontogenic IO coordinator with OntogenicIO struct
3. src/ontogenic_io/probes/mod.rs - Probe trait definition
4. src/meta_learning/mod.rs - Meta-learning controller stub

Each file should have:
- Proper module documentation
- Basic struct definition
- Constructor (new() method)
- use anyhow::Result imports

Also update src/lib.rs to export these new modules.
```

**Expected Result**: Cursor creates all directories and files with proper structure

### Step 3: Verify Compilation (Terminal in Cursor)

**Press**: `Cmd/Ctrl + J` (open terminal)

```bash
cargo check
```

**Success Criteria**: Project compiles without errors

---

## 🚀 PHASE 1: LLM Consensus Implementation

### Part 1.1: Create LLM Consensus Bridge (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Create a new file: foundation/orchestration/integration/bridges/llm_consensus_bridge.rs

This file should implement the LLM consensus functionality that:

1. Queries multiple LLM clients (OpenAI, Claude, Gemini) in parallel
2. Applies quantum voting consensus (using existing charlie_integration.quantum_voting)
3. Applies thermodynamic consensus (using existing charlie_integration.thermodynamic_consensus)
4. Applies transfer entropy routing (using existing charlie_integration.transfer_entropy_router)
5. Fuses results with weighted combination

Key types needed:
- ConsensusRequest { query: String, models: Vec<String>, temperature: f32 }
- ConsensusResponse { text: String, confidence: f64, agreement_score: f64, model_responses: Vec<ModelResponse>, algorithm_weights: Vec<(String, f64)> }
- ModelResponse { model: String, text: String, tokens: usize, cost: f64 }

The implementation should be added as methods to the existing PrismAIOrchestrator struct in:
foundation/orchestration/integration/prism_ai_integration.rs

Methods to implement:
- pub async fn llm_consensus(&self, query: &str, models: &[&str]) -> Result<ConsensusResponse>
- async fn query_all_llms(&self, query: &str, models: &[&str]) -> Result<Vec<LLMResponse>>
- async fn apply_quantum_voting(&self, responses: &[LLMResponse]) -> Result<QuantumVoteResult>
- async fn apply_thermodynamic_consensus(&self, responses: &[LLMResponse]) -> Result<ThermodynamicResult>
- async fn apply_transfer_entropy_routing(&self, responses: &[LLMResponse]) -> Result<RoutedResponses>
- fn fuse_consensus_results(&self, quantum: &QuantumVoteResult, thermo: &ThermodynamicResult, routed: &RoutedResponses) -> Result<ConsensusResponse>

Use the existing LLM client infrastructure from the charlie_integration field.
Add proper error handling and logging (use log::info!, log::debug!).
```

**What Cursor Will Do**:
- Create the bridge file
- Implement all methods
- Add necessary imports
- Connect to existing infrastructure

**Your Job**:
- Review the code Cursor generates
- Accept or modify implementation
- Test compilation

### Part 1.2: Update Module Exports (Inline Edit)

**Navigate to**: `foundation/orchestration/integration/mod.rs`

**Press**: `Cmd/Ctrl + K` (inline edit)

**Cursor Prompt**:
```
Add export for the bridges module:
pub mod bridges;
```

**Navigate to**: Create `foundation/orchestration/integration/bridges/mod.rs`

**Use Composer** (`Cmd/Ctrl + I`):
```
Create foundation/orchestration/integration/bridges/mod.rs that exports:
pub mod llm_consensus_bridge;

Re-export the main types:
pub use llm_consensus_bridge::{ConsensusRequest, ConsensusResponse, ModelResponse};
```

### Part 1.3: Create Main Executable (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Create src/bin/prism_mec.rs as the main executable for the MEC system.

It should:
1. Use clap for CLI parsing with these commands:
   - consensus <query> [--models <models>] - Run LLM consensus
   - diagnostics - Show system status

2. Initialize PrismAIOrchestrator with OrchestratorConfig

3. For the consensus command:
   - Parse comma-separated model names
   - Call orchestrator.llm_consensus()
   - Display results beautifully with:
     * The consensus text
     * Confidence percentage
     * Agreement score
     * Algorithm contribution breakdown

4. For diagnostics:
   - Show which LLM clients are available
   - Show which algorithms are active
   - Confirm system operational

Use proper async/await with tokio::main.
Add env_logger initialization.
Include nice formatting with colored output if possible.
```

**What Cursor Will Do**:
- Create complete executable
- Set up CLI with clap
- Implement both commands
- Add nice formatting

### Part 1.4: Test and Debug (Chat Mode)

**Open Chat**: `Cmd/Ctrl + L`

**Ask Cursor**:
```
Help me test the LLM consensus implementation. What do I need to do to:
1. Set up API keys
2. Run a test query
3. Debug any compilation errors
4. Verify the quantum voting is being called
```

**Then in Terminal**:
```bash
# Set up API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Compile
cargo build --release

# Test
./target/release/prism_mec consensus "What is consciousness?" --models gpt-4,claude,gemini
```

---

## 🔬 PHASE 2: Materials Discovery Implementation

### Part 2.1: Create Materials Bridge (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Create foundation/orchestration/integration/bridges/materials_bridge.rs

This should implement materials discovery by:

1. Creating a materials_to_cma_problem() method that converts MaterialProperties to a CMA Problem
2. Implementing discover_materials() method in PrismAIOrchestrator that:
   - Converts target properties to CMA problem
   - Calls CausalManifoldAnnealing::solve()
   - Applies MaterialsAdapter::discover_material()
   - Optionally applies quantum refinement
   - Validates with thermodynamic network
   - Returns MaterialDiscoveryResult

Types needed:
- MaterialDiscoveryRequest
- MaterialDiscoveryResult { candidates: Vec<MaterialCandidate>, total_evaluated: usize, best_score: f64, computation_time_ms: f64 }

The MaterialsAdapter already exists at src/cma/applications/mod.rs - use it.
The CausalManifoldAnnealing engine exists at src/cma/mod.rs - use it.

Add logging for each step.
Track computation time.
```

### Part 2.2: Add Materials Command (Inline Edit)

**Navigate to**: `src/bin/prism_mec.rs`

**Press**: `Cmd/Ctrl + K`

**Cursor Prompt**:
```
Add a new Materials command to the CLI:

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
}

Implement the handler that calls orchestrator.discover_materials() and displays results.
```

### Part 2.3: Test Materials Discovery

**Terminal**:
```bash
cargo build --release
./target/release/prism_mec materials --bandgap 1.5 --conductivity 1e6
```

---

## 🎤 PHASE 3: Ontogenic IO - Text Tone Probe

### Part 3.1: Text Tone Probe (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Create src/ontogenic_io/probes/text_tone.rs

This should implement a text tone analysis probe that extracts:
- Caps ratio (fraction of uppercase letters)
- Punctuation entropy (Shannon entropy of punctuation distribution)
- Valence (emotional positivity/negativity from simple word lists)
- Arousal (emotional intensity)
- Exclamation and question counts

Create these types:
- TextToneProbe struct
- ToneFeatures struct with all the above fields

Implement:
- new() constructor
- analyze(&self, text: &str) -> Result<ToneFeatures>
- Helper methods: caps_ratio, punct_entropy, lexicon_valence, lexicon_arousal

This should have NO external dependencies - just use std library.
Keep it simple - we'll add complexity later.

For lexicon valence/arousal, just use hardcoded word lists of common positive/negative words.
```

**What Cursor Will Do**:
- Create complete probe implementation
- Implement all feature extraction
- Add simple word lists
- Pure Rust, no dependencies

### Part 3.2: Test Text Tone Probe

**Use Chat**: `Cmd/Ctrl + L`

**Ask Cursor**:
```
Create a unit test for TextToneProbe that tests:
1. Analyzing positive text
2. Analyzing negative text
3. Analyzing neutral text
4. Edge cases (empty string, only punctuation)

Add the tests to src/ontogenic_io/probes/text_tone.rs
```

**Terminal**:
```bash
cargo test text_tone
```

---

## 🖱️ PHASE 3: Ontogenic IO - Cursor Probe

### Part 3.3: Cursor Dynamics Probe (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Create src/ontogenic_io/probes/cursor.rs

This probe tracks mouse cursor dynamics and extracts features:
- Velocity (mean and std dev)
- Acceleration (mean)
- Jerk (mean)
- Hesitation count (velocity drops below threshold)
- Path efficiency (straight-line distance / actual path)

Types needed:
- CursorProbe { history: Vec<CursorPoint>, max_history: usize }
- CursorPoint { x: f32, y: f32, timestamp_ms: u64 }
- CursorFeatures { velocity_mean, velocity_std, acceleration_mean, jerk_mean, hesitation_count, path_efficiency }

Methods:
- new() -> Self
- add_point(&mut self, x: f32, y: f32)
- extract_features(&self) -> Result<CursorFeatures>
- calculate_velocities(&self) -> Vec<f32>
- calculate_accelerations(&self, velocities: &[f32]) -> Vec<f32>
- calculate_jerks(&self, accelerations: &[f32]) -> Vec<f32>
- count_hesitations(&self, velocities: &[f32]) -> usize
- calculate_efficiency(&self) -> f32

Use simple math - no external dependencies.
Keep max_history at 100 points.
```

---

## 🧬 PHASE 4: Basic MEC Loop

### Part 4.1: MEC Engine Core (Composer Mode)

**Open Composer**: `Cmd/Ctrl + I`

**Cursor Prompt**:
```
Implement the core MEC engine in src/mec/mod.rs

The MetaEmergentComputation struct should have:
- orchestrator: PrismAIOrchestrator
- current_parameters: HashMap<String, f64>
- performance_history: Vec<PerformanceRecord>
- mutation_rate: f64

Implement these methods:

1. pub async fn evolve_cycle(&mut self) -> Result<EvolutionReport>
   This is the main evolution loop that:
   - Monitors current performance
   - Generates a parameter mutation
   - Tests the mutation in a sandbox
   - Applies it if better
   - Records the change

2. fn generate_mutation(&self) -> ParameterMutation
   - Randomly select a parameter
   - Apply Gaussian noise scaled by mutation_rate
   - Return the mutated parameters

3. async fn test_mutation(&self, mutation: &ParameterMutation) -> Result<f64>
   - Run a test task with mutated parameters
   - Measure performance (speed, accuracy, etc.)
   - Return fitness score

4. fn apply_mutation(&mut self, mutation: ParameterMutation) -> Result<()>
   - Update current_parameters
   - Record in performance_history

Types needed:
- ParameterMutation { param_name: String, old_value: f64, new_value: f64 }
- PerformanceRecord { timestamp: u64, fitness: f64, parameters: HashMap<String, f64> }
- EvolutionReport { improved: bool, old_fitness: f64, new_fitness: f64, mutation: Option<ParameterMutation> }

Start simple - just mutate numeric parameters, test on a simple task.
```

### Part 4.2: Add Evolution Command (Inline Edit)

**Navigate to**: `src/bin/prism_mec.rs`

**Press**: `Cmd/Ctrl + K`

**Cursor Prompt**:
```
Add an Evolve command:

Evolve {
    /// Number of evolution cycles to run
    #[arg(long, default_value = "10")]
    cycles: usize,
}

Implement the handler that:
1. Creates a MetaEmergentComputation instance
2. Runs the specified number of evolution cycles
3. Prints progress after each cycle
4. Shows final performance improvement
```

---

## 💡 CURSOR WORKFLOW TIPS

### Best Practices:

#### 1. **Use Composer for New Features**
```
Cmd/Ctrl + I → Describe the entire feature → Let Cursor scaffold
```

**When to use**:
- Creating new files
- Implementing new methods across multiple files
- Major refactoring
- Integration work

**Example**:
```
"Create a new materials discovery bridge that connects the CMA solver to the MaterialsAdapter, with proper error handling and logging"
```

#### 2. **Use Inline Edit for Quick Changes**
```
Cmd/Ctrl + K → Describe the specific change → Apply
```

**When to use**:
- Adding a method to existing struct
- Fixing a bug
- Updating a type signature
- Small refactors

**Example**:
```
"Add a timeout parameter to this function with a default value of 30 seconds"
```

#### 3. **Use Chat for Architecture Decisions**
```
Cmd/Ctrl + L → Ask about design → Get suggestions
```

**When to use**:
- Not sure how to structure something
- Need to understand existing code
- Want alternatives
- Debugging complex issues

**Example**:
```
"What's the best way to integrate the ontogenic IO context into the orchestrator's decision-making?"
```

#### 4. **Use Codebase Context**
```
@codebase - References entire codebase
@file - References specific file
@folder - References specific folder
```

**Example**:
```
"@codebase How does the quantum voting algorithm work? I need to call it from the LLM consensus bridge"
```

---

## 🔄 ITERATIVE DEVELOPMENT WORKFLOW

### Day 1 Example: LLM Consensus

**Morning** (9am - 12pm):
1. Open Composer (`Cmd/Ctrl + I`)
2. Prompt: "Create LLM consensus bridge" (from guide)
3. Review generated code
4. Make adjustments with inline edit
5. Test compilation

**Afternoon** (1pm - 5pm):
1. Create main executable with Composer
2. Set up API keys
3. Test with real LLMs
4. Debug issues with Chat
5. Polish output formatting
6. **Demo working feature!** 🎉

### Day 2-3: Iterate

Repeat same pattern for each new component:
- **Composer** for scaffold
- **Inline Edit** for tweaks
- **Chat** for questions
- **Test** continuously
- **Commit** when working

---

## 📝 CURSOR PROMPTS CHEAT SHEET

### For Composer (New Features):

```
Create [file/feature] that:
1. [First requirement]
2. [Second requirement]
3. [Third requirement]

Types needed:
- [Type 1]
- [Type 2]

Methods to implement:
- [Method 1]
- [Method 2]

Use existing [component] from [location].
Add proper error handling and logging.
```

### For Inline Edit (Quick Changes):

```
Add [thing] to this [struct/function/module]

Update [thing] to [new behavior]

Fix [bug] by [solution]

Refactor [thing] to use [approach]
```

### For Chat (Questions):

```
How should I [design decision]?

What's the best way to [implementation challenge]?

Help me understand [existing code]

I'm getting [error] - how do I fix it?

Review this implementation - any issues?
```

---

## 🎯 PHASE-BY-PHASE CURSOR WORKFLOW

### Phase 1: LLM Consensus (Days 1-5)

**Day 1**:
- Composer: Create bridges/llm_consensus_bridge.rs
- Composer: Update prism_ai_integration.rs with methods
- Inline: Add module exports
- Chat: "How do I test this?"

**Day 2-3**:
- Composer: Create src/bin/prism_mec.rs
- Inline: Add CLI commands
- Terminal: Test compilation
- Terminal: Run with real APIs

**Day 4**:
- Inline: Fix bugs
- Inline: Improve output formatting
- Chat: "How can I make the output prettier?"
- Add colored output

**Day 5**:
- Polish
- Document
- **Demo!** 🎉

### Phase 2: Materials Discovery (Days 6-10)

**Day 6**:
- Composer: Create bridges/materials_bridge.rs
- Chat: "@codebase How does MaterialsAdapter work?"
- Composer: Implement discover_materials()

**Day 7-8**:
- Inline: Wire up CMA solver
- Inline: Add Materials command to CLI
- Terminal: Test compilation
- Debug issues

**Day 9**:
- Terminal: Test with synthetic targets
- Inline: Tune parameters
- Chat: "Results don't look right - why?"

**Day 10**:
- Polish
- Validate results
- **Demo!** 🎉

### Phase 3: Context (Days 11-17)

**Day 11-12**: Text tone probe
- Composer: Create text_tone.rs
- Composer: Add unit tests
- Terminal: Run tests

**Day 13-14**: Cursor probe
- Composer: Create cursor.rs
- Composer: Add tests
- Terminal: Verify features

**Day 15-16**: Integration
- Inline: Connect to orchestrator
- Chat: "How do I make context affect decisions?"
- Implement context influence

**Day 17**:
- Test end-to-end
- **Demo context-aware behavior!** 🎉

---

## 🚀 ACCELERATION TIPS

### 1. Use Cursor's Autocomplete Aggressively
As you type, accept suggestions with `Tab`. Cursor will often complete entire functions.

### 2. Multi-File Edits in Composer
```
Update the following files to add [feature]:
1. src/mec/mod.rs - Add [thing]
2. src/lib.rs - Export [thing]
3. src/bin/prism_mec.rs - Use [thing]
```

### 3. Batch Similar Changes
Instead of one at a time:
```
Create all probe files at once:
- src/ontogenic_io/probes/text_tone.rs
- src/ontogenic_io/probes/cursor.rs
- src/ontogenic_io/probes/audio.rs

Each should follow the same pattern with a [ProbeType]Probe struct and extract_features() method.
```

### 4. Ask Cursor to Generate Tests
```
Generate comprehensive unit tests for [file] that cover:
- Normal cases
- Edge cases
- Error conditions
```

### 5. Use Cursor for Refactoring
```
Refactor this code to:
- Use async/await properly
- Add better error handling
- Improve performance
- Make it more idiomatic Rust
```

---

## ⚠️ COMMON PITFALLS & SOLUTIONS

### Pitfall 1: Cursor Generates Too Much
**Solution**: Be more specific in prompts
```
❌ "Create materials discovery"
✅ "Create materials_bridge.rs with ONLY the discover_materials() method that calls existing MaterialsAdapter"
```

### Pitfall 2: Code Doesn't Compile
**Solution**: Use Chat for debugging
```
"I'm getting error [paste error]. How do I fix it? Here's the context: [paste code]"
```

### Pitfall 3: Lost Context
**Solution**: Reference explicitly
```
"Using the existing PrismAIOrchestrator at foundation/orchestration/integration/prism_ai_integration.rs..."
```

### Pitfall 4: Unclear on Architecture
**Solution**: Ask before implementing
```
"Before implementing, help me understand: should the ontogenic IO context be stored in the orchestrator or passed as parameters?"
```

---

## ✅ SUCCESS CHECKLIST

After each phase, verify:

### Phase 1 (LLM Consensus):
- [ ] Can query GPT-4, Claude, Gemini
- [ ] Quantum voting executes
- [ ] Thermodynamic consensus converges
- [ ] Output is coherent
- [ ] CLI works smoothly
- [ ] **Can demo to someone**

### Phase 2 (Materials):
- [ ] Can specify target properties
- [ ] CMA solver runs
- [ ] MaterialsAdapter produces results
- [ ] Results are chemically plausible
- [ ] CLI command works
- [ ] **Can demo discovery**

### Phase 3 (Context):
- [ ] Text tone extracts features
- [ ] Cursor dynamics work
- [ ] Features are sensible
- [ ] Context feeds into decisions
- [ ] **Can show context influence**

### Phase 4 (Evolution):
- [ ] MEC loop runs
- [ ] Mutations are generated
- [ ] Fitness improves over time
- [ ] Changes are logged
- [ ] **Can show self-improvement**

---

## 🎬 START RIGHT NOW

### Step 1 (Next 5 min): Open Cursor
```bash
cursor /home/diddy/Desktop/PRISM-FINNAL-PUSH
```

### Step 2 (Next 10 min): Create Structure
Press `Cmd/Ctrl + I` and paste:
```
Create directory structure and stub files for MEC system:
- src/mec/mod.rs
- src/ontogenic_io/mod.rs
- src/ontogenic_io/probes/mod.rs
- src/meta_learning/mod.rs

Update src/lib.rs to export these modules.
Each file should have basic struct and documentation.
```

### Step 3 (Next 1 hour): Start Phase 1
Press `Cmd/Ctrl + I` and paste:
```
Create foundation/orchestration/integration/bridges/llm_consensus_bridge.rs

Implement LLM consensus by wiring together:
- Existing LLM clients from charlie_integration
- Existing quantum voting
- Existing thermodynamic consensus
- Existing transfer entropy router

[Then continue with detailed requirements from Phase 1...]
```

---

## 💪 YOU'VE GOT THIS!

**Remember**:
- Let Cursor do the heavy lifting
- You guide the architecture
- Test continuously
- Iterate quickly
- Celebrate small wins

**The 30% you need to build is achievable with Cursor in weeks, not months!**

---

*Cursor IDE Guide created: October 25, 2024*
*Start with: Phase 0 → Phase 1 → LLM Consensus in 5 days!*
*Let Cursor be your co-pilot* 🚀
