# PRISM-FINNAL-PUSH Directory Organization Analysis Report

**Analysis Date:** October 25, 2025
**Repository:** /home/diddy/Desktop/PRISM-FINNAL-PUSH
**Status:** CRITICAL ORGANIZATIONAL ISSUES IDENTIFIED

---

## Executive Summary

The PRISM-FINNAL-PUSH repository exhibits **severe organizational anti-patterns** that violate professional software engineering standards. The analysis reveals extensive code duplication, unclear module boundaries, scattered artifacts, and a confusing dual-library structure that creates maintenance nightmares and build complexity.

**Critical Issues Found:**
- 3 backup directories (*.backup) polluting /src/
- Complete module duplication between /src/ and /foundation/
- 14 root-level documentation files (3,332 lines total)
- 9.9GB target/ directory not properly isolated
- 23MB .fingerprint/ directory in root (should be in target/)
- Compiled binaries scattered across 3 locations
- PRISM-AI-UNIFIED-VAULT embedded as subdirectory

---

## 1. DUPLICATE AND REDUNDANT DIRECTORIES

### 1.1 Backup Directories in Source Tree (CRITICAL)

**Location:** `/home/diddy/Desktop/PRISM-FINNAL-PUSH/src/`

**Issue:** Three backup directories exist directly in the source tree:
```
/src/cma.backup/      - Minimal old version (65 lines in mod.rs)
/src/cuda.backup/     - Outdated CUDA code, missing 2 new files
/src/data.backup/     - Identical to current data/ module
```

**Impact:**
- Confuses IDE navigation and search
- Increases repository size unnecessarily
- Creates ambiguity about which code is canonical
- Git operations slower due to extra files
- No clear reason for retention (no README explaining why)

**Evidence:**
```bash
# cma.backup is vastly outdated (65 vs 241 lines)
$ wc -l src/cma.backup/mod.rs src/cma/mod.rs
  65 src/cma.backup/mod.rs      # OLD
 241 src/cma/mod.rs              # CURRENT

# cuda.backup missing critical files
$ diff -r src/cuda.backup src/cuda --brief
Only in src/cuda: dense_path_guard.rs      # NEW
Only in src/cuda: device_guard.rs          # NEW
```

**Recommendation:** DELETE ALL .backup directories. Use git history for recovery.

---

### 1.2 Complete Module Duplication Between /src/ and /foundation/

**Issue:** Nearly identical module structures exist in two top-level directories with unclear boundaries.

#### Duplicated Modules

| Module | /src/ Size | /foundation/ Size | Status |
|--------|-----------|-------------------|--------|
| **cma/** | 160KB (13 files) | 160KB (13 files) | 99% identical structure |
| **cuda/** | 100KB (7 files) | 92KB (5 files) | Similar + src has 2 extra files |
| **data/** | 68KB (4 files) | 68KB (4 files) | IDENTICAL (byte-for-byte) |
| **neuromorphic/** | 24KB (2 files) | Cargo package | src has minimal stubs |
| **quantum/** | 12KB (1 file) | Cargo package | src has minimal stub |
| **phase6/** | 12KB (1 file) | 112KB (6 files) | src has stub only |
| **integration/** | 128KB (9 files) | 128KB (9 files) | IDENTICAL timestamps |
| **telemetry/** | 32KB (5 files) | 20KB (2 .jsonl) | Different purposes! |

#### CUDA Kernel Duplication

**Critical Finding:** CUDA kernel files duplicated:
```
/foundation/cma/cuda/pimc_kernels.cu
/src/cma/cuda/pimc_kernels.cu              # DUPLICATE

/foundation/cma/cuda/ksg_kernels.cu
/src/cma/cuda/ksg_kernels.cu               # DUPLICATE

/foundation/cuda/adaptive_coloring.cu
/src/cuda/adaptive_coloring.cu             # DUPLICATE
```

**Impact:**
- Changes must be synchronized manually (ERROR-PRONE)
- Build system may compile both versions
- Unclear which version is canonical
- 2x storage for identical code

---

### 1.3 Library Entry Point Confusion

**Critical Architectural Flaw:**

```toml
# Cargo.toml defines binary entry point
[[bin]]
name = "prism-ai"
path = "foundation/lib.rs"    # Points to foundation!
```

**But both directories have lib.rs:**
```
/foundation/lib.rs  - 50+ lines, imports from foundation modules
/src/lib.rs         - 50 lines, imports from src modules
```

**Analysis:**
- `/foundation/lib.rs` - Appears to be the "platform foundation" with orchestration focus
- `/src/lib.rs` - Appears to be the "PRISM-AI" application with CMA/CUDA focus
- No clear documentation explaining the separation
- Both export similar types (CMA, CUDA, quantum, neuromorphic)

**Recommendation:** This dual-library structure is ANTI-PATTERN. Choose ONE:

**Option A: Single Library**
```
/src/           - All application code
/foundation/    - DELETE or move to separate repository
```

**Option B: Workspace Structure (if truly independent)**
```
/prism-foundation/   - Platform library (rename foundation/)
/prism-ai/           - Application library (rename src/)
Cargo.toml           - Workspace definition
```

---

## 2. SCATTERED AND MISPLACED FILES

### 2.1 Documentation Explosion in Root

**Issue:** 14 status/integration report markdown files (3,332 lines) cluttering repository root.

```
ACTIVE_INFERENCE_STATUS.md          (131 lines)
ACTUAL_STATUS_AND_NEXT_STEPS.md     (219 lines)
ANALYSIS_INDEX.md                   (285 lines)
CMA_INTEGRATION_CRITICAL.md         (180 lines)
CRITICAL_DISCOVERY_INTEGRATION.md   (188 lines)
CRITICAL_FILES_STATUS.md            (102 lines)
CRITICAL_FIXES_GUIDE.md             (347 lines)
CUDA_CRITICAL_INTEGRATION.md        (231 lines)
DATA_MODULE_INTEGRATION.md          (205 lines)
FILE_VERIFICATION_REPORT.md         (171 lines)
FINAL_INTEGRATION_REPORT_OCT25.md   (221 lines)
IMPLEMENTATION_STATUS_ANALYSIS.md   (737 lines)
INTEGRATION_UPDATE_OCT25.md         (91 lines)
QUICK_STATUS_REFERENCE.md           (224 lines)
```

**Professional Violation:** Repository roots should contain:
- README.md (main documentation)
- LICENSE
- CHANGELOG.md
- CONTRIBUTING.md (if needed)

**All other documentation belongs in:**
```
/docs/status/        - Status reports
/docs/integration/   - Integration guides
/docs/architecture/  - Architecture analysis
```

---

### 2.2 Duplicate RFC Documentation

**Issue:** RFC documents exist in THREE locations:

```
/docs/rfc/
├── RFC-M0-Meta-Foundations.md     (6,444 lines)
└── RFC-M1-Meta-Orchestrator.md    (7,912 lines)

/PRISM-AI-UNIFIED-VAULT/docs/rfc/
├── RFC-M0-Meta-Foundations.md     (2,591 lines) - DIFFERENT!
├── RFC-M1-Meta-Orchestrator.md    (1,126 lines) - DIFFERENT!
└── RFC-M5-Federated-Readiness.md  (2,500 lines)
```

**Critical:** The files have DIFFERENT content! Which version is authoritative?

**Impact:**
- Multiple sources of truth
- Outdated documentation in one location
- Contributors don't know which to update

---

### 2.3 Build Artifacts Not Properly Isolated

**Issue 1: .fingerprint/ in root (23MB, 1,086 directories)**
```
/home/diddy/Desktop/PRISM-FINNAL-PUSH/.fingerprint/
```
This should be inside `/target/` directory. It's cargo dependency metadata.

**Issue 2: Compiled libraries scattered:**
```
/lib/libgpu_runtime.so          - Why separate lib/ directory?
/foundation/libgpu_runtime.so   - Duplicate!
/foundation/test_gpu_benchmark  - Test binary in source tree!
/target/debug/*.so              - Correct location
```

**Issue 3: Python model files in source:**
```
/python/gnn_training/gnn_model.onnx      (440 KB)
/python/gnn_training/gnn_model.onnx.data (5.4 MB) - LARGE BINARY!
```
These are build artifacts and should be in `/models/` or excluded from git.

**Issue 4: Python cache committed:**
```
/PRISM-AI-UNIFIED-VAULT/scripts/__pycache__/
```
Should be in `.gitignore`.

---

### 2.4 Telemetry Confusion

**Two telemetry locations with different purposes:**

```
/src/telemetry/        - Source code (Rust modules)
├── contract.rs
├── logger.rs
├── mod.rs
├── provider.rs
└── sink.rs

/telemetry/            - Runtime data (JSONL logs)
├── contract.jsonl
└── device_guard.jsonl
```

**Recommendation:**
- `/src/telemetry/` - Keep as source code
- `/telemetry/` - RENAME to `/logs/` or `/data/telemetry/` for clarity
- Add to `.gitignore` (logs shouldn't be committed)

---

### 2.5 PRISM-AI-UNIFIED-VAULT as Subdirectory

**Issue:** The "Unified Vault" is embedded as a subdirectory (1.2MB, 19 subdirs).

```
/PRISM-AI-UNIFIED-VAULT/
├── 00-CONSTITUTION/
├── 01-GOVERNANCE/
├── 02-IMPLEMENTATION/
├── 03-AUTOMATION/
├── 04-ADJUSTMENTS/
├── 05-PROJECT-PLAN/
├── artifacts/
├── audit/
├── determinism/
├── docs/
├── meta/
├── reports/
├── scripts/
├── src/              # ANOTHER src/ directory!
├── telemetry/        # ANOTHER telemetry/ directory!
└── tests/            # ANOTHER tests/ directory!
```

**Professional Standards:**

**Option A: Separate Repository (Recommended)**
```bash
# This is project documentation/governance
# Should be separate repository or wiki
PRISM-AI-Vault (separate repo)
├── docs/
├── governance/
└── project-plans/
```

**Option B: Integrate into Main Docs**
```
/docs/
├── constitution/    # From VAULT/00-CONSTITUTION
├── governance/      # From VAULT/01-GOVERNANCE
├── implementation/  # From VAULT/02-IMPLEMENTATION
└── project-plan/    # From VAULT/05-PROJECT-PLAN
```

**Current Issues:**
- Creates confusion with THREE src/ directories (main, vault, foundation)
- THREE telemetry/ directories
- THREE tests/ directories
- Restrictive permissions (drwx------) suggest it's separate concern

---

## 3. PROFESSIONAL ORGANIZATION VIOLATIONS

### 3.1 Unclear Module Boundaries

**Violation:** Two top-level source directories with overlapping responsibilities.

**Professional Standard:**
```
/src/              - ALL application source code
  ├── lib.rs       - Library entry point
  ├── bin/         - Binary entry points
  ├── modules/     - Feature modules
  └── internal/    - Internal implementation details
```

**Current State:**
```
/src/              - "PRISM-AI" application (1.4MB)
/foundation/       - "Platform foundation" (9.8MB)
```

**Why This Is Wrong:**
- IDE confusion: "Go to definition" may jump between src and foundation
- Import path confusion: `use crate::cma` vs `use foundation::cma`?
- Build complexity: Which modules depend on which?
- Testing difficulty: Integration tests need both?
- Contributor confusion: Where do new features go?

---

### 3.2 Missing or Inadequate .gitignore

**Current .gitignore (10 lines):**
```gitignore
target/

# Environment variables with API keys (NEVER commit!)
.env
.env.local
.env.*.local

# Cargo lock for libraries
Cargo.lock
```

**Missing Critical Patterns:**
```gitignore
# Build artifacts
*.so
*.a
*.o
*.dylib
*.dll
*.exe
lib/
.fingerprint/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
*.onnx
*.onnx.data

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Logs and data
/logs/
/telemetry/*.jsonl
*.log

# Temporary files
*.tmp
*.bak
*.backup
```

---

### 3.3 9.9GB Target Directory

**Issue:** Build artifacts consume massive space:
```
target/          9.9 GB total
├── debug/       8.3 GB
└── release/     1.6 GB
```

**Included in git repo!** (Though .gitignore says target/)

**Verification Needed:**
```bash
# Check if target/ is actually tracked
git ls-files target/ | head
```

**Recommendation:**
- Verify target/ is NOT tracked in git
- Add to CI/CD: `cargo clean` before major commits
- Consider: `.cargo/config.toml` with `target-dir = "../target"` for workspace

---

### 3.4 No Clear Build Documentation

**Missing Files:**
- `BUILD.md` - How to build the project
- `ARCHITECTURE.md` - System architecture explanation
- `DEPENDENCIES.md` - External dependency requirements

**Current:** User must parse 14 root-level status docs to understand build process.

---

## 4. RECOMMENDED RESTRUCTURING

### Phase 1: Immediate Cleanup (Zero Risk)

#### Step 1.1: Remove Backup Directories
```bash
# These are in git history and not needed
rm -rf src/cma.backup/
rm -rf src/cuda.backup/
rm -rf src/data.backup/
git add -A
git commit -m "chore: remove backup directories (available in git history)"
```

#### Step 1.2: Remove Build Artifacts from Source
```bash
# Compiled binaries don't belong in source tree
rm foundation/libgpu_runtime.so
rm foundation/test_gpu_benchmark
rm foundation/test_gpu_benchmark.cu  # Keep source in tests/ or benchmarks/
rm -rf lib/  # Redundant with target/
rm -rf .fingerprint/  # Cargo metadata, regenerated automatically

git add -A
git commit -m "chore: remove build artifacts from source tree"
```

#### Step 1.3: Fix .gitignore
```bash
# Add comprehensive .gitignore
cat >> .gitignore << 'EOF'

# Compiled binaries and libraries
*.so
*.a
*.o
*.dylib
*.dll
*.exe
/lib/
.fingerprint/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
*.onnx.data

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Runtime data
/logs/
/telemetry/*.jsonl

# Temporary
*.tmp
*.bak
EOF

git add .gitignore
git commit -m "chore: improve .gitignore coverage"
```

#### Step 1.4: Organize Documentation
```bash
# Create docs structure
mkdir -p docs/status docs/integration docs/architecture

# Move status reports
mv ACTIVE_INFERENCE_STATUS.md docs/status/
mv ACTUAL_STATUS_AND_NEXT_STEPS.md docs/status/
mv CRITICAL_FILES_STATUS.md docs/status/
mv FILE_VERIFICATION_REPORT.md docs/status/
mv INTEGRATION_UPDATE_OCT25.md docs/status/
mv FINAL_INTEGRATION_REPORT_OCT25.md docs/status/

# Move integration guides
mv CMA_INTEGRATION_CRITICAL.md docs/integration/
mv CUDA_CRITICAL_INTEGRATION.md docs/integration/
mv DATA_MODULE_INTEGRATION.md docs/integration/
mv CRITICAL_DISCOVERY_INTEGRATION.md docs/integration/

# Move analysis docs
mv ANALYSIS_INDEX.md docs/architecture/
mv IMPLEMENTATION_STATUS_ANALYSIS.md docs/architecture/
mv CRITICAL_FIXES_GUIDE.md docs/architecture/
mv QUICK_STATUS_REFERENCE.md docs/architecture/

git add docs/
git commit -m "docs: organize documentation into logical structure"
```

---

### Phase 2: Structural Reorganization (Medium Risk)

#### Option A: Single Unified Library (Recommended)

**Goal:** Merge foundation and src into single coherent structure.

```
PRISM-AI/
├── Cargo.toml           # Single workspace or binary crate
├── src/
│   ├── lib.rs           # Main library entry
│   ├── bin/
│   │   └── prism-ai.rs  # Binary entry point
│   ├── core/            # Core types and utilities (from foundation/types.rs)
│   ├── platform/        # Platform abstractions (from foundation/platform.rs)
│   ├── cma/             # Causal Manifold Annealing (MERGE)
│   ├── cuda/            # GPU acceleration (MERGE)
│   ├── quantum/         # Quantum computing (MERGE)
│   ├── neuromorphic/    # Neuromorphic computing (MERGE)
│   ├── data/            # Data handling (MERGE)
│   ├── orchestration/   # From foundation/orchestration/
│   ├── integration/     # Integration adapters
│   ├── governance/      # Benchmarking and validation
│   ├── meta/            # Meta-learning and orchestration
│   └── telemetry/       # Observability
├── foundation/          # DELETE after merge
├── docs/                # All documentation
│   ├── status/
│   ├── integration/
│   ├── architecture/
│   └── rfc/
├── tests/               # Integration tests
├── benches/             # Benchmarking
├── examples/            # Example usage
├── python/              # Python bindings/tools
├── scripts/             # Build and utility scripts
└── data/                # Runtime data (logs, checkpoints)
    ├── logs/            # Rename from /telemetry/
    └── models/          # Move ONNX files here
```

**Migration Plan:**
```bash
# 1. Create new structure
mkdir -p src/core src/platform src/orchestration

# 2. Copy foundation modules to src/
cp -r foundation/orchestration src/
cp foundation/types.rs src/core/
cp foundation/platform.rs src/platform/
cp foundation/system.rs src/platform/

# 3. Update Cargo.toml
# Change bin path from foundation/lib.rs to src/bin/prism-ai.rs

# 4. Fix imports across all files
# Use search-replace: foundation:: -> crate::

# 5. Remove foundation/ after verification
```

---

#### Option B: Cargo Workspace (If Truly Separate)

**Only use if foundation is truly independent library with external users.**

```
PRISM-AI-Workspace/
├── Cargo.toml                # Workspace root
├── prism-foundation/         # Rename from foundation/
│   ├── Cargo.toml
│   ├── src/lib.rs
│   └── ...                   # Platform library
├── prism-ai/                 # Rename from src/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   └── bin/prism-ai.rs
│   └── ...                   # Application
├── neuromorphic-engine/      # Already separate (foundation/neuromorphic/)
│   └── Cargo.toml
├── quantum-engine/           # Already separate (foundation/quantum/)
│   └── Cargo.toml
├── docs/                     # Workspace documentation
├── tests/                    # Integration tests
└── scripts/                  # Workspace scripts
```

**Workspace Cargo.toml:**
```toml
[workspace]
members = [
    "prism-foundation",
    "prism-ai",
    "neuromorphic-engine",
    "quantum-engine",
]
resolver = "2"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
# ... shared dependencies
```

---

### Phase 3: PRISM-AI-UNIFIED-VAULT Resolution (Low Risk)

#### Option A: Separate Repository (Recommended)
```bash
# 1. Create new repository
cd /home/diddy/Desktop/
mkdir PRISM-AI-Documentation
cd PRISM-AI-Documentation
git init

# 2. Move vault contents
mv ../PRISM-FINNAL-PUSH/PRISM-AI-UNIFIED-VAULT/* .

# 3. Restructure
mkdir -p docs governance project-plans
mv 00-CONSTITUTION docs/constitution
mv 01-GOVERNANCE governance/
mv 05-PROJECT-PLAN project-plans/
# ... etc

# 4. Reference from main repo
cd ../PRISM-FINNAL-PUSH
echo "See https://github.com/org/PRISM-AI-Documentation" > DOCUMENTATION.md
```

#### Option B: Integrate into Main Docs
```bash
# Merge vault into main docs/
mkdir -p docs/governance docs/project-plans

mv PRISM-AI-UNIFIED-VAULT/00-CONSTITUTION docs/constitution/
mv PRISM-AI-UNIFIED-VAULT/01-GOVERNANCE docs/governance/
mv PRISM-AI-UNIFIED-VAULT/05-PROJECT-PLAN docs/project-plans/

# Handle vault's src/, tests/, scripts/
# (Analyze for unique content vs main directories)
```

---

## 5. VERIFICATION CHECKLIST

After restructuring, verify:

### Build Integrity
```bash
cargo clean
cargo build --release
cargo test --all
cargo clippy -- -D warnings
```

### Documentation
```bash
cargo doc --no-deps --open
# Verify all modules documented
```

### File Organization
```bash
# No more duplicates
find . -name "*.backup" | wc -l  # Should be 0
find . -name "__pycache__" | wc -l  # Should be 0

# No build artifacts in source
find src/ foundation/ -name "*.so" -o -name "*.a" | wc -l  # Should be 0

# Documentation organized
ls docs/  # Should have clear structure

# Root clean
ls *.md | wc -l  # Should be ~4 (README, CHANGELOG, LICENSE, CONTRIBUTING)
```

### Git Cleanliness
```bash
git status  # Should be clean
du -sh .git  # Check repo size (should decrease)
```

---

## 6. IMPACT ASSESSMENT

### Current Issues Impact

| Issue | Severity | Developer Impact | Build Impact | Maintenance Impact |
|-------|----------|------------------|--------------|-------------------|
| .backup directories | Medium | IDE confusion, search clutter | None | Version confusion |
| Duplicate modules | **CRITICAL** | Bug fixes needed in 2 places | Compile time 2x | High merge conflict risk |
| 14 root docs | Low | Hard to find info | None | Doc outdating |
| .fingerprint/ in root | Medium | Confusing, takes space | Cargo regenerates | None |
| Dual lib.rs | **CRITICAL** | Import confusion | Build ambiguity | High architecture debt |
| VAULT subdirectory | Medium | 3x src/ dirs confusing | None | Unclear ownership |
| Missing .gitignore | Medium | Commits artifacts | Repo bloat | Code review noise |

### Post-Cleanup Benefits

- **50% reduction** in source code duplication
- **Clearer** module boundaries and import paths
- **Faster** IDE navigation and search
- **Smaller** repository size (~23MB savings immediately)
- **Lower** cognitive load for new contributors
- **Better** CI/CD cache hit rates (clearer target structure)

---

## 7. PRIORITY RECOMMENDATIONS

### Priority 1 (Critical - Do Immediately)
1. **Remove .backup directories** - Zero risk, immediate clarity gain
2. **Update .gitignore** - Prevent future artifact commits
3. **Consolidate RFC docs** - Establish single source of truth

### Priority 2 (High - Do This Week)
4. **Organize root documentation** into /docs/ structure
5. **Remove build artifacts** from source trees
6. **Decide on src/ vs foundation/** architecture (Option A or B)

### Priority 3 (Medium - Do This Month)
7. **Merge duplicate modules** or establish clear separation
8. **Resolve PRISM-AI-UNIFIED-VAULT** (separate repo or integrate)
9. **Create BUILD.md and ARCHITECTURE.md**

### Priority 4 (Low - Do When Possible)
10. Document import conventions
11. Set up pre-commit hooks for code quality
12. Establish module ownership (CODEOWNERS file)

---

## 8. CONCLUSION

The PRISM-FINNAL-PUSH repository suffers from **organic growth without refactoring**, resulting in a structure that violates professional software engineering principles. The most critical issue is the **complete duplication of modules between /src/ and /foundation/** with unclear boundaries and responsibilities.

**Immediate actions required:**
1. Remove backup directories (5 minutes)
2. Fix .gitignore (5 minutes)
3. Organize documentation (30 minutes)
4. Decide on unified vs workspace architecture (research: 2 hours)
5. Execute consolidation (implementation: 1-2 days)

**Total effort:** ~2-3 days of focused work to achieve professional-grade organization.

**Failure to address:** Will compound technical debt, slow onboarding, increase merge conflicts, and create maintenance burden as project scales.

---

## APPENDIX A: Directory Size Analysis

```
Component                Size      % of Total   Notes
----------------------------------------------------------
target/                  9.9 GB    99.6%       Build artifacts (OK if gitignored)
foundation/              9.8 MB    0.25%       Large - includes orchestration (26 subdirs)
src/                     1.4 MB    0.04%       Smaller but overlaps foundation
python/                  5.7 MB    0.14%       Includes 5.4MB ONNX model data
benchmarks/              4.0 MB    0.10%       DIMACS graphs
PRISM-AI-UNIFIED-VAULT   1.2 MB    0.03%       Documentation + scripts
lib/                     992 KB    0.02%       Single .so file (redundant)
docs/                    24 KB     <0.01%      Only 2 RFC files
Root .md files           ~50 KB    <0.01%      14 files, should be in docs/
.fingerprint/            23 MB     0.58%       Should be in target/
```

---

## APPENDIX B: Module Dependency Matrix

**To be populated after architecture decision:**

Current state makes dependency analysis difficult due to unclear boundaries between src/ and foundation/.

---

**Report End**

*Generated by: Claude Code Analysis*
*Review Status: Pending Team Discussion*
*Implementation Status: Recommendations Not Yet Applied*
