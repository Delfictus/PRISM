# PRISM-FINNAL-PUSH Cleanup Action Plan

**Created:** October 25, 2025
**Priority:** HIGH
**Effort:** 2-3 days focused work
**Risk:** Low (with proper git workflow)

---

## Quick Summary

Your repository has significant organizational issues:
- 3 backup directories polluting /src/
- Complete module duplication between /src/ and /foundation/
- 14 documentation files in root (should be in /docs/)
- Build artifacts scattered in source trees
- 23MB .fingerprint/ directory in wrong location

**See:** `DIRECTORY_ORGANIZATION_ANALYSIS.md` for full analysis

---

## Immediate Actions (Today - 1 Hour)

### 1. Create Feature Branch
```bash
cd /home/diddy/Desktop/PRISM-FINNAL-PUSH
git checkout -b chore/directory-cleanup
```

### 2. Remove Backup Directories (5 min)
```bash
# These are in git history - no need to keep in working tree
rm -rf src/cma.backup/
rm -rf src/cuda.backup/
rm -rf src/data.backup/

git add -A
git commit -m "chore: remove backup directories (available in git history)"
```

### 3. Remove Build Artifacts from Source (10 min)
```bash
# Compiled libraries don't belong in source tree
rm -f foundation/libgpu_runtime.so
rm -f foundation/test_gpu_benchmark
rm -f foundation/test_gpu_benchmark.cu
rm -rf lib/

# Python cache
rm -rf PRISM-AI-UNIFIED-VAULT/scripts/__pycache__/

# Remove .fingerprint from root (cargo regenerates it)
rm -rf .fingerprint/

git add -A
git commit -m "chore: remove build artifacts from source tree"
```

### 4. Update .gitignore (5 min)
```bash
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
*.log

# Temporary
*.tmp
*.bak
*.backup
EOF

git add .gitignore
git commit -m "chore: improve .gitignore coverage"
```

### 5. Organize Root Documentation (30 min)
```bash
# Create structure
mkdir -p docs/status docs/integration docs/architecture

# Status reports
mv ACTIVE_INFERENCE_STATUS.md docs/status/
mv ACTUAL_STATUS_AND_NEXT_STEPS.md docs/status/
mv CRITICAL_FILES_STATUS.md docs/status/
mv FILE_VERIFICATION_REPORT.md docs/status/
mv INTEGRATION_UPDATE_OCT25.md docs/status/
mv FINAL_INTEGRATION_REPORT_OCT25.md docs/status/

# Integration guides
mv CMA_INTEGRATION_CRITICAL.md docs/integration/
mv CUDA_CRITICAL_INTEGRATION.md docs/integration/
mv DATA_MODULE_INTEGRATION.md docs/integration/
mv CRITICAL_DISCOVERY_INTEGRATION.md docs/integration/

# Architecture analysis
mv ANALYSIS_INDEX.md docs/architecture/
mv IMPLEMENTATION_STATUS_ANALYSIS.md docs/architecture/
mv CRITICAL_FIXES_GUIDE.md docs/architecture/
mv QUICK_STATUS_REFERENCE.md docs/architecture/

# Add this new analysis
mv DIRECTORY_ORGANIZATION_ANALYSIS.md docs/architecture/

git add docs/
git commit -m "docs: organize documentation into logical structure"
```

### 6. Test Everything Still Builds
```bash
cargo clean
cargo build --release
cargo test

# If all passes:
git push origin chore/directory-cleanup
```

**After push:** Review changes before merging to integration/m2

---

## Short-term Actions (This Week - 4 Hours)

### 7. Consolidate Duplicate RFC Documentation (1 hour)

**Issue:** RFC docs exist in 2 places with different content:
- `/docs/rfc/` (larger, more complete)
- `/PRISM-AI-UNIFIED-VAULT/docs/rfc/` (smaller, different)

**Action:**
```bash
# Compare and merge
diff docs/rfc/RFC-M0-Meta-Foundations.md \
     PRISM-AI-UNIFIED-VAULT/docs/rfc/RFC-M0-Meta-Foundations.md

# Decision needed: Which is authoritative?
# Then: Keep one, reference from other location
```

### 8. Rename /telemetry/ to /logs/ (15 min)

**Clarify purpose:** Runtime logs vs source code

```bash
mkdir -p data/logs
mv telemetry/*.jsonl data/logs/
rmdir telemetry/

# Update .gitignore
echo "/data/logs/" >> .gitignore

git add -A
git commit -m "refactor: clarify telemetry source vs runtime logs"
```

### 9. Move Large Binary Files (30 min)

```bash
# Create models directory
mkdir -p data/models

# Move ONNX files
mv python/gnn_training/gnn_model.onnx data/models/
mv python/gnn_training/gnn_model.onnx.data data/models/

# Update Python scripts to reference new location
# (grep for 'gnn_model.onnx' in python/ and update paths)

# Gitignore large model files
echo "/data/models/*.onnx.data" >> .gitignore

git add -A
git commit -m "refactor: move model artifacts to data directory"
```

### 10. Create Missing Documentation (2 hours)

Create these essential files:

**BUILD.md:**
```markdown
# Building PRISM-AI

## Requirements
- Rust 1.70+
- CUDA 11.8+
- Python 3.9+ (for GNN training)

## Quick Start
\`\`\`bash
cargo build --release --features cuda
\`\`\`

## Detailed Instructions
[Add specific build steps, environment variables, etc.]
```

**ARCHITECTURE.md:**
```markdown
# PRISM-AI Architecture

## Overview
[High-level system architecture]

## Module Structure
- /src/cma/ - Causal Manifold Annealing
- /src/cuda/ - GPU acceleration
- [etc.]

## Module Dependencies
[Dependency graph or description]
```

---

## Medium-term Decision (This Month - Research Required)

### Critical: Resolve src/ vs foundation/ Duplication

**The Problem:**
- Both `/src/` and `/foundation/` contain similar modules (cma, cuda, data, etc.)
- Some are identical, some have slight differences
- Unclear which is canonical
- Cargo.toml points binary to `foundation/lib.rs`

**Two Options:**

#### Option A: Single Unified Library (Recommended)
**Pros:**
- Clear module boundaries
- Single source of truth
- Simpler build process
- Easier for contributors

**Cons:**
- Requires careful merging
- May need to reconcile differences
- 1-2 days of work

**Action:**
```bash
# Merge foundation modules into src/
# Update Cargo.toml binary path to src/bin/prism-ai.rs
# Delete foundation/ after verification
```

#### Option B: Cargo Workspace
**Pros:**
- Keeps foundation as separate library
- Good if foundation used externally
- Clear separation of concerns

**Cons:**
- More complex build setup
- Requires workspace management
- Only beneficial if foundation truly independent

**Action:**
```bash
# Restructure as workspace
# Create prism-foundation/ and prism-ai/ crates
# Update Cargo.toml to workspace format
```

**Decision Criteria:**
- Is `foundation/` used by other projects? → Option B
- Is `foundation/` only internal? → Option A

**Time needed:**
- Research and decision: 2-4 hours
- Implementation: 1-2 days

---

## Long-term Actions (When Possible)

### 11. Resolve PRISM-AI-UNIFIED-VAULT

**Current:** Embedded as subdirectory with 19 subdirs

**Options:**
1. **Separate repo** - Move to PRISM-AI-Documentation repository
2. **Integrate** - Merge into main /docs/ structure

**Considerations:**
- VAULT has restrictive permissions (drwx------)
- Contains governance, not code
- Has its own src/, tests/, telemetry/ (confusing)

### 12. Set Up Pre-commit Hooks

```bash
# Install tools
cargo install cargo-fmt cargo-clippy

# Create .git/hooks/pre-commit
#!/bin/bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

### 13. Create CODEOWNERS

Define module ownership:
```
/src/cma/          @quantum-team
/src/cuda/         @gpu-team
/src/neuromorphic/ @neuro-team
/docs/             @all-team
```

---

## Verification After Each Step

After each change, run:

```bash
# Build verification
cargo clean
cargo build --release
cargo test --all

# Code quality
cargo clippy -- -D warnings
cargo fmt --all -- --check

# Documentation
cargo doc --no-deps

# Git status
git status  # Should be clean after commit
```

---

## Success Metrics

**After immediate actions:**
- ✅ Zero .backup directories
- ✅ Root has <5 .md files (README, LICENSE, CHANGELOG, CONTRIBUTING)
- ✅ No build artifacts in src/ or foundation/
- ✅ Comprehensive .gitignore
- ✅ Documentation organized in /docs/

**After medium-term actions:**
- ✅ Single clear module structure (src/ OR workspace)
- ✅ No duplicate modules
- ✅ Clear architecture documentation
- ✅ Build process documented

**After long-term actions:**
- ✅ VAULT resolved (separate or integrated)
- ✅ Pre-commit hooks in place
- ✅ CODEOWNERS defined

---

## Risk Mitigation

### Low-Risk Changes
All immediate actions are LOW RISK:
- Removing .backup dirs (in git history)
- Removing build artifacts (regenerated)
- Moving docs (just files)
- Updating .gitignore (only affects future)

### Medium-Risk Changes
Module consolidation (medium-term):
- Work on feature branch
- Test thoroughly before merge
- Pair program or get code review
- Can revert if issues

### Emergency Rollback
```bash
# If anything goes wrong:
git reset --hard HEAD~1  # Undo last commit
git clean -fd             # Remove untracked files

# Or abandon branch and start over:
git checkout integration/m2
git branch -D chore/directory-cleanup
```

---

## Timeline Summary

| Phase | Effort | Risk | When |
|-------|--------|------|------|
| Immediate cleanup | 1 hour | Low | Today |
| Short-term actions | 4 hours | Low | This week |
| Architecture decision | 2-4 hours | Medium | This week (research) |
| Module consolidation | 1-2 days | Medium | Next week |
| VAULT resolution | 4 hours | Low | When convenient |

**Total: ~3 days of focused work for professional-grade organization**

---

## Questions to Answer Before Proceeding

1. **Is /foundation/ used by other projects or external users?**
   - Yes → Workspace approach
   - No → Merge into /src/

2. **Is PRISM-AI-UNIFIED-VAULT actively maintained separately?**
   - Yes → Separate repository
   - No → Integrate into /docs/

3. **Are the duplicated modules intentional (versioning)?**
   - Yes → Need versioning strategy
   - No → Remove duplication

4. **Which RFC documents are authoritative?**
   - /docs/rfc/ vs /PRISM-AI-UNIFIED-VAULT/docs/rfc/

---

## Next Steps

1. **Review this plan with team** (if applicable)
2. **Start immediate actions** (can do solo, low risk)
3. **Answer key questions** above
4. **Schedule medium-term work** based on answers
5. **Track progress** in project management tool

---

**Ready to start?** Begin with the "Immediate Actions" section above.

**Questions?** See full analysis in `docs/architecture/DIRECTORY_ORGANIZATION_ANALYSIS.md`
