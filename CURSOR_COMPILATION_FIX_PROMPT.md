# Cursor Prompt: Fix PRISM-AI Compilation Errors
## Copy-Paste This Exact Prompt into Cursor Composer

**Press**: `Cmd/Ctrl + I` (Composer Mode)

---

## 🔧 COMPLETE FIX PROMPT

**Copy everything below and paste into Cursor Composer**:

```
I need to fix compilation errors in the PRISM-AI codebase. There are 4 main issues:

ISSUE #1: Missing information_theory module
-------------------------------------------
The code references `crate::information_theory` but this module doesn't exist at the crate root.

Looking at the codebase, information_theory exists at:
foundation/information_theory/

Fix by:
1. Check if foundation/lib.rs exports the information_theory module
2. If not, add to foundation/lib.rs:
   pub mod information_theory;

3. Then in src/lib.rs or wherever it's referenced, change imports from:
   use crate::information_theory::...
   TO:
   use foundation::information_theory::...

OR if foundation is already re-exported, use:
   use crate::foundation::information_theory::...

Search the codebase for all occurrences of "use.*information_theory" and fix the import paths.


ISSUE #2: Missing active_inference module
-----------------------------------------
The code references `crate::active_inference` but this module doesn't exist at the crate root.

Looking at the codebase, active_inference exists at:
foundation/active_inference/

Fix by:
1. Check if foundation/lib.rs exports the active_inference module
2. If not, add to foundation/lib.rs:
   pub mod active_inference;

3. Then fix all imports from:
   use crate::active_inference::...
   TO:
   use foundation::active_inference::...

Search for all "use.*active_inference" and update the paths.


ISSUE #3: PRISMPipeline vs PrismPipeline naming inconsistency
-------------------------------------------------------------
There's a naming conflict between PRISMPipeline and PrismPipeline.

Looking at src/cuda/mod.rs, it exports:
pub use prism_pipeline::PrismPipeline;

But src/lib.rs tries to export:
pub use cuda::PRISMPipeline;

Fix by choosing ONE consistent name:
1. Check src/cuda/prism_pipeline.rs - what is the actual struct name?
2. If it's "PrismPipeline", then update src/lib.rs to:
   pub use cuda::PrismPipeline;

3. Search the entire codebase for "PRISMPipeline" (all caps) and replace with "PrismPipeline"

OR if the struct is actually called PRISMPipeline, then update the export in src/cuda/mod.rs


ISSUE #4: Neuromorphic engine GPU features not enabled
------------------------------------------------------
The error says:
"could not find `gpu_reservoir` in `neuromorphic_engine`
note: found an item that was configured out
the item is gated behind the `cuda` feature"

This means the neuromorphic engine has gpu_reservoir but it's only available with the "cuda" feature.

Fix by:
1. Check foundation/neuromorphic/Cargo.toml - does it define a "cuda" feature?

2. Check the main Cargo.toml - when we enable features for neuromorphic_engine, we need:
   neuromorphic_engine = { path = "foundation/neuromorphic", package = "neuromorphic-engine", features = ["cuda"] }

3. Look for the neuromorphic_engine dependency and ensure "cuda" is in the features list

4. The imports in src/cuda/prism_pipeline.rs that use gpu_reservoir should be:
   #[cfg(feature = "cuda")]
   use neuromorphic_engine::gpu_reservoir::{GpuReservoirComputer, GpuConfig};

   Make sure this conditional compilation is in place.


ADDITIONAL: Fix any other import paths
--------------------------------------
After fixing the above, run cargo check and if there are still errors about missing modules:
- Check if they exist in foundation/
- Update foundation/lib.rs to export them
- Fix import paths throughout the codebase

Apply all fixes needed to make the project compile successfully.
```

---

## 🎯 WHAT CURSOR WILL DO

When you paste this prompt, Cursor will:

1. ✅ Add missing module exports to `foundation/lib.rs`
2. ✅ Fix all `information_theory` import paths
3. ✅ Fix all `active_inference` import paths
4. ✅ Resolve PRISMPipeline vs PrismPipeline naming
5. ✅ Add "cuda" feature to neuromorphic_engine dependency
6. ✅ Show you all the changes it made

---

## ✅ VERIFICATION STEPS

After Cursor makes the changes:

1. **Check the diff** - Review what Cursor changed
2. **Accept changes** - Click Accept if they look good
3. **Test compilation**:
   ```bash
   cargo check --all-features 2>&1 | head -50
   ```

4. **If more errors**, copy them and paste into Cursor Chat:
   ```
   Still getting these errors:
   [paste errors]

   Fix these remaining issues.
   ```

---

## 🔄 ITERATIVE FIX PROCESS

If Cursor doesn't fix everything in one go:

### **Round 1**: Use the prompt above
### **Round 2**: If errors remain, use this follow-up:

**Cursor Chat (`Cmd/Ctrl + L`)**:
```
I still have compilation errors after your fixes:

[paste the errors from cargo check]

Looking at the files you modified, I need you to:
1. Show me what imports are failing
2. Trace where those modules actually exist in the codebase
3. Fix the import paths
4. Check if any modules need to be added to lib.rs exports
```

### **Round 3**: Specific fixes

**For each remaining error, use Inline Edit (`Cmd/Ctrl + K`)**:
```
Fix this import error by changing the path to the correct location
```

---

## 💡 LIKELY FIXES CURSOR WILL MAKE

### **Fix 1: foundation/lib.rs**
```rust
// ADD these exports:
pub mod information_theory;
pub mod active_inference;
pub mod statistical_mechanics;

// Ensure neuromorphic is exported with cuda feature
#[cfg(feature = "cuda")]
pub use neuromorphic_engine::gpu_reservoir;
```

### **Fix 2: Cargo.toml**
```toml
# UPDATE neuromorphic_engine dependency:
neuromorphic_engine = {
    path = "foundation/neuromorphic",
    package = "neuromorphic-engine",
    features = ["cuda"]  # <-- ADD THIS
}
```

### **Fix 3: Import paths in prism_ai_integration.rs**
```rust
// CHANGE FROM:
use crate::information_theory::...;
use crate::active_inference::...;

// TO:
use crate::foundation::information_theory::...;
use crate::foundation::active_inference::...;
```

### **Fix 4: src/lib.rs**
```rust
// CHANGE FROM:
pub use cuda::PRISMPipeline;

// TO:
pub use cuda::PrismPipeline;
```

---

## 🎬 EXECUTE NOW

1. **Open Cursor IDE**
2. **Press `Cmd/Ctrl + I`** (Composer)
3. **Copy the entire "COMPLETE FIX PROMPT" section** above
4. **Paste** into Composer
5. **Press Enter/Execute**
6. **Review changes** Cursor makes
7. **Accept** the fixes
8. **Run**: `cargo check --all-features`
9. **If errors remain**, use Round 2 prompt

---

## ✅ SUCCESS CRITERIA

**You'll know it worked when**:
```bash
$ cargo check --all-features

   Compiling prism-ai v0.1.0
   ...
   ✅ Finished dev [unoptimized + debuginfo] target(s)
```

**No more errors about**:
- ❌ `could not find information_theory`
- ❌ `could not find active_inference`
- ❌ `no PRISMPipeline in cuda`
- ❌ `could not find gpu_reservoir`

---

*Cursor fix prompt ready*
*Copy-paste and execute now!*
*Should resolve all 4 compilation issues* 🔧
