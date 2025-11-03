# Cursor IDE Quick Reference Card
## Keep This Open While Building MEC

---

## ⌨️ ESSENTIAL SHORTCUTS

```
┌────────────────────────────────────────────┐
│  Cmd/Ctrl + I    Composer (Big Changes)    │
│  Cmd/Ctrl + K    Inline Edit (Quick Edits) │
│  Cmd/Ctrl + L    Chat (Ask Questions)      │
│  Cmd/Ctrl + J    Terminal (Run Commands)   │
│  Tab             Accept Autocomplete        │
└────────────────────────────────────────────┘
```

---

## 🎯 DECISION TREE

```
Need to...

Create new file/feature?
    → Use COMPOSER (Cmd/Ctrl + I)

Modify existing code?
    → Use INLINE EDIT (Cmd/Ctrl + K)

Have a question?
    → Use CHAT (Cmd/Ctrl + L)

Run a command?
    → Use TERMINAL (Cmd/Ctrl + J)
```

---

## 📋 PHASE CHECKLIST

```
☐ Phase 0: Setup [30 min]
  ☐ Composer: Create structure (Prompt 0.1)
  ☐ Terminal: cargo check
  ✓ DONE when: Project compiles

☐ Phase 1: LLM Consensus [2 days]
  ☐ Composer: LLM bridge (Prompt 1.1)
  ☐ Composer: Update orchestrator (Prompt 1.4)
  ☐ Composer: Main executable (Prompt 1.5)
  ☐ Terminal: Set API keys
  ☐ Terminal: cargo build --release
  ☐ Terminal: ./target/release/prism_mec consensus "test query"
  ✓ DONE when: Gets real consensus from 3 LLMs

☐ Phase 2: Materials [2 days]
  ☐ Composer: Materials bridge (Prompt 2.1)
  ☐ Inline: Add command (Prompt 2.2)
  ☐ Terminal: Test
  ✓ DONE when: Discovers material candidates

☐ Phase 3: Context [3 days]
  ☐ Composer: Text tone (Prompt 3.1)
  ☐ Composer: Cursor probe (Prompt 3.2)
  ☐ Inline: Wire to orchestrator
  ✓ DONE when: Context affects decisions

☐ Phase 4: MEC Loop [5 days]
  ☐ Composer: MEC engine
  ☐ Composer: Evolution methods
  ☐ Terminal: Test evolution
  ✓ DONE when: System improves itself
```

---

## 🎨 COMPOSER MODE TEMPLATE

**When to use**: Creating new files or major features

**Press**: `Cmd/Ctrl + I`

**Template**:
```
Create [file path]

This should implement [feature] that:
1. [Requirement 1]
2. [Requirement 2]
3. [Requirement 3]

Types needed:
- [Type 1]
- [Type 2]

Methods:
- [Method 1 signature and behavior]
- [Method 2 signature and behavior]

Use [existing component] from [location].
Add proper error handling with anyhow::Result.
Add logging with log::info! and log::debug!.
```

---

## ✏️ INLINE EDIT TEMPLATE

**When to use**: Quick modifications

**Press**: `Cmd/Ctrl + K`

**Template**:
```
Add [thing] to this [struct/enum/impl]

Update [thing] to [new behavior]

Fix [issue] by [solution]
```

---

## 💬 CHAT MODE TEMPLATE

**When to use**: Questions and debugging

**Press**: `Cmd/Ctrl + L`

**Template**:
```
I'm working on [component].

Question: [Your question]?

Context:
- [Relevant file 1]
- [Relevant file 2]

Help me understand [thing] or debug [issue].
```

---

## 🔥 SPEED RUN GUIDE

### Hour 1: Setup
```
1. Open Cursor
2. Cmd+I → Paste Prompt 0.1 → Accept
3. Cmd+J → cargo check
4. ✅ Structure ready
```

### Hour 2-4: Start LLM Bridge
```
1. Cmd+I → Paste Prompt 1.1 → Accept
2. Review code
3. Cmd+I → Paste Prompt 1.4 → Accept
4. cargo check
5. Fix any errors with Cmd+L (chat)
```

### Hour 5-8: Main Executable
```
1. Cmd+I → Paste Prompt 1.5 → Accept
2. cargo build --release
3. Set up API keys
4. Test run
5. Debug with Cmd+L
```

### Day 2-5: Finish Phase 1
```
1. Polish implementation
2. Add error handling
3. Improve output
4. Test thoroughly
5. 🎉 Working demo!
```

---

## 📊 PROGRESS TRACKER

```
Phase 0: [    ] 0%    →  [████] 100% ✅
         Setup              Ready

Phase 1: [    ] 0%    →  [████] 100% ✅
         Start              LLM Working

Phase 2: [    ] 0%    →  [████] 100% ✅
         Start              Materials Working

Phase 3: [    ] 0%    →  [████] 100% ✅
         Start              Context Working

Phase 4: [    ] 0%    →  [████] 100% ✅
         Start              Evolution Working

Phase 5: [    ] 0%    →  [████] 100% ✅
         Start              Complete!
```

---

## 🎯 TODAY'S GOAL

```
┌─────────────────────────────────┐
│  END OF TODAY:                  │
│                                 │
│  ✅ Phase 0 complete            │
│  ✅ Phase 1 started             │
│  ✅ LLM bridge scaffolded       │
│  ✅ Compiles without errors     │
│                                 │
│  TOMORROW: Wire it all up!      │
└─────────────────────────────────┘
```

---

## 🔗 QUICK LINKS

Open these in tabs:
1. `MEC_BUILD_PRIORITY_GUIDE.md` - Overall plan
2. `CURSOR_PROMPTS_READY_TO_USE.md` - Prompts to copy
3. This file - Quick reference

---

## ⚡ REMEMBER

```
1. Composer for NEW code
2. Inline for EDITS
3. Chat for HELP
4. Terminal for TESTING

Let Cursor write code.
You guide architecture.
Test continuously.
Ship fast! 🚀
```

---

*Keep this open while coding!*
*Created: October 25, 2024*
