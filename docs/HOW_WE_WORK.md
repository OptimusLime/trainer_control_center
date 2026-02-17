# How We Work

This document describes our collaboration process for building features, fixing bugs, and shipping quality code.

## Core Principles

### 1. The Dual Nature of All Work

Every phase and task has two equally important components:

1. **User Functionality:** What the user can DO when this is done.
2. **Backend Foundation:** What abstractions exist that accelerate future work.

When designing phases, state both explicitly:
- **Outcome:** What can the user do?
- **Foundation:** What abstraction does this build? What future work does it enable?

**The Library-Centric Rule:** When choosing between feature-specific code and a library abstraction, choose the abstraction. Feature-specific code is tech debt.

**Test:** After this phase, can the next phase be implemented by USING the abstractions, or by WRITING new code? If writing, the foundation is missing.

### 2. Incremental Building with Verification

We build in **phases**. Each phase has **subtasks**. Both phases and tasks are **verifiably complete**.

- No vague items like "implemented screen" or "finished feature"
- Use **SMART principles** (Specific, Measurable, Achievable, Relevant, Time-bound)
- Emphasis on **Specific** and **Measurable**: at the end, we can say definitively "this is done"

### 3. Verification is a First-Class Citizen

Verification **designs** our phases. If we cannot easily verify at the end of a phase that it's done, the phase is improperly designed.

- Each phase must have an **end-of-phase verification** that guarantees the work is done
- If verification is complicated (e.g., requires a big test script and lots of steps), that defeats the purpose
- If it's hard to verify, it's a bad phase
- If we need artifacts to make verification easier (e.g., a test harness), those are **dependencies** and belong in earlier phases
- The harness may start empty, then we build it up and complexify over time

### 4. Complexity Over Time, Not Upfront

Prefer **end-to-end pipelines** that demonstrate the full path through the code early. Then complexify those pipelines over time.

**Facade Pattern Approach:**
1. Design the API and functions
2. Initial implementations may return trivial values (e.g., constant 0.5 AO, placeholder)
3. Validate the pipeline setup works
4. Verify the pipeline behaves as expected
5. Later fill in details with heuristics or increasingly complex logic

**Critical guarantee:** Phases build on each other. Verification means we know at the end of the phase that the work was done, completed, and verified absolutely correct. We will **not** proceed through multiple phases only to discover in phase 3 that phase 1 was incomplete or incorrect.

### 5. Tasks Must Be Specific and Measurable

Tasks are verifiable, specific, and measurable.

**Bad:** "implement GTAO algorithm"  
**Good:** "modify `assets/shaders/gtao.wgsl` to output view-space depth as grayscale, verify near=dark far=bright"

When listing files:
- List specific file paths
- Specify exact names
- Include the directory structure

### 6. Naming Principles

Follow the principles from *Naming Things* by Tom Benner.

- Names (variables, files, images, folders/directories) **convey documentation**
- We document as we go through clear, informative naming
- Names should be:
  - **Descriptive:** what it is or does
  - **Consistent:** same concepts use same naming patterns
  - **Discoverable:** easy to find via search
  - **Unambiguous:** one meaning only

---

## Hypothesis-Driven Debugging

When something doesn't work, we do NOT abandon it. We do NOT substitute simpler alternatives. We follow a systematic process:

### The Hypothesis Loop

1. **Observe**: What exactly is happening? (e.g., "output is all white", "values are NaN")
2. **Hypothesize**: Form a specific, testable hypothesis about WHY
3. **Design Test**: Create a minimal test that validates or invalidates the hypothesis
4. **Execute**: Run the test
5. **Analyze**: What did we learn?
6. **Iterate**: If hypothesis was wrong, form new hypothesis. If right, fix the issue.

### What We Never Do

- **Never abandon a task** because it's hard or doesn't work immediately
- **Never substitute a "simpler" approach** without explicit user approval
- **Never skip verification steps** to move faster
- **Never assume something works** without testing
- **Never work on tangential issues** when the core task is incomplete
- **Never leave stub implementations with "not implemented yet" comments** - either implement it fully, return an explicit error, or create a tracking issue. Silent failures (returning false/empty) masquerading as working code are unacceptable.

### No Silent Stub Implementations

When porting code or building features:

1. **If a feature is not implemented**: Return an explicit error or panic with a clear message like `unimplemented!("WFC children execution")` - NEVER silently return false or skip functionality
2. **If intentionally unsupported**: Document it in code AND return an error (e.g., "3D overlap model not yet supported")
3. **Before marking any phase complete**: Search the code for "not implemented", "for now", "stub", "placeholder", "skip", "TODO" markers to ensure nothing was left incomplete
4. **Test the full path**: If the reference implementation has feature X, your port must either implement X or explicitly error when X is requested

**Example of BAD code (silent failure):**
```rust
if self.child_index >= 0 {
    // For now, just return false (no children implemented yet)  
    return false;  // WRONG: Silently skips critical functionality
}
```

**Example of GOOD code (explicit failure):**
```rust
if self.child_index >= 0 {
    unimplemented!("WFC children execution not yet implemented");
    // Or: return Err("WFC children not implemented".into());
}
```

### Debugging Graphics/Shaders

For rendering work, use these verification techniques:

1. **Constant Output Test**: Output a known constant (e.g., `return 0.5;`) to verify the pipeline runs
2. **Value Visualization**: Output intermediate values as colors (depth as grayscale, normals as RGB, UV as RG)
3. **Boundary Tests**: Test edge cases explicitly (UV = 0, UV = 1, depth = near, depth = far)
4. **Reference Comparison**: If porting from reference, verify intermediate values match at each step
5. **Incremental Buildup**: Start with constant output, add ONE variable at a time, verify each addition

### Example: Debugging GTAO

**Wrong approach:**
1. Port entire GTAO algorithm
2. Output is all white
3. "It doesn't work, let's try simpler SSAO instead" ❌

**Correct approach:**
1. Output is all white
2. **Hypothesis 1**: View-space position reconstruction is wrong
3. **Test**: Output `view_pos.z / far_plane` as grayscale
4. **Result**: All black → position reconstruction IS broken
5. **Hypothesis 2**: The view matrix is identity/wrong
6. **Test**: Output `view_matrix[0][0]` as brightness
7. **Result**: Value is 1.0 → matrix might be identity
8. **Fix**: Check uniform binding, find it's not being set
9. **Verify**: Now depth gradient shows correctly
10. **Continue** to next component

Each step produces evidence. We don't guess. We don't give up.

---

## Workflow: Feature Development

### Step 1: Create a Plan Document

Before any code, create a Markdown file in `docs/` describing the plan:

```markdown
# [Feature Name] Plan

## Summary
One paragraph describing what we're building.

## Context & Motivation  
Why are we building this? What problem does it solve?

## Naming Conventions for This PR
- File naming patterns we'll follow
- Component naming patterns
- Variable/function naming patterns

## Phases

### Phase 1: [Phase Name]
**Outcome:** [What the user can DO when this phase is complete]  
**Foundation:** [What abstraction this builds and what future phases it enables]
**Verification:** [How we prove it's done - must be simple and clear]

Tasks:
1. [Specific task with file path]
2. [Specific task with file path]
...

### Phase 2: [Phase Name]
...

## Phase Cleanup Notes

[Accumulated cleanup candidates from milestone audits. Review at phase end.]

### Cleanup Decision

At phase end, categorize accumulated items:
- **Do now:** Items that block next phase or create significant tech debt
- **Defer:** Nice-to-have items that don't block progress
- **Drop:** Items that turned out to be unnecessary

## Full Outcome Across All Phases
Summary of what the completed feature looks like.

## Directory Structure (Anticipated)
```
crates/studio_core/src/deferred/
├── gtao.rs
├── gtao_node.rs
...
```

## How to Review
Order of operations for reviewing this PR.
```

### Step 2: Create Feature Branch and Draft PR

Once the plan exists:

1. Create a feature branch: `git checkout -b feature/[descriptive-name]`
2. Commit the plan document
3. Push and open a **draft PR**
4. Fill in the PR description using `.github/PULL_REQUEST_TEMPLATE.md` **exactly**:
   - Do not leave anything out
   - Do not check items that aren't accurate (honesty over 100%)
   - If a section (e.g., screenshots) is empty, **do not remove it** - all sections must remain

### Step 3: Execute Phases

For each phase:
1. Work through tasks sequentially
2. Mark tasks complete as you go (not batched at the end)
3. **After each milestone:** Do a quick cleanup audit (5 min). Flag refactoring opportunities, don't necessarily fix them.
4. At phase end, perform verification
5. **At phase end:** Review accumulated cleanup notes. Decide what to do now, defer, or drop.
6. Only proceed to next phase after verification passes
7. **If verification fails**: Enter hypothesis-driven debugging loop. Do not skip ahead.

### Step 4: Complete PR

1. Mark PR as ready for review
2. Update PR description with any changes from the plan
3. Ensure all checklist items are accurate

---

## Workflow: Bug Fixes

### Step 1: Document the Bug

Create a brief plan or note:
- What is the bug?
- How to reproduce it?
- What is the expected behavior?

### Step 2: Create Branch and PR

1. Create branch: `git checkout -b fix/[descriptive-name]`
2. Open draft PR with:
   - Bug description
   - Reproduction steps
   - Root cause analysis (once identified)
   - Fix approach

### Step 3: Implement Fix

1. Write failing test (if applicable)
2. Implement fix
3. Verify fix resolves the issue
4. Verify no regressions

### Step 4: Complete PR

Same as feature development.

---

## Workflow: Refactoring

### Step 1: Document the Refactor

In `docs/refactor_roadmap/` or similar:
- Current state
- Target state
- Why the refactor matters
- Risk assessment

### Step 2: Phase the Work

Break into small, verifiable phases:
- Each phase leaves codebase in working state
- Each phase has clear verification
- Prefer many small PRs over one large PR

### Step 3: Execute

Same phase execution process as features.

---

## PR Checklist Standards

From `.github/PULL_REQUEST_TEMPLATE.md`:

### Testing
- [ ] I manually tested the new behavior (Mandatory)
- [ ] I added or updated automated tests where appropriate
- [ ] All tests (new and existing) pass locally

### Documentation
- [ ] I updated relevant docs if behavior changed
- [ ] I added or updated docstrings/comments for important functions

### Design Thoughtfulness
- [ ] I thought through the design before coding
- [ ] For major changes, I documented or had a design discussion

### Clarity & Structure
- [ ] Names, structure, and code flow are understandable and consistent
- [ ] Code is concise without sacrificing clarity
- [ ] Responsibilities are properly encapsulated

### Maintainability
- [ ] This PR leaves the codebase better than it found it
- [ ] The design supports future changes
- [ ] Removed code smells: unnecessary complexity, duplication, dead code

---

## Verification Examples

### Good Verification
- "Run `cargo run --example p20_gtao_test`, output shows depth gradient (near=dark, far=bright)"
- "Set DEBUG_MODE=1 in gtao.wgsl, normals display as RGB where +X=red, +Y=green, +Z=blue"
- "Corner geometry shows AO value 0.3-0.6, flat ground shows AO > 0.95"

### Bad Verification
- "GTAO works correctly" (not specific)
- "Test the feature" (not measurable)
- "Verify AO looks good" (not actionable)

---

## Realistic and Honest Evaluation

**CRITICAL: We evaluate outputs objectively, not hopefully.**

When reviewing visual output (screenshots, renders, etc.):

### What We Do
- **Look for defects FIRST**: Noise, artifacts, banding, incorrect values, missing effects
- **Compare against quality standards**: Is this production-ready? Or is it "kinda working"?
- **Be specific about problems**: "Patchy noise in flat areas", "AO missing from corners", "Visible banding at edges"
- **Use reference comparisons**: What SHOULD this look like? Compare to reference implementations or known-good outputs
- **Quantify when possible**: "AO values range 0.8-1.0 when they should be 0.3-0.7 in corners"

### What We NEVER Do
- **Wishful interpretation**: "The AO is subtle but working" when it's clearly broken
- **Hopeful evaluation**: Seeing what we want to see instead of what's actually there
- **Premature celebration**: Declaring success because something renders without crashing
- **Ignoring obvious defects**: Glossing over noise, artifacts, or incorrect behavior
- **Moving on too fast**: Proceeding to "tuning" when the fundamental algorithm is wrong

### Quality Gate Questions

Before marking any visual feature complete, answer honestly:

1. **Is this production quality?** Would you ship this in a game?
2. **Are there visible artifacts?** Noise, banding, patches, discontinuities?
3. **Does it match the reference?** If porting an algorithm, does output match?
4. **Would a user notice problems?** Not "can you see it if you look hard" but "is it obviously wrong"?

If ANY answer is negative, the task is NOT DONE. Return to debugging.

### Example: Evaluating GTAO Output

**Wrong evaluation:**
"GTAO is working! Corners are darker and flat surfaces are bright."

**Correct evaluation:**
"GTAO output shows:
- PROBLEMS: Visible patchy noise across flat surfaces, AO appears splotchy not smooth
- PROBLEMS: Corner darkening is inconsistent - some corners dark, others not
- PROBLEMS: Half-resolution artifacts visible as blocky patterns
- VERDICT: Algorithm fundamentally working but quality is unacceptable. Need to:
  1. Increase sample count or add denoising
  2. Check noise texture is sampling correctly
  3. Verify falloff parameters"

---

## Summary

1. **Dual nature** - Every phase has functionality AND foundation; state both explicitly
2. **Library-centric** - Choose abstractions over feature-specific code; feature-specific code is tech debt
3. **Plan first** - Create a plan document before coding
4. **Phase the work** - Break into verifiable phases
5. **Verify constantly** - Each phase ends with verification
6. **Cleanup audits** - After each milestone, flag refactoring opportunities; review at phase end
7. **Debug systematically** - Hypothesis → Test → Analyze → Iterate
8. **Never abandon** - If stuck, debug; don't substitute
9. **Name clearly** - Names are documentation
10. **Be specific** - File paths, exact names, measurable outcomes
11. **Stay honest** - PR checklists reflect reality, not aspirations
12. **Evaluate objectively** - See what IS, not what you HOPE to see
13. **Quality gates** - "Does it work?" is not the same as "Is it good enough?"

---

## Related Documents

- **[WRITING_MILESTONES.md](./WRITING_MILESTONES.md)** - How to write good milestones and construct build sequences. Milestones must be functionality-indexed ("I can X"), not backend-indexed ("system Y exists").

---

## Workflow: Testing New Features

**CRITICAL: Always run code to verify it works.**

When implementing a new feature that integrates with an existing application:

### Write Contained Examples First

1. **Create a standalone example** in `examples/` that tests the feature in isolation
2. **Run the example** with `cargo run --example <name>`
3. **Verify it works** before integrating into the main app
4. This catches dependency issues, missing plugins, resource conflicts early

### Why This Matters

The main app has complex plugin dependencies. A feature might:
- Depend on resources that aren't initialized
- Conflict with other systems
- Require specific plugin ordering

A contained example isolates the feature and reveals these issues immediately.

### Example: MarkovJunior 3D Rendering

**Wrong approach:**
1. Add rendering system to ScriptingPlugin
2. Assume it will work because the code compiles
3. Discover at runtime that `Assets<VoxelMaterial>` doesn't exist ❌

**Correct approach:**
1. Create `examples/p27_markov_imgui.rs` that tests the feature standalone
2. Run `cargo run --example p27_markov_imgui`
3. Verify it produces correct output (PNG + 3D screenshot)
4. Then integrate into main app
5. Run `cargo run` and verify the same behavior ✓

### Lesson Learned

In Phase 3.6, we added `render_generated_voxel_world` to ScriptingPlugin without testing. The main app crashed because `VoxelMaterialPlugin` wasn't loaded. Creating `p27_markov_imgui.rs` first would have:
- Caught the missing plugin dependency
- Proven the rendering code works
- Provided a reference for what plugins the main app needs
