# Writing Milestones

This document describes how to write good milestones and construct build sequences that actually work.

---

## The Dual Nature of Milestones

Every milestone has two equally important components:

1. **User Functionality:** What the user can DO when this is done.
2. **Backend Foundation:** What abstractions exist that accelerate future milestones.

These are not in tension. They are the same work, viewed from two angles.

**User functionality without foundation** creates tech debt. You ship features, then refactor endlessly to support the next feature.

**Foundation without functionality** creates ivory towers. You build abstractions nobody uses, then discover they're wrong when you finally ship.

**The goal:** Every milestone ships functionality AND leaves behind abstractions that make the next milestone trivial.

---

## The Library-Centric Rule

When you have a choice between:
- Code specific to this feature, or
- A library abstraction that supports this feature and future features

**Choose the library abstraction. Every time.**

This is not optional. This is not "nice to have." Feature-specific code is tech debt you're writing today.

**Test:** After this milestone, can the next milestone be implemented by USING abstractions, or by WRITING new code? If writing new code, the foundation is missing.

---

## Principles

### 1. State Both Components Explicitly

Every milestone must answer two questions:
- **"What can the user DO?"** — One sentence. "I can X."
- **"What foundation does this build?"** — Name the trait, the pattern, the abstraction.

If you can't answer both, the milestone is incomplete.

**Bad:** "In-memory material store" (foundation only, no functionality)
**Bad:** "I can pick from a list of materials" (functionality only, no foundation named)
**Good:** "I can pick from a list of materials. Foundation: `AssetStore<T>` trait with `get()`, `list()`, `search()`."

### 2. End-to-End From Milestone 1

The first milestone produces a working app. It runs. It shows output. It responds to input.

Static implementations are fine. Hardcoded values are fine. But the whole pipeline works. You see something. You click something. Something changes.

**Why:** You cannot validate foundations without functionality. You cannot validate functionality without running code. If you wait 5 milestones before anything works, you won't discover that milestone 1 was broken until milestone 5.

### 3. Coupled Functionality Ships Together

If feature A is useless without feature B, they are the same milestone.

**Example:** Lua scripting without hot reload is useless. Every edit requires an app restart. That defeats the purpose of external scripts. So "Lua materials" and "hot reload materials" are ONE milestone: "I can edit a Lua file and see materials update without restarting."

**Test:** Ask "Can I actually use this feature alone?" If no, it's not a milestone—it's half a milestone.

### 4. Foundations Enable Future Milestones

Before implementing, ask: "What will milestone N+1 need?"

Design your abstractions so that the next milestone is trivial. If this milestone introduces a trait, the next milestone's feature should be "just another implementation of that trait"—no special cases, no refactoring, no new abstractions required.

**Test:** Can M+1 be implemented by instantiating and composing what M built? If M+1 requires changing M's abstractions, M's foundation is incomplete.

### 5. Name Your APIs

Milestones reference specific traits, classes, and bindings. You don't need line-by-line detail, but you need to name what's being built.

**Bad:** "Generator defined in Lua file"
**Good:** "Implement `Generator` trait with `init(ctx)`, `step(ctx) -> bool`. Expose `ctx:set_voxel(x, y, mat_id)` binding."

### 6. Prove Integration Points Early

If two systems must integrate (Lua + Bevy, MCP + HTTP, file watcher + ECS), prove that integration works in a minimal form before building features on top.

Risky integrations fail in surprising ways. Don't discover that mlua doesn't play nice with Bevy's resource system in milestone 4.

### 7. Cleanup Audit After Each Milestone

After completing a milestone, do a quick non-blocking audit:

- **What abstractions did we introduce?** Can they be applied to existing code?
- **What tech debt did we notice?** Flag it, don't necessarily fix it now.
- **What could be refactored?** Now that the foundation exists, what old code could use it?

This is NOT a blocking gate. It's a 5-minute checklist that accumulates cleanup candidates. At phase end, review the accumulated list and decide what to address.

**Why:** Foundations are most valuable when applied broadly. If M4.5 introduces `AssetStore<T>`, but M5/M6 don't use it, you've built an abstraction nobody uses. The audit catches these opportunities.

---

## Milestone Template

```markdown
### M[N]: [Short Name]

**Functionality:** I can [what the user can do].

**Foundation:** Introduces [abstraction/trait/pattern] that [what it enables for future milestones].

- [Visible outcome 1]
- [Visible outcome 2]
- **Implements:** [specific APIs, traits, classes, bindings]
- **Proves:** [integration points validated, if any]

### M[N] Cleanup Audit

- [ ] [Question about applying new abstraction to existing code]
- [ ] [Question about tech debt noticed during implementation]
```

---

## Example: Map Editor Build Sequence

### M1: Static End-to-End
**Functionality:** I can pick from 2 materials and see a checkerboard update.

**Foundation:** Core data structures (`VoxelBuffer2D`, `Material`, `PlaybackState`) that all future milestones build on.

- Window shows rendered 32x32 grid
- Material picker with stone, dirt
- Click material → checkerboard changes
- Static everything (no Lua, no files)
- **Implements:** `VoxelBuffer2D`, `Material`, `MaterialPalette`, `PlaybackState`

### M2: Lua Materials + Hot Reload
**Functionality:** I can edit a Lua file, save it, and see my materials update without restarting.

**Foundation:** Hot-reload infrastructure (`FileWatcher`, Lua loading pattern) reused by all future Lua assets.

- Materials defined in `assets/materials.lua`
- Edit file → app updates within 1 second
- **Implements:** `MaterialStore` trait, `Material` Lua class, file watcher
- **Proves:** mlua + Bevy integration works

### M3: Lua Generator + Hot Reload
**Functionality:** I can edit a Lua generator script, save it, and see the terrain change without restarting.

**Foundation:** `Generator` trait with `init`/`step`/`reset` lifecycle—pattern for all procedural content.

- Generator defined in `assets/generator.lua`
- Edit file → terrain regenerates within 1 second
- **Implements:** `Generator` trait (`init`/`step`/`reset`), `ctx:set_voxel()` binding

### M4: External AI Access (MCP)
**Functionality:** An external AI can create materials and see the rendered output.

**Foundation:** MCP HTTP server pattern—extensible endpoint system for all future AI interactions.

- AI calls `create_material` → appears in picker
- AI calls `get_output` → receives PNG
- **Implements:** MCP server, HTTP endpoints (`/health`, `/mcp/list_materials`, `/mcp/get_output`)

---

## Audit Checklist

For each milestone, verify:

| Check | Question |
|-------|----------|
| Functionality | Can I state what the user can DO in one sentence? |
| Foundation | Can I name the abstraction this builds and what it enables? |
| Library-Centric | Is this a reusable abstraction, not feature-specific code? |
| Acceleration | Can M+1 be built by USING this foundation, not changing it? |
| End-to-End | Does M1 produce a working, visible app? |
| Coupled | Is every feature in this milestone usable on its own? |
| APIs Named | Did I name the specific traits/classes/bindings being built? |
| Integration | Are risky integrations proven before I build on them? |

If any check fails, revise the milestone.

---

## Common Mistakes

### Mistake: Backend-Indexed Milestones
"M1: Bevy shell. M2: In-memory materials. M3: Lua engine. M4: Generator. M5: Renderer."

**Problem:** 5 milestones before anything works. You don't know if M1 was broken until M5.

**Fix:** M1 is static end-to-end. Everything works. Then add complexity.

### Mistake: Functionality Without Foundation
"M2: I can pick materials from a list."

**Problem:** What abstraction does this build? If M3 needs to search materials, do we have to refactor?

**Fix:** "I can pick materials from a list. Foundation: `AssetStore<T>` trait with `get()`, `list()` that M3 extends with `search()`."

### Mistake: Foundation Without Functionality
"M2: Implement `AssetStore<T>` trait."

**Problem:** No user can do anything. No validation that the abstraction is correct.

**Fix:** Ship functionality that exercises the foundation. The foundation is validated by use.

### Mistake: Feature-Specific Code
"M2: `MaterialList` component that shows materials."

**Problem:** What if M3 needs to show generators? You'll write `GeneratorList`. Then `RendererList`. Tech debt.

**Fix:** "Foundation: Generic `AssetListUI<T: Asset>` that works for any asset type."

### Mistake: Splitting Coupled Features
"M3: Lua materials. M6: Hot reload materials."

**Problem:** Lua without hot reload is useless. You have to restart on every edit.

**Fix:** One milestone: "I can edit Lua and see materials update live."

### Mistake: Vague Functionality
"M2: Material system improvements"

**Problem:** What can I do? I have no idea.

**Fix:** "I can create materials with custom colors and see them in the picker."

### Mistake: Missing API Design
"M3: Generator from Lua"

**Problem:** What's the Generator API? What's the lifecycle? What bindings exist?

**Fix:** Either design the API in a prior milestone, or include "Design `Generator` base class" as an explicit deliverable.

---

## Summary

1. **Dual nature.** Every milestone has functionality AND foundation. State both explicitly.
2. **Library-centric.** Choose abstractions over feature-specific code. Every time.
3. **Acceleration.** Design foundations so M+1 is trivial. If M+1 requires refactoring M, the foundation is wrong.
4. **End-to-end from M1.** Working app immediately. Validate foundations through use.
5. **Coupled features ship together.** No half-usable milestones.
6. **Name your APIs.** Traits, classes, bindings—be specific.
7. **Prove integrations early.** Don't discover failures late.
8. **Cleanup audit.** After each milestone, flag refactoring opportunities. Review at phase end.

If you can't explain both "I can X" AND "This builds Y abstraction," the milestone is incomplete.
