# Local Patches / Operational Overrides

Local notes for `nexus-router` patches and behavior changes that matter across redeploys or rebuilds.

This file is for **router/project-local** changes.
OpenClaw runtime hotfixes belong in:
- `/home/ubuntu/.openclaw/workspace/PATCHES.md`

---

## 2026-03-29+ — `/route last|explain` clarity patch

### Problem
`/route explain` used to mix:
- router choice
- actual inspected execution result
- message footer expectations

That was confusing because the footer belongs to the **current command turn**, while `/route explain` inspects a **prior turn**.

### Local patch targets
- `/home/ubuntu/git/ghabs/nexus-router/plugin/src/index.ts`
- `/home/ubuntu/.openclaw/extensions/nexus-router/index.ts`

### Behavior
#### `/route last`
Router-centric summary only:
- routing mode
- classifier/task/confidence
- first pass
- selected model
- fallbacks
- short reason

#### `/route explain`
Adds explicit inspected execution details:
- `Inspected turn model: ...`
- `Execution override detected: yes|no`
- `Runtime status: success|error (+ duration)`
- `Runtime error: ...` when present

It also clarifies that the generic message footer may belong to the current `/route` command turn, not the inspected turn.

### Intent
Make route debugging honest:
- **Selected** = what the router chose
- **Inspected turn model** = what the inspected turn actually ran on
- **Footer usage model** = current command turn footer

---

## 2026-04-01 — explicit route-mode persistence / sticky behavior

### Intended behavior
Explicit route mode is authoritative and must not switch on its own:
- if set to `off`, it stays off
- if set to `auto`, `balanced`, `fast`, or `reasoning`, it stays that way
- only the user/caller should change it

### Local behavior changes
Route mode moved from transient plugin-only memory into router-managed persisted state.
Operationally this means:
- explicit route mode is persisted
- reset-time behavior no longer forces direct sessions back to `off`
- mode resolution can read persisted router state instead of only recent in-memory state

### Related state
- router SQLite state stores route-mode preference (`route_mode_preferences`)

### Intent
Make explicit routing mode sticky across resets/restarts and consistent with user intent.

---

## What we rely on in practice
Current reliable router observability:
- router feedback notifications
- `/route last`
- `/route explain`

These are the practical debugging tools even when generic OpenClaw usage footer behavior is imperfect.

### Important caveat
Without the OpenClaw runtime hotfix, a router-selected model override should be treated as **advisory, not guaranteed**.
The router may still select a model and report it correctly, but actual execution can still diverge if the OpenClaw runtime rejects or replaces that override during live-session selection/fallback handling.
