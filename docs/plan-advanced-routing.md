# Implementation Plan: Advanced Routing Features

## Priority 1: Core Transparency & Resilience
These features establish the foundation for observing and protecting the routing layer.

### 1. Route Explainability Payload (formerly #3)
**Goal:** Expose the exact reasoning behind model selection to callers so they can see why a decision was made.
- **Router side (nexus-router):**
  - Formalize the `reason` array currently returned by the `/route` endpoint into a structured `explainability` payload.
  - Schema additions to `RoutingDecision`: `rules_triggered`, `classifier_confidence`, `model_scores` (breakdown of task fit, health, preference, learned), and `exclusion_reasons`.
- **Consumer side (OpenClaw / Nexus ARC):**
  - Update `nexus-router` OpenClaw plugin to optionally emit the explainability dict as debug telemetry.
  - Nexus ARC: add a new slash command `/route-info <id>` or extend `/status` to dump the explainability payload for a given trace.

### 2. Fallback Memory + Cooldowns (formerly #4)
**Goal:** Smart, localized circuit breakers for models instead of global agent fuses.
- **Router side:**
  - Add `cooldown_manager.py` to track transient model failures (e.g. 5xx responses, timeouts) using Redis/memory.
  - Define triggers: e.g. 3 failures in 60s -> 5-minute cooldown for that specific model on that specific task.
  - In `scorer.py`, apply an immediate multiplier of `0.0` or exclude models currently in a cooldown state.

### 3. Task Taxonomy Auto-Refresh (formerly #7)
**Goal:** Periodically refine the router's internal task classification based on actual usage and feedback.
- **Router / Data side:**
  - Create a background job or periodic script (`scripts/taxonomy_refresh.py`) that fetches recent routes and feedback from the Redis streams.
  - Identify drift: when users frequently correct a classification (e.g., "this was coding, not reasoning").
  - Auto-generate or propose adjustments to the few-shot examples or embeddings used by the classifier to improve future accuracy.

---

## Priority 2: Optimization & Diagnostics
Once the core is transparent and resilient, we introduce experiments and deep observability.

### 4. A/B Router Policy Experiments (formerly #5)
**Goal:** Test different routing strategies side-by-side to scientifically optimize decisions.
- **Router side:**
  - Allow defining multiple active scoring policies (e.g., "Policy A: Cost-optimized", "Policy B: Quality-optimized").
  - Add a traffic-splitting mechanism in `router.py` (e.g., 50/50 split based on request ID hash).
  - Tag outgoing telemetry and `explainability` payload with the `policy_id` used.
  - **Data side:** Dashboard or script to compare feedback ("Correct" clicks) and latency/cost between the two policies.

### 5. Operator Command: `/doctor target=router` (formerly #8)
**Goal:** Dedicated health checks for the routing layer via Nexus.
- **Router side:**
  - Expose a `GET /diagnostics` endpoint on `nexus-router`.
  - Aggregate internal state: cache sizes/TTLs, quota sync freshness, current model health aggregates, active cooldowns.
- **Nexus ARC side:**
  - Intercept `/doctor router` in Nexus ARC.
  - Fetch data from `nexus-router:7771/diagnostics` and format as a clear Markdown report for Telegram.

---

## Backlog (The Rest)
These are deferred until the above priorities are mature.

- **Per-user adaptive routing (1):** Deferred. Nice-to-have once the core is rock solid. Involves tracking user-specific preferences instead of global ones.
- **Hard latency/cost SLO mode (2):** Backlog. Strict routing constraints (e.g., `<4s`, `<$0.01`). Involves pre-filtering the catalog before scoring based on historical latency/cost.
- **Cold-start safety policy (6):** Backlog. Special handling for models or tasks that haven't been used recently to avoid unexpected latency spikes on the first call.