# Nexus Router

Multilingual LLM router for OpenClaw. Routes each request to the best model
from your configured providers, based on task type, model capabilities,
provider health, and usage history.

## Philosophy
- **Classify first** — a Nexus classifier agent determines task type
- **Score everything** — capability priors + runtime health + learned history
- **Explain decisions** — every routing decision is logged with a reason
- **Override-friendly** — user can always override; router is advisory by default
- **Persistent mode state** — explicit route mode is saved in router state until changed

## Architecture

```
inbound message
      │
      ▼
┌─────────────────────┐
│  Pre-signals        │  (attachments, code blocks, context size, etc.)
│  (non-linguistic)   │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Classifier agent   │  (Nexus task classification → structured output)
│  (multilingual)     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Router             │  (scores models, selects primary + fallbacks)
│                     │  input: task type + model registry + provider health
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  OpenClaw adapter   │  (sets model for current turn)
└─────────────────────┘
```

## State model

| Layer | File | Description |
|---|---|---|
| Raw catalog | `~/.local/state/nexus-router/generated/openclaw-models.json` (host) / `/app/generated/openclaw-models.json` (container) | Fetched from OpenClaw |
| Normalized registry | `~/.local/state/nexus-router/generated/models.json` (host) / `/app/generated/models.json` (container) | Generated, with scores |
| Family priors | `policies/families.yaml` | Provider/pattern scoring |
| Routing policy | `policies/routing.yaml` | Task → preferred models |
| Overrides | `policies/overrides.yaml` | Per-model manual scores |
| Benchmark cache | `~/.local/state/nexus-router/cache/benchmark/` (host) / `/app/cache/benchmark/` (container) | Raw fetched benchmark cache |
| Router state | `~/.local/state/nexus-router/data/router.sqlite` (host) / `/app/data/router.sqlite` (container) | Outcomes, feedback, persisted route-mode preferences, provider health state, and benchmark scores (DB-only, no YAML fallback) |

## Classifier output schema

```json
{
  "taskType": "coding|code_review|reasoning|summarization|fast_utility|long_context|vision|general_chat",
  "subtype": "string|null",
  "complexity": "low|medium|high",
  "needsTools": true,
  "needsVision": false,
  "needsLongContext": false,
  "costProfile": "cheap|balanced|premium",
  "confidence": 0.91
}
```

## Router output schema

```json
{
  "taskType": "coding",
  "confidence": 0.91,
  "selectedModel": "openai-codex/gpt-5.4",
  "fallbacks": [
    "github-copilot/gpt-5.3-codex",
    "github-copilot/claude-sonnet-4.6"
  ],
  "reason": [
    "code blocks detected",
    "tool-capable coding model preferred",
    "provider healthy"
  ],
  "score": 0.847,
  "excludedModels": [
    { "model": "openai-codex/gpt-5.4-mini", "reason": "below complexity threshold" }
  ]
}
```

## Scoring formula

```
final_score =
  (task_fit    * 0.50)
+ (health      * 0.20)
+ (preference  * 0.10)
+ (learned     * 0.10)
+ (cost        * 0.05)
+ (speed       * 0.05)
```

Models below `health_hard_cutoff` (default: 0.30) are excluded.

### Task objective weights (latency-aware)

The scorer now blends the legacy composite with explicit per-task objectives:

- `fast_utility`: latency-first among healthy providers (fresh `latency_updated_at` required)
- `reasoning`: quality-first, latency is secondary
- `coding` / `code_review`: quality-first with reliability guardrails

Default objective profiles in `src/scorer.py`:

- `fast_utility`: quality 0.18, latency 0.52, reliability 0.15, cost 0.15
- `reasoning`: quality 0.62, latency 0.05, reliability 0.23, cost 0.10
- `coding`: quality 0.58, latency 0.08, reliability 0.22, cost 0.12
- `code_review`: quality 0.56, latency 0.08, reliability 0.24, cost 0.12

Latency freshness handling:
- latency older than 30 minutes is down-weighted
- latency older than 24 hours is treated as stale for objective scoring

## Supported task types

| Task | Description |
|---|---|
| `coding` | Code generation, implementation, debugging, refactoring |
| `code_review` | PR review, diff analysis, quality audit |
| `reasoning` | Planning, strategy, comparison, decision-making |
| `summarization` | Extraction, synthesis, TL;DR, distillation |
| `fast_utility` | Quick lookups, minor edits, short answers |
| `long_context` | Large documents, many files, transcripts |
| `vision` | Images, screenshots, diagrams |
| `general_chat` | Conversational, general questions |

## Quick start

```bash
# Install deps
pip install pyyaml

# Refresh model catalog
openclaw models list --all --json > ~/.local/state/nexus-router/generated/openclaw-models.json

# Generate normalized registry
python src/generate_registry.py
```

## Regenerate catalog

```bash
openclaw models list --all --json > ~/.local/state/nexus-router/generated/openclaw-models.json
python src/generate_registry.py
```

## Local ModernBERT classifier

A local ModernBERT classifier path is now wired into `/route` as the primary
classifier path when a usable ONNX artifact exists at
`artifacts/router-classifier/onnx/` (mounted in the container at
`/app/artifacts/router-classifier/onnx`).

Routing order is now:

1. explicit classifier hint
2. local ONNX classifier
3. heuristic classifier
4. LLM classifier fallback
5. generic fallback

The router response and logs now surface classifier provenance via:
- `classifier_source` (`local`, `heuristic`, `fallback`)
- `classifier_provider`
- `classifier_model`
- `classifier_debug` (local confidence / margin / artifact availability)

See `docs/local-classifier-modernbert.md` for build/export details.
