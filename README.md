# Nexus Router

Multilingual LLM router for OpenClaw. Routes each request to the best model
from your configured providers, based on task type, model capabilities,
provider health, and usage history.

## Philosophy
- **Classify first** — a Nexus classifier agent determines task type
- **Score everything** — capability priors + runtime health + learned history
- **Explain decisions** — every routing decision is logged with a reason
- **Override-friendly** — user can always override; router is advisory by default

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
| Raw catalog | `catalog/raw/openclaw-models.json` | Fetched from OpenClaw |
| Normalized registry | `catalog/normalized/models.json` | Generated, with scores |
| Family priors | `policies/families.yaml` | Provider/pattern scoring |
| Routing policy | `policies/routing.yaml` | Task → preferred models |
| Overrides | `policies/overrides.yaml` | Per-model manual scores |
| Runtime health | `~/.local/state/nexus-router/state/runtime-health.json` (host) / `/app/state/runtime-health.json` (container) | Auth/quota/error state |
| Routing history | `~/.local/state/nexus-router/data/routing-history.sqlite` (host) / `/app/data/routing-history.sqlite` (container) | Outcomes and feedback |

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
openclaw models list --all --json > catalog/raw/openclaw-models.json

# Generate normalized registry
python src/generate_registry.py
```

## Regenerate catalog

```bash
openclaw models list --all --json > catalog/raw/openclaw-models.json
python src/generate_registry.py
```
