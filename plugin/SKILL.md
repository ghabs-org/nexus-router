---
name: nexus-router
description: >
  OpenClaw skill for Nexus Router — routes each chat turn to the best model
  using task classification, model capability scoring, and provider health.
  Use when you want automatic model selection instead of a fixed default.
---

# Nexus Router Skill

Smart LLM router that picks the best model for each request based on:
- Task type (coding, review, reasoning, summarization, vision, etc.)
- Model capability scores
- Provider health and quota status
- Learned historical performance

## Prerequisites

The Nexus Router Python server must be running before this skill works:

```bash
cd ${NEXUS_ROUTER_SRC:-/home/ubuntu/git/ghabs/nexus-router}
venv/bin/python -m src.server --port 7771
```

To run it as a background service, see `systemd/nexus-router.service`.

## How it works

```
message → pre-signal extraction → heuristic classification
       → (optional LLM classifier for ambiguous cases)
       → scorer (task_fit × health × preference × learned × cost × speed)
       → selected_model + fallbacks + reason
```

## Configuration (openclaw.json)

```json
{
  "plugins": {
    "allow": ["nexus-router"],
    "entries": {
      "nexus-router": {
        "enabled": true,
        "config": {
          "routerUrl": "http://127.0.0.1:7771",
          "costProfile": "balanced",
          "useLlmClassifier": false,
          "debugMode": false,
          "minConfidence": 0.60
        }
      }
    }
  }
}
```

## Config options

| Key | Default | Description |
|---|---|---|
| `routerUrl` | `http://127.0.0.1:7771` | Nexus Router server URL |
| `costProfile` | `balanced` | cheap / balanced / premium |
| `useLlmClassifier` | `false` | Use LLM for ambiguous messages (costs a micro-call) |
| `debugMode` | `false` | Log routing decisions to console |
| `minConfidence` | `0.60` | Skip routing if classifier confidence is below this |

## CLI test

```bash
# Route a coding task
cd ${NEXUS_ROUTER_SRC:-/home/ubuntu/git/ghabs/nexus-router}
venv/bin/python -m src.cli route --task coding --complexity high --has-code

# Route a vision task
venv/bin/python -m src.cli route --task vision --has-image

# Check model stats after use
venv/bin/python -m src.cli stats
```

## Supported task types

| Type | Description |
|---|---|
| `coding` | Code generation, implementation, debugging |
| `code_review` | PR review, diff analysis |
| `reasoning` | Planning, strategy, architecture |
| `summarization` | Summaries, extraction, TL;DR |
| `fast_utility` | Quick answers, minor edits |
| `long_context` | Large documents, many files |
| `vision` | Images, screenshots, diagrams |
| `general_chat` | Conversational, general questions |

## Regenerate model registry

Run after `openclaw update` or when providers change:

```bash
cd ${NEXUS_ROUTER_SRC:-/home/ubuntu/git/ghabs/nexus-router}
openclaw models list --all --json > ~/.local/state/nexus-router/generated/openclaw-models.json
venv/bin/python src/generate_registry.py
```
