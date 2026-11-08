# Nexus Router

Multilingual LLM router for OpenClaw. Routes each request to the best model
from your configured providers, based on task type, model capabilities,
provider health, and usage history.

## CLI wire proxy

The same server accepts `POST /v1/messages`, `POST /v1/chat/completions`, and
`POST /v1/responses`. Start it with `python -m src.server --port 7771`, then attach
a local CLI shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:7771
export OPENAI_BASE_URL=http://127.0.0.1:7771/v1
curl -s http://127.0.0.1:7771/v1/nexus/cli-status
```

Configure ordered `provider:model` targets under `cli_proxy.routes` in
`policies/routing.yaml`. This section is always used by the wire proxy, independent
of `NEXUS_ROUTER_ENABLE_ROUTING_POLICY`. Upstream `base_url` values must point to
the actual provider, never back to this proxy. Set the configured credential
environment variables **in the server process** (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `NVIDIA_API_KEY` in the defaults). Client auth headers and cookies
are removed; each upstream receives only its own configured credential. Existing
CLI subscription/OAuth login is not automatically exchanged for upstream API
credentials. Codex configurations with an explicit provider `base_url` may need
that URL changed to the loopback `/v1` address as well.

Providers must also have enabled rows in the existing SQLite `providers` table;
YAML does not override DB controls. Unknown providers fail closed. To opt into
the example local Ollama provider, load the configured model and explicitly add
or enable the `local` provider in your router DB, for example:

```bash
python - <<'PY'
from src.db import _connect, ensure_schema
ensure_schema()
with _connect() as db:
    db.execute("INSERT INTO providers (provider, enabled, status) VALUES ('local', 1, 'enabled') "
               "ON CONFLICT(provider) DO UPDATE SET enabled=1, status='enabled'")
PY
```

The default local URL is `http://127.0.0.1:11434/v1`. Inside Docker, change it to
the reachable Ollama host. Container port publication remains `127.0.0.1:7771`;
standalone processes bind loopback unless `--host`/`NEXUS_ROUTER_HOST` is set.

Classification runs deterministic heuristics only, never the advisory router's
LLM classifier. Dedicated compaction/title/probe instructions and lightweight
workers use the inexpensive ordered tier without changing a main session's pin.
Empty tools alone are insufficient evidence for housekeeping. Explicit planning
models, enabled thinking, or Codex plan/debug collaboration markers select the
planning tier. All other requests use the stateful execution tier.

Main sessions use a hashed `x-session-id` (also Codex `session_id`) or a fingerprint
of system instructions plus the first user content. Pins remain stable across
policy reorderings; explicit phase changes or unhealthy DB state permit a new
assignment. Pins are persisted in SQLite (`cli_proxy_pins`, hashed session key only),
reloaded on restart, expire after 24 hours by default, and are bounded to 4096
entries; a reloaded pin whose target is no longer routed for its category is
dropped. Use stable session headers when compaction changes the initial prompt.
Responses `previous_response_id` requires an existing session pin and stays on
that exact provider/model; an unavailable pin returns an error rather than
replaying a server-side continuation elsewhere.

Native requests retain system/tool/message `cache_control` blocks, beta headers,
and all protocol-specific fields. Native SSE bytes are flushed as they arrive;
the proxy never accumulates a streamed completion. Main requests prefer native
fallbacks. Messages ↔ Chat Completions adapters support text, standard function
tools, usage, stop reasons, and incremental SSE text/tool deltas. Images, signed
thinking, server tools, interleaved parallel tool deltas, and unsupported protocol-specific options require a
native route; they are rejected for conversion. Responses requires a native
Responses provider, including on the inexpensive tier.

Before each attempt the proxy consults SQLite enablement, role controls, health,
and cooldown. 429s use the existing escalating circuit; server errors/529s and
transport timeouts open a 60-second DB cooldown. Expired transient circuits can
be tried again, while disabled/auth-expired/quota-exhausted providers remain
blocked. Failover is permitted before response headers are sent. A failure after
streaming starts closes that response and records health without replaying text
or tool calls against another provider.

A per-model breaker sits under the provider circuit: 5xx, transport timeouts and
404 (unknown model) count against `provider:model` for its call category, and
`cooldowns.failures` of them within `window_seconds` skip that target for
`cooldown_seconds` (top-level `cooldowns:` in `policies/routing.yaml`; shared with
the advisory router, see below). 401/403/429 stay provider-wide.

`cli_proxy.ordering: learned` reorders a category's targets for new sessions only:
a later target moves ahead when `health × smoothed 24h success − latency penalty`
beats an earlier one by `learned_margin` (0.15). Pins and native-protocol preference
still come first; client hang-ups (recorded as 499) are left out. Default `static`.

`GET /v1/models` lists the configured route models (OpenAI- and Anthropic-readable).
`POST /v1/messages/count_tokens` goes to the first healthy Messages-native target of
the request's category (the pin first), else 404 so Claude Code estimates locally.

**Subscription backends.** [llm-cli-proxy](../llm-cli-proxy) runs the local
claude/codex/agy/copilot/opencode logins behind Chat Completions on `:8811`. Add it
as a keyless provider (`subscription`, commented in `routing.yaml`) and route to
targets such as `subscription:cc:sonnet`; enable its `providers` row like `local`
above. Limits: chat only, so Messages turns with images or thinking cannot convert
to it (mostly housekeeping/worker and Codex chat); streaming is simulated
(keepalives, then the whole answer); tools are emulated in the prompt; a usage
limit hit after its keepalive commit cannot fail over. llm-cli-proxy ignores
loopback `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` so its CLIs cannot loop back here.

Responses include `X-Nexus-Routed-Model`, `X-Nexus-Call-Category`, and
`X-Nexus-Cache-Pinned` (`true` means an existing pin was reused). Every answered or
failed turn is a `cli_proxy_turns` row: category, target, failures, final status,
latency and token usage read from the forwarded bytes (Messages
`message_start`/`message_delta`, Chat final chunk, Responses `response.completed`;
input includes cache tokens) without buffering the stream. These rows are kept out
of `routing_decisions`, so learned stats and classifier training never see them.
`/v1/nexus/cli-status` shows active hashed pins, the last 100 turns, 24h usage per
target, the applied order, model cooldowns, DB health/controls, and export snippets.
It stores no prompt text or credentials.

Verify with `python -m pytest -q` (development dependencies include `pytest` plus
the repository's training dependencies). `tests/test_cli_proxy.py` starts real
loopback upstreams and the server handler, testing all three native endpoints,
streaming without completion buffering, adapters, pinning, and SQLite failover.
Where TCP listeners are prohibited, `NEXUS_TEST_SOCKETPAIR=1 python -m pytest -q`
uses connected Unix sockets with real HTTP serialization and handlers. This mode
does not establish that TCP binding or a live CLI attachment works.

## Explainability, diagnostics, cooldowns, experiments (advisory `/route`)

- **`/route` inputs:** `allowed_providers` (only score these providers; 409 when none
  is eligible) and `experiment_key` (stable A/B assignment; the plugin sends its
  conversation key).
- **`explain`** in every `/route` response and in `routing_decisions.explain_json`:
  rules triggered, classifier (source, task, effective task, confidence), top-5
  model scores with components/contributions, exclusion counts, policy id.
- **`GET /route-info?id=<decision_id>`:** stored decision + explain + outcome + tokens
  + feedback; never the message text. Plugin: `/route info [id]`.
- **`GET /diagnostics`:** registry/catalog age, local classifier, DB and outcome fill,
  policy YAML parse status, provider health and quota-sync freshness, model cooldowns,
  experiment, CLI proxy summary, and a `warnings` list. Plugin: `/route doctor`.
- **Model cooldowns:** a failed `/outcome` counts against the decision's model on its
  effective task type (a fallback outcome counts against the primary and is credited
  to the fallback); cooling models are excluded as `model_cooldown`.
- **A/B experiments:** `experiment:` in `routing.yaml` (commented example) deep-merges
  the chosen variant over `scoring:` for that call and tags the decision `policy_id`.
  Compare with `python scripts/policy_experiment_report.py --experiment <id>`.

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
- `long_context`: quality 0.60, latency 0.05, reliability 0.23, cost 0.12
- `vision`: quality 0.58, latency 0.08, reliability 0.22, cost 0.12

For `long_context` and `vision`, quality averages the task score (context
window, which scales +0.08 per doubling from 0.80 at 200k, or vision support)
with the model's reasoning score, so the stronger model wins among those that
can take the input.

### Where model scores come from

- Artificial Analysis: Intelligence Index → `reasoning`, Terminal-Bench Hard +
  SciCode → `coding`. Scores are relative to the leaderboard leader, mapped
  onto 0.5–1.0 (the range of the 0.70 default for models with no benchmark).
- `review` and `summarize` have no public benchmark; they derive from
  coding + reasoning and from reasoning.
- Free endpoints (`nvidia/*`, OpenRouter `:free`) score `cost` 1.0; other
  models keep 0.70. List prices are not used: the paid models here run on
  subscriptions, where the quota penalty limits use.
- `policies/overrides.yaml` sets scores by hand, and `available: false` there
  retires a model its provider has shut down but the catalog still lists.

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
python scripts/refresh_catalog.py

# Generate normalized registry
python src/generate_registry.py
```

## Regenerate catalog

The daily `nexus-router-benchmark-sync` job refreshes the catalog, the
benchmarks and the registry, then restarts the router (it loads the registry
once at startup). To do it by hand:

```bash
python scripts/refresh_catalog.py   # add --dry-run to only show the diff
python src/generate_registry.py
docker restart nexus-router-1
```

`refresh_catalog.py` keeps the previous catalog (exit 2) when the fetch looks
partial: no models, the total halving, or a provider losing most of its models
while its discovery reports a sign-in error. A provider that vanishes without a
discovery outcome is treated as disabled in OpenClaw. It also stores OpenClaw's
`agents.defaults.modelPolicy.allow` in the catalog. The registry includes only
models OpenClaw is configured to use: tagged `configured` and matching that
allowlist (some built-in models are tagged without being allowed). Pass
`--include-unconfigured` to `generate_registry.py` to keep the rest.

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
