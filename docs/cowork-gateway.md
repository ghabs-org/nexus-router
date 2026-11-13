# Claude Cowork gateway on Vertex AI

The standalone gateway gives Cowork an **Automatic (Nexus)** model entry. It
selects an allowed model and thinking/effort configuration when a task starts,
then retains that choice across tool calls and service restarts. Users can
instead select an explicitly allowed model and set that model's effort.

This is an inference gateway, not an MCP worker. Cowork keeps its existing file
access, permissions, tool execution, and agent loop. Vertex runs inference. No
second agent is launched and no extra model call is made to classify a task.
The existing advisory/OpenClaw server and CLI wire proxy remain separate.

## Install from this checkout

Python 3.11 or later is required. The gateway package installs HTTP, YAML, Google
authentication, and JWT libraries; it does not install ONNX, Torch, or OpenClaw.

```bash
uv tool install .
nexus-router-gateway init
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=europe-west1
gcloud auth application-default login
nexus-router-gateway check
nexus-router-gateway enable-provider vertex
nexus-router-gateway serve
```

`init` creates a private configuration and a private sibling `gateway.env` with
a generated gateway client token. It refuses to overwrite either file. `check`
validates configuration and client credentials without making an inference call.
`enable-provider` explicitly enables the provider in the shared router SQLite
controls. YAML never overrides a disabled provider.

ADC must identify an account permitted to invoke the enabled Claude models in
that GCP project and region. Attached service identities are preferable for a
shared GCP deployment; no service-account key file is needed. ADC access tokens
are cached and refreshed under a lock. `access_token_env` supports credentials
refreshed externally. The gateway never forwards Cowork's client credential to
Vertex, and it never trades a Claude subscription login for Google credentials.

## Configure Cowork

```bash
nexus-router-gateway cowork-config --output /tmp/cowork-nexus.json
```

The output contains the local gateway client token and has mode 0600. Import its
settings using Claude Desktop's enterprise configuration flow and export through
MDM for managed devices. It does not modify Claude Desktop settings automatically.
For a shared deployment, pass `--url https://gateway.example.corp`.

The generated settings set `inferenceProvider` to `gateway`, select `nexus-auto`
first, and map its `behavesAs` to the configured known Claude model. They also
list the actual allowed models with family tiers for explicit selection and
Cowork's tier-based subagents. Auto-discovery is disabled because it only
recognizes Claude model IDs. The alias mapping requires a desktop version that
supports `behavesAs` (documented as 4.29.0 or later).

In **Automatic (Nexus)** mode, Nexus owns reasoning settings. Cowork's incoming
top-level effort/thinking defaults are replaced by the selected profile. The
native picker cannot display a different resolved model/effort on every task.
For a manual effort choice, select a real allowed model, or have an API client
send `X-Nexus-Effort` at the beginning of a task. This header is not a Cowork UI
feature. Actual routing is exposed in response headers and the status endpoint.

The generated configuration selects an allowed model for Cowork's built-in
permission classifier and leaves server-side Auto mode checks off. That preserves
the built-in permission review; it does not disable approvals or safety checks.
If enabling server-side checks, first verify that the chosen Vertex model and
deployment support the required safeguards protocol. The gateway forwards the
native fields unchanged, but does not generate safeguards verdicts itself.

## Routing and reasoning contract

`src/gateway.example.yaml` is the complete starting policy. Copy/edit the generated
configuration, not the OpenClaw `policies/routing.yaml`. Restart after edits.

| Profile | Initial target | Reasoning |
| --- | --- | --- |
| utility | Haiku | Thinking disabled |
| standard | Sonnet | Adaptive thinking, medium effort |
| deep | Opus | Adaptive thinking, high effort |

These are editable examples, not claims that those models are optimal for every
organization. Confirm model availability in your Vertex region. Each allowed
model declares supported thinking modes, efforts, output ceiling, context window,
and defaults. Invalid combinations fail before inference. Manual thinking budgets
must fit below `max_tokens`; output caps include thinking and visible output.

Rules are ordered deterministic expressions over the first paragraph of the
opening user instruction, capped at 2,000 characters. The supplied rules cover a
small English/Italian set. Unknown instructions use **standard**, not utility.
The rule system is deliberately conservative; it is not a general semantic
classifier and is not a promise of unchanged answer quality. Document blocks,
later tool results, and generic system-prompt task descriptions do not select the
business profile. Existing CLI heuristics separately identify background work.

Profiles list ordered fallbacks. Every attempt checks existing SQLite provider
controls, health, and model cooldowns. A model 404 can select the next target.
Provider-wide authentication, quota, or cooldown blocks all models on that same
provider. Having two Vertex models is not redundancy for a Vertex outage.
No retry occurs after response headers are committed. Signed/redacted thinking
history is accepted only with an existing task pin and cannot fail over to a
different model. Do not strip signatures to force a cheaper model.

## Session identity and caching

The best identity is a stable `X-Session-ID`, `session_id`, or
`X-Claude-Code-Session-ID`. Supported Claude metadata session IDs are recognized
too. Otherwise the gateway fingerprints system instructions and first user
content. A fingerprint is best effort: identical openings can collide, and
compaction that changes the opening can change identity. Verify the deployed
Cowork version's session markers before relying on continuity after compaction.
Every identity is scoped to the authenticated principal before hashing.

Main tasks persist model/profile/reasoning snapshots. A different requested
model or explicit effort on an existing task returns 409; start a new task to
change it. Independent background calls never overwrite the main pin. Pins expire
after 24 hours by default, independently of provider cache TTL. A pin is routing
affinity, not proof that the provider cached anything.

Native Messages content is retained, including tools, cache breakpoints, TTLs,
thinking blocks, signatures, and tool results. Only routing-owned model/reasoning
fields and configured output ceilings change. Vertex moves the model into the
URL, adds `anthropic_version`, and chooses `rawPredict` or `streamRawPredict`.
Native SSE is forwarded incrementally without completion buffering. Beta headers
are retained; legacy deployments can set `beta_transport: body` explicitly.

The gateway never adds timestamps or routing explanations to the system prompt.
It keeps reasoning stable because changing thinking/effort can invalidate the
cached prefix. It does not claim cache sharing across models or regions. It also
does not silently enable per-message effort betas: support must be confirmed for
the exact Vertex model and endpoint before adding that optimization.

Provider responses retain all usage fields. The gateway separately records base
input, cache reads, cache writes, 5-minute/1-hour writes, and output. Check these
actual counters when judging savings; successful HTTP responses and session pins
alone say nothing about answer quality or cache hits.

## Authentication, limits, and accounting

Static client keys live in environment variables referenced by `clients`. Give
different users or teams different keys. Shared keys share attribution and limits.
Every endpoint except `/health` requires Bearer or `x-api-key` authentication.
Status results contain only that principal's usage and hashed task identifiers.

For user-level enterprise SSO, configure `oidc.issuer`, `oidc.audience`, and
`oidc.jwks_url`, and remove static clients if unnecessary. Only RS256/ES256 JWTs
with valid signatures, issuer, audience, expiry, and subject are accepted. An
optional subject allowlist further restricts access. Keys refresh from JWKS;
tokens and prompts are never stored. Generate Desktop SSO settings with
`cowork-config --oidc-client-id YOUR_PUBLIC_CLIENT_ID`. Match the gateway audience
to the ID-token audience; opaque access tokens are not supported by this verifier.

`max_output_tokens` is a hard per-request output ceiling. `max_tokens_24h` and
`max_task_tokens` are **admission gates based on reported input plus output**.
Already admitted requests can overshoot them, so these are not hard dollar
budgets. `max_concurrent` limits in-flight requests for a principal. With token
gates configured, incomplete usage from an interrupted response blocks further
requests until the administrator reconciles accounting. Do not describe these
controls as a prepaid balance or a billing-grade spending cap.

Optional `pricing` is your actual per-million-token Vertex/contract tariff.
Unconfigured prices are reported as unknown. If cache writes lack their TTL split,
cost remains unknown rather than inventing a tariff. Estimates exclude retries
whose provider usage is unavailable and are not the authoritative GCP bill.
SQLite stores metadata and usage only; no prompts, files, credentials, or raw
user/session IDs. Treat the state directory as operational data and protect it.

## HTTP interface

| Endpoint | Contract |
| --- | --- |
| `GET /health` | Public liveness only; no inference/credential readiness claim |
| `GET /v1/models` | Authenticated automatic alias and allowed model IDs |
| `POST /v1/messages` | Native Messages, JSON or incremental SSE |
| `POST /v1/messages/count_tokens` | 404; caller uses its local estimate |
| `POST /v1/nexus/route-preview` | Messages-shaped input; profile/choices, no pin or inference |
| `GET /v1/nexus/status` | Caller-scoped 24-hour usage and last 50 routing outcomes |

Successful inference responses include `X-Nexus-Routed-Model`, `X-Nexus-Profile`,
`X-Nexus-Effort`, and `X-Nexus-Cache-Pinned`. Unsupported models/options fail 400;
authentication fails 401/403; profile changes fail 409; admission gates return
402 with a Messages `billing_error`; concurrency gates return 429; no healthy
compatible target returns 503. Body size and read time are bounded. This service
does not enable CORS. Native OpenAI endpoints are also available when
protocol-specific profiles are configured; see [client gateways](client-gateways.md).

## Enterprise deployment

Use one gateway process and durable SQLite volume, behind an authenticated TLS
ingress with request/connection limits. Deploy the standalone `Dockerfile.gateway`
on a GCP service identity with only the required project/model permissions. Set
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and client-key variables, or
configure OIDC. Mount the policy at `/config/gateway.yaml` and a writable state
directory for UID 10001. Never publish a plain HTTP gateway on a public interface.

SQLite admission and in-process concurrency controls are **single-process**.
Do not scale this image to multiple independent replicas and claim shared pins,
limits, or quotas. Multi-replica deployment needs a shared transactional state
store. Pin the required Vertex region; fallback configuration must stay within
the organization's permitted provider and residency boundary.

## Distribution

Build the Python wheel/source archive using a PEP 517 frontend such as
`python -m build`. The wheel can be installed with pipx or uv. Package creation
does not publish to a registry or install on employee machines.

* npm: `npm pack ./packaging/npm --pack-destination ./dist`. The wrapper installs
  the matching Python release in a private environment. Before publication, set
  `NEXUS_GATEWAY_WHEEL` to a trusted local wheel or `NEXUS_GATEWAY_EXECUTABLE` to
  an already installed executable. Arguments are forwarded without a shell.
* Debian: `python packaging/build_deb.py`. Install with `apt install ./dist/...deb`
  on a distribution satisfying the declared Python-library versions. The package
  installs code and a systemd unit; create `/etc/nexus-router/gateway.yaml` and
  `/etc/nexus-router/gateway.env`, then enable the service explicitly. The unit
  loads the private policy through systemd credentials, so its dynamic user can
  read it without making the file public. No maintainer script starts it automatically. An apt
  repository and its signing are separate publication work.
* Homebrew: `python packaging/render_homebrew.py dist/nexus_router_gateway-0.2.0.tar.gz
  --url https://YOUR_RELEASE/source.tar.gz --output dist/nexus-router-gateway.rb`.
  The formula pins that archive's SHA-256. A local file URL supports smoke tests.
  Publish a reviewed tap for organization-wide brew installation. The generated
  formula resolves dependencies through pip; vendor pinned resources if your
  organization requires offline/reproducible installation.

## Verification and operational limits

Run `python scripts/run_gateway_checks.py` for gateway and wire-proxy checks.
Use `python scripts/run_gateway_checks.py --all --socketpair` for the full suite
where TCP listeners are forbidden. The runner isolates runtime state and patches
older HTTP fixtures that do not read `NEXUS_TEST_SOCKETPAIR` themselves.
The HTTP tests exercise real gateway handlers, HTTPX serialization, streaming,
Vertex-shaped upstream requests, cache accounting, tenant isolation, failover,
and persistence. Socket pairs do not establish TCP reachability or actual Vertex
feature acceptance. OIDC tests use locally signed JWTs; ADC tests use test
credentials and verify locking/refresh behavior, not Google authorization.

A real-device release check must still exercise Cowork with your Vertex project,
region, enabled models, beta features, session headers, compaction, and cached
follow-up turns. Compare completed-task quality and GCP costs with sensible fixed
model/effort defaults before claiming savings. No live credential is needed for
the local test suite and no benchmark result is fabricated by this implementation.

## Provider references

* [Cowork gateway integration](https://claude.com/docs/third-party/claude-desktop/gateway)
* [Third-party Desktop configuration](https://claude.com/docs/third-party/claude-desktop/configuration)
* [Claude on Google Cloud](https://platform.claude.com/docs/en/api/claude-on-vertex-ai)
* [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
* [Thinking and effort](https://platform.claude.com/docs/en/build-with-claude/thinking)
