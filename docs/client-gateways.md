# Codex and Antigravity integrations

Both integrations use the installed `nexus-router-gateway` package and the same
policy, provider-health controls, identity isolation, task pins and usage database.
No client settings are overwritten automatically.

## Codex: native Responses gateway

Create an OpenAI-only policy, or add its providers/models and `protocol_profiles`
sections to an existing enterprise policy:

```sh
nexus-router-gateway init --template codex --config /tmp/nexus-codex.yaml
nexus-router-gateway enable-provider openai --config /tmp/nexus-codex.yaml
nexus-router-gateway serve --config /tmp/nexus-codex.yaml
nexus-router-gateway codex-config --output /tmp/nexus-codex.toml
```

Set `OPENAI_API_KEY` in the gateway process. Export the generated gateway client
key as `NEXUS_GATEWAY_TOKEN` in the Codex process; it is a different credential.
Review and merge the generated TOML into your user-level Codex configuration.
Provider settings belong in user configuration, not an untrusted project file.
Example model IDs and effort levels must match your upstream account's capabilities.
The example deliberately makes no claim of model availability or pricing.

Native `/v1/responses` and `/v1/chat/completions` are accepted. The gateway never
converts Responses reasoning items, function calls, continuation IDs or cache
fields to Messages. Model and effort are selected together; the adapter writes
`reasoning.effort` or `reasoning_effort` and applies an output ceiling. All other
native fields are retained. Streaming Chat requests request usage explicitly.
OpenAI cached-input tokens are a subset of input tokens, so accounting does not
add them twice. This cache read category is retained separately from uncached input.

`protocol_profiles.responses` and `protocol_profiles.chat` each contain the same
utility/standard/deep profile keys as the primary policy, with native-compatible
targets. Cross-protocol fallbacks fail closed. Pins include the wire protocol in
the identity; an identically named Cowork and Codex task cannot overwrite each
other. Responses `previous_response_id` requires the existing session pin and
cannot fail over to another upstream model. Keep the client's session ID stable.
Native reasoning and continuation history is forwarded unchanged.

## Antigravity: MCP task delegation

The supported integration implemented here is a local stdio MCP server. It does
**not** intercept Antigravity's main inference traffic or change its model picker.
No supported main-model endpoint override was established in the documentation
review; this is explicitly task delegation, not transparent main-model routing.

```sh
nexus-router-gateway antigravity-config --output /tmp/nexus-antigravity.json
```

Merge the generated `mcpServers` entry into the configuration opened through
Antigravity's MCP manager. Its command is `nexus-router-gateway mcp`, which must
be on the app's PATH. Supply `NEXUS_GATEWAY_TOKEN` in that process environment
through your managed deployment. A GUI app does not necessarily inherit shell
exports. The gateway URL can be supplied with `--url`; remote URLs require HTTPS.
The snippet contains the environment variable's name, never its secret value.

The bridge exposes `nexus_run_task(task, session_id?)` and `nexus_status()`.
A task call makes one bounded native Messages request to the shared gateway,
which applies the configured model/effort profile, health controls, admission
limits and accounting. Task strings are bounded; a session ID allows follow-ups
to retain their route. Without one, each task gets its own identity. The bridge
has no filesystem or shell tools and never executes model-returned tool calls.
Use it for review, analysis, extraction and drafting. Antigravity retains its own
orchestration cost; this integration does not promise to eliminate that cost.

Run this bridge against a gateway with Messages profiles (the Cowork/Vertex
example, or a combined policy). An OpenAI-only Codex policy does not provide a
Messages upstream, and is intentionally not used through a lossy conversion.

## Verification

`python scripts/run_gateway_checks.py --all --socketpair` runs the existing and
new checks with isolated state when listening TCP sockets are prohibited.
New tests exercise real HTTP through the native gateway, model/effort mapping,
cache accounting, response continuations, and MCP delegation. A subprocess test
checks MCP initialization/tools discovery and generated client configurations.
These tests establish local contracts, not a live desktop attachment or provider
feature acceptance. Actual Codex, Antigravity and upstream accounts still require
an environment-specific attachment check before fleet rollout.

References:
- https://antigravity.google/docs/mcp
- https://developers.openai.com/codex/config-advanced
