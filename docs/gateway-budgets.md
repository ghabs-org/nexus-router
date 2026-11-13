# Persistent gateway budget reservations

The gateway now supports opt-in reservation-based cost limits in addition to the
existing soft token admission gates. They apply to Cowork/Vertex, native Codex
protocols, and Antigravity's delegated gateway calls. They do not limit the host
agent's own inference outside this gateway.

## Enable

Add limits to a static client or OIDC limits mapping:

```yaml
limits:
  max_cost_usd_24h: 25
  max_task_cost_usd: 5
  max_concurrent: 4
```

Every candidate, including fallbacks, must have a verified `context_window` and
all five per-million-token tariffs (`input`, `output`, `cache_read`,
`cache_write_5m`, `cache_write_1h`). Rates must reflect the endpoint and applicable
contract/region. Prices are not fetched or guessed. Missing bounds/tariffs reject
strict requests before inference. Existing token limits remain admission gates,
not hard token reservations.

Before any upstream attempt, SQLite `BEGIN IMMEDIATE` atomically checks spend,
active reservations and the new request's conservative ceiling, then reserves
funds. The ceiling includes each candidate's entire declared context at the
highest input/cache tariff, plus its capped output at the output tariff. This
can reserve substantially more than normal consumption and reject a task that
would probably fit. This is intentional; it does not use a cheap token estimate
as a provable ceiling. Ledger amounts are integer micro-USD; charges round up and
allowances round down. All processes must share the same database and compatible
policies for shared limits. This is not independent-replica or distributed-store
support.

## Settlement and unknown usage

A single successful attempt with complete usage and known cache-write TTLs
settles to its estimated tariff cost and releases the unused amount. OpenAI cache
reads remain a subset of input, while Anthropic cache categories are separate.
No network request occurs before its attempt marker is durable.

Failures, interrupted streams, unknown usage, and successful answers following
an earlier attempt retain the **entire request reservation** as uncertain. The
last successful attempt cannot make earlier unknown charges disappear. A request
that failed before any attempt releases its reservation. Process crashes leave
the durable reservation in place; reservations are never silently timed out into
free credit. Uncertain/active/breached obligations continue to consume the rolling
allowance even when their start time is older than 24 hours. Settled charges use
their settlement time for that window. Task charges accumulate for the stable
task identity.

A provider-reported cost above the declared ceiling is recorded as a breach and
blocks further strict admission for that principal until reconciliation. This
means the guarantee depends on truthful tariffs/model bounds and provider cap
compliance. The gateway cannot undo a provider overcharge or enforce a vendor
invoice. Separate fees, taxes, paid server tools, opaque server-side permission
checks and internal inference loops are outside the ceiling; unsupported paid
or opaque server-tool requests are rejected in strict mode. Only ordinary
client-executed function tools are accepted. This is not a prepaid billing system.

Legacy gateway turns without reservation coverage fail closed when enabling a
cost limit. Use a separate state database for strict-budget rollout rather than
silently treating the old usage as zero. SQLite tables are added safely to an
existing database, but legacy billing import is not automated.

## Reconciliation

`GET /v1/nexus/status` adds a caller-scoped `budget` object with held micro-USD,
reservation IDs, states, attempt counts and hashed task IDs. Successful responses
also expose the reservation identifier through `X-Nexus-Request-ID`. No prompts or tokens
are stored in the ledger. The command below is an operator action requiring
filesystem access to the gateway state database, not an end-user HTTP endpoint:

```sh
nexus-router-gateway reconcile-budget \
  --state-dir /path/to/gateway-state \
  --request-id RESERVATION_ID \
  --actual-cost-usd 0.12 \
  --note 'Reconciled against provider usage export'
```

Use independently reviewed provider usage/billing, including failed attempts.
Every adjustment appends an audit entry with old/new amounts and the operator's
note. Keep notes free of prompt text, personal data and credentials. Active
reservations whose owner PID is alive cannot be reconciled; crashed owners may
be reconciled once the process is gone. PID reuse or unavailable process checks
fail closed. This recovery check assumes one host; do not reconcile a shared
network database using PID identities from another host.

## Verification scope

Tests cover tariff settlement, cached token accounting, insufficient capacity,
cross-instance/concurrent reservation checks, retained crash/stream obligations,
legacy usage rejection, paid-tool rejection, provider overruns and audited
reconciliation. Local fixture rates are synthetic, not measured savings or actual
cloud prices. Live provider billing reconciliation remains unverified.

Cache-economic route selection, automatic escalation and a team dashboard are
separate developments; they are not supplied by this budget ledger.
