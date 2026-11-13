# Economic routing, controlled escalation and operations dashboard

These controls build on the existing classification, task pins, native protocol
adapters and reservation ledger. They do not introduce automatic replay of model
or tool actions, or bypass the gateway's authentication and cost limits.

## Economic selection

Opt in through gateway policy:

```yaml
economics:
  enabled: true
  minimum_saving_fraction: 0.10
  expected_output_tokens: 512
  hint_ttl_seconds: 60
  quality_floor: 0.90
  max_quality_drop: 0
```

Each candidate needs all five configured tariffs and `quality_pass_rate` (0..1,
exclusive of zero) in its model entry (a profile choice can override `quality_pass_rate` for its
specific task/effort configuration). Supply pass rates from representative
reviewed evaluations; HTTP success is not a quality score. Nothing fetches or
fabricates those rates. Missing/invalid quality metadata or tariffs leaves the
administrator's original ordering intact and explains why optimization was skipped.

For fresh automatic tasks, the optimizer estimates input size from serialized
request bytes, caps expected output, and compares estimated cost per quality pass
inside the already approved task profile. Cold input uses the highest configured
input/cache-write tariff. Quality eligibility is the higher of the configured
floor and the first candidate's pass rate minus the permitted drop. With complete
metadata, ineligible candidates are removed from this new assignment, including
fallbacks; no eligible candidate produces 503 before inference. Existing pins
remain unchanged. Only a sufficient estimated saving reorders eligible candidates;
if the original first candidate is below the floor, an eligible candidate replaces it.
Explicit model requests and escalation tickets do not use economic reordering.

Cache hints come from reported cache reads on new assignments, not a guessed
"warm" flag. Their keys hash the authenticated principal, wire protocol, provider
configuration, target/reasoning options and stable system/tool prefix. Prefix
text is never stored. Hints expire after 60 seconds by default (at most 300), and
cache-read credit is capped by an approximate shared-prefix size. Pinned turns
do not refresh hints against their original prefix when later content may have
changed. Models, providers, reasoning configurations and principals do not share
hint keys. These are predictions, not proof of cache reuse. Credential/account or
provider cache behavior can invalidate an apparent hit; inspect actual usage.

The explanation records candidate estimates, supplied quality rates, eligibility,
cache hint sizes and whether ordering changed. No forecast is used to reduce a
strict spending reservation: the ledger still reserves configured worst-case
input/cache/output costs for every remaining candidate. Real savings and quality
preservation still require live workload measurement.

## Feedback and escalation

`POST /v1/nexus/feedback` accepts `source_session` from caller-scoped status and a
rating of `good`, `poor_quality` or `too_expensive`. It records metadata only,
returns no inference result and never starts work. Feedback does not automatically
change the supplied model evaluation rates.

`POST /v1/nexus/escalations` accepts a source session and wire (`messages`,
`responses` or `chat`). The source must belong to the caller and have a recorded
wire explanation. It returns a pending ticket, new session ID, next profile and
level. Legacy pins without a recorded wire cannot be escalated through this API.

The route ladder is utility to standard to deep, with at most two issued steps
per original task. Standard tasks have only the step to deep. Pending tickets
expire after ten minutes and can be retrieved again before use; a used/expired
step cannot be issued again to loop the same task indefinitely. Tickets are
principal- and protocol-scoped. Claiming is transactional and single-use.

Submit a fresh task with the automatic alias, `X-Session-ID` set to the returned
new session ID, and `X-Nexus-Escalation-Ticket` set to the ticket. Prior assistant
turns, tool results, native reasoning items and `previous_response_id` are rejected.
The client supplies a fresh user instruction; the server never retrieves or
replays the old prompt. Admission and budget checks happen before the ticket is
consumed; an admitted ticket is consumed before upstream work starts. Even a
subsequent upstream failure does not automatically replay it. Subsequent ordinary
turns on the new session use the resulting pin without sending the consumed ticket.

Strict per-task cost reservations retain the original task's budget root through
all escalation levels, including follow-ups. Creating a new escalation session
does not grant a new monetary allowance. Ordinary soft token admission gates
retain their existing per-session semantics.

The MCP bridge adds `nexus_escalate_task(source_session, task)`. It issues a ticket
and submits one fresh Messages task with no filesystem/shell tools. It exposes no
automatic retry loop. This remains delegated work in Antigravity; its main-model
traffic and orchestration costs are unchanged.

## Operations dashboard

Open `/dashboard` on the gateway (by default `http://127.0.0.1:7772/dashboard`).
The static page is public, but data APIs require authentication. Enter the gateway
token into the password field; it is retained only in page memory and cleared by
Clear session or reload. No token is embedded in the page, URL, cookie, local
storage or session storage. Static assets use a restrictive Content Security
Policy, `no-store` and `nosniff`. Status values are rendered with text nodes,
never HTML interpolation.

The dashboard shows caller-scoped usage, estimated cost, unknown-price counts,
recent model/effort choices and explanations, feedback controls, and budget
reservations. Request stronger route issues a ticket and displays the fresh-task
parameters; it does not run inference from the dashboard. A shared team key sees
that key's aggregate usage; OIDC identities see their own data. There is no
company-wide administrator role or cross-principal browsing in this release.
The page is an operations console, not a replacement for the host client's model
picker or an enterprise end-user workflow. Use a TLS endpoint for remote access.
OIDC bearer JWTs can be supplied, but the page itself does not implement browser SSO.

## Verification boundaries

Tests exercise quality eligibility, evaluated-cost ordering, cache hint identity
and expiry, pin retention, scoped feedback, bounded escalation, concurrent ticket
consumption, fresh-input checks, MCP delegation, and preservation of the original
monetary allowance. Dashboard assets, authentication boundaries and safe DOM
patterns are checked; JavaScript syntax is checked with Node. On 2026-11-13,
interactive browser checks against a temporary local gateway verified login,
rendered usage and routes, quality feedback, escalation ticket creation, session
clearing, and refusal to refresh without a token. The browser reported no console
errors. These checks used two synthetic routes and did not run provider inference.
See [browser verification](verification/README.md) for evidence and scope.
Provider caches, declared quality rates and actual cloud savings remain
unverified against live inference.
