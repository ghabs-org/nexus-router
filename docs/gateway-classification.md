# Gateway classification and explanations

This milestone adds task classification and inspectable routing decisions to all
native gateway protocols. It does not implement strict spending reservations,
automatic retries/escalation, a team dashboard, or a cache-economic optimizer.
Existing admission limits and session/model affinity keep their previous meaning.

## Decision order

For a new automatically routed task, the hybrid classifier examines only the
first paragraph of the first user instruction, capped at 2,000 characters:

1. Conservative multilingual complexity indicators select the reasoning profile.
   This takes precedence over a broad cheap rule, such as translating a root-cause
   analysis. The indicators cover a limited English, Italian, French, German and
   Spanish vocabulary; they are not a multilingual language model.
2. Ordered administrator rules select their configured profile. These remain
   explicit policy decisions, without invented confidence scores.
3. If local inference is enabled, the existing ONNX router classifier supplies
   task type, complexity, top-label score and label margin. Both configured score
   gates must pass. Unknown labels, invalid scores, low confidence, missing model
   artifacts and runtime failures use the default profile. No paid inference
   classifier call or automatic model download is introduced.
4. Without local inference, conservative multilingual utility/summary indicators
   apply; ambiguous instructions use the default profile.

Explicit model selection, background-worker classification and existing task pins
retain their previous precedence. A follow-up cannot silently downgrade a pinned
model or alter reasoning settings. Neither quoted documents after the paragraph
boundary nor later tool results are inspected by this business-task classifier.
An attacker-controlled instruction can still influence routing: this is a cost
selection mechanism, not a security or permission boundary.

## Configuration

Both shipped policies now include:

```yaml
classifier:
  mode: hybrid
  use_local: false
  min_confidence: 0.8
  min_margin: 0.15
```

`mode: rules` selects administrator rules and the default, with no multilingual
indicators. `use_local` is opt-in. The package's optional `classifier` dependency
extra adds numpy, ONNX Runtime and tokenizers, without the training stack.
Deploy your existing trained artifacts and configure
`NEXUS_ROUTER_LOCAL_CLASSIFIER_DIR` in the gateway environment. No new trained
weights or calibrated quality claim are part of this change.

`task_profiles` can override task-to-profile mappings. Defaults map fast utility
to utility, reasoning to deep, and other recognized task types to standard.
Absent conventional profiles fall back to the configured default; explicitly
configured mappings to nonexistent profiles are rejected. Score gates must be
finite values between zero and one. Model scores are not calibrated correctness
probabilities, and the explanation reports `calibrated: false`.

## Explanations and privacy

`POST /v1/nexus/route-preview` returns `explanation` beside profile, reason and
choices. Native clients can specify `?wire=responses` or `?wire=chat`, using that
protocol's request body and profiles. Preview makes no upstream call and writes
no task pin. Optional local classification may run.

Explanations contain source, task type, complexity, score/margin when available,
fallback reason, matched administrator-rule index and chosen profile. Existing
pins report `pin_reused`. Source identifiers are also exposed through
`X-Nexus-Routing-Source`. No matched text, document content or prompt is inserted
into the explanation, response system prompt, headers or SQLite metadata.

Task explanations survive restarts in an additive SQLite table. Each recorded
turn has its own explanation, returned through caller-scoped
`GET /v1/nexus/status`; this avoids rewriting historical decisions when a later
turn reuses the pin. Old database rows without explanations remain readable.
Expired/evicted task explanation entries are removed on pin lookup.

## Offline routing benchmark

```sh
python scripts/benchmark_gateway_routing.py
python scripts/benchmark_gateway_routing.py --config src/gateway.codex.yaml
```

The shipped JSONL has 15 synthetic multilingual and ambiguity examples. The
benchmark reports profile matches, decision sources, failures and elapsed time.
It returns a nonzero status for mismatches. It never calls an inference provider.
A match on these authored regression examples is not evidence of broad semantic
accuracy, answer quality, latency at production load, or monetary savings.
Use `--cases` with a reviewed representative routing dataset for broader coverage.

## Comparing evaluated outcomes

The same command accepts `--outcomes reviewed-results.jsonl`. Each case must
have one baseline and one routed record, for example:

```json
{"case_id":"review-01","strategy":"baseline","cost_usd":0.10,"cost_basis":"estimated","quality_score":0.9,"success":true,"latency_ms":2000}
{"case_id":"review-01","strategy":"routed","cost_usd":0.07,"cost_basis":"estimated","quality_score":0.9,"success":true,"latency_ms":1500}
```

These numbers are schema examples, not measured results. Supply your own human-
evaluated or task-assertion-derived quality score (0..1) and success outcome.
Include all attempts/retries in each strategy's task cost. Use comparable tasks
and evaluation criteria, and the same estimated/billed cost basis throughout.
Duplicate or unpaired cases, invalid/nonfinite values and mixed cost bases fail
validation. The comparison reports total cost, cost per successful task, quality,
success rate, latency and deltas; zero-success runs have no finite cost-per-success.
Caller-supplied evaluations are labeled as such and are not independently verified.

No real-provider quality or cost benchmark was run in this milestone. That needs
representative enterprise tasks, review criteria and the approved provider setup.
