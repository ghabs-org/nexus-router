# Feedback reason_tag taxonomy (closed vocabulary)

Purpose: keep `route_feedback.reason_tag` countable and stable for weekly calibration.

## Canonical tags

- `task_mismatch` - classifier task label was wrong.
- `task_and_model_mismatch` - both task and model choice were wrong.
- `quality` - model quality/output quality issue (or quality praise when verdict is correct).
- `too_cheap` - model was underpowered for the request.
- `too_powerful` - model was overpowered/too expensive for the request.
- `latency` - model response speed/latency issue.
- `tooling` - tool use/tool calling behavior issue.
- `format` - answer format/structure mismatch.
- `policy` - safety/compliance/policy mismatch.
- `confirmation` - explicit correct confirmation with no issue.
- `other` - fallback bucket for unknown custom tags.

## Normalization rules

On feedback ingest (`src/db.py`), reason tags are normalized using `src/feedback_taxonomy.py`:

1. If `reason_tag` is provided and already canonical, keep it.
2. If alias is provided (`bad`, `cheap`, `too-cheap`, etc.), map to canonical tag.
3. If unknown custom value is provided, map to `other`.
4. If empty, infer from payload:
   - `corrected_task` + model signal => `task_and_model_mismatch`
   - `corrected_task` only => `task_mismatch`
   - `model_verdict=too_cheap|too_powerful` => matching canonical tag
   - `model_verdict=bad|good|neutral` => `quality`
   - `verdict=correct` with no other signal => `confirmation`
   - otherwise => `other`

This keeps weekly calibration reports comparable over time without adding new labels.
