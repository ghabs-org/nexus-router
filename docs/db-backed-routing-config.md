## DB-Backed Routing Config

Versioned YAML and generated JSON remain seed/default catalog inputs. Live mutable routing state should live in SQLite when a DB-backed field exists.

Current source-of-truth rules:

- `generated/models.json` provides the static model catalog and fallback metadata.
- `policies/overrides.yaml` is used only when regenerating the static catalog.
- `model_metadata.is_free` is mutable DB state and overrides generated `features.is_free` when present.
- If no `model_metadata` row exists for a model, routing falls back to generated/static `features.is_free`.
- A DB `is_free` value of `NULL` is an explicit unknown value and does not count as free.
- `route_mode_preferences.free_filter` remains DB-backed user/route preference state.
- Route mode `free` and `free_only=True` hard-filter to models whose effective `is_free` is `True`.
