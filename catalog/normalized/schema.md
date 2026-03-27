# Normalized Model Registry Schema

## Purpose
One entry per model. Generated from the OpenClaw model catalog (`openclaw models list --all --json`),
enriched with family scoring priors from `policies/families.yaml`, and optional per-model
overrides from `policies/overrides.yaml`.

## Generation
Run: `python src/generate_registry.py` (or ts equivalent)

Input:
- `catalog/raw/openclaw-models.json`
- `policies/families.yaml`
- `policies/overrides.yaml`

Output:
- `catalog/normalized/models.json`

## Schema

```json
{
  "generatedAt": "ISO-8601",
  "totalModels": 0,
  "models": [
    {
      "id": "openai-codex/gpt-5.4",
      "provider": "openai-codex",
      "modelId": "gpt-5.4",
      "name": "GPT-5.4",
      "features": {
        "supportsVision": true,
        "supportsTools": true,
        "supportsReasoning": false,
        "contextWindow": 266000,
        "inputModalities": ["text", "image"]
      },
      "availability": {
        "authed": true,
        "available": true,
        "local": false
      },
      "scores": {
        "coding": 0.96,
        "review": 0.82,
        "reasoning": 0.85,
        "summarize": 0.74,
        "fast": 0.68,
        "cost": 0.55,
        "context": 0.80,
        "vision": 0.75,
        "tools": 0.92,
        "multilingual": 0.78
      },
      "scoreSource": {
        "coding": "pattern:gpt-5.4",
        "fast": "family:default",
        "context": "family:default"
      }
    }
  ]
}
```

## Notes
- `scores.*` are 0.0–1.0 (higher = better for that dimension)
- `scoreSource` tracks where each score came from (for auditability)
- `features.supportsVision` is derived from `input` containing "image"
- `features.contextWindow` is raw from OpenClaw catalog
- `availability.authed` is from the OpenClaw catalog `available` flag
