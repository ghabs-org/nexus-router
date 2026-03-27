# Normalized Model Registry

This directory contains the generated model registry used by the Nexus Router.

## Regenerate

```bash
cd /home/ubuntu/.openclaw/workspace/nexus-router
pip install pyyaml
python src/generate_registry.py
```

## Input sources
- `catalog/raw/openclaw-models.json` — raw catalog from `openclaw models list --all --json`
- `policies/families.yaml` — provider/model family scoring priors
- `policies/overrides.yaml` — per-model manual overrides

## Output
- `catalog/normalized/models.json` — all models, fully scored

## When to regenerate
- After `openclaw update`
- When models appear/disappear from providers
- When you edit `policies/families.yaml`
- Periodically (weekly is fine for stability)
