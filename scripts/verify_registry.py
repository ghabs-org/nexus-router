#!/usr/bin/env python3
import json
import sys
import urllib.request
from pathlib import Path

REGISTRY = Path('/home/ubuntu/.local/state/nexus-router/generated/models.json')
WANT = {
    'openai-codex/gpt-5.4',
    'github-copilot/gpt-5.4',
    'github-copilot/o3',
    'github-copilot/o4-mini',
    'google-gemini-cli/gemini-3.1-pro-preview',
    'google-gemini-cli/gemini-2.5-pro-preview',
}

with REGISTRY.open() as f:
    data = json.load(f)

print('=== registry ===')
for m in sorted(data.get('models', []), key=lambda x: x.get('id', '')):
    if m.get('id') in WANT:
        print(m['id'])
        print('  reasoning=', m.get('scores', {}).get('reasoning'))
        print('  scoreSource.reasoning=', m.get('scoreSource', {}).get('reasoning'))
        print('  authed=', m.get('availability', {}).get('authed'))

print('=== live route ===')
req = urllib.request.Request(
    'http://127.0.0.1:7771/route',
    data=json.dumps({
        'message': 'Compare two architectures and recommend one with tradeoffs.',
        'route_mode': 'reasoning',
        'use_llm_classifier': False,
    }).encode(),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=15) as r:
    route = json.loads(r.read())

print('selected_model=', route.get('selected_model'))
print('selected_provider=', route.get('selected_provider'))
print('fallbacks=', route.get('fallbacks'))
print('score=', route.get('score'))
print('reason=', route.get('reason'))
