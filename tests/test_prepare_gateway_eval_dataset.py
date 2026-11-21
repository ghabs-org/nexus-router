from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_gateway_routing, prepare_gateway_eval_dataset as prep


def test_sanitize_prompt_redacts_secrets_and_preserves_paths():
    raw = (
        "Debug /home/ubuntu/git/ghabs/nexus-router/src/gateway.py with "
        "Authorization: Bearer abc123secret_token_xyz and key nvapi-1234567890abcdef "
        "at https://example.com/cb?token=supersecretvalue&ok=1"
    )
    cleaned = prep.sanitize_prompt(raw)
    assert "/home/ubuntu/git/ghabs/nexus-router/src/gateway.py" in cleaned
    assert "abc123secret_token_xyz" not in cleaned
    assert "nvapi-1234567890abcdef" not in cleaned
    assert "supersecretvalue" not in cleaned
    assert "Bearer [REDACTED]" in cleaned
    assert "[REDACTED_KEY]" in cleaned


def test_extract_prompts_picks_first_substantive_per_session_and_dedupes(tmp_path):
    claude = tmp_path / "claude_history.jsonl"
    claude.write_text(
        "\n".join([
            json.dumps({"display": "/compact", "sessionId": "s1", "timestamp": 100}),
            json.dumps({"display": "ok", "sessionId": "s1", "timestamp": 101}),
            json.dumps({"display": "Refactor the auth middleware for OIDC across routes", "sessionId": "s1", "timestamp": 102}),
            json.dumps({"display": "Follow up message in same session", "sessionId": "s1", "timestamp": 103}),
            json.dumps({"display": "/home/ubuntu/repo/src/app.py has a race condition on shutdown", "sessionId": "s2", "timestamp": 104}),
        ]) + "\n"
    )
    codex = tmp_path / "codex_history.jsonl"
    codex.write_text(
        "\n".join([
            json.dumps({"text": "Refactor the auth middleware for OIDC across routes", "session_id": "c1", "ts": 200}),
            json.dumps({"text": "Translate this changelog into German", "session_id": "c2", "ts": 201}),
        ]) + "\n"
    )
    out = tmp_path / "prompts.jsonl"
    summary = prep.extract_prompts(
        claude_history=claude,
        codex_history=codex,
        output_path=out,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    tasks = [r["task"] for r in rows]
    assert summary["written"] == 3
    assert "Refactor the auth middleware for OIDC across routes" in tasks
    assert "/home/ubuntu/repo/src/app.py has a race condition on shutdown" in tasks
    assert "Translate this changelog into German" in tasks


def test_distill_prompts_dry_run_and_resumable_caller(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join([
            json.dumps({"id": "c-1", "source": "claude", "task": "Rewrite this release note for clarity"}),
            json.dumps({"id": "c-2", "source": "codex", "task": "Investigate distributed cache invalidation architecture"}),
        ]) + "\n"
    )
    out = tmp_path / "distilled.jsonl"

    dry = prep.distill_prompts(input_path=prompts, output_path=out, dry_run=True)
    assert dry["mode"] == "dry_run"
    assert dry["pending"] == 2
    assert not out.exists()

    calls: list[str] = []

    def fake_caller(_url, _key, _model, task):
        calls.append(task)
        if "Rewrite" in task:
            return "utility", 0.94, "short rewrite task"
        # Deliberately low confidence + mismatch test
        return "standard", 0.72, "borderline debugging task"

    import os
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-test-placeholder-key-123456"
    try:
        res_part = prep.distill_prompts(
            input_path=prompts,
            output_path=out,
            provider="openai",
            limit=1,
            caller=fake_caller,
        )
        assert res_part["written"] == 1
        assert len(calls) == 1

        res_rest = prep.distill_prompts(
            input_path=prompts,
            output_path=out,
            provider="openai",
            caller=fake_caller,
        )
        assert res_rest["already_done"] == 1
        assert res_rest["written"] == 1
        assert len(calls) == 2

        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) == 2
        assert rows[0]["needs_review"] is False  # matches utility rule & conf 0.94
        assert rows[1]["needs_review"] is True   # conf 0.72 < 0.85
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key


def test_benchmark_compare_local_reports_delta(tmp_path, monkeypatch, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join([
            json.dumps({"id": "1", "task": "Rewrite this note", "profile": "utility"}),
            json.dumps({"id": "2", "task": "Debug async deadlock in gateway worker", "profile": "deep"}),
        ]) + "\n"
    )
    monkeypatch.setattr(
        "src.gateway_classifier.local_prediction",
        lambda text: (
            {"task_type": "reasoning", "complexity": "high", "confidence": 0.95, "margin": 0.4}
            if "deadlock" in text
            else {"task_type": "fast_utility", "complexity": "low", "confidence": 0.95, "margin": 0.4}
        ),
    )
    rc = benchmark_gateway_routing.main([
        "--config", str(ROOT / "src/gateway.codex.yaml"),
        "--cases", str(cases),
        "--compare-local",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "offline_gateway_comparison"
    assert out["rules_only"]["profile_accuracy"] == 0.5
    assert out["with_local_model"]["profile_accuracy"] == 1.0
    assert out["accuracy_delta"] == 0.5
    assert out["disagreements"]


def test_resolve_distill_target_ignores_openclaw_secretref_dict(tmp_path, monkeypatch):
    oc_dir = tmp_path / ".openclaw"
    oc_dir.mkdir()
    (oc_dir / "openclaw.json").write_text(
        json.dumps({"models": {"providers": {"nvidia": {"apiKey": {"source": "env", "provider": "default", "id": "NVIDIA_API_KEY"}}}}})
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert "NVIDIA_API_KEY" not in prep._load_env_file_keys()
    assert prep.resolve_distill_target("nvidia") is None
