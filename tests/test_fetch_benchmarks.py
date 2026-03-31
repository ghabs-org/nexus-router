import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fetch_benchmarks as fb


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _http_error(url: str, code: int, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "boom", headers, io.BytesIO(b"boom"))


def test_map_public_model_name_uses_catalog_aliases():
    assert fb._map_public_model_name("Claude Sonnet 4.6 Thinking") == "github-copilot/claude-sonnet-4.6"
    assert fb._map_public_model_name("Gemini 2.5 Pro Preview 03-25") == "google-gemini-cli/gemini-2.5-pro"


def test_fetch_with_cache_falls_back_to_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)

    calls = {"count": 0}

    def succeed_once(request, timeout=30):
        calls["count"] += 1
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr(fb.urllib.request, "urlopen", succeed_once)
    first = fb._fetch_json("https://example.test/data.json")
    assert first == {"ok": True}

    def always_rate_limited(request, timeout=30):
        raise _http_error("https://example.test/data.json", 429, retry_after="0")

    monkeypatch.setattr(fb.urllib.request, "urlopen", always_rate_limited)
    monkeypatch.setattr(fb.time, "sleep", lambda *_args, **_kwargs: None)

    cached = fb._fetch_json("https://example.test/data.json")
    assert cached == {"ok": True}
    assert calls["count"] == 1


def test_fetch_with_cache_retries_on_transient_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fb.time, "sleep", lambda *_args, **_kwargs: None)

    attempts = {"count": 0}

    def flaky(request, timeout=30):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _http_error("https://example.test/retry.json", 503)
        return _FakeResponse('{"ok": true, "attempts": 3}')

    monkeypatch.setattr(fb.urllib.request, "urlopen", flaky)
    payload = fb._fetch_json("https://example.test/retry.json")

    assert payload["ok"] is True
    assert attempts["count"] == 3
