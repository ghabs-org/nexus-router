"""Native Vertex Messages transport; no Chat Completions conversion."""

import copy
import os
import re
import threading
from urllib.parse import quote, urlparse


class VertexCredentials:
    """Reuse and refresh ADC safely across the threaded HTTP handlers."""

    def __init__(self):
        self._credentials = None
        self._lock = threading.Lock()

    def token(self):
        import google.auth
        from google.auth.transport.requests import Request

        with self._lock:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not self._credentials.valid:
                self._credentials.refresh(Request())
            return self._credentials.token


def validate_provider(provider):
    if provider.get("transport", "messages") != "vertex":
        return
    for key in ("project", "region"):
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", provider.get(key, "")):
            raise ValueError(f"Vertex {key} is required and must be a literal identifier")
    base = urlparse(provider["base_url"])
    local = base.hostname in {"127.0.0.1", "localhost", "::1"}
    if base.scheme != "https" and not local:
        raise ValueError("Vertex requires HTTPS (HTTP is allowed only for loopback tests)")


def prepare(provider, model, body, credentials):
    validate_provider(provider)
    payload = copy.deepcopy(body)
    payload.pop("model", None)
    streaming = payload.pop("stream", False)
    payload["anthropic_version"] = provider.get("anthropic_version", "vertex-2023-10-16")
    action = "streamRawPredict" if streaming else "rawPredict"
    base = provider["base_url"].rstrip("/")
    project, region = quote(provider["project"], safe=""), quote(provider["region"], safe="")
    url = f"{base}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{quote(model, safe='')}:{action}"
    token_env = provider.get("access_token_env")
    token = os.environ.get(token_env, "") if token_env else credentials.token()
    if not token:
        raise ValueError("Vertex access token is unavailable")
    return url, payload, {"content-type": "application/json", "authorization": f"Bearer {token}",
                          "accept-encoding": "identity"}
