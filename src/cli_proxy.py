"""Loopback wire proxy, sharing the router's SQLite provider health authority."""

import asyncio
import json
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import yaml

from .cli_adapters import UnsupportedConversion, convert_request, convert_response, convert_stream
from .cli_classifier import CallCategory, CliPayloadClassifier
from .db import load_providers
from .health import load_provider_health
from .health_updater import observe_turn_outcome
from .paths import POLICIES_ROOT


PATHS = {"/v1/messages": "messages", "/v1/chat/completions": "chat", "/v1/responses": "responses"}
WIRE_PATH = {v: k for k, v in PATHS.items()}
BACKGROUND = {CallCategory.STATELESS_HOUSEKEEPING, CallCategory.SUBAGENT_WORKER}
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}


@dataclass(frozen=True)
class Target:
    provider: str
    model: str

    @property
    def label(self):
        return f"{self.provider}:{self.model}"


@dataclass
class Pin:
    target: Target
    category: CallCategory
    touched: float


class CliProxy:
    def __init__(self, config=None):
        if config is None:
            with open(POLICIES_ROOT / "routing.yaml") as file:
                config = (yaml.safe_load(file) or {}).get("cli_proxy", {})
        self.config = config
        self.providers = config.get("providers", {})
        self.routes = {}
        for category in CallCategory:
            targets = []
            for entry in config.get("routes", {}).get(category.value, []):
                provider, separator, model = entry.partition(":")
                if not separator or not model or provider not in self.providers:
                    raise ValueError(f"Invalid CLI target: {entry}")
                targets.append(Target(provider, model))
            self.routes[category] = targets
        for name, provider in self.providers.items():
            url = urlparse(provider["base_url"])
            if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment or url.username:
                raise ValueError(f"Invalid upstream base_url for {name}")
            if not set(provider.get("protocols", [])) <= set(WIRE_PATH):
                raise ValueError(f"Invalid protocols for {name}")
        self.classifier = CliPayloadClassifier()
        self.pins = OrderedDict()
        self.decisions = deque(maxlen=100)
        self.lock = threading.RLock()
        # Bounded striped locks serialize concurrent turns of a session, including
        # phase transitions. Unrelated sessions still forward concurrently.
        self.session_locks = [threading.Lock() for _ in range(256)]
        self.max_pins = max(1, int(config.get("max_pins", 4096)))
        self.pin_ttl = max(1, float(config.get("pin_ttl_seconds", 86400)))

    def _expire(self):
        now = time.monotonic()
        for key in list(self.pins):
            if now - self.pins[key].touched > self.pin_ttl:
                del self.pins[key]

    def _pin(self, classification, target):
        if classification.category in BACKGROUND:
            return
        with self.lock:
            self._expire()
            self.pins[classification.session] = Pin(target, classification.category, time.monotonic())
            self.pins.move_to_end(classification.session)
            while len(self.pins) > self.max_pins:
                self.pins.popitem(last=False)

    def _healthy(self, target, fallback=False):
        # Reload on every candidate attempt: provider controls/cooldown changes
        # must affect active pinned sessions as well as new sessions.
        controls = load_providers([target.provider]).get(target.provider)
        health = load_provider_health([target.provider]).get(target.provider)
        if not controls or not controls.get("enabled") or controls.get("status") in {"disabled", "maintenance"}:
            return False
        if not controls.get("allow_fallback" if fallback else "allow_primary", 1):
            return False
        # Allow a half-open attempt once the persisted circuit expires. Error
        # EWMA only decays on successful observations, so a hard score gate alone
        # could otherwise keep a recovered provider excluded forever.
        cooldown = health.rate_limit_cooldown_until if health else None
        if cooldown and health.health_score > 0 and health.auth not in {"expired", "missing"} and health.quota != "exhausted":
            until = datetime.fromisoformat(cooldown)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if until <= datetime.now(timezone.utc):
                return True
        if controls.get("health_status") in {"degraded", "unhealthy"}:
            return False
        return bool(health and health.health_score >= float(self.config.get("min_health_score", 0.8)))

    def _wire(self, target, source):
        protocols = self.providers[target.provider].get("protocols", [])
        if source in protocols:
            return source
        if source == "messages" and "chat" in protocols:
            return "chat"
        if source == "chat" and "messages" in protocols:
            return "messages"
        raise UnsupportedConversion("No compatible upstream protocol")

    def _headers(self, incoming, target, source, wire):
        incoming = {k.lower(): v for k, v in incoming.items()}
        blocked = HOP_HEADERS | {"host", "content-length", "authorization", "x-api-key", "cookie", "accept-encoding"}
        blocked.update(x.strip().lower() for x in incoming.get("connection", "").split(","))
        headers = {k: v for k, v in incoming.items() if k not in blocked}
        headers.update({"content-type": "application/json", "accept-encoding": "identity"})
        if source != wire:
            headers = {k: v for k, v in headers.items() if not k.startswith("anthropic-")}
        provider = self.providers[target.provider]
        env = provider.get("api_key_env")
        token = os.getenv(env) if env else None
        if env and not token:
            raise UnsupportedConversion(f"Missing credential environment variable: {env}")
        if token:
            if wire == "messages":
                headers["x-api-key"] = token
            else:
                headers["authorization"] = f"Bearer {token}"
        if wire == "messages":
            headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    def status(self, port):
        with self.lock:
            self._expire()
            pins = [{"session": key, "target": pin.target.label, "category": pin.category.value,
                     "idle_seconds": round(time.monotonic() - pin.touched, 1)} for key, pin in self.pins.items()]
            decisions = list(self.decisions)
        return {"active_session_pins": pins, "recent_decisions": decisions,
                "provider_health": {key: asdict(value) for key, value in load_provider_health().items()},
                "provider_controls": {key: {field: value.get(field) for field in
                    ("enabled", "status", "allow_primary", "allow_fallback", "health_status", "cooldown_until")}
                    for key, value in load_providers().items()},
                "exports": [f"export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}",
                            f"export OPENAI_BASE_URL=http://127.0.0.1:{port}/v1"]}

    def handle(self, handler, body, path):
        from .server import _json_response
        try:
            self._validate(body)
            classification = self.classifier.classify(body, dict(handler.headers))
        except (ValueError, TypeError, AttributeError) as exc:
            _json_response(handler, 400, {"error": {"type": "invalid_request_error", "message": str(exc)}})
            return
        guard = self.session_locks[int(classification.session[:8], 16) % len(self.session_locks)]
        # Background calls never acquire the main session lock or mutate its pin.
        if classification.category in BACKGROUND:
            asyncio.run(self._forward(handler, body, path, classification))
        else:
            with guard:
                asyncio.run(self._forward(handler, body, path, classification))

    @staticmethod
    def _validate(body):
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        for key in ("messages", "tools"):
            if key in body and (not isinstance(body[key], list) or any(not isinstance(x, dict) for x in body[key])):
                raise ValueError(f"{key} must be an array of objects")
        if "input" in body and not isinstance(body["input"], (str, list)):
            raise ValueError("input must be text or an array")
        if "stream" in body and not isinstance(body["stream"], bool):
            raise ValueError("stream must be a boolean")

    def _observe(self, target, status, start, error=None):
        observe_turn_outcome(target.provider, status, int((time.monotonic() - start) * 1000), error)

    async def _forward(self, handler, body, path, classification):
        from .server import _json_response
        source = PATHS[path]
        with self.lock:
            self._expire()
            pin = self.pins.get(classification.session)
        main = classification.category not in BACKGROUND
        candidates = list(self.routes[classification.category])
        if main:
            # Preserve native schema/cache breakpoints ahead of conversions.
            candidates.sort(key=lambda t: source not in self.providers[t.provider].get("protocols", []))
        pinned = bool(main and pin and pin.category == classification.category)
        if pinned:
            candidates = [pin.target] + [t for t in candidates if t != pin.target]
        # Responses server-side continuations cannot move across providers/models.
        continuation = bool(body.get("previous_response_id"))
        if continuation:
            if not pinned:
                _json_response(handler, 409, {"error": {"message": "Responses continuation needs an active x-session-id pin"}})
                return
            candidates = [pin.target]
        errors = []
        timeout = httpx.Timeout(float(self.config.get("read_timeout_seconds", 300)),
                                connect=float(self.config.get("connect_timeout_seconds", 5)))
        # A request-scoped client keeps async resources on the handler's loop.
        # All fallback attempts reuse its connections; no upstream retries occur
        # after response headers have been committed to the CLI.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            for index, target in enumerate(candidates):
                if not self._healthy(target, fallback=index > 0):
                    errors.append(f"{target.label}: unhealthy_or_disabled")
                    continue
                try:
                    wire = self._wire(target, source)
                    payload = convert_request(body, source, wire)
                    payload["model"] = target.model
                    headers = self._headers(dict(handler.headers), target, source, wire)
                except (UnsupportedConversion, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{target.label}: {exc}")
                    continue
                base = self.providers[target.provider]["base_url"].rstrip("/")
                parsed_base = urlparse(base)
                if parsed_base.hostname in {"127.0.0.1", "localhost", "::1"} and parsed_base.port == handler.server.server_port:
                    errors.append(f"{target.label}: upstream URL points back to this proxy")
                    continue
                endpoint = WIRE_PATH[wire]
                if base.endswith("/v1"):
                    endpoint = endpoint[3:]
                start = time.monotonic()
                committed = False
                try:
                    async with client.stream("POST", base + endpoint, json=payload, headers=headers) as response:
                        status = response.status_code
                        if status in {401, 403, 429} or status >= 500:
                            error = "auth" if status in {401, 403} else "rate_limit" if status == 429 else "server"
                            self._observe(target, status, start, error)
                            errors.append(f"{target.label}: HTTP {status}")
                            continue
                        converted = wire != source
                        streaming = bool(body.get("stream") and status < 400 and "text/event-stream" in response.headers.get("content-type", ""))
                        if body.get("stream") and status < 400 and not streaming:
                            self._observe(target, 502, start, "server")
                            errors.append(f"{target.label}: expected SSE response")
                            continue
                        content = None
                        if converted and not streaming:
                            content = json.dumps(convert_response(json.loads(await response.aread()), source, target.model)).encode()
                        if status < 400:
                            self._pin(classification, target)
                        with self.lock:
                            self.decisions.append({"session": classification.session, "category": classification.category.value,
                                "reason": classification.reason, "target": target.label, "cache_pinned": pinned and target == pin.target,
                                "failures": errors.copy(), "at": time.time()})
                        handler.send_response(status)
                        blocked = HOP_HEADERS | {"content-length", "content-encoding", "x-nexus-routed-model", "x-nexus-call-category", "x-nexus-cache-pinned"}
                        blocked.update(x.strip().lower() for x in response.headers.get("connection", "").split(","))
                        for key, value in response.headers.multi_items():
                            if key.lower() not in blocked and not (converted and key.lower() == "content-type"):
                                handler.send_header(key, value)
                        if converted:
                            handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
                        handler.send_header("X-Nexus-Routed-Model", target.label)
                        handler.send_header("X-Nexus-Call-Category", classification.category.value)
                        handler.send_header("X-Nexus-Cache-Pinned", str(bool(pinned and target == pin.target)).lower())
                        handler.send_header("Connection", "close")
                        if streaming:
                            handler.send_header("X-Accel-Buffering", "no")
                        if content is not None:
                            handler.send_header("Content-Length", str(len(content)))
                        handler.end_headers()
                        handler.close_connection = True
                        committed = True
                        if content is not None:
                            handler.wfile.write(content)
                        else:
                            chunks = convert_stream(response, source, target.model) if converted and streaming else response.aiter_bytes()
                            async for chunk in chunks:
                                # No chunk_size: never wait to fill a buffer before
                                # forwarding the first upstream SSE bytes.
                                handler.wfile.write(chunk)
                                handler.wfile.flush()
                        if status < 400:
                            self._observe(target, status, start)
                        return
                except (BrokenPipeError, ConnectionResetError):
                    return  # Client disconnect is not a provider health failure.
                except (httpx.HTTPError, UnsupportedConversion, ValueError, KeyError) as exc:
                    self._observe(target, None, start, "timeout" if isinstance(exc, httpx.TimeoutException) else "server")
                    errors.append(f"{target.label}: {type(exc).__name__}")
                    if committed:
                        # Retrying after a partial answer would duplicate text/tool calls.
                        handler.close_connection = True
                        return
        with self.lock:
            self.decisions.append({"session": classification.session, "category": classification.category.value,
                "reason": classification.reason, "target": None, "failures": errors, "at": time.time()})
        _json_response(handler, 503, {"error": {"type": "overloaded_error", "message": "No healthy compatible CLI upstream", "attempts": errors}})
