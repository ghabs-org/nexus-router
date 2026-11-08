"""Loopback wire proxy, sharing the router's SQLite provider health authority."""

import asyncio
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import yaml

from . import cooldowns, db
from .cli_adapters import UnsupportedConversion, convert_request, convert_response, convert_stream
from .cli_classifier import CallCategory, CliPayloadClassifier
from .db import load_providers
from .health import load_provider_health
from .health_updater import observe_turn_outcome
from .paths import POLICIES_ROOT


PATHS = {"/v1/messages": "messages", "/v1/chat/completions": "chat", "/v1/responses": "responses"}
WIRE_PATH = {v: k for k, v in PATHS.items()}
BACKGROUND = {CallCategory.STATELESS_HOUSEKEEPING, CallCategory.SUBAGENT_WORKER}
CLIENT_CLOSED = 499  # nginx convention: the client went away before the answer ended
# A turn counts as failed when no upstream answered or it answered/broke with >= 400
# (client hang-ups excluded).
FAILED_SQL = "(http_status IS NULL OR (http_status >= 400 AND http_status != 499))"
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
    touched: float  # wall-clock seconds, so pins survive a restart


class UsageTap:
    """Reads token usage from forwarded bytes without holding any of them back.

    SSE: scans complete `data:` lines (Messages message_start/message_delta,
    Chat final chunk, Responses response.completed). JSON: keeps a bounded copy.
    input_tokens includes Anthropic cache read/creation tokens.
    """

    MAX_BODY = 4 << 20

    def __init__(self, streaming):
        self.streaming, self.tail, self.body, self.usage = streaming, b"", bytearray(), {}

    def feed(self, chunk):
        if self.streaming:
            lines = (self.tail + chunk).split(b"\n")
            self.tail = lines.pop()[-65536:]
            for line in lines:
                if line.startswith(b"data:") and b"usage" in line:
                    self._take(line[5:])
        elif self.body is not None:
            self.body = self.body + chunk if len(self.body) + len(chunk) <= self.MAX_BODY else None

    def finish(self):
        if self.streaming and self.tail.startswith(b"data:"):
            self._take(self.tail[5:])
        elif not self.streaming and self.body:
            self._take(bytes(self.body))
        return self.usage

    def _take(self, raw):
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        usage = data.get("usage") or (data.get("message") or {}).get("usage") or (data.get("response") or {}).get("usage")
        if not isinstance(usage, dict):
            return
        parts = [usage.get(k) for k in ("input_tokens", "prompt_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
        if any(type(v) is int for v in parts):
            self.usage["input_tokens"] = sum(v for v in parts if type(v) is int)
        output = usage.get("output_tokens", usage.get("completion_tokens"))
        if type(output) is int:
            self.usage["output_tokens"] = output


class CliProxy:
    def __init__(self, config=None):
        # Built from routing.yaml (not an explicit dict): reloaded when the file changes.
        self._config_file = POLICIES_ROOT / "routing.yaml" if config is None else None
        self._config_mtime = None
        if config is None:
            config, self._config_mtime = self._read_config()
        self._apply(config)
        self.classifier = CliPayloadClassifier()
        self.pins = OrderedDict()
        self.lock = threading.RLock()
        # Bounded striped locks serialize concurrent turns of a session, including
        # phase transitions. Unrelated sessions still forward concurrently.
        self.session_locks = [threading.Lock() for _ in range(256)]
        self._load_pins()

    def _read_config(self):
        mtime = self._config_file.stat().st_mtime
        with open(self._config_file) as file:
            return (yaml.safe_load(file) or {}).get("cli_proxy", {}), mtime

    def _apply(self, config):
        """Validate a cli_proxy section completely before any of it takes effect."""
        providers = config.get("providers", {})
        routes = {}
        for category in CallCategory:
            targets = []
            for entry in config.get("routes", {}).get(category.value, []):
                provider, separator, model = entry.partition(":")
                if not separator or not model or provider not in providers:
                    raise ValueError(f"Invalid CLI target: {entry}")
                targets.append(Target(provider, model))
            routes[category] = targets
        for name, provider in providers.items():
            url = urlparse(provider["base_url"])
            if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment or url.username:
                raise ValueError(f"Invalid upstream base_url for {name}")
            if not set(provider.get("protocols", [])) <= set(WIRE_PATH):
                raise ValueError(f"Invalid protocols for {name}")
        if config.get("ordering", "static") not in {"static", "learned"}:
            raise ValueError("cli_proxy.ordering must be static or learned")
        self.config, self.providers, self.routes = config, providers, routes
        self.max_pins = max(1, int(config.get("max_pins", 4096)))
        self.pin_ttl = max(1, float(config.get("pin_ttl_seconds", 86400)))

    def _reload_if_changed(self):
        """Hot reload routing.yaml's cli_proxy section; a bad edit keeps the running config."""
        if self._config_file is None:
            return
        try:
            if self._config_file.stat().st_mtime == self._config_mtime:
                return
            config, mtime = self._read_config()
            with self.lock:
                self._apply(config)
                self._config_mtime = mtime
                for session in [k for k, pin in self.pins.items()
                                if pin.target not in self.routes.get(pin.category, [])]:
                    del self.pins[session]  # its target is no longer routed for that category
            print("[cli-proxy] reloaded cli_proxy routes from routing.yaml", flush=True)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            print(f"[cli-proxy] reload failed, keeping previous config: {exc}", flush=True)

    # ── Persistence (SQLite; tables in schema.sql) ──────────────────────────

    def _db(self, sql, args=(), fetch=False):
        """Best effort: wire traffic must keep flowing if the state DB hiccups."""
        try:
            with closing(db._connect()) as conn, conn:
                rows = conn.execute(sql, args)
                return [dict(row) for row in rows] if fetch else None
        except sqlite3.Error as exc:
            print(f"[cli-proxy] state db: {exc}", flush=True)
            return [] if fetch else None

    def _load_pins(self):
        cutoff = time.time() - self.pin_ttl
        self._db("DELETE FROM cli_proxy_pins WHERE touched_at < ?", (cutoff,))
        rows = self._db("SELECT session, provider, model, category, touched_at FROM cli_proxy_pins "
                        "ORDER BY touched_at DESC LIMIT ?", (self.max_pins,), fetch=True)
        categories = {c.value: c for c in CallCategory}
        for row in reversed(rows):
            target, category = Target(row["provider"], row["model"]), categories.get(row["category"])
            # A pin to a target no longer configured for its category is dropped.
            if category and target in self.routes.get(category, []):
                self.pins[row["session"]] = Pin(target, category, row["touched_at"])

    def _record_turn(self, classification, target, pinned, errors, status=None, latency_ms=None, usage=None, converted=False):
        usage = usage or {}
        self._db("INSERT INTO cli_proxy_turns (created_at, session, category, reason, target, cache_pinned, failures_json, "
                 "http_status, latency_ms, input_tokens, output_tokens, converted) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (time.time(), classification.session, classification.category.value, classification.reason,
                  target.label if target else None, int(bool(pinned)), json.dumps(errors), status, latency_ms,
                  usage.get("input_tokens"), usage.get("output_tokens"), int(converted)))

    def _expire(self):
        now = time.time()
        for key in list(self.pins):
            if now - self.pins[key].touched > self.pin_ttl:
                del self.pins[key]

    def _pin(self, classification, target):
        if classification.category in BACKGROUND:
            return
        now = time.time()
        with self.lock:
            self._expire()
            self.pins[classification.session] = Pin(target, classification.category, now)
            self.pins.move_to_end(classification.session)
            evicted = []
            while len(self.pins) > self.max_pins:
                evicted.append(self.pins.popitem(last=False)[0])
        self._db("INSERT INTO cli_proxy_pins (session, provider, model, category, touched_at) VALUES (?,?,?,?,?) "
                 "ON CONFLICT(session) DO UPDATE SET provider=excluded.provider, model=excluded.model, "
                 "category=excluded.category, touched_at=excluded.touched_at",
                 (classification.session, target.provider, target.model, classification.category.value, now))
        for session in evicted:
            self._db("DELETE FROM cli_proxy_pins WHERE session=?", (session,))

    # ── Learned ordering ─────────────────────────────────────────────────────

    def _target_scores(self, targets):
        """health × smoothed 24h success − small latency penalty, per target."""
        since = time.time() - 86400
        rows = self._db("SELECT target, COUNT(*) AS n, SUM(http_status < 400) AS ok, "
                        "AVG(CASE WHEN http_status < 400 THEN latency_ms END) AS latency FROM cli_proxy_turns "
                        "WHERE created_at >= ? AND target IS NOT NULL AND http_status IS NOT NULL "
                        "AND http_status != 499 GROUP BY target",
                        (since,), fetch=True)
        stats = {row["target"]: row for row in rows}
        health = load_provider_health(sorted({t.provider for t in targets}))
        scores = {}
        for target in targets:
            row = stats.get(target.label) or {}
            success = ((row.get("ok") or 0) + 1) / ((row.get("n") or 0) + 2)
            latency = min(1.0, (row.get("latency") or 0) / 60000)
            h = health.get(target.provider)
            scores[target] = round((h.health_score if h else 0.85) * success - 0.05 * latency, 4)
        return scores

    def _ordered(self, category):
        """Configured order, unless learned mode and a later target beats an earlier one by the margin."""
        targets = list(self.routes[category])
        if self.config.get("ordering", "static") != "learned" or len(targets) < 2:
            return targets, {}
        scores = self._target_scores(targets)
        margin = float(self.config.get("learned_margin", 0.15))
        for i in range(1, len(targets)):
            j = i
            while j > 0 and scores[targets[j]] - scores[targets[j - 1]] >= margin:
                targets[j - 1], targets[j] = targets[j], targets[j - 1]
                j -= 1
        return targets, scores

    def _healthy(self, target, fallback=False, category=None):
        # Reload on every candidate attempt: provider controls/cooldown changes
        # must affect active pinned sessions as well as new sessions.
        controls = load_providers([target.provider]).get(target.provider)
        health = load_provider_health([target.provider]).get(target.provider)
        if not controls or not controls.get("enabled") or controls.get("status") in {"disabled", "maintenance"}:
            return False
        if not controls.get("allow_fallback" if fallback else "allow_primary", 1):
            return False
        if category is not None and target.label in cooldowns.cooling_models(category.value):
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

    def _usage_24h(self):
        return self._db(f"SELECT target, COUNT(*) AS turns, SUM({FAILED_SQL}) AS failed, "
                        "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
                        "ROUND(AVG(latency_ms)) AS avg_latency_ms FROM cli_proxy_turns WHERE created_at >= ? "
                        "GROUP BY target ORDER BY turns DESC", (time.time() - 86400,), fetch=True)

    def status(self, port):
        self._reload_if_changed()
        now = time.time()
        with self.lock:
            self._expire()
            pins = [{"session": key, "target": pin.target.label, "category": pin.category.value,
                     "idle_seconds": round(now - pin.touched, 1)} for key, pin in self.pins.items()]
        decisions = self._db("SELECT session, category, reason, target, cache_pinned, failures_json, http_status, "
                             "latency_ms, input_tokens, output_tokens, created_at FROM cli_proxy_turns "
                             "ORDER BY id DESC LIMIT 100", fetch=True)
        for item in decisions:
            item["cache_pinned"] = bool(item["cache_pinned"])
            item["failures"] = json.loads(item.pop("failures_json") or "[]")
            item["at"] = item.pop("created_at")
        order = {}
        for category in CallCategory:
            targets, scores = self._ordered(category)
            order[category.value] = [{"target": t.label, **({"score": scores[t]} if scores else {})} for t in targets]
        return {"active_session_pins": pins, "recent_decisions": decisions,
                "ordering": self.config.get("ordering", "static"), "applied_order": order,
                "usage_24h": self._usage_24h(),
                "model_cooldowns": [c for c in cooldowns.active() if c["scope"] in {x.value for x in CallCategory}],
                "provider_health": {key: asdict(value) for key, value in load_provider_health().items()},
                "provider_controls": {key: {field: value.get(field) for field in
                    ("enabled", "status", "allow_primary", "allow_fallback", "health_status", "cooldown_until")}
                    for key, value in load_providers().items()},
                "exports": self._exports(port)}

    @staticmethod
    def _exports(port):
        from . import server  # token lives on the server module (env or embedded launch)

        lines = [f"export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}",
                 f"export OPENAI_BASE_URL=http://127.0.0.1:{port}/v1"]
        if server.ROUTER_TOKEN:
            # The router token doubles as the CLI's API key; its value is never printed.
            lines += ["export ANTHROPIC_API_KEY=$NEXUS_ROUTER_TOKEN", "export OPENAI_API_KEY=$NEXUS_ROUTER_TOKEN"]
        return lines

    def summary(self):
        """Compact state for /diagnostics."""
        with self.lock:
            self._expire()
            pins = len(self.pins)
        usage = self._usage_24h()
        return {"routes": {c.value: [t.label for t in targets] for c, targets in self.routes.items()},
                "ordering": self.config.get("ordering", "static"), "active_pins": pins,
                "turns_24h": sum(row["turns"] for row in usage),
                "failed_24h": sum(row["failed"] or 0 for row in usage)}

    def models(self):
        """GET /v1/models: configured route models, in an OpenAI- and Anthropic-readable list."""
        seen = {}
        for targets in self.routes.values():
            for target in targets:
                seen.setdefault(target.model, target.provider)
        data = [{"id": model, "object": "model", "type": "model", "display_name": model,
                 "created": 0, "created_at": "1970-01-01T00:00:00Z", "owned_by": provider}
                for model, provider in seen.items()]
        return {"object": "list", "data": data, "has_more": False,
                "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}

    def count_tokens(self, handler, body):
        """Forward to the first healthy messages-native target of the request's category; else 404."""
        from .server import _json_response
        try:
            self._validate(body)
            classification = self.classifier.classify(body, dict(handler.headers))
        except (ValueError, TypeError, AttributeError) as exc:
            _json_response(handler, 400, {"error": {"type": "invalid_request_error", "message": str(exc)}})
            return
        timeout = httpx.Timeout(float(self.config.get("read_timeout_seconds", 300)),
                                connect=float(self.config.get("connect_timeout_seconds", 5)))
        with self.lock:
            pin = self.pins.get(classification.session)
        candidates, _ = self._ordered(classification.category)
        if pin and pin.category == classification.category:
            candidates = [pin.target] + [t for t in candidates if t != pin.target]
        for index, target in enumerate(candidates):
            if "messages" not in self.providers[target.provider].get("protocols", []):
                continue
            if not self._healthy(target, fallback=index > 0, category=classification.category):
                continue
            try:
                headers = self._headers(dict(handler.headers), target, "messages", "messages")
            except UnsupportedConversion:
                continue
            base = self.providers[target.provider]["base_url"].rstrip("/")
            endpoint = "/messages/count_tokens" if base.endswith("/v1") else "/v1/messages/count_tokens"
            try:
                with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
                    response = client.post(base + endpoint, json={**body, "model": target.model}, headers=headers)
            except httpx.HTTPError:
                continue
            if response.status_code >= 500 or response.status_code in {401, 403, 404, 429}:
                continue
            content = response.content
            handler.send_response(response.status_code)
            handler.send_header("Content-Type", response.headers.get("content-type", "application/json"))
            handler.send_header("Content-Length", str(len(content)))
            handler.send_header("X-Nexus-Routed-Model", target.label)
            handler.end_headers()
            handler.wfile.write(content)
            return
        # Claude Code falls back to its own estimate on 404.
        _json_response(handler, 404, {"error": {"type": "not_found_error",
                                                "message": "No healthy messages-native upstream for count_tokens"}})

    def handle(self, handler, body, path):
        from .server import _json_response
        self._reload_if_changed()
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

    def _observe(self, target, status, start, error=None, category=None):
        latency = int((time.monotonic() - start) * 1000)
        observe_turn_outcome(target.provider, status, latency, error)
        if category is not None:
            # Model-level breaker: 5xx/timeouts/404 are this model's problem;
            # 401/403/429 stay with the provider-wide circuit above.
            if error in {"server", "timeout"}:
                cooldowns.record_failure(target.label, category.value)
            elif error is None and status is not None and status < 400:
                cooldowns.record_success(target.label, category.value)
        return latency

    async def _forward(self, handler, body, path, classification):
        from .server import _json_response
        source = PATHS[path]
        with self.lock:
            self._expire()
            pin = self.pins.get(classification.session)
        main = classification.category not in BACKGROUND
        category = classification.category
        # Learned order only reorders new assignments; an existing pin still goes first.
        candidates, _ = self._ordered(category)
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
                if not self._healthy(target, fallback=index > 0, category=category):
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
                            self._observe(target, status, start, error, category)
                            errors.append(f"{target.label}: HTTP {status}")
                            continue
                        converted = wire != source
                        streaming = bool(body.get("stream") and status < 400 and "text/event-stream" in response.headers.get("content-type", ""))
                        if body.get("stream") and status < 400 and not streaming:
                            self._observe(target, 502, start, "server", category)
                            errors.append(f"{target.label}: expected SSE response")
                            continue
                        content = None
                        if converted and not streaming:
                            content = json.dumps(convert_response(json.loads(await response.aread()), source, target.model)).encode()
                        if status < 400:
                            self._pin(classification, target)
                        cache_pinned = bool(pinned and target == pin.target)
                        tap = UsageTap(streaming)
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
                        handler.send_header("X-Nexus-Cache-Pinned", str(cache_pinned).lower())
                        handler.send_header("Connection", "close")
                        if streaming:
                            handler.send_header("X-Accel-Buffering", "no")
                        if content is not None:
                            handler.send_header("Content-Length", str(len(content)))
                        handler.end_headers()
                        handler.close_connection = True
                        committed = True
                        # 502 if the upstream breaks mid-answer; 499 if the CLI hangs up (Esc),
                        # which says nothing about the target and is left out of learned stats.
                        outcome_status, recorded = 502, False
                        try:
                            if content is not None:
                                # Content-Length reply: the CLI may read it before control returns
                                # here, so the turn is recorded before the write.
                                tap.feed(content)
                                outcome_status = status
                                self._record_turn(classification, target, cache_pinned, errors.copy(), status,
                                                  int((time.monotonic() - start) * 1000), tap.finish(), converted)
                                recorded = True
                                handler.wfile.write(content)
                            else:
                                chunks = convert_stream(response, source, target.model) if converted and streaming else response.aiter_bytes()
                                async for chunk in chunks:
                                    # No chunk_size: never wait to fill a buffer before
                                    # forwarding the first upstream SSE bytes.
                                    handler.wfile.write(chunk)
                                    handler.wfile.flush()
                                    tap.feed(chunk)
                            outcome_status = status
                        except (BrokenPipeError, ConnectionResetError):
                            outcome_status = CLIENT_CLOSED
                            raise
                        finally:
                            # Recorded once the answer ends (or breaks), with what was seen.
                            if not recorded:
                                self._record_turn(classification, target, cache_pinned, errors.copy(), outcome_status,
                                                  int((time.monotonic() - start) * 1000), tap.finish(), converted)
                        if status < 400:
                            self._observe(target, status, start, category=category)
                        elif status == 404:  # unknown model: model breaker only, not provider health
                            cooldowns.record_failure(target.label, category.value)
                        return
                except (BrokenPipeError, ConnectionResetError):
                    return  # Client disconnect is not a provider health failure.
                except (httpx.HTTPError, UnsupportedConversion, ValueError, KeyError) as exc:
                    self._observe(target, None, start, "timeout" if isinstance(exc, httpx.TimeoutException) else "server", category)
                    errors.append(f"{target.label}: {type(exc).__name__}")
                    if committed:
                        # Retrying after a partial answer would duplicate text/tool calls.
                        handler.close_connection = True
                        return
        self._record_turn(classification, None, False, errors)
        _json_response(handler, 503, {"error": {"type": "overloaded_error", "message": "No healthy compatible CLI upstream", "attempts": errors}})
