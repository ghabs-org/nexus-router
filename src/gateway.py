"""Standalone Cowork gateway using Nexus's existing streaming/health machinery."""

import asyncio
import copy
from dataclasses import replace
import json
import logging
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import yaml

from . import db
from .cli_adapters import UnsupportedConversion
from .cli_proxy import BACKGROUND, CliProxy, UsageTap
from .gateway_auth import GatewayAuth
from .gateway_policy import GatewayPolicy, PolicyError
from .gateway_state import GatewayState
from .http_helpers import json_response
from .vertex import VertexCredentials, prepare as prepare_vertex

LOG = logging.getLogger("nexus.gateway")


def load_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Gateway configuration must be an object")
    for provider in config.get("providers", {}).values():
        if provider.get("transport") == "vertex":
            for key in ("project", "region"):
                if provider.get(f"{key}_env"):
                    provider[key] = os.environ.get(provider[f"{key}_env"], "")
            region = provider.get("region", "")
            provider.setdefault("base_url", f"https://{'' if region == 'global' else region + '-'}aiplatform.googleapis.com")
        provider["protocols"] = provider.get("protocols", ["responses", "chat"] if provider.get("transport") == "openai" else ["messages"])
    return config


class GatewayProxy(CliProxy):
    def __init__(self, config):
        self.policy = GatewayPolicy(config)
        self.policies = {"messages": self.policy}
        for wire, profiles in config.get("protocol_profiles", {}).items():
            self.policies[wire] = GatewayPolicy({**config, "profiles": profiles})
        self.state = GatewayState(config)
        self.credentials = VertexCredentials()
        targets = list(self.policy.targets)
        proxy_config = {**config, "routes": {category: targets for category in (
            "MAIN_EXECUTION", "MAIN_PLANNING", "STATELESS_HOUSEKEEPING", "SUBAGENT_WORKER")}}
        super().__init__(proxy_config)

    def policy_request(self, body, headers, wire):
        policy = self.policies.get(wire)
        if policy is None:
            raise PolicyError("No policy configured for this protocol")
        normalized = copy.deepcopy(body)
        if wire != "messages":
            items = body.get("input", body.get("messages", []))
            if isinstance(items, str):
                items = [{"role": "user", "content": items}]
            normalized["messages"] = [m for m in items if isinstance(m, dict) and m.get("role") in {"user", "assistant", "system", "developer"}]
            normalized["system"] = body.get("instructions", "")
            normalized["max_tokens"] = body.get("max_output_tokens", body.get("max_completion_tokens", body.get("max_tokens", self.config.get("max_output_tokens", 16384))))
            normalized["output_config"] = {"effort": (body.get("reasoning") or {}).get("effort", body.get("reasoning_effort"))}
            if normalized["output_config"]["effort"] is None:
                normalized.pop("output_config")
        # Scope pins by wire format, including when clients reuse their session IDs.
        headers = dict(headers)
        identity = policy.identity(normalized, headers, "wire:" + wire)
        headers["x-session-id"] = identity
        return policy, normalized, headers

    def classification(self, body, headers, principal, wire="messages"):
        self._validate(body)
        request_body = body
        raw_headers = {k.lower(): v for k,v in headers.items()}
        policy, body, headers = self.policy_request(body, headers, wire)
        session = policy.identity(body, headers, principal)
        pin = self.state.pin(session)
        result = policy.classify(body, headers, principal, pin)
        ticket = raw_headers.get("x-nexus-escalation-ticket")
        if ticket:
            from .gateway_escalation import Escalations
            escalation = Escalations().read(ticket, principal, wire, request_body, raw_headers)
            if pin or result.requested_model != policy.alias:
                raise PolicyError("Escalation requires a fresh automatic task",409)
            choices = policy.profiles.get(escalation["profile"])
            if not choices:
                raise PolicyError("Escalation profile is not configured")
            result = replace(result, profile=escalation["profile"], choices=tuple(copy.deepcopy(choices)),
                             reason="explicit_escalation", explanation={"source":"explicit_escalation","ticket":ticket,"level":escalation["level"],"budget_root":escalation["root"]})
        else:
            from .gateway_economics import order
            result = order(result, body, policy, wire, pin is not None)
        result = replace(result, explanation={**result.explanation, "wire":wire})
        for choice in result.choices:
            target = policy.targets[choice["target"]]
            if wire not in self.providers[target.provider].get("protocols", []):
                raise PolicyError("Profile contains a model without native protocol support")
            policy.apply(body, result, target)
        return result

    def handle(self, handler, body, path):
        admitted = False
        reservation = None
        try:
            self._validate(body)
            wire = {"/v1/messages": "messages", "/v1/responses": "responses", "/v1/chat/completions": "chat"}[path]
            policy, normalized, scoped_headers = self.policy_request(body, dict(handler.headers), wire)
            session = policy.identity(normalized, scoped_headers, handler.principal)
            guard = self.session_locks[int(session[:8], 16) % len(self.session_locks)]
            # Resolve both model and effort inside the same lock as forwarding.
            # Concurrent first requests cannot commit different task profiles.
            with guard:
                classification = self.classification(body, dict(handler.headers), handler.principal, wire)
                from .gateway_budget import BudgetLedger
                if BudgetLedger().reserve(classification, handler.limits, self.config, normalized):
                    reservation = classification.request_id
                self.state.admit(handler.principal, classification.session, handler.limits)
                admitted = True
                ticket = classification.explanation.get("ticket")
                if ticket and not classification.explanation.get("pin_reused"):
                    from .gateway_escalation import Escalations
                    Escalations().claim(ticket,handler.principal,classification.session)
                asyncio.run(self._forward(handler, body, path, classification))
        except (PolicyError, ValueError, TypeError, AttributeError) as exc:
            if not handler.committed:
                error_response(handler, getattr(exc, "status", 400), str(exc))
        except Exception as exc:
            LOG.error("Gateway request failed: %s", type(exc).__name__)
            if not handler.committed:
                error_response(handler, 503, "Gateway state or transport is unavailable")
        finally:
            if reservation:
                from .gateway_budget import BudgetLedger
                BudgetLedger().finish(reservation)
            if admitted:
                self.state.release(handler.principal)

    def _ordered_for(self, classification):
        choices = [self.policy.targets[c["target"]] for c in classification.choices]
        pin = self.state.pin(classification.session)
        if pin and classification.category not in BACKGROUND:
            choices.sort(key=lambda target: target.label != pin["target"])
        return choices, {}

    def _wire(self, target, source):
        if source not in self.providers[target.provider].get("protocols", []):
            raise UnsupportedConversion("Gateway routes require a native upstream protocol")
        return source

    def _retry_status(self, status):
        return status == 404 or super()._retry_status(status)

    def _before_attempt(self, classification, target):
        from .gateway_budget import BudgetLedger
        BudgetLedger().attempt(classification.request_id)

    def _usage_tap(self, streaming):
        return UsageTap(streaming, detailed=True)

    def _pin(self, classification, target):
        if classification.category not in BACKGROUND:
            # Persist reasoning before accepting the upstream response. SQLite
            # failure must not silently turn a managed task into unpinned traffic.
            self.state.save_pin(classification, target)
        super()._pin(classification, target)

    def _prepare_upstream(self, handler, body, target, source, wire, classification):
        if source == "messages":
            payload = self.policy.apply(body, classification, target)
        else:
            policy, normalized, _ = self.policy_request(body, dict(handler.headers), source)
            applied = policy.apply(normalized, classification, target)
            payload = copy.deepcopy(body)
            payload["model"] = target.model
            if source == "responses":
                payload["max_output_tokens"] = applied["max_tokens"]
                if applied.get("output_config", {}).get("effort"):
                    payload["reasoning"] = {**payload.get("reasoning", {}), "effort": applied["output_config"]["effort"]}
            else:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = applied["max_tokens"]
                if payload.get("stream"):
                    payload["stream_options"] = {**payload.get("stream_options", {}), "include_usage": True}
                if applied.get("output_config", {}).get("effort"):
                    payload["reasoning_effort"] = applied["output_config"]["effort"]
        provider = self.providers[target.provider]
        if provider.get("transport") == "vertex":
            try:
                url, payload, headers = prepare_vertex(provider, target.model, payload, self.credentials)
            except Exception:
                raise UnsupportedConversion("Vertex credentials are unavailable") from None
            for key in ("anthropic-beta", "anthropic-version"):
                if handler.headers.get(key):
                    headers[key] = handler.headers[key]
            # Legacy deployments can explicitly request body-encoded betas.
            if provider.get("beta_transport") == "body" and headers.get("anthropic-beta"):
                payload["anthropic_beta"] = [b.strip() for b in headers.pop("anthropic-beta").split(",") if b.strip()]
            return url, payload, headers
        url, payload, headers = super()._prepare_upstream(handler, payload, target, source, wire, classification)
        # Never forward client identity, session headers, cookies or routing overrides upstream.
        headers = {k: v for k, v in headers.items() if k in {
            "content-type", "accept-encoding", "authorization", "x-api-key", "anthropic-version", "anthropic-beta"}}
        return url, payload, headers

    def _extra_headers(self, classification, target):
        choice = next(c for c in classification.choices if c["target"] == target.label)
        return {"X-Nexus-Request-ID": classification.request_id, "X-Nexus-Profile": classification.profile, "X-Nexus-Effort": choice.get("effort", "model-default"),
                "X-Nexus-Routing-Source": classification.explanation.get("source", "unknown")}

    def _record_turn(self, classification, target, pinned, errors, status=None, latency_ms=None, usage=None, converted=False):
        self.state.record(classification, target, status, latency_ms, usage)
        if status is not None and status<300:
            from .gateway_economics import observe
            observe(classification,target,usage or {})
        super()._record_turn(classification, target, pinned, errors, status, latency_ms, usage, converted)

    def count_tokens(self, handler, body):
        # Vertex countTokens is a separate API. Advertising an approximate count
        # as exact can break compaction. Cowork falls back to its local estimate.
        error_response(handler, 404, "Token counting is not advertised by this gateway")


def error_response(handler, status, message):
    kind = {400: "invalid_request_error", 401: "authentication_error", 402: "billing_error",
            403: "permission_error", 404: "not_found_error", 409: "invalid_request_error",
            413: "request_too_large", 429: "rate_limit_error", 503: "overloaded_error"}.get(status, "api_error")
    json_response(handler, status, {"type": "error", "error": {"type": kind, "message": message}})


class GatewayHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.gateway_config.get("client_timeout_seconds", 300))
        self.committed = False

    def send_response(self, code, message=None):
        self.committed = True
        super().send_response(code, message)

    def log_message(self, *args):
        pass

    def authenticate(self):
        try:
            self.principal, self.limits = self.server.gateway_auth.authenticate(self.headers)
            return True
        except PolicyError as exc:
            error_response(self, exc.status, str(exc))
            return False

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            json_response(self, 200, {"status": "ok", "service": "nexus-router-gateway"})
            return
        if path in {"/dashboard", "/dashboard.js", "/dashboard.css"}:
            from importlib.resources import files
            names={"/dashboard":("gateway_dashboard.html","text/html"),"/dashboard.js":("gateway_dashboard.js","text/javascript"),"/dashboard.css":("gateway_dashboard.css","text/css")}
            name,mime=names[path]
            content=files(__package__).joinpath(name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",mime+"; charset=utf-8")
            self.send_header("Content-Length",str(len(content)))
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Security-Policy","default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers(); self.wfile.write(content)
            return
        if not self.authenticate():
            return
        if path == "/v1/models":
            json_response(self, 200, self.server.gateway.policy.model_list())
        elif path == "/v1/nexus/status":
            json_response(self, 200, self.server.gateway.state.status(self.principal))
        else:
            error_response(self, 404, "Unknown gateway endpoint")

    def do_POST(self):
        if not self.authenticate():
            self.close_connection = True
            return
        path = urlparse(self.path).path
        try:
            if self.headers.get("Transfer-Encoding"):
                raise PolicyError("Transfer-Encoding is not supported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise PolicyError("Exactly one Content-Length is required")
            length = int(lengths[0])
            if not 0 < length <= self.server.gateway_config.get("max_body_bytes", 32 * 1024 * 1024):
                raise PolicyError("Request body exceeds the configured limit", 413)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise PolicyError("Incomplete request body")
            body = json.loads(raw)
            if path in {"/v1/messages", "/v1/responses", "/v1/chat/completions"}:
                self.server.gateway.handle(self, body, path)
            elif path == "/v1/nexus/feedback":
                if not isinstance(body,dict) or set(body)!={"source_session","rating"} or not isinstance(body["source_session"],str):
                    raise PolicyError("Expected source_session and rating")
                from .gateway_escalation import Escalations
                Escalations().feedback(self.principal,body["source_session"],body["rating"])
                json_response(self,200,{"recorded":True})
            elif path == "/v1/nexus/escalations":
                if not isinstance(body,dict) or set(body)-{"source_session","wire"}:
                    raise PolicyError("Expected source_session and wire")
                wire=body.get("wire","messages")
                if wire not in self.server.gateway.policies or not isinstance(body.get("source_session"),str):
                    raise PolicyError("Invalid escalation protocol or source")
                from .gateway_escalation import Escalations
                row=Escalations().issue(self.principal,body["source_session"],wire)
                json_response(self,200,{key:row[key] for key in ("ticket","new_session_id","profile","wire","level")})
            elif path == "/v1/messages/count_tokens":
                self.server.gateway.count_tokens(self, body)
            elif path == "/v1/nexus/route-preview":
                wire = parse_qs(urlparse(self.path).query).get("wire", ["messages"])[0]
                classification = self.server.gateway.classification(body, dict(self.headers), self.principal, wire)
                json_response(self, 200, {"profile": classification.profile, "reason": classification.reason,
                                          "choices": classification.choices, "explanation": classification.explanation, "writes_pin": False})
            else:
                error_response(self, 404, "Unknown gateway endpoint")
        except (ValueError, TypeError, PolicyError) as exc:
            error_response(self, getattr(exc, "status", 400), str(exc))
            self.close_connection = True
        except (socket.timeout, TimeoutError):
            self.close_connection = True


def create_server(config, host="127.0.0.1", port=7772):
    db.ensure_schema()
    auth, gateway = GatewayAuth(config), GatewayProxy(config)
    server = ThreadingHTTPServer((host, port), GatewayHandler)
    server.daemon_threads = True
    server.gateway_config, server.gateway_auth, server.gateway = config, auth, gateway
    return server
