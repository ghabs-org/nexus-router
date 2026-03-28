"""
server.py — Lightweight HTTP API server for Nexus Router.

Exposes the router over HTTP so the OpenClaw plugin (Node.js) can call it.
Runs on localhost only. Not intended for external exposure.

Endpoints:
  POST /route         → route a turn, returns RoutingDecision
  POST /outcome       → record turn outcome
  GET  /health        → server + provider health check
  GET  /stats         → model stats summary
  POST /probe         → probe all provider auth status

Usage:
  python -m src.server [--port 7771]
"""

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from .router import Router
from .classifier import extract_pre_signals, heuristic_classify, classify_with_model
from .types import ClassifierOutput, PreSignals
from .health import load_provider_health
from .health_updater import probe_all_providers, observe_turn_outcome
from .db import ensure_schema, update_outcome, load_model_stats


def should_reclassify_with_llm(classifier: ClassifierOutput | None, use_llm: bool, conversation_context: str | None) -> bool:
    return bool(
        use_llm
        and classifier is not None
        and classifier.task_type == "fast_utility"
        and conversation_context
        and conversation_context.strip()
    )

DEFAULT_PORT = 7771

# Lazy-init router
_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(persist=True)
    return _router


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any):
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length))


class RouterHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress default Apache-style logs; use print for important events only
        pass

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            provider_health = load_provider_health()
            _json_response(self, 200, {
                "status": "ok",
                "providers": {
                    p: {
                        "auth": ph.auth,
                        "quota": ph.quota,
                        "health_score": ph.health_score,
                    }
                    for p, ph in provider_health.items()
                },
            })

        elif path == "/stats":
            stats = load_model_stats()
            _json_response(self, 200, {
                "total_models": len(stats),
                "models": [
                    {
                        "model": m,
                        "total_selected": s.get("total_selected", 0),
                        "success_rate": s.get("success_rate"),
                        "avg_latency_ms": s.get("avg_latency_ms"),
                        "total_override": s.get("total_override", 0),
                    }
                    for m, s in sorted(stats.items(), key=lambda x: -(x[1].get("total_selected") or 0))
                ],
            })

        else:
            _json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            body = _read_body(self)
        except Exception as e:
            _json_response(self, 400, {"error": f"invalid_body: {e}"})
            return

        if path == "/route":
            self._handle_route(body)
        elif path == "/outcome":
            self._handle_outcome(body)
        elif path == "/probe":
            self._handle_probe(body)
        else:
            _json_response(self, 404, {"error": "not_found"})

    def _handle_route(self, body: dict):
        try:
            message = body.get("message", "")
            has_image = bool(body.get("has_image", False))
            cost_profile = body.get("cost_profile", "balanced")
            use_llm = bool(body.get("use_llm_classifier", False))
            classifier_model = body.get("classifier_model", "openai-codex/gpt-5.4-mini")
            raw_classifier_context = body.get("conversation_context")
            if raw_classifier_context is None:
                classifier_context = None
            elif isinstance(raw_classifier_context, str):
                classifier_context = raw_classifier_context
            else:
                _json_response(self, 400, {"error": "invalid_conversation_context_type"})
                return
            nexus_context = body.get("nexus_context")
            route_mode = body.get("route_mode")

            # Extract pre-signals
            pre_signals = extract_pre_signals(message, has_image_attachment=has_image)

            # Classify — explicit classifier hint (e.g. from Nexus) always wins over heuristic
            raw_cls = body.get("classifier")
            classifier = None
            classifier_source = "fallback"

            if raw_cls:
                classifier = ClassifierOutput(
                    task_type=raw_cls.get("task_type", "general_chat"),
                    subtype=raw_cls.get("subtype"),
                    complexity=raw_cls.get("complexity", "medium"),
                    needs_tools=bool(raw_cls.get("needs_tools", True)),
                    needs_vision=bool(raw_cls.get("needs_vision", False)),
                    needs_long_context=bool(raw_cls.get("needs_long_context", False)),
                    cost_profile=raw_cls.get("cost_profile", "balanced"),
                    confidence=float(raw_cls.get("confidence", 0.75)),
                    detected_language=raw_cls.get("detected_language"),
                )
                classifier_source = "explicit"

            if classifier is None:
                classifier = heuristic_classify(message, pre_signals)
                if classifier is not None:
                    classifier_source = "heuristic"

            if should_reclassify_with_llm(classifier, use_llm, classifier_context):
                classifier = None
                classifier_source = "fallback"

            if classifier is None and use_llm:
                classifier = classify_with_model(
                    message=message,
                    pre_signals=pre_signals,
                    conversation_context=classifier_context,
                    model=classifier_model,
                )
                if classifier is not None:
                    classifier_source = "llm"

            reply_context_used = (
                classifier_source == "llm"
                and bool(classifier_context and classifier_context.strip())
            )

            if classifier is None:
                classifier = ClassifierOutput(
                    task_type="general_chat",
                    cost_profile=cost_profile,
                    confidence=0.60,
                )
                classifier_source = "fallback"
            else:
                classifier.cost_profile = cost_profile

            decision = _get_router().route(
                classifier=classifier,
                pre_signals=pre_signals,
                nexus_context=nexus_context or {},
                route_mode=route_mode,
            )

            print(
                "route"
                f" task={decision.task_type}"
                f" complexity={classifier.complexity}"
                f" confidence={decision.confidence:.2f}"
                f" model={decision.selected_model}"
                f" provider={decision.selected_provider}"
                f" score={decision.score:.3f}"
                f" prompt_len={len(message)}"
                f" est_tokens={pre_signals.estimated_tokens}",
                flush=True,
            )

            _json_response(self, 200, {
                "decision_id": decision.decision_id,
                "task_type": decision.task_type,
                "confidence": decision.confidence,
                "selected_model": decision.selected_model,
                "selected_provider": decision.selected_provider,
                "fallbacks": decision.fallbacks,
                "score": decision.score,
                "reason": decision.reason,
                "classifier_source": classifier_source,
                "reply_context_used": reply_context_used,
                "pre_signals": {
                    "has_image": pre_signals.has_image,
                    "has_code": pre_signals.has_code,
                    "has_diff": pre_signals.has_diff,
                    "has_logs": pre_signals.has_logs,
                    "estimated_tokens": pre_signals.estimated_tokens,
                },
            })

        except Exception as e:
            _json_response(self, 500, {"error": str(e), "trace": traceback.format_exc()})

    def _handle_outcome(self, body: dict):
        try:
            decision_id = body.get("decision_id")
            if not decision_id:
                _json_response(self, 400, {"error": "decision_id required"})
                return

            update_outcome(
                decision_id=decision_id,
                success=bool(body.get("success", True)),
                latency_ms=body.get("latency_ms"),
                fallback_used=bool(body.get("fallback_used", False)),
                fallback_model=body.get("fallback_model"),
                user_override=bool(body.get("user_override", False)),
                user_override_model=body.get("user_override_model"),
            )

            provider = body.get("provider")
            if provider:
                observe_turn_outcome(
                    provider=provider,
                    http_status=body.get("http_status", 200 if body.get("success") else 500),
                    latency_ms=body.get("latency_ms"),
                    error_type=body.get("error_type"),
                )

            _json_response(self, 200, {"ok": True})

        except Exception as e:
            _json_response(self, 500, {"error": str(e)})

    def _handle_probe(self, body: dict):
        try:
            providers = body.get("providers")
            results = probe_all_providers(providers)
            _json_response(self, 200, {"results": results})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})


def run(port: int = DEFAULT_PORT):
    ensure_schema()
    # Bind to all interfaces inside Docker — external access is restricted
    # by the port mapping in docker-compose.yml (127.0.0.1:7771->7771)
    bind_addr = "0.0.0.0"
    server = HTTPServer((bind_addr, port), RouterHandler)
    print(f"Nexus Router server listening on http://127.0.0.1:{port}")
    print("Endpoints: POST /route  POST /outcome  POST /probe  GET /health  GET /stats")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus Router HTTP server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run(args.port)
