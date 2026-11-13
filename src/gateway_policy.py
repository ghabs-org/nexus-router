"""Joint model/reasoning policy for Cowork. No inference is used to route a turn."""

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass

from .cli_classifier import CallCategory, CliClassification, CliPayloadClassifier, text_blocks
from .cli_proxy import Target
from .vertex import validate_provider


class PolicyError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GatewayClassification(CliClassification):
    principal: str
    profile: str
    requested_model: str
    choices: tuple


class GatewayPolicy:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.alias = config.get("auto_model", "nexus-auto")
        self.models = config["models"]
        self.profiles = config["profiles"]
        self.default = config.get("default_profile", "standard")
        self.background = config.get("background_profile", "utility")
        if self.default not in self.profiles or self.background not in self.profiles:
            raise ValueError("Default and background profiles must exist")
        self.targets = {}
        for key in ("max_output_tokens", "max_body_bytes", "max_pins", "pin_ttl_seconds"):
            if key in config and (type(config[key]) is not int or config[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        for client in list(config.get("clients", {}).values()) + ([config["oidc"]] if config.get("oidc") else []):
            for key, limit in client.get("limits", {}).items():
                if key not in {"max_tokens_24h", "max_task_tokens", "max_concurrent"} or type(limit) is not int or limit < 1:
                    raise ValueError(f"Invalid client limit: {key}")
        public_ids = set()
        for label, model in self.models.items():
            provider, separator, model_id = label.partition(":")
            if not separator or not re.fullmatch(r"[a-zA-Z0-9._@-]+", model_id):
                raise ValueError(f"Invalid gateway model target: {label}")
            if provider not in config["providers"] or model_id in public_ids or model_id == self.alias:
                raise ValueError(f"Unknown provider or duplicate public model ID: {label}")
            public_ids.add(model_id)
            self.targets[label] = Target(provider, model_id)
            if not isinstance(model.get("max_output_tokens"), int) or model["max_output_tokens"] < 1:
                raise ValueError(f"max_output_tokens is required for {label}")
            self.validate_options(label, model.get("defaults", {}))
            if model.get("pricing"):
                prices = model["pricing"]
                expected = {"input", "output", "cache_read", "cache_write_5m", "cache_write_1h"}
                if set(prices) != expected or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
                                                 for value in prices.values()):
                    raise ValueError(f"Invalid per-million-token tariffs for {label}")
        if not self.models:
            raise ValueError("At least one model is required")
        for provider in config["providers"].values():
            validate_provider(provider)
            if provider.get("beta_transport", "header") not in {"header", "body"}:
                raise ValueError("beta_transport must be header or body")
            if provider.get("transport", "messages") not in {"vertex", "messages", "openai"}:
                raise ValueError("Gateway requires a native Messages or OpenAI transport")
        for name, choices in self.profiles.items():
            if not choices:
                raise ValueError(f"Empty gateway profile: {name}")
            for choice in choices:
                if choice["target"] not in self.targets:
                    raise ValueError(f"Unknown profile target: {choice['target']}")
                self.validate_options(choice["target"], choice)
        self.rules = []
        for rule in config.get("rules", []):
            if rule["profile"] not in self.profiles:
                raise ValueError("Rule profile does not exist")
            self.rules.append((re.compile(rule["pattern"], re.I), rule["profile"]))
        behaves_as = config.get("behaves_as")
        if behaves_as is not None and behaves_as not in public_ids:
            raise ValueError("behaves_as must identify an allowed Claude model")

    def validate_options(self, label, options):
        model = self.models[label]
        effort = options.get("effort")
        if effort is not None and effort not in model.get("efforts", []):
            raise PolicyError(f"Unsupported effort for {label}: {effort}")
        thinking = options.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict) or thinking.get("type") not in model.get("thinking_modes", []):
                raise PolicyError(f"Unsupported thinking configuration for {label}")
            if thinking["type"] == "enabled":
                budget = thinking.get("budget_tokens")
                if type(budget) is not int or not 1024 <= budget < model["max_output_tokens"]:
                    raise PolicyError("Thinking budget must be >=1024 and below the model output limit")

    def identity(self, body, headers, principal):
        headers = {k.lower(): v for k, v in headers.items()}
        identity = headers.get("x-session-id") or headers.get("session_id") or headers.get("x-claude-code-session-id")
        metadata = body.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise PolicyError("metadata must be an object")
        user_id = metadata.get("user_id", "")
        if not identity and isinstance(user_id, str):
            try:
                parsed = json.loads(user_id)
                identity = parsed.get("session_id") if isinstance(parsed, dict) else None
            except ValueError:
                match = re.search(r"_session_([a-zA-Z0-9-]+)$", user_id)
                identity = match[1] if match else None
        first_user = next((m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user"), "")
        if not identity:
            identity = json.dumps([body.get("system"), first_user], sort_keys=True)
        return hashlib.blake2s(json.dumps([principal, identity]).encode(), digest_size=16).hexdigest()

    def classify(self, body, headers, principal, pin=None):
        requested = body.get("model")
        if not isinstance(requested, str):
            raise PolicyError("model must be an allowed model ID or the automatic alias")
        session = self.identity(body, headers, principal)
        wire_class = CliPayloadClassifier().classify(body, headers)
        # A generic Cowork system prompt mentions many tasks. Only the user's
        # opening instruction selects the business profile; untrusted files do not.
        first_user = next((m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user"), "")
        instruction = text_blocks(first_user).split("\n\n", 1)[0][:2000]
        background = wire_class.category in {CallCategory.STATELESS_HOUSEKEEPING, CallCategory.SUBAGENT_WORKER}
        category = wire_class.category if background else CallCategory.MAIN_EXECUTION
        effort_override = {k.lower(): v for k, v in headers.items()}.get("x-nexus-effort")
        if pin and not background:
            if requested != pin["requested_model"]:
                raise PolicyError("Model changes require a new task/session ID", 409)
            choices = pin["choices"]
            for choice in choices:
                if choice["target"] not in self.targets:
                    raise PolicyError("Task route was removed; start a new task", 409)
                self.validate_options(choice["target"], choice)
            if effort_override and any(c.get("effort") != effort_override for c in choices):
                raise PolicyError("Effort changes require a new task/session ID", 409)
            if requested != self.alias:
                incoming = self._explicit_options(body, self.models[choices[0]["target"]])
                if any(incoming.get(k) != choices[0].get(k) for k in ("thinking", "effort")):
                    raise PolicyError("Thinking changes require a new task/session ID", 409)
            profile, reason = pin["profile"], "existing_task_profile"
        elif requested == self.alias:
            profile = self.background if background else self.default
            reason = "background" if background else "default_profile"
            if not background:
                for pattern, candidate in self.rules:
                    if pattern.search(instruction):
                        profile, reason = candidate, "task_rule"
                        break
            choices = copy.deepcopy(self.profiles[profile])
        else:
            label = next((label for label, target in self.targets.items() if target.model == requested), None)
            if label is None:
                raise PolicyError("Requested model is not in the administrator's allowlist")
            choices = [{"target": label, **self._explicit_options(body, self.models[label])}]
            profile, reason = "explicit", "explicit_model"
        choices = copy.deepcopy(choices)
        if effort_override:
            for choice in choices:
                choice["effort"] = effort_override
        for choice in choices:
            self.validate_options(choice["target"], choice)
        # Never replay a signed thinking continuation to another model. A new
        # cold-cache request is safe only before model-specific state exists.
        signed = any(block.get("type") in {"thinking", "redacted_thinking"}
                     for m in body.get("messages", []) for block in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(block, dict))
        if signed:
            if not pin:
                raise PolicyError("Thinking continuation needs an existing task pin", 409)
            choices = [c for c in choices if c["target"] == pin["target"]]
        return GatewayClassification(category, session, reason, principal, profile, requested, tuple(choices))

    @staticmethod
    def _explicit_options(body, model):
        output = body.get("output_config") or {}
        if not isinstance(output, dict):
            raise PolicyError("output_config must be an object")
        options = copy.deepcopy(model.get("defaults", {}))
        if "thinking" in body:
            options["thinking"] = copy.deepcopy(body["thinking"])
        if "effort" in output:
            options["effort"] = output["effort"]
        return options

    def apply(self, body, classification, target):
        choice = next(c for c in classification.choices if c["target"] == target.label)
        result = copy.deepcopy(body)
        result["model"] = target.model
        if "thinking" in choice:
            result["thinking"] = copy.deepcopy(choice["thinking"])
        else:
            result.pop("thinking", None)
        output = result.get("output_config") or {}
        if not isinstance(output, dict):
            raise PolicyError("output_config must be an object")
        output = dict(output)
        output.pop("effort", None)
        if choice.get("effort"):
            output["effort"] = choice["effort"]
        if output:
            result["output_config"] = output
        else:
            result.pop("output_config", None)
        maximum = result.get("max_tokens")
        if type(maximum) is not int or maximum < 1:
            raise PolicyError("max_tokens must be a positive integer")
        cap = min(self.models[target.label]["max_output_tokens"], self.config.get("max_output_tokens", maximum))
        result["max_tokens"] = min(maximum, cap)
        thinking = result.get("thinking") or {}
        if thinking.get("type") == "enabled" and thinking["budget_tokens"] >= result["max_tokens"]:
            raise PolicyError("Thinking budget must be below max_tokens")
        return result

    def model_list(self):
        caps = list(self.models.values())
        data = [{"id": self.alias, "type": "model", "display_name": "Automatic (Nexus)",
                 "description": "Administrator-managed model and effort routing",
                 "max_input_tokens": min(m.get("context_window", 200000) for m in caps)}]
        for label, target in self.targets.items():
            model = self.models[label]
            data.append({"id": target.model, "type": "model", "display_name": target.model,
                         "anthropic_family_tier": model.get("tier"),
                         "max_input_tokens": model.get("context_window", 200000)})
        for item in data:
            item["created_at"] = "2026-11-12T00:00:00Z"
        return {"object": "list", "data": data, "has_more": False, "first_id": data[0]["id"], "last_id": data[-1]["id"]}
