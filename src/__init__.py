"""
__init__.py — Nexus Router public API
"""

from .router import Router
from .types import (
    ClassifierOutput,
    PreSignals,
    ProviderHealth,
    RoutingDecision,
    ModelScore,
)
from .health import record_observation, load_provider_health
from .db import ensure_schema, write_decision, update_outcome, log_provider_observation

__all__ = [
    "Router",
    "ClassifierOutput",
    "PreSignals",
    "ProviderHealth",
    "RoutingDecision",
    "ModelScore",
    "record_observation",
    "load_provider_health",
    "ensure_schema",
    "write_decision",
    "update_outcome",
    "log_provider_observation",
]
