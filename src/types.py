"""
types.py — shared data types for Nexus Router
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassifierOutput:
    """
    Structured output from the Nexus classifier agent.
    The router consumes this to determine model selection.
    """
    task_type: str                    # coding|code_review|reasoning|summarization|fast_utility|long_context|vision|general_chat
    subtype: Optional[str] = None     # e.g. implementation|debugging|refactor
    complexity: str = "medium"        # low|medium|high
    needs_tools: bool = True
    needs_vision: bool = False
    needs_long_context: bool = False
    cost_profile: str = "balanced"    # cheap|balanced|premium
    confidence: float = 0.75
    detected_language: Optional[str] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None


@dataclass
class PreSignals:
    """
    Non-linguistic pre-signals extracted before classifier runs.
    Used as fast-path heuristics and as input to classifier context.
    """
    has_image: bool = False
    has_code: bool = False
    has_diff: bool = False
    has_logs: bool = False
    has_url: bool = False
    has_file_refs: bool = False
    estimated_tokens: int = 0
    message_length: int = 0


@dataclass
class ProviderHealth:
    """Runtime health snapshot for a provider."""
    provider: str
    auth: str = "unknown"             # ok|expired|missing|unknown
    quota: str = "unknown"            # healthy|low|exhausted|unknown
    quota_remaining_ratio: Optional[float] = None  # 0.0–1.0 when known
    recent_error_rate: float = 0.0    # 0.0–1.0
    rate_limit_risk: float = 0.0      # 0.0–1.0
    latency_ms_p50: Optional[float] = None
    last_failure_at: Optional[str] = None
    last_check_at: Optional[str] = None
    health_score: float = 1.0         # composite 0.0–1.0


@dataclass
class ModelScore:
    """Final computed score for one candidate model."""
    model_id: str
    provider: str
    total_score: float                # composite 0.0–1.0
    task_fit: float
    health: float
    preference: float
    learned: float
    cost: float
    speed: float
    excluded: bool = False
    exclusion_reason: Optional[str] = None


@dataclass
class RoutingDecision:
    """Final output of the router."""
    task_type: str
    confidence: float
    selected_model: str
    selected_provider: str
    decision_id: Optional[str] = None
    fallbacks: list[str] = field(default_factory=list)
    score: float = 0.0
    reason: list[str] = field(default_factory=list)
    excluded_models: list[dict] = field(default_factory=list)
    all_scores: list[ModelScore] = field(default_factory=list)
