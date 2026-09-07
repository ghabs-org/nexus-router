from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np

from .local_classifier_labels import ID_TO_LABEL
from .types import ClassifierOutput, PreSignals

from .paths import LOCAL_CLASSIFIER_DIR

DEFAULT_LOCAL_CLASSIFIER_DIR = LOCAL_CLASSIFIER_DIR
DEFAULT_LOCAL_CLASSIFIER_MODEL_FILE = os.environ.get(
    "NEXUS_ROUTER_LOCAL_CLASSIFIER_MODEL_FILE", "model.onnx"
)
DEFAULT_LOCAL_CLASSIFIER_MIN_CONFIDENCE = float(
    os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MIN_CONFIDENCE", "0.70")
)
DEFAULT_LOCAL_CLASSIFIER_MARGIN = float(
    os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MARGIN", "0.05")
)
DEFAULT_LOCAL_CLASSIFIER_MIN_CONFIDENCE_BY_LABEL: dict[str, float] = {
    "code_review": float(os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MIN_CONFIDENCE_CODE_REVIEW", "0.55")),
    "long_context": float(os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MIN_CONFIDENCE_LONG_CONTEXT", "0.50")),
}
DEFAULT_LOCAL_CLASSIFIER_MARGIN_BY_LABEL: dict[str, float] = {
    "code_review": float(os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MARGIN_CODE_REVIEW", "0.02")),
    "long_context": float(os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MARGIN_LONG_CONTEXT", "0.02")),
}
DEFAULT_LOCAL_CLASSIFIER_MAX_LENGTH = int(
    os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_MAX_LENGTH", "512")
)


@dataclass(frozen=True)
class LocalClassifierResult:
    classifier: ClassifierOutput
    top_label: str
    confidence: float
    margin: float
    top2: list[tuple[str, float]]
    artifact_dir: str


def _normalize_context_snippet(conversation_context: str | None, message: str) -> str | None:
    raw = str(conversation_context or "").strip()
    if not raw:
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    current = str(message or "").strip().lower()
    candidate = ""
    for line in reversed(lines):
        lowered = line.lower()
        # Skip exact duplicate of current turn.
        if current and lowered == current:
            continue
        # Drop common envelope labels/prefixes.
        cleaned = re.sub(r"^(user|assistant|system)\s*[:\-]\s*", "", line, flags=re.IGNORECASE).strip()
        if cleaned:
            candidate = cleaned
            break

    if not candidate:
        return None

    # Keep this short and focused: previous message only.
    words = candidate.split()
    if len(words) > 48:
        candidate = " ".join(words[-48:])
    return candidate


class LocalRouteClassifier:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = Path(artifact_dir or DEFAULT_LOCAL_CLASSIFIER_DIR)
        self.model_path = self.artifact_dir / DEFAULT_LOCAL_CLASSIFIER_MODEL_FILE
        self.tokenizer_path = self.artifact_dir / "tokenizer.json"
        self.config_path = self.artifact_dir / "config.json"
        self._load_error: str | None = None
        self._session = None
        self._tokenizer = None
        self._label_map = dict(ID_TO_LABEL)

    @property
    def available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"missing ONNX model: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"missing tokenizer.json: {self.tokenizer_path}")

        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore

        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        if self.config_path.exists():
            try:
                cfg = json.loads(self.config_path.read_text())
                id2label = cfg.get("id2label")
                if isinstance(id2label, dict):
                    parsed: dict[int, str] = {}
                    for key, value in id2label.items():
                        try:
                            parsed[int(key)] = str(value)
                        except Exception:
                            continue
                    if parsed:
                        self._label_map = parsed
            except Exception:
                pass

    def classify(
        self,
        message: str,
        pre_signals: PreSignals,
        conversation_context: str | None = None,
    ) -> LocalClassifierResult | None:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._session is not None

        prev = _normalize_context_snippet(conversation_context, message)
        text_for_classifier = message if not prev else f"prev: {prev}\ncurrent: {message}"

        encoding = self._tokenizer.encode(text_for_classifier)
        ids = encoding.ids[:DEFAULT_LOCAL_CLASSIFIER_MAX_LENGTH]
        attention = encoding.attention_mask[:DEFAULT_LOCAL_CLASSIFIER_MAX_LENGTH]
        type_ids = encoding.type_ids[:DEFAULT_LOCAL_CLASSIFIER_MAX_LENGTH] if getattr(encoding, 'type_ids', None) else []

        if not ids:
            return None

        ort_inputs: dict[str, Any] = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.asarray([attention or ([1] * len(ids))], dtype=np.int64),
        }

        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = np.asarray([type_ids or ([0] * len(ids))], dtype=np.int64)

        outputs = self._session.run(None, ort_inputs)
        if not outputs:
            return None

        logits = np.asarray(outputs[0][0], dtype=np.float64)
        probs = _softmax(logits)
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_label = self._label_map.get(top_idx, ID_TO_LABEL.get(top_idx, "general_chat"))
        confidence = float(probs[top_idx])
        margin = float(probs[top_idx] - probs[second_idx]) if len(order) > 1 else confidence
        top2 = [
            (self._label_map.get(int(idx), ID_TO_LABEL.get(int(idx), "general_chat")), float(probs[int(idx)]))
            for idx in order[:2]
        ]

        classifier = _build_classifier_output(top_label, confidence, margin, pre_signals)
        classifier.classifier_provider = "local"
        classifier.classifier_model = f"onnx:{self.model_path}"
        return LocalClassifierResult(
            classifier=classifier,
            top_label=top_label,
            confidence=confidence,
            margin=margin,
            top2=top2,
            artifact_dir=str(self.artifact_dir),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    total = np.sum(exps)
    if total <= 0:
        return np.zeros_like(logits)
    return exps / total


def _build_classifier_output(task_type: str, confidence: float, margin: float, pre_signals: PreSignals) -> ClassifierOutput:
    needs_tools = task_type in {"coding", "code_review"}
    needs_vision = task_type == "vision" or pre_signals.has_image
    needs_long_context = task_type == "long_context" or pre_signals.estimated_tokens > 200_000

    if confidence >= 0.88:
        complexity = "high"
    elif confidence >= 0.72:
        complexity = "medium"
    else:
        complexity = "low"

    cost_profile = "cheap" if task_type == "fast_utility" else "balanced"
    subtype = None
    if task_type == "coding" and pre_signals.has_logs:
        subtype = "debugging"
    elif task_type == "code_review" and pre_signals.has_diff:
        subtype = "review"
    elif task_type == "reasoning":
        subtype = "comparison" if margin < 0.18 else "planning"

    return ClassifierOutput(
        task_type=task_type,
        subtype=subtype,
        complexity=complexity,
        needs_tools=needs_tools,
        needs_vision=needs_vision,
        needs_long_context=needs_long_context,
        cost_profile=cost_profile,
        confidence=confidence,
    )


_LOCAL_CLASSIFIER: LocalRouteClassifier | None = None


def get_local_classifier() -> LocalRouteClassifier:
    global _LOCAL_CLASSIFIER
    if _LOCAL_CLASSIFIER is None:
        _LOCAL_CLASSIFIER = LocalRouteClassifier()
    return _LOCAL_CLASSIFIER


def _threshold_for_label(label: str, global_threshold: float, per_label: dict[str, float]) -> float:
    specific = per_label.get(label)
    if specific is None:
        return global_threshold
    return min(global_threshold, float(specific))


def classify_with_local_model(
    message: str,
    pre_signals: PreSignals,
    min_confidence: float = DEFAULT_LOCAL_CLASSIFIER_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_LOCAL_CLASSIFIER_MARGIN,
    conversation_context: str | None = None,
) -> LocalClassifierResult | None:
    classifier = get_local_classifier()
    if not classifier.available:
        return None

    try:
        result = classifier.classify(
            message,
            pre_signals,
            conversation_context=conversation_context,
        )
    except TypeError:
        # Keep simple fakes and older custom classifier adapters compatible.
        result = classifier.classify(message, pre_signals)
    if result is None:
        return None

    effective_min_confidence = _threshold_for_label(
        result.top_label,
        min_confidence,
        DEFAULT_LOCAL_CLASSIFIER_MIN_CONFIDENCE_BY_LABEL,
    )
    effective_min_margin = _threshold_for_label(
        result.top_label,
        min_margin,
        DEFAULT_LOCAL_CLASSIFIER_MARGIN_BY_LABEL,
    )

    if result.confidence < effective_min_confidence:
        return None
    if result.margin < effective_min_margin:
        return None
    return result
