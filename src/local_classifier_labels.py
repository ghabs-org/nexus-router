"""Shared label definitions for the local router classifier.

This module intentionally mirrors the router's top-level task types so a
fine-tuned local classifier can emit stable labels that map directly into the
existing routing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUTER_TASK_LABELS: tuple[str, ...] = (
    "coding",
    "code_review",
    "reasoning",
    "summarization",
    "fast_utility",
    "long_context",
    "vision",
    "general_chat",
)

LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(ROUTER_TASK_LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_HF_CACHE_DIR = "/home/ubuntu/.openclaw/workspace/.cache/huggingface"
DEFAULT_DATA_DIR = "artifacts/router-classifier/data"
DEFAULT_OUTPUT_DIR = "artifacts/router-classifier/checkpoints"
DEFAULT_ONNX_DIR = "artifacts/router-classifier/onnx"
MAX_SEQUENCE_LENGTH = 512


@dataclass(frozen=True)
class RouterLabelSet:
    labels: tuple[str, ...] = ROUTER_TASK_LABELS

    @property
    def label_to_id(self) -> dict[str, int]:
        return LABEL_TO_ID

    @property
    def id_to_label(self) -> dict[int, str]:
        return ID_TO_LABEL
