"""Correction memory: give a new message the task you already gave similar ones.

Routing feedback reaches the classifier only at its next successful retrain.
Until then, a message that closely matches messages labelled through feedback
(a task the user picked, or a confirmed pick) takes that task straight away.

Matching is character-trigram cosine over the newest labelled feedback (stdlib
only). A match changes the task only when all of these hold:

- similarity >= 0.6, and the winning task holds >= 60% of the similarity-weighted vote;
- at least one supporting example is a task the user picked themselves, not only
  confirmations of the router's own pick;
- the message has at least 3 words and is not a short reply that starts by agreeing
  ("yes go ahead", "ok do that"): those take their task from the conversation.

Set NEXUS_ROUTER_CORRECTION_MEMORY=off to disable it.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from . import db

SIMILARITY = 0.6
MIN_SHARE = 0.6
MIN_WORDS = 3
AGREEMENT_MAX_WORDS = 8
AGREEMENT_WORDS = frozenset({
    "yes", "ok", "okay", "sure", "yep", "yeah", "si", "sì", "certo", "va",
    "perfect", "perfetto", "great", "good", "alright", "fine", "go",
})
# Replayed over this router's own feedback, 73% of overrides were right (89% since
# August 2026): reported as the decision's confidence, so the low-confidence gate
# does not turn the corrected task back into general_chat.
CONFIDENCE = 0.75
MAX_EXAMPLES = 2000
CACHE_SECONDS = 300
MESSAGE_CHARS = 512  # routing_decisions keeps the first 512 characters

_lock = threading.Lock()
_cache: tuple[str, float, list[tuple[dict, tuple[Counter, float]]]] | None = None


@dataclass(frozen=True)
class Match:
    task_type: str
    similarity: float  # best similarity among the examples voting for task_type
    share: float  # task_type's share of the similarity-weighted vote
    support: int  # examples voting for task_type
    feedback_ids: tuple[str, ...]  # up to three strongest of them

    def explain(self, from_task: str, from_source: str) -> dict:
        return {
            "from_task": from_task,
            "from_source": from_source,
            "task_type": self.task_type,
            "similarity": self.similarity,
            "share": self.share,
            "support": self.support,
            "feedback_ids": list(self.feedback_ids),
        }


def enabled() -> bool:
    return os.getenv("NEXUS_ROUTER_CORRECTION_MEMORY", "on").strip().lower() not in {"off", "0", "false", "no"}


def invalidate() -> None:
    """Forget the cached examples, e.g. after new feedback."""
    global _cache
    with _lock:
        _cache = None


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def _vector(text: str) -> tuple[Counter, float]:
    padded = f" {' '.join(_words(text))} "
    grams = Counter(padded[i:i + 3] for i in range(len(padded) - 2))
    return grams, math.sqrt(sum(v * v for v in grams.values())) or 1.0


def _cosine(a: tuple[Counter, float], b: tuple[Counter, float]) -> float:
    (small, small_norm), (large, large_norm) = sorted((a, b), key=lambda v: len(v[0]))
    return sum(count * large.get(gram, 0) for gram, count in small.items()) / (small_norm * large_norm)


def eligible(text: str) -> bool:
    """Whether a message's own words, not the conversation, can carry its task."""
    words = _words(text)
    if len(words) < MIN_WORDS:
        return False
    return not (words[0] in AGREEMENT_WORDS and len(words) <= AGREEMENT_MAX_WORDS)


def _examples() -> list[tuple[dict, tuple[Counter, float]]]:
    global _cache
    with _lock:
        source = str(db.DB_PATH)
        if _cache is None or _cache[0] != source or time.monotonic() - _cache[1] > CACHE_SECONDS:
            examples = [
                (example, _vector(example["message_text"][:MESSAGE_CHARS]))
                for example in db.load_labeled_feedback(MAX_EXAMPLES)
                if len(_words(example["message_text"])) >= MIN_WORDS
            ]
            _cache = (source, time.monotonic(), examples)
        return _cache[2]


def summary() -> dict:
    examples = _examples() if enabled() else []
    return {
        "enabled": enabled(),
        "examples": len(examples),
        "user_picked": sum(1 for example, _ in examples if example["user_picked"]),
    }


def lookup(message: str) -> Match | None:
    """The task earlier feedback gives this message, or None when it gives none."""
    if not enabled() or not eligible(message):
        return None
    query = _vector(message[:MESSAGE_CHARS])
    votes: dict[str, float] = defaultdict(float)
    support: Counter = Counter()
    picked: Counter = Counter()
    matches: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for example, vector in _examples():
        similarity = _cosine(query, vector)
        if similarity < SIMILARITY:
            continue
        task = example["task_type"]
        votes[task] += similarity
        support[task] += 1
        picked[task] += bool(example["user_picked"])
        matches[task].append((similarity, example["feedback_id"]))
    if not votes:
        return None
    task, weight = max(votes.items(), key=lambda item: item[1])
    share = weight / sum(votes.values())
    if share < MIN_SHARE or not picked[task]:
        return None
    strongest = sorted(matches[task], reverse=True)[:3]
    return Match(
        task_type=task,
        similarity=round(strongest[0][0], 3),
        share=round(share, 3),
        support=support[task],
        feedback_ids=tuple(feedback_id for _, feedback_id in strongest),
    )
