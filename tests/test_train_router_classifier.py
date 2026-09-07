from scripts.train_router_classifier import compute_class_weights
from src.local_classifier_labels import LABEL_TO_ID, ROUTER_TASK_LABELS


def _rows(label: str, n: int):
    return [{"text": f"{label}-{i}", "label": LABEL_TO_ID[label]} for i in range(n)]


def test_compute_class_weights_boosts_rare_labels():
    rows = []
    rows += _rows("general_chat", 120)
    rows += _rows("coding", 80)
    rows += _rows("reasoning", 60)
    rows += _rows("code_review", 4)
    rows += _rows("long_context", 2)

    weights = compute_class_weights(rows)
    assert len(weights) == len(ROUTER_TASK_LABELS)
    assert weights[LABEL_TO_ID["long_context"]] > weights[LABEL_TO_ID["general_chat"]]
    assert weights[LABEL_TO_ID["code_review"]] > weights[LABEL_TO_ID["coding"]]


def test_compute_class_weights_normalized_mean_near_one():
    rows = _rows("general_chat", 10) + _rows("reasoning", 10)
    weights = compute_class_weights(rows)
    mean = sum(weights) / len(weights)
    assert 0.99 <= mean <= 1.01
