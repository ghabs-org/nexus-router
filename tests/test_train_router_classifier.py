from scripts.train_router_classifier import (
    build_tokenized_dataset,
    compute_class_weights,
    freeze_bottom_layers,
)
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


def test_build_tokenized_dataset_without_hf_datasets():
    class FakeTokenizer:
        def __call__(self, texts, truncation=True, max_length=256):
            return {
                "input_ids": [[101, idx + 1, 102] for idx, _ in enumerate(texts)],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    rows = [
        {"text": "fix failing unit test", "label": LABEL_TO_ID["coding"]},
        {"text": "compare consensus algorithms", "label": LABEL_TO_ID["reasoning"]},
    ]
    ds = build_tokenized_dataset(rows, FakeTokenizer(), max_length=128)
    assert len(ds) == 2
    assert ds[0] == {"input_ids": [101, 1, 102], "attention_mask": [1, 1, 1], "labels": LABEL_TO_ID["coding"]}
    assert ds[1]["labels"] == LABEL_TO_ID["reasoning"]


def test_freeze_bottom_layers_keeps_upper_blocks_and_head_trainable():
    class FakeParam:
        def __init__(self, n: int):
            self.n = n
            self.requires_grad = True

        def numel(self) -> int:
            return self.n

    class FakeBlock:
        def __init__(self, n: int):
            self._p = [FakeParam(n)]

        def parameters(self):
            return list(self._p)

    class FakeModel:
        def __init__(self):
            self.model = type("Encoder", (), {
                "embeddings": FakeBlock(10),
                "layers": [FakeBlock(5), FakeBlock(5), FakeBlock(5), FakeBlock(5)],
            })()
            self.classifier = FakeBlock(2)

        def parameters(self):
            return (
                self.model.embeddings.parameters()
                + [p for layer in self.model.layers for p in layer.parameters()]
                + self.classifier.parameters()
            )

    model = FakeModel()
    stats = freeze_bottom_layers(model, 2)
    assert stats == {"frozen_layers": 2, "trainable_params": 12, "total_params": 32}
    assert model.model.embeddings.parameters()[0].requires_grad is False
    assert model.model.layers[1].parameters()[0].requires_grad is False
    assert model.model.layers[2].parameters()[0].requires_grad is True
    assert model.classifier.parameters()[0].requires_grad is True
