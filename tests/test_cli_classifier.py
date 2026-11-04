import copy
import json
from pathlib import Path

import pytest

from src.cli_classifier import CliPayloadClassifier


FIXTURES = json.loads((Path(__file__).parent / "fixtures/cli_payloads.json").read_text())


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_cli_payload_fixtures(fixture):
    original = copy.deepcopy(fixture["body"])
    result = CliPayloadClassifier().classify(fixture["body"], {})
    assert result.category.value == fixture["category"]
    assert fixture["body"] == original


def test_fingerprint_stable_across_turns_and_phase():
    classifier = CliPayloadClassifier()
    body = {"system": "You are a coding assistant", "messages": [{"role": "user", "content": "Build a parser"}]}
    first = classifier.classify(body, {})
    body["messages"] += [{"role": "assistant", "content": "Working"}, {"role": "user", "content": "Continue"}]
    body["thinking"] = {"type": "enabled"}
    assert classifier.classify(body, {}).session == first.session
    body["messages"][0]["content"] = "Another project"
    assert classifier.classify(body, {}).session != first.session


def test_explicit_session_header_case_insensitive_and_private():
    classifier = CliPayloadClassifier()
    a = classifier.classify({"input": "first"}, {"X-Session-ID": "secret-session"})
    b = classifier.classify({"input": "next", "instructions": "changed"}, {"x-session-id": "secret-session"})
    assert a.session == b.session
    assert "secret-session" not in a.session


def test_codex_wire_collaboration_marker_is_planning():
    result = CliPayloadClassifier().classify({"input": [
        {"role": "developer", "content": [{"type": "input_text", "text": "<collaboration_mode>Plan Mode</collaboration_mode>"}]},
        {"role": "user", "content": "Design a parser"},
    ]}, {})
    assert result.category.value == "MAIN_PLANNING"


@pytest.mark.parametrize("instruction", [
    "You are a helpful assistant tasked with summarizing conversations.",
    "Create a detailed summary of the conversation so far.",
    "Summarise the transcript for the next context window.",
])
def test_compaction_instruction_variants(instruction):
    result = CliPayloadClassifier().classify({"system": instruction, "max_tokens": 2048, "tools": []}, {})
    assert result.category.value == "STATELESS_HOUSEKEEPING"
