from scripts.general_chat_audit import _cluster_name


def test_cluster_name_brief_chitchat():
    row = {
        "message_text": "hey there",
        "has_code": 0,
        "has_diff": 0,
        "has_logs": 0,
        "has_image": 0,
        "estimated_tokens": 30,
    }
    assert _cluster_name(row) == "brief_chitchat"


def test_cluster_name_technical_embedded_wins():
    row = {
        "message_text": "what does this stack trace mean?",
        "has_code": 0,
        "has_diff": 0,
        "has_logs": 1,
        "has_image": 0,
        "estimated_tokens": 40,
    }
    assert _cluster_name(row) == "technical_embedded"
