import asyncio
import json

import httpx
import pytest

from src.cli_adapters import UnsupportedConversion, convert_request, convert_response, convert_stream


def test_tool_roundtrip_preserves_call_ids_arguments_and_results():
    body = {"model": "claude", "max_tokens": 500,
            "system": [{"type": "text", "text": "Search"}],
            "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "search"},
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "call1", "name": "search", "input": {"path": "src"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call1", "content": "Found"}]}]}
    chat = convert_request(body, "messages", "chat")
    assert chat["messages"][1]["tool_calls"][0]["id"] == "call1"
    assert chat["messages"][2] == {"role": "tool", "tool_call_id": "call1", "content": "Found"}
    back = convert_request(chat, "chat", "messages")
    assert back["messages"] == body["messages"]
    assert back["tool_choice"] == body["tool_choice"]
    assert back["tools"][0]["input_schema"] == body["tools"][0]["input_schema"]


@pytest.mark.parametrize("body", [
    {"thinking": {"type": "enabled"}},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
    {"tools": [{"name": "web_search", "type": "web_search_20250305"}]},
])
def test_protocol_specific_fields_are_rejected(body):
    with pytest.raises(UnsupportedConversion):
        convert_request(body, "messages", "chat")


def test_response_tools_usage_stop_reason():
    chat = {"id": "x", "choices": [{"message": {"content": None, "tool_calls": [{"id": "c", "function": {"name": "search", "arguments": '{"q":"abc"}'}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8}}
    message = convert_response(chat, "messages", "cheap")
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0]["input"] == {"q": "abc"}
    back = convert_response(message, "chat", "cheap")
    assert back["choices"][0]["finish_reason"] == "tool_calls"
    assert back["usage"]["total_tokens"] == 20


def test_stream_translation_handles_fragmented_tool_json_and_usage():
    events = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call1", "function": {"name": "search", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"foo"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
    ]
    content = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
    async def run():
        response = httpx.Response(200, content=content)
        return b"".join([chunk async for chunk in convert_stream(response, "messages", "small")])
    output = asyncio.run(run()).decode()
    assert output.count("event: content_block_start") == 1
    assert '"partial_json": "{\\\"q\\\":"' in output
    assert '"stop_reason": "tool_use"' in output
    assert '"output_tokens": 5' in output
    assert output.endswith('data: {"type": "message_stop"}\n\n')


def test_truncated_stream_is_not_success():
    async def run():
        response = httpx.Response(200, content='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        return [chunk async for chunk in convert_stream(response, "messages", "small")]
    with pytest.raises(UnsupportedConversion, match="finish_reason"):
        asyncio.run(run())
