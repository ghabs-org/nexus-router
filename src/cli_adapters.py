"""Conservative Messages <-> Chat adapters; native routes never pass here.

Cross-protocol conversion deliberately rejects features it cannot represent.
Responses continuation IDs, thinking signatures and caching are wire-specific.
"""

import json
import uuid


class UnsupportedConversion(ValueError):
    pass


def _text(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or any(b.get("type") not in {"text", "input_text"} for b in value):
        raise UnsupportedConversion("Only text content can cross protocols")
    return "\n".join(b.get("text", "") for b in value)


def convert_request(body: dict, source: str, target: str) -> dict:
    if source == target:
        return dict(body)
    if {source, target} != {"messages", "chat"}:
        raise UnsupportedConversion("Responses requires a native Responses upstream")
    allowed = {"model", "messages", "stream", "temperature", "top_p", "tools", "tool_choice",
               "max_tokens", "max_completion_tokens", "system", "stop_sequences", "stop",
               "metadata", "stream_options"}
    if set(body) - allowed:
        raise UnsupportedConversion("Request has protocol-specific fields; use a native upstream")
    result = {k: body[k] for k in ("model", "stream", "temperature", "top_p") if k in body}
    messages = []
    if source == "messages":
        if body.get("system"):
            messages.append({"role": "system", "content": _text(body["system"])})
        for message in body.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                messages.append(dict(message))
                continue
            texts, calls, results = [], [], []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    texts.append(block["text"])
                elif kind == "tool_use":
                    calls.append({"id": block["id"], "type": "function", "function": {
                        "name": block["name"], "arguments": json.dumps(block["input"])}})
                elif kind == "tool_result" and not block.get("is_error"):
                    results.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": _text(block["content"])})
                else:
                    raise UnsupportedConversion(f"Unsupported Messages content: {kind}")
            messages.extend(results)
            if texts or calls:
                item = {"role": message["role"], "content": "\n".join(texts) or None}
                if calls:
                    item["tool_calls"] = calls
                messages.append(item)
        result["max_tokens"] = body.get("max_tokens", 4096)
        if "stop_sequences" in body:
            result["stop"] = body["stop_sequences"]
        if body.get("tools"):
            if any(t.get("type") or set(t) - {"name", "description", "input_schema", "cache_control"} for t in body["tools"]):
                raise UnsupportedConversion("Server tools require a native upstream")
            result["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
                for t in body["tools"]]
        choice = body.get("tool_choice")
        if choice:
            if set(choice) - {"type", "name"}:
                raise UnsupportedConversion("Unsupported tool_choice options")
            result["tool_choice"] = ({"type": "function", "function": {"name": choice["name"]}}
                                     if choice["type"] == "tool" else {"any": "required"}.get(choice["type"], choice["type"]))
        if body.get("stream"):
            result["stream_options"] = {"include_usage": True}
    else:
        systems = []
        for message in body.get("messages", []):
            role = message["role"]
            if set(message) - {"role", "content", "tool_calls", "tool_call_id"}:
                raise UnsupportedConversion("Message contains protocol-specific fields")
            if role in {"system", "developer"}:
                systems.append(_text(message["content"]))
            elif role == "tool":
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": _text(message["content"])}]})
            else:
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": _text(message["content"])})
                for call in message.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": json.loads(call["function"]["arguments"])})
                messages.append({"role": role, "content": blocks})
        if systems:
            result["system"] = "\n".join(systems)
        result["max_tokens"] = body.get("max_completion_tokens", body.get("max_tokens", 4096))
        if "stop" in body:
            stop = body["stop"]
            result["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if body.get("tools"):
            if any(t.get("type") != "function" or set(t["function"]) - {"name", "description", "parameters"} for t in body["tools"]):
                raise UnsupportedConversion("Only standard function tools can cross protocols")
            result["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""), "input_schema": t["function"].get("parameters", {"type": "object"})} for t in body["tools"]]
        choice = body.get("tool_choice")
        if choice:
            result["tool_choice"] = ({"type": "tool", "name": choice["function"]["name"]}
                                     if isinstance(choice, dict) else {"type": {"required": "any"}.get(choice, choice)})
    result["messages"] = messages
    return result


TO_MESSAGES_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
TO_CHAT_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def convert_response(body: dict, source: str, model: str) -> dict:
    """source is the client's protocol; the upstream uses the other protocol."""
    if "error" in body:
        return {"type": "error", "error": body["error"]} if source == "messages" else {"error": body["error"]}
    usage = body.get("usage") or {}
    if source == "messages":
        choice = body["choices"][0]
        message = choice["message"]
        blocks = [{"type": "text", "text": message["content"]}] if message.get("content") else []
        for call in message.get("tool_calls", []):
            blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": json.loads(call["function"]["arguments"])})
        return {"id": body.get("id", "msg_" + uuid.uuid4().hex), "type": "message", "role": "assistant",
                "model": model, "content": blocks, "stop_reason": TO_MESSAGES_STOP.get(choice.get("finish_reason"), "end_turn"),
                "stop_sequence": None, "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}
    message = {"role": "assistant", "content": ""}
    for block in body.get("content", []):
        if block["type"] == "text":
            message["content"] += block["text"]
        elif block["type"] == "tool_use":
            message.setdefault("tool_calls", []).append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": json.dumps(block["input"])}})
        else:
            raise UnsupportedConversion("Upstream returned non-portable content")
    inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {"id": body.get("id"), "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": TO_CHAT_STOP.get(body.get("stop_reason"), "stop")}],
            "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}}


def _sse(data, event=None):
    return ((f"event: {event}\n" if event else "") + "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n").encode()


async def convert_stream(response, source: str, model: str):
    """Translate one SSE event at a time, without buffering the completion.

    Text and function-call deltas are supported; advanced reasoning stays native.
    """
    message_id = "msg_" + uuid.uuid4().hex
    started = False
    blocks = {}
    active_block = None
    closed_blocks = set()
    input_tokens = output_tokens = 0

    def chat(delta, finish=None, usage=None):
        item = {"id": message_id, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            item["usage"] = usage
        return _sse(item)

    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        data = json.loads(raw)
        if data.get("error"):
            yield _sse({"type": "error", "error": data["error"]}, "error" if source == "messages" else None)
            return
        if source == "messages":
            if not started:
                yield _sse({"type": "message_start", "message": {"id": message_id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}, "message_start")
                started = True
            usage = data.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                additions = []
                if delta.get("content"):
                    additions.append(("text", {"type": "text", "text": ""}, {"type": "text_delta", "text": delta["content"]}))
                for call in delta.get("tool_calls", []):
                    key = ("tool", call["index"])
                    fn = call.get("function", {})
                    additions.append((key, {"type": "tool_use", "id": call.get("id"), "name": fn.get("name"), "input": {}},
                                      {"type": "input_json_delta", "partial_json": fn.get("arguments", "")}))
                for key, block, change in additions:
                    if key not in blocks:
                        if active_block is not None:
                            closed_blocks.add(active_block)
                            yield _sse({"type": "content_block_stop", "index": blocks[active_block]}, "content_block_stop")
                        if block["type"] == "tool_use" and (not block.get("id") or not block.get("name")):
                            raise UnsupportedConversion("Tool name/id must precede argument deltas")
                        blocks[key] = len(blocks)
                        active_block = key
                        yield _sse({"type": "content_block_start", "index": blocks[key], "content_block": block}, "content_block_start")
                    if key in closed_blocks:
                        raise UnsupportedConversion("Interleaved parallel tool deltas require a native route")
                    yield _sse({"type": "content_block_delta", "index": blocks[key], "delta": change}, "content_block_delta")
                if choice.get("finish_reason"):
                    finish = TO_MESSAGES_STOP.get(choice["finish_reason"], "end_turn")
            # Usage normally arrives after finish_reason; emit the closing events at EOF.
        else:
            kind = data.get("type")
            if kind == "message_start":
                message_id = data["message"]["id"]
                input_tokens = data["message"].get("usage", {}).get("input_tokens", 0)
                yield chat({"role": "assistant", "content": ""})
            elif kind == "content_block_start":
                block = data["content_block"]
                if block["type"] == "tool_use":
                    index = len(blocks)
                    blocks[data["index"]] = index
                    yield chat({"tool_calls": [{"index": index, "id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": ""}}]})
                elif block["type"] != "text":
                    raise UnsupportedConversion("Upstream streamed non-portable content")
            elif kind == "content_block_delta":
                delta = data["delta"]
                if delta["type"] == "text_delta":
                    yield chat({"content": delta["text"]})
                elif delta["type"] == "input_json_delta":
                    yield chat({"tool_calls": [{"index": blocks[data["index"]], "function": {"arguments": delta["partial_json"]}}]})
                else:
                    raise UnsupportedConversion("Upstream streamed non-portable delta")
            elif kind == "message_delta":
                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                yield chat({}, TO_CHAT_STOP.get(data["delta"].get("stop_reason"), "stop"))
            elif kind == "message_stop":
                yield _sse({"id": message_id, "object": "chat.completion.chunk", "model": model, "choices": [],
                            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}})
                yield _sse("[DONE]")
                return
    if source == "messages" and started:
        if "finish" not in locals():
            raise UnsupportedConversion("Upstream stream ended without finish_reason")
        if active_block is not None:
            yield _sse({"type": "content_block_stop", "index": blocks[active_block]}, "content_block_stop")
        yield _sse({"type": "message_delta", "delta": {"stop_reason": finish, "stop_sequence": None},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}, "message_delta")
        yield _sse({"type": "message_stop"}, "message_stop")
    else:
        raise UnsupportedConversion("Upstream stream ended without a terminal event")
