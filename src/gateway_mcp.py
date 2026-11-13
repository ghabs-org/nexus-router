"""Stdio MCP task delegation for Antigravity. Never executes model tool calls."""
import json
import os
import sys
import uuid
from urllib.parse import urlparse
import httpx

TOOLS = [
    {"name": "nexus_run_task", "description": "Delegate a bounded text analysis, review, extraction or writing task to the administrator-routed model. No filesystem or shell access.",
     "inputSchema": {"type": "object", "properties": {"task": {"type": "string", "maxLength": 100000}, "session_id": {"type": "string", "maxLength": 256}}, "required": ["task"], "additionalProperties": False}},
    {"name":"nexus_escalate_task", "description":"Explicitly request one stronger route for a fresh bounded task after a poor result. Never replays prior tool actions.",
     "inputSchema":{"type":"object","properties":{"task":{"type":"string","maxLength":100000},"source_session":{"type":"string","maxLength":64}},"required":["task","source_session"],"additionalProperties":False}},
    {"name": "nexus_status", "description": "Show your gateway usage and recent routes.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]

def dispatch(request, client):
    method = request.get("method")
    if method == "initialize":
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "nexus-router", "version": "0.2.0"}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method != "tools/call":
        raise ValueError("Unsupported MCP method")
    params = request.get("params", {})
    name, args = params.get("name"), params.get("arguments", {})
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    if name == "nexus_status" and not args:
        response = client.get("/v1/nexus/status")
    elif name == "nexus_escalate_task":
        if set(args)!={"task","source_session"} or not isinstance(args['task'],str) or not 0<len(args['task'])<=100000 or not isinstance(args['source_session'],str) or len(args['source_session'])>64:
            raise ValueError('Invalid escalation arguments')
        issued=client.post('/v1/nexus/escalations',json={'source_session':args['source_session'],'wire':'messages'})
        if issued.is_error:
            return {"isError":True,"content":[{"type":"text","text":f"Escalation returned HTTP {issued.status_code}"}]}
        ticket=issued.json()
        response=client.post('/v1/messages',headers={'x-session-id':ticket['new_session_id'],'x-nexus-escalation-ticket':ticket['ticket']},
                             json={'model':'nexus-auto','max_tokens':4096,'tools':[],'messages':[{'role':'user','content':args['task']}]})
    elif name == "nexus_run_task":
        task = args.get("task")
        session = args.get("session_id", uuid.uuid4().hex)
        if set(args) - {"task", "session_id"} or not isinstance(task, str) or not 0 < len(task) <= 100000 or not isinstance(session, str) or not 0 < len(session) <= 256:
            raise ValueError("Invalid task arguments")
        response = client.post("/v1/messages", headers={"x-session-id": session}, json={"model": "nexus-auto", "max_tokens": 4096,
                               "messages": [{"role": "user", "content": task}], "tools": []})
    else:
        raise ValueError("Unknown tool or invalid arguments")
    if response.is_error:
        return {"isError": True, "content": [{"type": "text", "text": f"Gateway returned HTTP {response.status_code}"}]}
    return {"content": [{"type": "text", "text": json.dumps(response.json())}]}

def serve(url, token_env):
    parsed = urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Remote gateway requires HTTPS")
    token = os.environ.get(token_env)
    if not token:
        raise ValueError("Gateway token environment variable is missing")
    with httpx.Client(base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=300, trust_env=False, follow_redirects=False) as client:
        while True:
            line = sys.stdin.buffer.readline(1024 * 1024 + 1)
            if not line:
                return
            if len(line) > 1024 * 1024:
                raise ValueError("MCP request too large")
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("MCP request must be an object")
                if "id" not in request:
                    continue
                reply = {"jsonrpc": "2.0", "id": request["id"], "result": dispatch(request, client)}
            except Exception:
                reply = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                         "error": {"code": -32602, "message": "Invalid request or gateway unavailable"}}
            print(json.dumps(reply), flush=True)
