"""HTTP helpers shared by the advisory server and the standalone gateway."""

import json


def json_response(handler, status, data):
    body = json.dumps(data, default=str).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass
