"""Opt-in HTTP transport for sandboxes that prohibit listening TCP sockets.

This replaces only connect_tcp with connected Unix socket pairs. HTTPX/httpcore
still serialize/parse real HTTP bytes, and the real BaseHTTPRequestHandlers run
on server threads. It does NOT verify TCP binding or external CLI connectivity.
The default integration harness uses TCP; enable with NEXUS_TEST_SOCKETPAIR=1.
"""

import asyncio
import itertools
import socket
import threading

import httpcore
from httpcore._backends.auto import AutoBackend
from httpcore._backends.base import AsyncNetworkStream
from httpcore._backends.sync import SyncBackend, SyncStream


class PairStream(AsyncNetworkStream):
    def __init__(self, sock):
        self.stream = SyncStream(sock)

    async def read(self, max_bytes, timeout=None):
        return await asyncio.to_thread(self.stream.read, max_bytes, timeout)

    async def write(self, buffer, timeout=None):
        await asyncio.to_thread(self.stream.write, buffer, timeout)

    async def aclose(self):
        self.stream.close()

    def get_extra_info(self, name):
        return self.stream.get_extra_info(name)


def install(monkeypatch):
    registry = {}
    ports = itertools.count(40000)

    class PairServer:
        def __init__(self, address, handler):
            self.server_port = next(ports)
            self.handler = handler
            self.threads = []
            registry[self.server_port] = self

        def connect(self):
            client, accepted = socket.socketpair()

            def serve():
                try:
                    self.handler(accepted, ("127.0.0.1", 1), self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    accepted.close()

            thread = threading.Thread(target=serve, daemon=True)
            self.threads.append(thread)
            thread.start()
            return client

        def serve_forever(self):
            pass

        def shutdown(self):
            for thread in self.threads:
                thread.join(4)

        def server_close(self):
            registry.pop(self.server_port, None)

    def connect(port):
        if port not in registry:
            raise httpcore.ConnectError("No socket-pair server at this port")
        return registry[port].connect()

    def sync_connect(self, host, port, **kwargs):
        return SyncStream(connect(port))

    async def async_connect(self, host, port, **kwargs):
        return PairStream(connect(port))

    monkeypatch.setattr(SyncBackend, "connect_tcp", sync_connect)
    monkeypatch.setattr(AutoBackend, "connect_tcp", async_connect)
    return PairServer
