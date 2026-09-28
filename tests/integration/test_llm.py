"""`LocalRewriter` over real HTTP: a loopback stub (IT-022) and the real Ollama (IT-023, gpu)."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import typing
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from talktype.config import LlmSettings
from talktype.llm import LocalRewriter, RewriteSkipped, drift_ok

pytestmark = pytest.mark.integration


class Stub(ThreadingHTTPServer):
    """An Ollama-like ``/api/chat`` on 127.0.0.1 that answers `content` after `delay_s`."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, content: str, delay_s: float = 0.0) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.content = content
        self.delay_s = delay_s
        self.bodies: list[dict[str, object]] = []

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = typing.cast(Stub, self.server)
        length = int(self.headers.get("Content-Length", "0"))
        server.bodies.append(json.loads(self.rfile.read(length)))
        time.sleep(server.delay_s)
        body = json.dumps({"message": {"role": "assistant", "content": server.content}})
        payload = body.encode("utf-8")
        try:
            self.send_response(200 if self.path == "/api/chat" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError:  # the client gave up (timeout case)
            pass

    def log_message(self, format: str, *args: object) -> None:
        del format, args  # keep the test output quiet


def serve(content: str, delay_s: float = 0.0) -> Iterator[Stub]:
    stub = Stub(content, delay_s)
    thread = threading.Thread(target=stub.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        stub.shutdown()
        stub.server_close()


@pytest.fixture
def stub() -> Iterator[Stub]:
    yield from serve("Texto de teste para o stub.")


@pytest.fixture
def slow_stub() -> Iterator[Stub]:
    yield from serve("Texto de teste para o stub.", delay_s=6.0)


def local(endpoint: str, **settings: object) -> LocalRewriter:
    return LocalRewriter(LlmSettings.model_validate({"endpoint": endpoint, **settings}))


def test_it022_ollama_path_returns_stub_content(stub: Stub) -> None:
    rw = local(stub.endpoint)
    try:
        assert rw.rewrite("texto de teste para o stub") == "Texto de teste para o stub."
    finally:
        rw.close()

    body = stub.bodies[0]
    assert body["stream"] is False
    assert body["keep_alive"] == "30m"
    assert body["options"] == {"temperature": 0}


def test_it022_slow_server_times_out(slow_stub: Stub) -> None:
    rw = local(slow_stub.endpoint, timeout_s=5)
    started = time.monotonic()
    try:
        with pytest.raises(RewriteSkipped) as info:
            rw.rewrite("texto de teste para o stub")
    finally:
        rw.close()
    elapsed = time.monotonic() - started

    assert info.value.reason == "timeout"
    assert 5.0 <= elapsed <= 5.5


def test_it022_stopped_server_is_unreachable() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again before the request
    rw = local(f"http://127.0.0.1:{port}")
    try:
        with pytest.raises(RewriteSkipped) as info:
            rw.rewrite("texto")
    finally:
        rw.close()

    assert info.value.reason == "unreachable"


# --------------------------------------------------------------------------------------
# IT-023: the real Ollama with qwen2.5:7b-instruct
# --------------------------------------------------------------------------------------

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b-instruct"


def _ollama_has_model() -> bool:
    try:
        tags = httpx.get(f"{OLLAMA}/api/tags", timeout=2, trust_env=False).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(m.get("name") == MODEL for m in tags.get("models", []))


@pytest.mark.gpu
def test_it023_real_ollama_cleans_without_drifting() -> None:
    if not _ollama_has_model():
        pytest.skip(f"Ollama with {MODEL} is not running on {OLLAMA}")
    original = "é então hum eu preciso fazer o deploy tipo hoje"
    rw = local(OLLAMA, enabled=True, model=MODEL)
    try:
        rw.prewarm()  # a cold load takes longer than the 5 s rewrite timeout
        assert rw.prewarm_thread is not None
        rw.prewarm_thread.join(120)
        output = rw.rewrite(original)
    finally:
        rw.close()

    assert drift_ok(original, output)
    assert "deploy" in output.lower()
    assert re.search(r"\bhum\b", output.lower()) is None
