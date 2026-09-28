"""`LocalRewriter`, framing strip, drift guard and loopback checks (UT-041, UT-080 … UT-093)."""

from __future__ import annotations

import httpx
import pytest

from talktype.config import DEFAULT_LLM_INSTRUCTION, LlmSettings
from talktype.llm import (
    MAX_INPUT_CHARS,
    LocalRewriter,
    RewriteSkipped,
    drift_ok,
    is_loopback_endpoint,
    strip_framing,
)
from tests.fakes import FakeHttp, ollama_reply, openai_reply

pytestmark = pytest.mark.unit


def rewriter(http: FakeHttp, **settings: object) -> LocalRewriter:
    return LocalRewriter(LlmSettings.model_validate(settings), http=http.client)


def skipped(rw: LocalRewriter, text: str) -> str:
    with pytest.raises(RewriteSkipped) as info:
        rw.rewrite(text)
    return info.value.reason


# --- Endpoint -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434", "http://localhost:1234", "http://[::1]:11434", "http://127.5.5.5"],
)
def test_ut041_loopback_endpoints(url: str) -> None:
    assert is_loopback_endpoint(url) is True


@pytest.mark.parametrize(
    "url",
    ["http://10.0.0.2", "http://example.com", "http://0.0.0.0:11434", "ftp://127.0.0.1"],
)
def test_ut041_non_loopback_endpoints(url: str) -> None:
    assert is_loopback_endpoint(url) is False


def test_ut092_remote_endpoint_is_refused() -> None:
    with pytest.raises(ValueError):
        LocalRewriter(LlmSettings(endpoint="http://192.168.0.5:11434"))


def test_ut093_own_client_ignores_proxy_environment() -> None:
    rw = LocalRewriter(LlmSettings())
    try:
        assert rw.http.trust_env is False
    finally:
        rw.close()


# --- Providers ----------------------------------------------------------------------------


def test_ut080_ollama_request_and_answer() -> None:
    http = FakeHttp(ollama_reply("Texto limpo."))
    rw = rewriter(http)

    assert rw.rewrite("texto limpo") == "Texto limpo."

    request = http.requests[0]
    body = http.payload()
    assert request.method == "POST"
    assert request.url.path == "/api/chat"
    assert str(request.url).startswith("http://127.0.0.1:11434/")
    assert body["options"]["temperature"] == 0
    assert body["keep_alive"] == "30m"
    assert body["stream"] is False
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["messages"][0] == {"role": "system", "content": DEFAULT_LLM_INSTRUCTION}
    assert body["messages"][1] == {"role": "user", "content": "texto limpo"}
    assert "authorization" not in request.headers


def test_ut081_openai_compatible_request() -> None:
    http = FakeHttp(openai_reply("Texto limpo."))
    rw = rewriter(http, provider="openai_compatible", endpoint="http://localhost:1234", api_key="k")

    assert rw.rewrite("texto limpo") == "Texto limpo."

    request = http.requests[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    assert http.payload()["temperature"] == 0
    assert http.payload()["messages"][1]["content"] == "texto limpo"


def test_openai_compatible_endpoint_with_v1_prefix() -> None:
    http = FakeHttp(openai_reply("Texto limpo."))
    rw = rewriter(http, provider="openai_compatible", endpoint="http://localhost:1234/v1")

    rw.rewrite("texto limpo")

    assert http.requests[0].url.path == "/v1/chat/completions"
    assert "authorization" not in http.requests[0].headers


# --- Errors -------------------------------------------------------------------------------


def test_ut082_read_timeout() -> None:
    http = FakeHttp(error=lambda r: httpx.ReadTimeout("timed out", request=r))

    assert skipped(rewriter(http), "texto") == "timeout"


def test_ut083_connection_refused() -> None:
    http = FakeHttp(error=lambda r: httpx.ConnectError("refused", request=r))

    assert skipped(rewriter(http), "texto") == "unreachable"


@pytest.mark.parametrize("status", [401, 403])
def test_ut084_auth_error(status: int) -> None:
    http = FakeHttp({"error": "unauthorized"}, status=status)

    assert skipped(rewriter(http, provider="openai_compatible", api_key="x"), "texto") == "auth"


@pytest.mark.parametrize("body", [{"error": "model not found"}, None])
def test_other_http_errors(body: object) -> None:
    assert skipped(rewriter(FakeHttp(body, status=404)), "texto") == "http_error"
    assert skipped(rewriter(FakeHttp({"unexpected": 1})), "texto") == "http_error"


def test_ut085_blank_answer_is_empty() -> None:
    assert skipped(rewriter(FakeHttp(ollama_reply("   "))), "texto") == "empty"


def test_ut088_drifting_answer_is_refused() -> None:
    http = FakeHttp(ollama_reply("I need to deploy today"))

    assert skipped(rewriter(http), "é então eu preciso fazer o deploy hoje") == "drift"


def test_ut089_input_length_limit() -> None:
    http = FakeHttp(error=AssertionError("no request expected"))
    too_long = "a " * (MAX_INPUT_CHARS // 2) + "b"
    assert len(too_long) == MAX_INPUT_CHARS + 1

    assert skipped(rewriter(http), too_long) == "too_long"
    assert http.requests == []

    at_limit = too_long[:-1]
    echo = FakeHttp(ollama_reply(at_limit))
    assert rewriter(echo).rewrite(at_limit) == at_limit.strip()
    assert len(echo.requests) == 1


def test_ut090_empty_instruction_sends_the_preset() -> None:
    http = FakeHttp(ollama_reply("Olá."))

    rewriter(http, instruction="").rewrite("olá")

    assert http.payload()["messages"][0]["content"] == DEFAULT_LLM_INSTRUCTION


def test_custom_instruction_is_the_system_message() -> None:
    http = FakeHttp(ollama_reply("Olá."))

    rewriter(http, instruction="Seja formal.").rewrite("olá")

    assert http.payload()["messages"][0]["content"] == "Seja formal."


# --- Pre-warm -----------------------------------------------------------------------------


def test_ut091_prewarm_failure_is_logged_not_raised(logs: list[str]) -> None:
    http = FakeHttp(error=lambda r: httpx.ConnectError("refused", request=r))
    rw = rewriter(http)

    rw.prewarm()
    assert rw.prewarm_thread is not None
    rw.prewarm_thread.join(5)

    assert len(http.requests) == 1
    assert http.payload()["options"]["num_predict"] == 1
    assert any(line.startswith("rewrite_prewarm_failed") and "unreachable" in line for line in logs)


def test_prewarm_openai_compatible_asks_for_one_token() -> None:
    http = FakeHttp(openai_reply("O"))
    rw = rewriter(http, provider="openai_compatible", endpoint="http://localhost:1234")

    rw.prewarm()
    assert rw.prewarm_thread is not None
    rw.prewarm_thread.join(5)

    assert len(http.requests) == 1
    assert http.payload()["max_tokens"] == 1


# --- Framing and drift ----------------------------------------------------------------------


def test_ut086_strip_framing() -> None:
    assert strip_framing('Aqui está o texto limpo:\n"Olá, tudo bem."') == "Olá, tudo bem."
    assert strip_framing("```\nOlá\n```") == "Olá"
    assert strip_framing("Here is the cleaned text:\n```text\nOlá\n```") == "Olá"
    assert strip_framing("“Olá.”") == "Olá."


def test_strip_framing_keeps_real_content() -> None:
    assert strip_framing('Ele disse "oi" e saiu.') == 'Ele disse "oi" e saiu.'
    intro = "Aqui está o relatório:\nTudo certo."
    assert strip_framing(intro, original="aqui está o relatório:\ntudo certo") == intro


def test_ut087_drift_examples() -> None:
    original = "é então eu preciso fazer o deploy hoje"

    assert drift_ok(original, "I need to deploy today") is False
    assert drift_ok(original, "Então, eu preciso fazer o deploy hoje.") is True


@pytest.mark.parametrize(
    ("length", "expected"), [(50, True), (160, True), (49, False), (161, False)]
)
def test_ut087_length_ratio_bounds(length: int, expected: bool) -> None:
    original = "ab " * 33 + "a"
    assert len(original) == 100
    output = "ab a" + "." * (length - 4)

    assert drift_ok(original, output) is expected


@pytest.mark.parametrize(("kept", "expected"), [(50, True), (49, False)])
def test_ut087_token_recall_bounds(kept: int, expected: bool) -> None:
    words = [f"w{i:03d}" for i in range(100)]
    original = " ".join(words)
    output = " ".join(words[:kept]) + " " + "-" * 100

    assert drift_ok(original, output) is expected
