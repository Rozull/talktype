"""`LocalRewriter`: the optional AI rewrite through a local LLM server.

- **Local only.** The endpoint must be ``localhost`` or a loopback address; anything else
  raises ValueError here and is refused by config validation.
- **Providers.** ``ollama`` posts to ``/api/chat`` with ``keep_alive``; ``openai_compatible``
  posts to ``/v1/chat/completions`` with ``Authorization: Bearer`` when an API key is set.
  Both use ``temperature = 0``.
- **Transport.** Synchronous `httpx` with ``timeout = timeout_s`` and ``trust_env=False``, so
  a corporate proxy never sees loopback traffic. No retries.
- **Guards.** Input over 6000 characters is not sent. The output is stripped of framing and
  accepted only when `drift_ok()` holds. Every failure raises `RewriteSkipped(reason)`, and
  the caller inserts the text without the rewrite.
- **Pre-warm.** `prewarm()` sends a one-token request on a daemon thread and never raises.

`strip_framing()` and `drift_ok()` are pure functions. `is_loopback_endpoint()` lives in
`config.py` (validation needs it) and is re-exported here.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from fractions import Fraction
from typing import Any, Literal

import httpx

from talktype.config import DEFAULT_LLM_INSTRUCTION, LlmSettings, is_loopback_endpoint
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg

logger = get_logger("llm")

MAX_INPUT_CHARS = 6000
MIN_LENGTH_RATIO = Fraction(1, 2)
MAX_LENGTH_RATIO = Fraction(8, 5)
MIN_TOKEN_RECALL = Fraction(1, 2)
PREWARM_TEXT = "Olá."
PREWARM_TIMEOUT_S = 60.0  # a cold model load takes about 18 s

SkipReason = Literal["timeout", "unreachable", "http_error", "empty", "drift", "too_long", "auth"]


class RewriteSkipped(Exception):  # noqa: N818 - public name
    """The rewrite was not used; `reason` says why."""

    def __init__(self, reason: SkipReason) -> None:
        super().__init__(reason)
        self.reason: SkipReason = reason


# --------------------------------------------------------------------------------------
# Pure guards
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+")
_FENCE = re.compile(r"^```[^\n]*\n(?P<body>.*?)\n?[ \t]*```$", re.DOTALL)
_FRAMING_LINE = re.compile(
    r"^(?:(?:claro|certo|ok|sure)[!,.]?[ \t]+)?"
    r"(?:aqui\s+(?:está|esta|vai)|here(?:'s|\s+is)|"
    r"(?:o\s+)?texto\s+(?:revisado|corrigido|limpo)|(?:a\s+)?vers[ãa]o\s+(?:revisada|corrigida)|"
    r"(?:the\s+)?(?:revised|cleaned(?:[- ]up)?|corrected)\s+text)"
    r"\b[^\n]*:[ \t]*(?:\n|$)",
    re.IGNORECASE,
)
_QUOTES = {'"': '"', "“": "”", "'": "'", "«": "»", "‘": "’"}


def _unquote(text: str) -> str:
    if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
        inner = text[1:-1]
        if text[0] not in inner and text[-1] not in inner:
            return inner.strip()
    return text


def strip_framing(output: str, *, original: str = "") -> str:
    """Remove what models wrap around the answer.

    That is a leading "Aqui está…:" / "Here is…:" line, code fences and surrounding quotes.
    A leading line is kept when `original` starts with such a line itself.
    """
    keep_intro = bool(_FRAMING_LINE.match(original.strip()))
    text = output.strip()
    while True:
        before = text
        if not keep_intro and (match := _FRAMING_LINE.match(text)):
            text = text[match.end() :].strip()
        if fence := _FENCE.match(text):
            text = fence.group("body").strip()
        text = _unquote(text)
        if text == before:
            return text


def drift_ok(original: str, output: str) -> bool:
    """Whether `output` is still a cleanup of `original` rather than a new text.

    The character length ratio must be within 0.5..1.6, and at least half of the input's
    distinct lowercase word tokens must appear in the output. Bounds are inclusive.
    """
    source, result = original.strip(), output.strip()
    if not source or not result:
        return False
    ratio = Fraction(len(result), len(source))
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return False
    tokens = set(_WORD.findall(source.lower()))
    if not tokens:
        return True
    kept = tokens & set(_WORD.findall(result.lower()))
    return Fraction(len(kept), len(tokens)) >= MIN_TOKEN_RECALL


# --------------------------------------------------------------------------------------
# Rewriter
# --------------------------------------------------------------------------------------


class LocalRewriter:
    """Rewrites one text at a time; safe to share between the delivery and pre-warm threads."""

    def __init__(self, s: LlmSettings, http: httpx.Client | None = None) -> None:
        if not is_loopback_endpoint(s.endpoint):
            raise ValueError(Msg.LLM_ENDPOINT_NOT_LOCAL.text)
        self.settings = s
        self._owns_http = http is None
        self.http = httpx.Client(timeout=s.timeout_s, trust_env=False) if http is None else http
        self.prewarm_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        base = self.settings.endpoint.rstrip("/")
        if self.settings.provider == "ollama":
            return f"{base}/api/chat"
        if base.endswith("/v1"):  # LM Studio style endpoints already carry the prefix
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    # -- requests ----------------------------------------------------------------------

    def _payload(self, text: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        s = self.settings
        messages = [
            {"role": "system", "content": s.instruction.strip() or DEFAULT_LLM_INSTRUCTION},
            {"role": "user", "content": text},
        ]
        if s.provider == "ollama":
            options: dict[str, Any] = {"temperature": 0}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            return {
                "model": s.model,
                "messages": messages,
                "stream": False,
                "keep_alive": s.keep_alive,
                "options": options,
            }
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _headers(self) -> dict[str, str]:
        s = self.settings
        if s.provider == "openai_compatible" and s.api_key:
            return {"Authorization": f"Bearer {s.api_key}"}
        return {}

    def _content(self, data: Any) -> str:
        if self.settings.provider == "ollama":
            content = data["message"]["content"]
        else:
            content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not a string")
        return content

    def _complete(self, payload: dict[str, Any], timeout_s: float) -> str:
        """POST the chat request and return the model's text. Raises `RewriteSkipped`."""
        try:
            response = self.http.post(
                self.url, json=payload, headers=self._headers(), timeout=timeout_s
            )
        except httpx.TimeoutException as exc:
            raise RewriteSkipped("timeout") from exc
        except httpx.NetworkError as exc:  # connection refused or reset
            raise RewriteSkipped("unreachable") from exc
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise RewriteSkipped("http_error") from exc
        if response.status_code in (401, 403):
            raise RewriteSkipped("auth")
        if not response.is_success:
            log_event(
                logger, "rewrite_http_status", level=logging.WARNING, status=response.status_code
            )
            raise RewriteSkipped("http_error")
        try:
            return self._content(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RewriteSkipped("http_error") from exc

    # -- public API --------------------------------------------------------------------

    def rewrite(self, text: str) -> str:
        """The rewritten `text`. Raises `RewriteSkipped`; never anything else from I/O."""
        if len(text) > MAX_INPUT_CHARS:
            raise RewriteSkipped("too_long")
        raw = self._complete(self._payload(text), self.settings.timeout_s)
        output = strip_framing(raw, original=text)
        if not output:
            raise RewriteSkipped("empty")
        if not drift_ok(text, output):
            raise RewriteSkipped("drift")
        return output

    def prewarm(self) -> None:
        """Load the model in the background with a one-token request. Never raises."""
        try:
            thread = threading.Thread(
                target=self._prewarm, name="talktype-llm-prewarm", daemon=True
            )
            self.prewarm_thread = thread
            thread.start()
        except Exception as exc:
            log_event(
                logger,
                "rewrite_prewarm_failed",
                level=logging.WARNING,
                reason="thread",
                exc=type(exc).__name__,
            )

    def _prewarm(self) -> None:
        started = time.monotonic()
        try:
            self._complete(self._payload(PREWARM_TEXT, max_tokens=1), PREWARM_TIMEOUT_S)
        except RewriteSkipped as exc:
            log_event(logger, "rewrite_prewarm_failed", level=logging.WARNING, reason=exc.reason)
            return
        except Exception as exc:
            log_event(
                logger,
                "rewrite_prewarm_failed",
                level=logging.WARNING,
                reason="error",
                exc=type(exc).__name__,
            )
            return
        elapsed_ms = round((time.monotonic() - started) * 1000)
        log_event(logger, "rewrite_prewarmed", model=self.settings.model, ms=elapsed_ms)
