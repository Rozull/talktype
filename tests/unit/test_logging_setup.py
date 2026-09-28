"""Log redaction (UT-200; UT-201 is merged into it)."""

from __future__ import annotations

import io
import logging
from typing import Any

import pytest

from talktype import logging_setup
from talktype.logging_setup import KeyValueFormatter, log_event

pytestmark = pytest.mark.unit


def render(**fields: Any) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(KeyValueFormatter())
    logger = logging.getLogger("talktype.test.redaction")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        log_event(logger, "dictation_finished", **fields)
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


def test_ut200_text_is_redacted_unless_log_text() -> None:
    logging_setup.set_log_text(False)
    redacted = render(id=3, text="segredo", status="inserted")

    assert " INFO dictation_finished " in redacted
    assert "id=3" in redacted
    assert "text_len=7" in redacted
    assert "status=inserted" in redacted
    assert "segredo" not in redacted

    logging_setup.set_log_text(True)
    visible = render(id=3, text="segredo", status="inserted")

    assert "text=segredo" in visible


def test_ut200_explicit_text_len_is_not_duplicated() -> None:
    line = render(text="segredo", text_len=7)

    assert line.count("text_len=") == 1
    assert "segredo" not in line


def test_values_with_spaces_are_quoted() -> None:
    logging_setup.set_log_text(True)

    line = render(text='olá "mundo"\nfim', device="Headset (Jabra)")

    assert 'text="olá \\"mundo\\"\\nfim"' in line
    assert 'device="Headset (Jabra)"' in line
    assert line.count("\n") == 1
