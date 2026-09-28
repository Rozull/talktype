"""Test seams are inert by default (UT-204)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from talktype import seams

pytestmark = pytest.mark.unit

SEAM_VARS = (seams.AUDIO_SOURCE_ENV, seams.ACCEPT_INJECTED_ENV, seams.NO_MSGBOX_ENV)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in SEAM_VARS:
        monkeypatch.delenv(name, raising=False)
    seams.current.cache_clear()
    yield monkeypatch
    seams.current.cache_clear()


def test_ut204_no_seams_by_default(clean_env: pytest.MonkeyPatch) -> None:
    current = seams.current()

    assert seams.active() == []
    assert current.accept_injected is False
    assert current.no_msgbox is False
    assert current.audio_wav is None


def test_ut204_audio_source_is_listed_and_logged(
    clean_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clean_env.setenv(seams.AUDIO_SOURCE_ENV, "wav:C:/fixtures/deploy_pr.wav")
    logger = logging.getLogger("tests.seams")

    with caplog.at_level(logging.INFO, logger="tests.seams"):
        names = seams.log_active(logger)

    assert names == ["audio_source"]
    assert seams.active() == ["audio_source"]
    assert seams.current().audio_wav == Path("C:/fixtures/deploy_pr.wav")
    assert [r.getMessage() for r in caplog.records] == ["test_seam_active"]


def test_boolean_seams_need_the_value_1() -> None:
    assert seams.read({seams.ACCEPT_INJECTED_ENV: "0"}).active() == []
    both = seams.read({seams.ACCEPT_INJECTED_ENV: "1", seams.NO_MSGBOX_ENV: "1"})
    assert both.active() == ["accept_injected", "no_msgbox"]
