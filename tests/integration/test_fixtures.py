"""The committed audio fixtures match `expected.toml`."""

from __future__ import annotations

import tomllib
import wave

import numpy as np
import pytest

from tests.conftest import FIXTURES_AUDIO

pytestmark = pytest.mark.integration

FIXTURES = (
    "deploy_pr",
    "curta_sim",
    "nova_linha",
    "hesitacao",
    "silencio_3s",
    "ruido_3s",
)


def expected() -> dict[str, dict[str, object]]:
    return tomllib.loads((FIXTURES_AUDIO / "expected.toml").read_text(encoding="utf-8"))


def samples(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES_AUDIO / f"{name}.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def test_every_fixture_is_16khz_mono_and_listed() -> None:
    table = expected()

    assert set(table) == set(FIXTURES)
    for name in FIXTURES:
        entry = table[name]
        assert entry["file"] == f"{name}.wav"
        duration = len(samples(name)) / 16000
        assert duration == pytest.approx(entry["duration_s"], abs=0.01)


def test_spoken_fixtures_have_text_and_key_terms() -> None:
    table = expected()

    assert 14.0 <= float(table["deploy_pr"]["duration_s"]) <= 17.0  # type: ignore[arg-type]
    for name in ("deploy_pr", "curta_sim", "nova_linha", "hesitacao"):
        entry = table[name]
        text = str(entry["text"]).casefold()
        assert entry["speech"] is True
        terms = entry["key_terms"]
        assert isinstance(terms, list) and terms
        assert all(str(term) in text for term in terms)
    assert {"deploy", "pull request", "github", "pipeline", "ci"} <= set(
        table["deploy_pr"]["key_terms"]  # type: ignore[arg-type]
    )
    assert "hum" in str(table["hesitacao"]["text"]).casefold()


def test_silence_and_noise() -> None:
    assert not samples("silencio_3s").any()
    noise = samples("ruido_3s")
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(noise**2)))
    assert rms_dbfs == pytest.approx(-40.0, abs=0.5)
    assert expected()["ruido_3s"]["speech"] is False
