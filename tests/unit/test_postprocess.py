"""Cleanup rules, spoken commands, order, idempotency and speed (UT-058 … UT-079, UT-227)."""

from __future__ import annotations

import time
import tomllib

import pytest

from talktype.config import (
    CleanupSettings,
    CommandItem,
    CommandsSettings,
    Settings,
    Substitution,
)
from talktype.postprocess import apply_cleanup, apply_commands, postprocess
from tests.conftest import FIXTURES_AUDIO

pytestmark = pytest.mark.unit

DEFAULT_CLEANUP = CleanupSettings()
DEFAULT_COMMANDS = CommandsSettings()


def substitutions(*pairs: tuple[str, str]) -> CleanupSettings:
    return CleanupSettings(
        substitutions=[Substitution.model_validate({"from": a, "to": b}) for a, b in pairs]
    )


def commands(*items: tuple[str, str]) -> CommandsSettings:
    return CommandsSettings(items=[CommandItem(phrase=p, text=t) for p, t in items])


def fixture_texts() -> list[str]:
    data = tomllib.loads((FIXTURES_AUDIO / "expected.toml").read_text(encoding="utf-8"))
    return [table["text"] for table in data.values()]


# --- Cleanup ------------------------------------------------------------------------------


def test_ut058_hesitations_are_removed() -> None:
    assert apply_cleanup("é… eu acho que hum funciona", DEFAULT_CLEANUP) == "eu acho que funciona"


def test_ut059_hesitation_removal_can_be_switched_off() -> None:
    s = CleanupSettings(remove_hesitations=False)

    assert apply_cleanup("é… eu acho que hum funciona", s) == "é… eu acho que hum funciona"


def test_ut060_e_without_ellipsis_is_kept() -> None:
    assert apply_cleanup("o bug é que nada acontece", DEFAULT_CLEANUP) == (
        "o bug é que nada acontece"
    )


def test_ut060_e_with_three_dots_is_removed() -> None:
    assert apply_cleanup("é... funciona", DEFAULT_CLEANUP) == "funciona"


def test_ut061_substitution_applies() -> None:
    s = substitutions(("ponto e vírgula", ";"))

    assert apply_cleanup("a ponto e vírgula b", s) == "a; b"


def test_ut061_hesitations_removed_without_substitutions() -> None:
    assert apply_cleanup("a hum b", CleanupSettings(substitutions=[])) == "a b"


def test_ut062_substitutions_are_not_chained() -> None:
    s = substitutions(("A", "B"), ("B", "C"))

    assert apply_cleanup("A", s) == "B"
    assert apply_cleanup("B", s) == "C"


def test_hesitation_takes_one_adjacent_comma() -> None:
    text = "Hum, eu acho que, ahn, a gente pode revisar isso amanhã, hum, depois da reunião."

    assert apply_cleanup(text, DEFAULT_CLEANUP) == (
        "eu acho que, a gente pode revisar isso amanhã, depois da reunião."
    )
    assert apply_cleanup("funciona, hum.", DEFAULT_CLEANUP) == "funciona."


def test_tipo_and_ne_are_never_removed() -> None:
    assert apply_cleanup("é tipo isso, né", DEFAULT_CLEANUP) == "é tipo isso, né"


def test_hesitations_are_whole_words() -> None:
    assert apply_cleanup("humor e humildade", DEFAULT_CLEANUP) == "humor e humildade"


@pytest.mark.parametrize("text", fixture_texts())
def test_ut064_postprocess_is_idempotent_on_fixture_texts(text: str) -> None:
    once = postprocess(text, Settings())

    assert postprocess(once, Settings()) == once


def test_ut065_cleanup_disabled_still_runs_commands() -> None:
    settings = Settings(cleanup=CleanupSettings(enabled=False))

    assert postprocess("hum  nova linha teste", settings) == "hum \nteste"


def test_ut066_whitespace_is_normalized() -> None:
    assert apply_cleanup("  a   b ,c  ", DEFAULT_CLEANUP) == "a b, c"


def test_whitespace_keeps_decimal_commas_and_line_breaks() -> None:
    assert apply_cleanup("custa 1,5 mil", DEFAULT_CLEANUP) == "custa 1,5 mil"
    assert apply_cleanup("a \n b", DEFAULT_CLEANUP) == "a\nb"


# --- Spoken commands ------------------------------------------------------------------------


def test_ut067_nova_linha() -> None:
    assert apply_commands("primeira nova linha segunda", DEFAULT_COMMANDS) == "primeira\nsegunda"


def test_ut068_command_consumes_adjacent_punctuation_and_case() -> None:
    assert apply_commands("fim. Nova linha. Começo", DEFAULT_COMMANDS) == "fim.\nComeço"


def test_ut069_novo_paragrafo() -> None:
    assert apply_commands("a novo parágrafo b", DEFAULT_COMMANDS) == "a\n\nb"


def test_ut070_commands_disabled() -> None:
    s = CommandsSettings(enabled=False)

    assert apply_commands("a nova linha b", s) == "a nova linha b"
    assert postprocess("a nova linha b", Settings(commands=s)) == "a nova linha b."


def test_ut071_commands_match_whole_phrases_only() -> None:
    assert apply_commands("renova linhas", DEFAULT_COMMANDS) == "renova linhas"


def test_ut072_repeated_command() -> None:
    assert apply_commands("a nova linha nova linha b", DEFAULT_COMMANDS) == "a\n\nb"


def test_ut073_longest_phrase_wins() -> None:
    s = commands(("nova", "X"), ("nova linha", "\n"))

    assert apply_commands("a nova linha b", s) == "a\nb"


def test_ut074_user_command() -> None:
    s = commands(("abre parênteses", "("))

    assert apply_commands("f abre parênteses x", s) == "f (x"


def test_ut075_no_command_items() -> None:
    assert apply_commands("a nova linha b", CommandsSettings(items=[])) == "a nova linha b"


def test_ut076_cleanup_runs_before_commands() -> None:
    assert postprocess("hum nova linha", Settings()) == "\n"


def test_ut077_only_hesitations_gives_empty_text() -> None:
    assert postprocess("hum… hmm", Settings()) == ""


def test_ut078_trailing_line_break_is_kept() -> None:
    assert postprocess("tchau nova linha", Settings()) == "tchau\n"


def test_ut079_emoji_command() -> None:
    s = commands(("carinha feliz", "😀"))

    assert apply_commands("oi carinha feliz", s) == "oi 😀"


def test_commands_are_not_chained() -> None:
    s = commands(("um", "nova linha"), ("nova linha", "\n"))

    assert apply_commands("a um b", s) == "a nova linha b"


# --- Limits -------------------------------------------------------------------------------


def test_ut227_long_text_is_fast_and_complete() -> None:
    unit = "o deploy de hoje hum passou em todos os testes nova linha "  # 1 hesitation, 1 command
    text = "".join(
        unit if i < 100 else "o deploy de hoje hum passou em todos os testes e " for i in range(200)
    )
    text += "x" * (30_000 - len(text))
    assert len(text) == 30_000
    settings = Settings()
    postprocess("aquecimento hum nova linha", settings)  # compile the cached rules

    started = time.perf_counter()
    result = postprocess(text, settings)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.020
    assert result.count("\n") == 100
    assert "nova linha" not in result
    assert " hum " not in result


# --- Final period ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("vamos subir o deploy", "vamos subir o deploy."),
        ("são 3", "são 3."),
        ("já terminou.", "já terminou."),
        ("tudo certo?", "tudo certo?"),
        ("que legal!", "que legal!"),
        ("e aí…", "e aí…"),
        ("veja o item (dois)", "veja o item (dois)"),
        ("tchau nova linha", "tchau\n"),
        ("", ""),
        ("hum", ""),
    ],
)
def test_final_period_is_added_only_after_a_letter_or_digit(text: str, expected: str) -> None:
    assert postprocess(text, Settings()) == expected


def test_final_period_is_idempotent() -> None:
    once = postprocess("sem ponto", Settings())
    assert postprocess(once, Settings()) == once == "sem ponto."


def test_final_period_follows_its_switch_and_the_cleanup_layer() -> None:
    assert postprocess("texto", Settings(cleanup=CleanupSettings(final_period=False))) == "texto"
    assert postprocess("texto", Settings(cleanup=CleanupSettings(enabled=False))) == "texto"
