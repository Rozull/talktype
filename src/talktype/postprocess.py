"""Post-processing layers: cleanup rules, then spoken commands.

Pure functions with no I/O. The order is fixed (cleanup, then commands; the AI rewrite runs
afterwards in `delivery.py`), each layer honours its own `enabled` switch, no substitution is
applied to the output of another, and running the layers twice changes nothing more.

Cleanup, in this order:
- **Substitutions**: the user's ``from → to`` pairs, matched as whole phrases, ignoring case,
  in one left-to-right pass (``A→B`` and ``B→C`` turn "A" into "B", never "C").
- **Hesitations**: each listed token is removed as a whole word, together with one adjacent
  comma or ellipsis (the one after it, otherwise the one before it). The filler "é" is removed
  only when an ellipsis (``...`` or ``…``) follows it immediately. "tipo" and "né" are not in
  the default list and are left to the AI rewrite.
- **Whitespace**: runs of spaces collapse to one, no space before ``,.;:!?…``, one space after
  ``,;!?`` when a letter follows, no spaces around line breaks, and the ends are trimmed. Line
  breaks themselves are kept.
- **Final period**: a period is added when the text ends in a letter or a digit
  (`final_period`). It runs after the commands, so a trailing line break stays as it is.

Commands: whole-phrase, case-insensitive, longest
phrase first, one left-to-right pass. The punctuation right after a phrase is consumed. The
single space before it is consumed when the replacement attaches to the left (it starts with
whitespace or closing punctuation, like a line break), and the single space after it when the
replacement attaches to the right (it ends with whitespace or opening punctuation, like "(").
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from talktype.config import CleanupSettings, CommandsSettings, Settings

_ELLIPSIS = r"(?:\.\.\.|…)"
_PAUSE = rf"(?:,|{_ELLIPSIS})"
_HSPACE = r"[^\S\r\n]"  # whitespace other than line breaks
_COMMAND_PUNCT = r"[,.;:!?…]*"

# Replacement edges that glue to the neighbouring word.
_ATTACH_LEFT = frozenset(",.;:!?…)]}»”’")
_ATTACH_RIGHT = frozenset("([{«“‘¿¡")

_SPACES = re.compile(rf"{_HSPACE}+")
_SPACE_BEFORE_PUNCT = re.compile(rf"{_HSPACE}+(?=[,.;:!?…])")
_MISSING_SPACE = re.compile(r"([,;!?])(?=[^\W\d_])")
_SPACES_AROUND_BREAK = re.compile(rf"{_HSPACE}*(\r?\n){_HSPACE}*")


def _key(phrase: str) -> str:
    """Lookup form of a phrase: lowercase, single spaces."""
    return " ".join(phrase.lower().split())


def _phrase_pattern(phrase: str) -> str:
    """A whole-phrase regex: word edges must not touch other word characters."""
    words = phrase.split()
    body = r"\s+".join(re.escape(word) for word in words)
    head = r"(?<!\w)" if re.match(r"\w", words[0]) else ""
    tail = r"(?!\w)" if re.search(r"\w$", words[-1]) else ""
    return head + body + tail


def _alternation(phrases: Iterable[str]) -> str:
    """Alternatives ordered longest first, so the longest phrase wins at each position."""
    ordered = sorted(set(phrases), key=lambda p: (-len(p), p))
    return "|".join(_phrase_pattern(p) for p in ordered)


def _table(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Phrase key -> replacement; empty phrases are dropped and the first duplicate wins."""
    table: dict[str, str] = {}
    for phrase, replacement in pairs:
        if phrase.strip():
            table.setdefault(_key(phrase), replacement)
    return table


# --------------------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _substitution_rule(
    pairs: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], dict[str, str]] | None:
    table = _table(pairs)
    if not table:
        return None
    return re.compile(_alternation(table), re.IGNORECASE), table


@lru_cache(maxsize=32)
def _hesitation_rule(tokens: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = [_phrase_pattern(t) for t in sorted({_key(t) for t in tokens if t.strip()})]
    alternatives.sort(key=len, reverse=True)
    alternatives.append(rf"(?<!\w)é(?={_ELLIPSIS})")  # "é" only right before an ellipsis
    return re.compile(
        rf"(?P<before>{_PAUSE}{_HSPACE}*)?(?:{'|'.join(alternatives)})"
        rf"(?P<after>{_HSPACE}*{_PAUSE})?",
        re.IGNORECASE,
    )


def _drop_hesitation(match: re.Match[str]) -> str:
    # One adjacent pause goes with the token: the one after it when present (keeping the
    # one before), otherwise the one before it.
    return (match.group("before") or "") if match.group("after") else ""


def _substitute(text: str, s: CleanupSettings) -> str:
    rule = _substitution_rule(tuple((sub.from_, sub.to) for sub in s.substitutions))
    if rule is None:
        return text
    pattern, table = rule
    return pattern.sub(lambda m: table[_key(m.group(0))], text)


def _remove_hesitations(text: str, s: CleanupSettings) -> str:
    return _hesitation_rule(tuple(s.hesitations)).sub(_drop_hesitation, text)


def normalize_whitespace(text: str) -> str:
    """Collapse spaces, fix spacing around punctuation and line breaks, trim the ends."""
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _SPACES_AROUND_BREAK.sub(r"\1", text)
    return text.strip(" ")


def apply_cleanup(text: str, s: CleanupSettings) -> str:
    """Substitutions, then hesitation removal, then whitespace normalization."""
    if not s.enabled:
        return text
    text = _substitute(text, s)
    if s.remove_hesitations:
        text = _remove_hesitations(text, s)
    if s.normalize_whitespace:
        text = normalize_whitespace(text)
    return text


# --------------------------------------------------------------------------------------
# Spoken commands
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _command_rule(
    items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], dict[str, str]] | None:
    table = _table(items)
    if not table:
        return None
    pattern = re.compile(
        rf"(?P<lead>{_HSPACE}?)(?P<phrase>{_alternation(table)})"
        rf"(?P<punct>{_COMMAND_PUNCT})(?P<trail>{_HSPACE}?)",
        re.IGNORECASE,
    )
    return pattern, table


def _command_replacement(match: re.Match[str], table: dict[str, str]) -> str:
    text = table[_key(match.group("phrase"))]
    if not text:  # a command that deletes its phrase leaves one space
        return match.group("trail") if match.group("lead") else ""
    first, last = text[0], text[-1]
    lead = "" if first.isspace() or first in _ATTACH_LEFT else match.group("lead")
    trail = "" if last.isspace() or last in _ATTACH_RIGHT else match.group("trail")
    return lead + text + trail


def apply_commands(text: str, s: CommandsSettings) -> str:
    """Replace spoken commands ("nova linha", …) by their text."""
    if not s.enabled:
        return text
    rule = _command_rule(tuple((item.phrase, item.text) for item in s.items))
    if rule is None:
        return text
    pattern, table = rule
    return pattern.sub(lambda m: _command_replacement(m, table), text)


def ensure_final_period(text: str) -> str:
    """End `text` with a period when it ends in a letter or a digit.

    Text that already ends in punctuation, a closing bracket or quote, or a line break (a
    trailing "nova linha") is left alone, and so is text without any letter or digit.
    """
    body = text.rstrip(" ")
    if body and body[-1].isalnum():
        return body + "."
    return text


def postprocess(text: str, settings: Settings) -> str:
    """Cleanup, then commands, each only when its layer is enabled; then the final period
    (part of cleanup, applied last so it sees the text the commands produced)."""
    text = apply_cleanup(text, settings.cleanup)
    text = apply_commands(text, settings.commands)
    if settings.cleanup.enabled and settings.cleanup.final_period:
        text = ensure_final_period(text)
    return text
