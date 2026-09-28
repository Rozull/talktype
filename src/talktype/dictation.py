"""Dictation data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from talktype.config import Settings


class DictationMode(StrEnum):
    HOLD = "hold"
    HANDSFREE = "handsfree"


class DictationState(StrEnum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    POSTPROCESSING = "postprocessing"
    INSERTING = "inserting"
    INSERTED = "inserted"
    FAILED_INSERT = "failed_insert"
    CANCELLED = "cancelled"
    DISCARDED = "discarded"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class Dictation:
    id: int  # monotonically increasing per process
    mode: DictationMode
    settings: Settings  # immutable snapshot taken at commit
    committed_at_ms: int
    state: DictationState = DictationState.RECORDING


@dataclass(frozen=True, slots=True)
class AsrOutcome:
    dictation_id: int
    kind: Literal["text", "no_speech", "error"]
    text: str = ""
    speech_s: float = 0.0
    asr_ms: int = 0
    used_cpu_fallback: bool = False
