"""Smoke tests for the ctypes Win32 bindings against the real APIs."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import uuid
import wave
import winreg

import pytest

from talktype import win32

pytestmark = pytest.mark.integration


def test_structure_layouts_match_x64_win32() -> None:
    assert ctypes.sizeof(win32.INPUT) == 40
    assert ctypes.sizeof(win32.KBDLLHOOKSTRUCT) == 24
    assert ctypes.sizeof(win32.MSLLHOOKSTRUCT) == 32
    assert ctypes.sizeof(win32.WNDCLASSEXW) == 80
    assert ctypes.sizeof(win32.LASTINPUTINFO) == 8


def test_clock_and_input_info() -> None:
    tick = win32.GetTickCount()
    assert tick > 0
    assert win32.GetTickCount64() >= tick
    assert (tick - win32.GetLastInputInfo()) % 2**32 < 2**31
    assert win32.GetCurrentThreadId() > 0
    assert win32.GetCurrentProcessId() == os.getpid()


def test_named_mutex_reports_already_exists() -> None:
    name = f"Local\\talktype-test-{uuid.uuid4()}"
    first, first_existed = win32.CreateMutexW(name)
    second, second_existed = win32.CreateMutexW(name)
    try:
        assert (first_existed, second_existed) == (False, True)
    finally:
        win32.CloseHandle(second)
        win32.CloseHandle(first)


def test_message_only_window_receives_posted_message() -> None:
    received: list[int] = []

    def wndproc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32.WM_APP_REPLAY:
            received.append(wparam)
            return 0
        return win32.DefWindowProcW(hwnd, msg, wparam, lparam)

    proc = win32.WNDPROC(wndproc)
    class_name = f"talktype-test-{uuid.uuid4()}"
    hwnd = win32.create_message_window(class_name, proc)
    try:
        assert win32.IsWindow(hwnd)
        win32.PostMessageW(hwnd, win32.WM_APP_REPLAY, 42, 0)
        assert win32.pump_pending_messages() is True
        assert received == [42]
    finally:
        win32.destroy_message_window(hwnd, class_name)
    assert not win32.IsWindow(hwnd)


def test_process_token_elevation_matches_shell32() -> None:
    assert win32.current_process_is_elevated() == win32.IsUserAnAdmin()
    assert win32.process_is_elevated(os.getpid()) == win32.current_process_is_elevated()


def test_unicode_inputs_use_surrogate_pairs() -> None:
    inputs = win32.unicode_inputs("a😀")

    assert len(inputs) == 6  # "a" plus two surrogates, each down and up
    scans = [i.ki.wScan for i in inputs]
    assert scans == [0x61, 0x61, 0xD83D, 0xD83D, 0xDE00, 0xDE00]
    assert all(i.ki.dwFlags & win32.KEYEVENTF_UNICODE for i in inputs)
    assert [bool(i.ki.dwFlags & win32.KEYEVENTF_KEYUP) for i in inputs[:2]] == [False, True]


def test_clipboard_format_registration_and_text_bytes() -> None:
    html = win32.RegisterClipboardFormatW(win32.CFSTR_HTML)
    assert html >= 0xC000
    assert win32.GetClipboardFormatNameW(html) == win32.CFSTR_HTML
    assert win32.GetClipboardFormatNameW(win32.CF_UNICODETEXT) is None
    data = win32.unicode_text_bytes("linha 1\r\nlinha 2")
    assert win32.text_from_unicode_bytes(data) == "linha 1\r\nlinha 2"
    handle = win32.global_from_bytes(data)
    try:
        assert win32.global_to_bytes(handle)[: len(data)] == data
    finally:
        win32.GlobalFree(handle)


def test_registry_round_trip_under_a_test_key() -> None:
    parent = r"Software\talktype-tests"
    subkey = rf"{parent}\{uuid.uuid4()}"
    try:
        assert win32.reg_read_str(win32.HKEY_CURRENT_USER, subkey, "valor") is None
        win32.reg_write_str(win32.HKEY_CURRENT_USER, subkey, "valor", "olá")
        assert win32.reg_read_str(win32.HKEY_CURRENT_USER, subkey, "valor") == "olá"
        assert win32.reg_delete_value(win32.HKEY_CURRENT_USER, subkey, "valor") is True
        assert win32.reg_delete_value(win32.HKEY_CURRENT_USER, subkey, "valor") is False
    finally:
        winreg.DeleteKey(win32.HKEY_CURRENT_USER, subkey)
        with contextlib.suppress(OSError):  # still holds keys of a concurrent run
            winreg.DeleteKey(win32.HKEY_CURRENT_USER, parent)


def test_win32_errors_surface_as_oserror() -> None:
    with pytest.raises(OSError) as info:
        win32.GetClassNameW(0)
    assert info.value.winerror == win32.ERROR_INVALID_WINDOW_HANDLE  # type: ignore[attr-defined]


def test_play_sound_from_memory() -> None:
    image = io.BytesIO()
    with wave.open(image, "wb") as target:
        target.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        target.writeframes(b"\0\0" * 1600)  # 0.1 s of silence
    sound = image.getvalue()
    flags = win32.SND_MEMORY | win32.SND_ASYNC | win32.SND_NODEFAULT

    assert win32.PlaySoundW(b"not a wav", flags) is False
    played = win32.PlaySoundW(sound, flags)
    win32.PlaySoundW(None, 0)
    if not played:  # a machine without an audio output device
        pytest.skip("no audio output device")
