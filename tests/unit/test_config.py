"""Config validation, fallback, editing, export and import (UT-034 … UT-057 and related)."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pydantic
import pytest

from talktype import config
from talktype.config import (
    DEFAULT_LLM_INSTRUCTION,
    DEFAULT_REPASTE_HOTKEY,
    ConfigError,
    ConfigManager,
    SaveResult,
    Settings,
    Substitution,
    load_or_create,
    parse_hotkey,
    resolve_preview,
    set_value,
    template_bytes,
)
from talktype.strings import Msg
from talktype.win32 import MOD_ALT, MOD_CONTROL, MOD_SHIFT
from tests.conftest import FakeClock

pytestmark = pytest.mark.unit


def write_config(home: Path, text: str) -> Path:
    path = home / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def make_manager(home: Path, clock: FakeClock, text: str | None = None) -> ConfigManager:
    if text is not None:
        write_config(home, text)
    return ConfigManager(home, clock=clock)


def reload_with(manager: ConfigManager, text: str) -> config.ValidationReport:
    write_config(manager.paths.home, text)
    return manager.reload()


def error_keys(report: config.ValidationReport) -> list[str]:
    return [e.key for e in report.errors]


# --- Load and template ------------------------------------------------------------------


def test_ut034_first_run_creates_config_from_template(home: Path, clock: FakeClock) -> None:
    settings = load_or_create(home, clock=clock)

    assert (home / "config.toml").read_bytes() == template_bytes()
    assert settings == Settings()
    assert settings.asr.language == "pt"
    assert settings.asr.model == "large-v3-turbo"
    assert settings.trigger.hold_threshold_ms == 1000


def test_ut035_template_documents_every_leaf_key() -> None:
    lines = template_bytes().decode("utf-8").splitlines()
    table = ""
    documented: dict[str, bool] = {}
    previous = ""
    for line in lines:
        stripped = line.strip()
        header = re.fullmatch(r"\[([a-z_]+)\]", stripped)
        if header:
            table = header.group(1)
        else:
            key = re.match(r"([a-z_]+)\s*=", stripped)
            if key and table and not line.startswith((" ", "\t")):
                documented[f"{table}.{key.group(1)}"] = previous.startswith("#")
        if stripped:
            previous = stripped

    leaves = config.leaf_keys()
    assert len(leaves) >= 40
    missing = [k for k in leaves if k not in documented]
    uncommented = [k for k in leaves if not documented.get(k, False)]
    assert missing == []
    assert uncommented == []


# --- Reload: syntax, per-field fallback, unknown keys ---------------------------------------


def test_ut036_syntax_error_rejects_reload_with_line_number(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[injection]\nmode = "type"\n')
    before = manager.settings
    lines = [f"# comentário {n}" for n in range(1, 11)] + ["[injection]", 'mode = "paste']

    report = reload_with(manager, "\n".join(lines) + "\n")

    assert report.fatal is True
    assert len(report.errors) == 1
    error = report.errors[0]
    assert error.key == str(home / "config.toml")
    assert error.reason.startswith("linha 12: ")
    assert manager.settings is before
    assert manager.settings.injection.mode == "type"


def test_ut037_invalid_key_keeps_previous_value_and_applies_others(
    home: Path, clock: FakeClock
) -> None:
    manager = make_manager(home, clock, '[trigger]\nkey = "right_alt"\n')

    report = reload_with(manager, '[trigger]\nkey = "banana"\n[injection]\nmode = "type"\n')

    assert manager.settings.trigger.key == "right_alt"
    assert manager.settings.injection.mode == "type"
    assert error_keys(report) == ["trigger.key"]
    assert report.fatal is False


@pytest.mark.parametrize("bad", ["0", "-5", '"x"'])
def test_ut038_max_seconds_invalid_values_fall_back(home: Path, clock: FakeClock, bad: str) -> None:
    manager = make_manager(home, clock, "[recording]\nmax_seconds = 120\n")

    report = reload_with(manager, f"[recording]\nmax_seconds = {bad}\n")

    assert manager.settings.recording.max_seconds == 120
    assert error_keys(report) == ["recording.max_seconds"]


def test_ut038_max_seconds_boundaries(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, "[recording]\nmax_seconds = 1800\n")
    assert manager.settings.recording.max_seconds == 1800
    assert manager.last_report.errors == []
    assert manager.last_report.warnings == []

    report = reload_with(manager, "[recording]\nmax_seconds = 1801\n")

    assert manager.settings.recording.max_seconds == 1800
    assert report.errors == []
    assert report.warnings == ["recording.max_seconds limitado a 1800"]


def test_ut039_unknown_key_is_a_warning(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(
        manager, '[trigger]\nhold_treshold_ms = 900\nkey = "f13"\n[injection]\nmode = "type"\n'
    )

    assert report.warnings == ["chave desconhecida: trigger.hold_treshold_ms"]
    assert report.errors == []
    assert manager.settings.trigger.key == "f13"
    assert manager.settings.injection.mode == "type"
    assert manager.settings.trigger.hold_threshold_ms == 1000


def test_ut040_remote_llm_endpoint_disables_the_rewrite(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(
        manager, '[llm]\nenabled = true\nendpoint = "http://example.com:11434"\nmodel = "m"\n'
    )

    assert report.errors == [ConfigError("llm.endpoint", Msg.LLM_ENDPOINT_NOT_LOCAL.text)]
    assert manager.settings.llm.enabled is False
    assert manager.settings.llm.endpoint == "http://127.0.0.1:11434"
    assert manager.settings.llm.model == "m"


def test_ut040_local_llm_endpoint_is_accepted(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(manager, '[llm]\nenabled = true\nendpoint = "http://localhost:1234/v1"\n')

    assert report.errors == []
    assert manager.settings.llm.enabled is True
    assert manager.settings.llm.endpoint == "http://localhost:1234/v1"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:1234", True),
        ("http://[::1]:11434", True),
        ("http://127.5.5.5", True),
        ("http://10.0.0.2", False),
        ("http://example.com", False),
        ("http://0.0.0.0:11434", False),
        ("ftp://127.0.0.1", False),
    ],
)
def test_ut041_is_loopback_endpoint(url: str, expected: bool) -> None:
    assert config.is_loopback_endpoint(url) is expected


# --- Editing ------------------------------------------------------------------------------


def test_ut042_set_value_rewrites_only_that_value(home: Path) -> None:
    original = (
        template_bytes()
        .decode("utf-8")
        .replace("vocabulary = []", 'vocabulary = ["GitHub", "CI"]')
        .replace("[injection]\n", "[injection]\n# meu comentário: prefiro colar\n")
    )
    path = write_config(home, original)
    original_bytes = path.read_bytes()

    result = set_value(path, "injection.mode", "type")

    assert result == SaveResult(saved=True, path=path)
    expected = original_bytes.replace(b'mode = "paste"', b'mode = "type"', 1)
    assert path.read_bytes() == expected


def test_ut042_set_value_keeps_inline_comment(home: Path) -> None:
    path = write_config(home, '[injection]\nmode = "paste"  # escolha minha\n')

    set_value(path, "injection.mode", "type")

    assert path.read_text(encoding="utf-8") == '[injection]\nmode = "type"  # escolha minha\n'


def test_ut043_read_only_file_keeps_change_in_memory(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)
    path = home / "config.toml"
    before = path.read_bytes()
    path.chmod(stat.S_IREAD)
    try:
        result = manager.set_value("injection.mode", "type")
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert result.saved is False
    assert result.reason == "read_only"
    assert manager.settings.injection.mode == "type"
    assert path.read_bytes() == before


def test_ut044_fresh_file_is_reread_after_250ms(home: Path) -> None:
    path = write_config(home, '[injection]\nmode = "paste"\n')
    stamp = 1_000_000.0
    os.utime(path, (stamp, stamp))
    clock = FakeClock(stamp + 3600)
    manager = ConfigManager(home, clock=clock)
    assert manager.settings.injection.mode == "paste"

    def finish_save() -> None:
        path.write_text('[injection]\nmode = "type"\n', encoding="utf-8")
        os.utime(path, (stamp, stamp))

    path.write_text('[injection]\nmode = "pas', encoding="utf-8")  # half-written
    os.utime(path, (stamp, stamp))
    clock.set(stamp + 0.100)
    clock.on_sleep = finish_save

    report = manager.reload()

    assert clock.sleeps == [0.25]
    assert report.fatal is False
    assert manager.settings.injection.mode == "type"


def test_ut045_reload_of_unchanged_file_is_a_no_op(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)
    first = reload_with(manager, '[asr]\nmodel = "small"\n')
    assert first.changed is True
    assert first.model_changed is True

    second = manager.reload()

    assert second.changed is False
    assert second.model_changed is False


def test_ut046_deleted_file_is_recreated_with_defaults(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[injection]\nmode = "type"\n')
    (home / "config.toml").unlink()

    report = manager.reload()

    assert (home / "config.toml").read_bytes() == template_bytes()
    assert manager.settings == Settings()
    assert report.ok


# --- Export and import --------------------------------------------------------------------


def backups(home: Path) -> list[Path]:
    return sorted(home.glob("config.backup-*.toml"))


def test_ut047_export_is_byte_identical(home: Path, clock: FakeClock, tmp_path: Path) -> None:
    manager = make_manager(home, clock, '# meu\n[injection]\nmode = "type"\n')
    dest = tmp_path / "exportado.toml"

    result = manager.export_config(dest)

    assert result.saved is True
    assert dest.read_bytes() == (home / "config.toml").read_bytes()


def test_ut048_import_valid_file_backs_up_and_applies(
    home: Path, clock: FakeClock, tmp_path: Path
) -> None:
    manager = make_manager(home, clock, '[injection]\nmode = "paste"\n')
    old = (home / "config.toml").read_bytes()
    src = tmp_path / "outra.toml"
    src.write_text('[injection]\nmode = "type"\n[asr]\nvocabulary = ["PR"]\n', encoding="utf-8")

    result = manager.import_config(src)

    assert result.applied is True
    [backup] = backups(home)
    assert re.fullmatch(r"config\.backup-\d{8}-\d{6}\.toml", backup.name)
    assert result.backup == backup
    assert backup.read_bytes() == old
    assert (home / "config.toml").read_bytes() == src.read_bytes()
    assert manager.settings.injection.mode == "type"
    assert manager.settings.asr.vocabulary == ["PR"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("", "empty"),
        ("[foo]\nbar = 1\n", "not_talktype"),
        ('[injection]\nmode = "type\n', "syntax"),
    ],
)
def test_ut049_invalid_import_changes_nothing(
    home: Path, clock: FakeClock, tmp_path: Path, content: str, reason: str
) -> None:
    manager = make_manager(home, clock, '[injection]\nmode = "paste"\n')
    before = (home / "config.toml").read_bytes()
    src = tmp_path / "ruim.toml"
    src.write_text(content, encoding="utf-8")

    result = manager.import_config(src)

    assert result.applied is False
    assert result.reason == reason
    assert result.message
    assert (home / "config.toml").read_bytes() == before
    assert backups(home) == []
    assert manager.settings.injection.mode == "paste"


def test_ut050_importing_twice_gives_two_backups(
    home: Path, clock: FakeClock, tmp_path: Path
) -> None:
    manager = make_manager(home, clock)
    src = tmp_path / "outra.toml"
    src.write_text('[injection]\nmode = "type"\n', encoding="utf-8")

    first = manager.import_config(src)
    clock.advance(1)
    second = manager.import_config(src)

    assert first.applied and second.applied
    assert first.backup != second.backup
    assert len(backups(home)) == 2
    assert (home / "config.toml").read_bytes() == src.read_bytes()


# --- Field rules ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preview", "device", "expected"),
    [("auto", "cuda", True), ("auto", "cpu", False), ("on", "cpu", True), ("off", "cuda", False)],
)
def test_ut051_resolve_preview(preview: str, device: str, expected: bool) -> None:
    assert resolve_preview(preview, device) is expected


def test_ut052_unknown_language_falls_back_to_pt(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(manager, '[asr]\nlanguage = "xx"\n')

    assert manager.settings.asr.language == "pt"
    assert error_keys(report) == ["asr.language"]


def test_ut053_vocabulary_is_normalized(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[asr]\nvocabulary = ["PR", "", "  ", "PR", "Docker"]\n')

    assert manager.settings.asr.vocabulary == ["PR", "Docker"]
    assert manager.last_report.errors == []


def test_ut054_parse_hotkey() -> None:
    assert parse_hotkey("ctrl+alt+shift+v") == (MOD_CONTROL | MOD_ALT | MOD_SHIFT, 0x56)
    for bad in ["ctrl+banana", "v", "ctrl+", "ctrl+ctrl+v", "shift"]:
        with pytest.raises(ValueError):
            parse_hotkey(bad)


def test_ut054_invalid_repaste_hotkey_falls_back(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(manager, '[history]\nrepaste_hotkey = "ctrl+banana"\n')

    assert manager.settings.history.repaste_hotkey == DEFAULT_REPASTE_HOTKEY
    assert error_keys(report) == ["history.repaste_hotkey"]


def test_ut055_empty_instruction_uses_preset(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[llm]\ninstruction = ""\n')

    assert manager.settings.llm.instruction == DEFAULT_LLM_INSTRUCTION
    assert "Não mude o sentido" in manager.settings.llm.instruction


def test_ut056_microphone_name_is_kept_verbatim(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[recording]\nmicrophone = "Headset (Jabra)"\n')

    assert manager.settings.recording.microphone == "Headset (Jabra)"


def test_ut057_settings_are_frozen(home: Path, clock: FakeClock) -> None:
    settings = make_manager(home, clock).settings

    with pytest.raises(pydantic.ValidationError):
        settings.trigger.key = "f13"  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        settings.debug = config.DebugSettings(log_text=True)  # type: ignore[misc]


def test_ut063_empty_substitution_is_rejected_others_apply(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(
        manager,
        "[cleanup]\nsubstitutions = [\n"
        '  { from = "", to = "x" },\n'
        '  { from = "pê erre", to = "PR" },\n'
        "]\n",
    )

    assert error_keys(report) == ["cleanup.substitutions[0]"]
    assert manager.settings.cleanup.substitutions == [
        Substitution.model_validate({"from": "pê erre", "to": "PR"})
    ]


def test_ut099_unknown_device_falls_back_to_auto(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(manager, '[asr]\ndevice = "gpu"\n')

    assert manager.settings.asr.device == "auto"
    assert error_keys(report) == ["asr.device"]


def test_ut220_invalid_preview_keeps_previous(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[overlay]\npreview = "off"\n')

    report = reload_with(manager, '[overlay]\npreview = "sim"\n')

    assert manager.settings.overlay.preview == "off"
    assert error_keys(report) == ["overlay.preview"]


def test_ut229_unsupported_model_keeps_current(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock, '[asr]\nmodel = "small"\n')

    report = reload_with(manager, '[asr]\nmodel = "whisper-xl"\n')

    assert manager.settings.asr.model == "small"
    assert error_keys(report) == ["asr.model"]
    assert report.model_changed is False


def test_ut237_toggle_recreates_missing_config(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)
    path = home / "config.toml"
    path.unlink()

    result = manager.set_value("injection.mode", "type")

    assert result.saved is True
    assert path.read_bytes() == template_bytes().replace(b'mode = "paste"', b'mode = "type"', 1)
    assert manager.settings.injection.mode == "type"
    manager.reload()
    assert manager.settings.injection.mode == "type"


def test_ut238_unreadable_config_keeps_last_valid(
    home: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(home, clock, '[injection]\nmode = "type"\n')
    before = manager.settings
    path = home / "config.toml"
    real_read_bytes = Path.read_bytes

    def denied(self: Path) -> bytes:
        if self == path:
            raise PermissionError(13, "Acesso negado", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", denied)

    report = manager.reload()

    assert manager.settings is before
    assert report.fatal is True
    assert report.errors == [ConfigError(str(path), Msg.CONFIG_UNREADABLE.format(path=path))]


def test_report_notice_lists_rejected_keys(home: Path, clock: FakeClock) -> None:
    manager = make_manager(home, clock)

    report = reload_with(manager, '[trigger]\nkey = "banana"\n')

    assert "trigger.key" in report.notice()
    assert report.notice().startswith("configuração recarregada")
