"""`Tray`: the system-tray icon and menu.

The icon shows the app state through a distinct shape and a tooltip, not color alone.
The menu holds the model status, the quick toggles (bound to the live settings and saved
through `ConfigManager.set_value`), the microphone and history submenus, and the config
actions. The tray owns no app logic: every other action is a signal that `App` connects.
Qt GUI thread only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFileDialog, QMenu, QSystemTrayIcon

from talktype.asr_device import Device
from talktype.config import ConfigManager, resolve_preview
from talktype.history import HistoryEntry, HistoryStatus, label
from talktype.strings import Msg

ICON_SIZE = 64


class TrayState(StrEnum):
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    PAUSED = "paused"
    ERROR = "error"


class IconShape(StrEnum):
    RING = "ring"  # loading or downloading
    MIC = "mic"  # ready
    PAUSE = "pause"  # paused
    ALERT = "alert"  # error


STATE_SHAPES = {
    TrayState.DOWNLOADING: IconShape.RING,
    TrayState.LOADING: IconShape.RING,
    TrayState.READY: IconShape.MIC,
    TrayState.PAUSED: IconShape.PAUSE,
    TrayState.ERROR: IconShape.ALERT,
}

# Toggle keys (the `set_value` keys the tray writes).
KEY_MODE = "injection.mode"
KEY_PREVIEW = "overlay.preview"
KEY_CLEANUP = "cleanup.enabled"
KEY_COMMANDS = "commands.enabled"
KEY_REWRITE = "llm.enabled"
KEY_MICROPHONE = "recording.microphone"


# --------------------------------------------------------------------------------------
# Icons
# --------------------------------------------------------------------------------------


def _mic_renderer() -> QSvgRenderer:
    data = resources.files("talktype").joinpath("assets", "mic.svg").read_bytes()
    return QSvgRenderer(data)


def render_icon(shape: IconShape, size: int = ICON_SIZE) -> QPixmap:
    """The tray icon for `shape`: each state has its own outline, not just a color."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    full = QRectF(0, 0, size, size)
    inner = full.adjusted(size * 0.22, size * 0.18, -size * 0.22, -size * 0.18)
    if shape is IconShape.MIC:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2e7d32"))
        painter.drawEllipse(full.adjusted(1, 1, -1, -1))
        _mic_renderer().render(painter, inner)
    elif shape is IconShape.RING:
        pen = QPen(QColor("#78909c"), size * 0.12)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        margin = size * 0.08
        painter.drawArc(full.adjusted(margin, margin, -margin, -margin), 90 * 16, -270 * 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#78909c"))
        painter.drawEllipse(QPointF(size / 2, size / 2), size * 0.14, size * 0.14)
    elif shape is IconShape.PAUSE:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f9a825"))
        painter.drawRoundedRect(full.adjusted(2, 2, -2, -2), size * 0.18, size * 0.18)
        painter.setBrush(QColor("#212121"))
        bar_w, bar_h = size * 0.14, size * 0.5
        top = (size - bar_h) / 2
        painter.drawRect(QRectF(size * 0.30, top, bar_w, bar_h))
        painter.drawRect(QRectF(size * 0.56, top, bar_w, bar_h))
    else:
        path = QPainterPath()
        path.moveTo(size / 2, 2)
        path.lineTo(size - 2, size - 4)
        path.lineTo(2, size - 4)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#c62828"))
        painter.drawPath(path)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(round(size * 0.5))
        painter.setFont(font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(full.adjusted(0, size * 0.18, 0, 0), Qt.AlignmentFlag.AlignCenter, "!")
    painter.end()
    return pixmap


def state_text(state: TrayState, *, reason: str = "", percent: int = 0) -> str:
    if state is TrayState.DOWNLOADING:
        return Msg.TRAY_DOWNLOADING.format(percent=percent)
    if state is TrayState.LOADING:
        return Msg.TRAY_LOADING.text
    if state is TrayState.READY:
        return Msg.TRAY_READY.text
    if state is TrayState.PAUSED:
        return Msg.TRAY_PAUSED.text
    return Msg.TRAY_ERROR.format(reason=reason)


def device_label(device: str | Device) -> str:
    kind = device.kind if isinstance(device, Device) else device
    return "GPU" if kind == "cuda" else "CPU"


def history_label(entry: HistoryEntry) -> str:
    text = label(entry).replace("&", "&&")  # "&" would become a menu mnemonic
    if entry.status is HistoryStatus.FAILED_INSERT:
        return f"{text} ({Msg.HISTORY_NOT_INSERTED.text})"
    return text


def _default_copy(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


def _default_save_dialog() -> Path | None:
    name, _ = QFileDialog.getSaveFileName(
        None,
        Msg.DIALOG_EXPORT_TITLE.text,
        str(Path.home() / "talktype-config.toml"),
        Msg.DIALOG_TOML_FILTER.text,
    )
    return Path(name) if name else None


def _default_open_dialog() -> Path | None:
    name, _ = QFileDialog.getOpenFileName(
        None, Msg.DIALOG_IMPORT_TITLE.text, str(Path.home()), Msg.DIALOG_TOML_FILTER.text
    )
    return Path(name) if name else None


# --------------------------------------------------------------------------------------
# Tray
# --------------------------------------------------------------------------------------


class Tray(QSystemTrayIcon):
    setting_changed = Signal(str, object, bool)  # key, value, saved to the file
    pause_requested = Signal(bool)  # True = pause
    autostart_requested = Signal(bool)
    history_copied = Signal(str)
    history_insert_requested = Signal(str)
    history_clear_requested = Signal()
    open_config_requested = Signal()
    reload_config_requested = Signal()
    open_config_folder_requested = Signal()
    export_requested = Signal(object)  # Path chosen in the save dialog
    import_requested = Signal(object)  # Path chosen in the open dialog
    retry_download_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        config: ConfigManager,
        *,
        history: Callable[[], Sequence[HistoryEntry]],
        devices: Callable[[], Sequence[str]],
        autostart_enabled: Callable[[], bool],
        toggle_guard: Callable[[str, Any], bool] | None = None,
        copy_text: Callable[[str], None] | None = None,
        save_dialog: Callable[[], Path | None] | None = None,
        open_dialog: Callable[[], Path | None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._history = history
        self._devices = devices
        self._autostart_enabled = autostart_enabled
        self._guard = toggle_guard
        self._copy = copy_text or _default_copy
        self._save_dialog = save_dialog or _default_save_dialog
        self._open_dialog = open_dialog or _default_open_dialog
        self._icons = {shape: QIcon(render_icon(shape)) for shape in IconShape}
        self._device: str = "cpu"
        self._history_menus: list[QMenu] = []
        self.state = TrayState.LOADING
        self.icon_shape = IconShape.RING

        self.menu = QMenu()
        self._build_menu()
        self.setContextMenu(self.menu)
        self.menu.aboutToShow.connect(self.refresh)
        self.set_status(config.settings.asr.model, "cpu")
        self.set_state(TrayState.LOADING)
        self.refresh()

    # -- menu construction ---------------------------------------------------------------

    def _action(self, text: str, slot: Callable[[], None], *, checkable: bool = False) -> QAction:
        action = QAction(text, self.menu)
        action.setCheckable(checkable)
        action.triggered.connect(lambda _checked=False: slot())
        return action

    def _build_menu(self) -> None:
        menu = self.menu
        self.status_action = menu.addAction("")
        self.status_action.setEnabled(False)
        self.state_action = menu.addAction("")
        self.state_action.setEnabled(False)
        self.retry_action = self._action(
            Msg.MENU_RETRY_DOWNLOAD.text, self.retry_download_requested.emit
        )
        self.retry_action.setVisible(False)
        menu.addAction(self.retry_action)
        menu.addSeparator()

        self._mode_group = QActionGroup(menu)
        self._mode_group.setExclusive(True)
        self.mode_paste_action = self._action(
            Msg.MENU_MODE_PASTE.text, partial(self._toggle, KEY_MODE, "paste"), checkable=True
        )
        self.mode_type_action = self._action(
            Msg.MENU_MODE_TYPE.text, partial(self._toggle, KEY_MODE, "type"), checkable=True
        )
        for action in (self.mode_paste_action, self.mode_type_action):
            self._mode_group.addAction(action)
            menu.addAction(action)

        self.preview_action = self._action(
            Msg.MENU_PREVIEW.text, self._toggle_preview, checkable=True
        )
        self.cleanup_action = self._action(
            Msg.MENU_CLEANUP.text, partial(self._toggle_bool, KEY_CLEANUP), checkable=True
        )
        self.commands_action = self._action(
            Msg.MENU_COMMANDS.text, partial(self._toggle_bool, KEY_COMMANDS), checkable=True
        )
        self.rewrite_action = self._action(
            Msg.MENU_REWRITE.text, partial(self._toggle_bool, KEY_REWRITE), checkable=True
        )
        for action in (
            self.preview_action,
            self.cleanup_action,
            self.commands_action,
            self.rewrite_action,
        ):
            menu.addAction(action)
        menu.addSeparator()

        self.pause_action = self._action(Msg.MENU_PAUSE.text, self._toggle_pause)
        self.autostart_action = self._action(
            Msg.MENU_AUTOSTART.text, self._toggle_autostart, checkable=True
        )
        menu.addAction(self.pause_action)
        menu.addAction(self.autostart_action)
        menu.addSeparator()

        self.microphone_menu = QMenu(Msg.MENU_MICROPHONE.text, menu)
        self.microphone_menu.aboutToShow.connect(self.rebuild_microphones)
        self._mic_group = QActionGroup(self.microphone_menu)
        self._mic_group.setExclusive(True)
        menu.addMenu(self.microphone_menu)
        self.history_menu = QMenu(Msg.MENU_HISTORY.text, menu)
        self.history_menu.aboutToShow.connect(self.rebuild_history)
        menu.addMenu(self.history_menu)
        menu.addSeparator()

        for text, slot in (
            (Msg.MENU_OPEN_CONFIG.text, self.open_config_requested.emit),
            (Msg.MENU_RELOAD_CONFIG.text, self.reload_config_requested.emit),
            (Msg.MENU_OPEN_CONFIG_FOLDER.text, self.open_config_folder_requested.emit),
            (Msg.MENU_EXPORT_CONFIG.text, self._export),
            (Msg.MENU_IMPORT_CONFIG.text, self._import),
        ):
            menu.addAction(self._action(text, slot))
        menu.addSeparator()
        self.quit_action = self._action(Msg.MENU_QUIT.text, self.quit_requested.emit)
        menu.addAction(self.quit_action)
        self.rebuild_microphones()
        self.rebuild_history()

    # -- state and status ----------------------------------------------------------------

    def set_state(self, state: TrayState, *, reason: str = "", percent: int = 0) -> None:
        self.state = state
        self.icon_shape = STATE_SHAPES[state]
        text = state_text(state, reason=reason, percent=percent)
        self.setIcon(self._icons[self.icon_shape])
        self.setToolTip(f"talktype — {text}")
        self.state_action.setText(text)
        paused = state is TrayState.PAUSED
        self.pause_action.setText(Msg.MENU_RESUME.text if paused else Msg.MENU_PAUSE.text)

    def set_status(self, model: str, device: str | Device) -> None:
        """The "Modelo: <model> · GPU/CPU" line; the device also drives the preview check."""
        self._device = device.kind if isinstance(device, Device) else device
        self.status_action.setText(Msg.TRAY_STATUS.format(model=model, device=device_label(device)))
        self.refresh()

    @property
    def status_text(self) -> str:
        return self.status_action.text()

    def set_retry_visible(self, visible: bool) -> None:
        self.retry_action.setVisible(visible)

    def notify(self, text: str, *, error: bool = False) -> None:
        """A balloon notice from the tray icon."""
        if self.isVisible() and QSystemTrayIcon.supportsMessages():
            icon = (
                QSystemTrayIcon.MessageIcon.Warning
                if error
                else QSystemTrayIcon.MessageIcon.Information
            )
            self.showMessage("talktype", text, icon, 6000)

    def refresh(self) -> None:
        """Sync every check state with the live settings and the real autostart state."""
        settings = self._config.settings
        self.mode_paste_action.setChecked(settings.injection.mode == "paste")
        self.mode_type_action.setChecked(settings.injection.mode == "type")
        self.preview_action.setChecked(resolve_preview(settings.overlay.preview, self._device))
        self.cleanup_action.setChecked(settings.cleanup.enabled)
        self.commands_action.setChecked(settings.commands.enabled)
        self.rewrite_action.setChecked(settings.llm.enabled)
        self.autostart_action.setChecked(self._autostart_enabled())

    # -- toggles -------------------------------------------------------------------------

    def _toggle(self, key: str, value: Any) -> None:
        if self._guard is not None and not self._guard(key, value):
            self.refresh()
            return
        result = self._config.set_value(key, value)
        self.refresh()
        self.setting_changed.emit(key, value, result.saved)

    def _toggle_bool(self, key: str) -> None:
        table, name = key.split(".")
        current = bool(getattr(getattr(self._config.settings, table), name))
        self._toggle(key, not current)

    def _toggle_preview(self) -> None:
        shown = resolve_preview(self._config.settings.overlay.preview, self._device)
        self._toggle(KEY_PREVIEW, "off" if shown else "on")

    def _toggle_pause(self) -> None:
        self.pause_requested.emit(self.state is not TrayState.PAUSED)

    def _toggle_autostart(self) -> None:
        self.autostart_requested.emit(not self._autostart_enabled())
        self.autostart_action.setChecked(self._autostart_enabled())

    # -- submenus ------------------------------------------------------------------------

    def rebuild_microphones(self) -> None:
        """List "Padrão do Windows" plus every input device (refreshed on each open)."""
        menu = self.microphone_menu
        for action in self._mic_group.actions():
            self._mic_group.removeAction(action)
        menu.clear()
        pinned = self._config.settings.recording.microphone
        names = list(dict.fromkeys(self._devices()))
        if pinned and pinned not in names:
            names.append(pinned)
        for value, text in [("", Msg.MIC_DEFAULT.text), *((n, n) for n in names)]:
            action = QAction(text.replace("&", "&&"), menu)
            action.setCheckable(True)
            action.setChecked(value == pinned)
            action.triggered.connect(partial(self._select_microphone, value))
            self._mic_group.addAction(action)
            menu.addAction(action)

    def _select_microphone(self, name: str, _checked: bool = False) -> None:
        self._toggle(KEY_MICROPHONE, name)

    def rebuild_history(self) -> None:
        """Rebuild the history submenu from the store (on every open)."""
        menu = self.history_menu
        menu.clear()
        for submenu in self._history_menus:
            submenu.deleteLater()
        self._history_menus = []
        entries = list(self._history())
        if not entries:
            empty = menu.addAction(Msg.HISTORY_EMPTY.text)
            empty.setEnabled(False)
            return
        for entry in entries:
            submenu = QMenu(history_label(entry), menu)
            copy = submenu.addAction(Msg.MENU_HISTORY_COPY.text)
            copy.triggered.connect(partial(self._copy_entry, entry.text))
            insert = submenu.addAction(Msg.MENU_HISTORY_INSERT.text)
            insert.triggered.connect(partial(self._insert_entry, entry.text))
            menu.addMenu(submenu)
            self._history_menus.append(submenu)
        menu.addSeparator()
        clear = menu.addAction(Msg.MENU_HISTORY_CLEAR.text)
        clear.triggered.connect(lambda _checked=False: self.history_clear_requested.emit())

    def _copy_entry(self, text: str, _checked: bool = False) -> None:
        self._copy(text)
        self.history_copied.emit(text)

    def _insert_entry(self, text: str, _checked: bool = False) -> None:
        self.history_insert_requested.emit(text)

    # -- dialogs -------------------------------------------------------------------------

    def _export(self) -> None:
        path = self._save_dialog()
        if path is not None:
            self.export_requested.emit(path)

    def _import(self) -> None:
        path = self._open_dialog()
        if path is not None:
            self.import_requested.emit(path)

    # -- lookup (tests and e2e) ----------------------------------------------------------

    def find_action(self, text: str) -> QAction | None:
        """The top-level menu action (or submenu action) whose text is `text`."""
        for action in self.menu.actions():
            if action.text() == text:
                return action
        return None
