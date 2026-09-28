"""`DeliveryThread` ordering, cancellation, snapshots and failure handling.

UT-134 … UT-143, UT-224, UT-225, UT-228.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from talktype import delivery as delivery_module
from talktype.config import CleanupSettings, InjectionSettings, LlmSettings, Settings
from talktype.delivery import DeliveryOutcome, DeliveryStatus, DeliveryThread
from talktype.dictation import AsrOutcome, Dictation, DictationMode
from talktype.history import HistoryStatus, HistoryStore
from talktype.injector import InjectResult, InjectStatus
from talktype.llm import RewriteSkipped
from talktype.strings import Msg
from tests.fakes import FakeInjector, FakeRewriter

pytestmark = pytest.mark.unit

# These tests compare exact texts through the pipeline; the final period has its own tests
# in test_postprocess.py.
NO_PERIOD = CleanupSettings(final_period=False)
BASE = Settings(cleanup=NO_PERIOD)
LLM_ON = Settings(cleanup=NO_PERIOD, llm=LlmSettings(enabled=True))
TYPE_MODE = Settings(cleanup=NO_PERIOD, injection=InjectionSettings(mode="type"))


class Outcomes:
    """`on_outcome` that records outcomes (and their thread) and lets the test wait."""

    def __init__(self) -> None:
        self.items: list[DeliveryOutcome] = []
        self.threads: list[int] = []
        self.events: list[str] = []
        self._cond = threading.Condition()

    def __call__(self, outcome: DeliveryOutcome) -> None:
        with self._cond:
            self.items.append(outcome)
            self.threads.append(threading.get_ident())
            self.events.append(f"outcome:{outcome.dictation_id}")
            self._cond.notify_all()

    def note(self, event: str) -> None:
        with self._cond:
            self.events.append(event)

    def wait(self, count: int, timeout: float = 5.0) -> list[DeliveryOutcome]:
        with self._cond:
            assert self._cond.wait_for(lambda: len(self.items) >= count, timeout)
            return list(self.items)


class HistorySpy(HistoryStore):
    def __init__(self, home: Path, order: list[str]) -> None:
        super().__init__(home)
        self.order = order

    def add_pending(self, text: str) -> str:
        self.order.append("add_pending")
        return super().add_pending(text)

    def mark(self, entry_id: str, status: HistoryStatus | str) -> bool:
        self.order.append("mark")
        return super().mark(entry_id, status)


def dictation(
    did: int, text: str, settings: Settings | None = None
) -> tuple[Dictation, AsrOutcome]:
    d = Dictation(did, DictationMode.HOLD, settings or BASE, committed_at_ms=0)
    return d, AsrOutcome(did, "text", text)


class Env:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.order: list[str] = []
        self.injector = FakeInjector(order=self.order)
        self.rewriter = FakeRewriter(order=self.order)
        self.history = HistorySpy(home, self.order)
        self.outcomes = Outcomes()
        self.started_ids: list[int] = []
        self.inserting_ids: list[int] = []
        self._delivery: DeliveryThread | None = None

    def delivery(self, **kwargs: object) -> DeliveryThread:
        def started(did: int) -> None:
            self.started_ids.append(did)

        def inserting(did: int) -> None:
            self.inserting_ids.append(did)
            self.outcomes.note(f"inserting:{did}")

        options: dict[str, object] = {
            "injector": self.injector,
            "history": self.history,
            "on_outcome": self.outcomes,
            "rewriter_factory": self.rewriter.factory,
            "on_inject_started": started,
            "on_inserting": inserting,
        }
        options.update(kwargs)
        self._delivery = DeliveryThread(**options)  # type: ignore[arg-type]
        self._delivery.start()
        return self._delivery

    def submit(self, did: int, text: str, settings: Settings | None = None) -> Dictation:
        assert self._delivery is not None
        d, outcome = dictation(did, text, settings)
        self._delivery.submit(d, outcome)
        return d

    def close(self) -> None:
        if self._delivery is not None:
            self._delivery.stop()


@pytest.fixture
def env(home: Path) -> Iterator[Env]:
    environment = Env(home)
    yield environment
    for fake in (environment.injector, environment.rewriter):
        if fake.gate is not None:
            fake.gate.set()
    environment.close()


def statuses(history: HistoryStore) -> list[tuple[str, HistoryStatus]]:
    return [(e.text, e.status) for e in history.entries()]


# --- Pipeline order -----------------------------------------------------------------------


def test_ut134_pipeline_order(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    real = delivery_module.postprocess

    def spy(text: str, settings: Settings) -> str:
        env.order.append("postprocess")
        return real(text, settings)

    monkeypatch.setattr(delivery_module, "postprocess", spy)
    env.rewriter.output = "Eu acho que funciona."
    env.delivery()

    env.submit(1, "hum eu acho que funciona", LLM_ON)
    (outcome,) = env.outcomes.wait(1)

    assert env.order == ["postprocess", "rewrite", "add_pending", "inject", "mark"]
    assert env.rewriter.calls == ["eu acho que funciona"]
    assert env.injector.calls == [("Eu acho que funciona.", "paste")]
    assert outcome.status is DeliveryStatus.INSERTED
    assert outcome.text == "Eu acho que funciona."
    assert outcome.rewrite_skipped is None
    assert outcome.inject == env.injector.default
    assert outcome.message is None
    assert statuses(env.history) == [("Eu acho que funciona.", HistoryStatus.INSERTED)]
    assert outcome.history_id == env.history.entries()[0].id
    assert env.started_ids == [1]


def test_ut135_fifo_even_when_the_first_rewrite_is_slow(env: Env) -> None:
    env.rewriter.delay_s = 0.3
    env.delivery()

    env.submit(1, "primeiro ditado", LLM_ON)
    env.submit(2, "segundo ditado")
    env.outcomes.wait(2)

    assert [text for text, _ in env.injector.calls] == ["primeiro ditado", "segundo ditado"]
    assert [o.dictation_id for o in env.outcomes.items] == [1, 2]


def test_ut136_skipped_rewrite_inserts_postprocessed_text(env: Env, logs: list[str]) -> None:
    env.rewriter.error = RewriteSkipped("timeout")
    env.delivery()

    env.submit(1, "hum texto  ditado", LLM_ON)
    (outcome,) = env.outcomes.wait(1)

    assert env.injector.calls == [("texto ditado", "paste")]
    assert outcome.rewrite_skipped == "timeout"
    assert outcome.status is DeliveryStatus.INSERTED
    assert "rewrite_skipped id=1 reason=timeout" in logs


def test_unexpected_rewriter_error_is_a_skip(env: Env) -> None:
    env.rewriter.error = RuntimeError("bug")
    env.delivery()

    env.submit(1, "texto", LLM_ON)
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.INSERTED
    assert outcome.rewrite_skipped == "http_error"
    assert env.injector.calls == [("texto", "paste")]


def test_ut137_empty_postprocessed_text_is_no_speech(env: Env) -> None:
    env.delivery()

    env.submit(1, "hum… hmm", LLM_ON)
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.NO_SPEECH
    assert env.history.entries() == []
    assert env.injector.calls == []
    assert env.rewriter.calls == []


def test_ut138_failed_insert_is_recorded(env: Env) -> None:
    env.injector.results = [InjectStatus.FAILED_UNCONFIRMED]
    env.delivery()

    env.submit(1, "texto")
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.FAILED_INSERT
    assert outcome.message is Msg.INSERT_FAILED_CLIPBOARD
    assert statuses(env.history) == [("texto", HistoryStatus.FAILED_INSERT)]


# --- Cancellation -------------------------------------------------------------------------


def test_ut139_cancel_during_rewrite(env: Env) -> None:
    env.rewriter.gate = threading.Event()
    delivery = env.delivery()

    env.submit(1, "texto", LLM_ON)
    assert env.rewriter.started.wait(5)
    delivery.cancel(1)
    env.rewriter.gate.set()
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.CANCELLED
    assert env.injector.calls == []
    assert env.history.entries() == []


def test_cancel_while_queued(env: Env) -> None:
    env.injector.gate = threading.Event()
    delivery = env.delivery()

    env.submit(1, "primeiro")
    assert env.injector.started.wait(5)
    env.submit(2, "segundo")
    delivery.cancel(2)
    env.injector.gate.set()
    outcomes = env.outcomes.wait(2)

    assert [o.status for o in outcomes] == [DeliveryStatus.INSERTED, DeliveryStatus.CANCELLED]
    assert env.injector.calls == [("primeiro", "paste")]


def test_ut140_cancel_after_injection_is_a_no_op(env: Env) -> None:
    delivery = env.delivery()

    env.submit(1, "texto")
    env.outcomes.wait(1)
    delivery.cancel(1)
    env.submit(2, "outro")
    env.outcomes.wait(2)

    assert statuses(env.history) == [
        ("outro", HistoryStatus.INSERTED),
        ("texto", HistoryStatus.INSERTED),
    ]
    assert [o.status for o in env.outcomes.items] == [DeliveryStatus.INSERTED] * 2


def test_ut224_cancel_or_pause_during_injection_lets_it_complete(env: Env) -> None:
    env.injector.gate = threading.Event()
    delivery = env.delivery()

    env.submit(1, "digitando", TYPE_MODE)
    assert env.injector.started.wait(5)
    delivery.cancel(1)  # what the App may do when it pauses
    env.injector.gate.set()
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.INSERTED
    assert env.injector.calls == [("digitando", "type")]
    assert statuses(env.history) == [("digitando", HistoryStatus.INSERTED)]


def test_stop_finishes_the_current_job_and_saves_the_queued(env: Env) -> None:
    env.injector.gate = threading.Event()
    delivery = env.delivery()

    env.submit(1, "em andamento")
    assert env.injector.started.wait(5)
    env.submit(2, "na fila")
    delivery.stop(timeout_s=0)
    env.injector.gate.set()
    outcomes = env.outcomes.wait(2)

    assert [(o.dictation_id, o.status) for o in outcomes] == [
        (1, DeliveryStatus.INSERTED),
        (2, DeliveryStatus.FAILED_INSERT),
    ]
    assert outcomes[1].message is Msg.INSERT_FAILED_OPEN_HISTORY
    assert env.injector.calls == [("em andamento", "paste")]
    assert statuses(env.history) == [
        ("na fila", HistoryStatus.FAILED_INSERT),
        ("em andamento", HistoryStatus.INSERTED),
    ]
    delivery.stop()  # idempotent
    assert not delivery.running


# --- Crash, snapshots, switches -------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_ut141_history_is_pending_before_inject(env: Env, home: Path) -> None:
    """A crash inside the injection (simulated by `SystemExit`) kills the thread."""
    env.injector.errors = [SystemExit()]
    delivery = env.delivery()

    env.submit(1, "texto salvo")
    assert env.injector.started.wait(5)
    assert delivery._thread is not None
    delivery._thread.join(5)

    fresh = HistoryStore(home)
    assert [(e.text, e.status) for e in fresh.entries()] == [("texto salvo", HistoryStatus.PENDING)]


def test_ut142_each_dictation_uses_its_settings_snapshot(env: Env) -> None:
    env.injector.gate = threading.Event()
    env.delivery()

    env.submit(1, "um ditado antes")
    assert env.injector.started.wait(5)
    d2 = env.submit(2, "commit em colar")
    env.submit(3, "depois da troca", TYPE_MODE)  # the settings changed to type
    env.injector.gate.set()
    env.outcomes.wait(3)

    assert [mode for _, mode in env.injector.calls] == ["paste", "paste", "type"]
    assert env.injector.settings[1] is d2.settings.injection


def test_ut142_postprocess_uses_the_snapshot(env: Env) -> None:
    env.delivery()

    env.submit(1, "hum  texto", Settings(cleanup=CleanupSettings(enabled=False)))
    env.outcomes.wait(1)

    assert env.injector.calls == [("hum  texto", "paste")]


def test_ut143_rewrite_disabled_is_never_called(env: Env) -> None:
    env.delivery()

    env.submit(1, "texto")
    env.outcomes.wait(1)

    assert env.rewriter.calls == []
    assert env.rewriter.settings == []


def test_ut225_two_failures_leave_the_last_text(env: Env) -> None:
    env.injector.results = [InjectStatus.FAILED_UNCONFIRMED, InjectStatus.FAILED_UNCONFIRMED]
    env.delivery()

    env.submit(1, "primeiro")
    env.submit(2, "segundo")
    env.outcomes.wait(2)

    assert env.injector.clipboard == "segundo"
    assert statuses(env.history) == [
        ("segundo", HistoryStatus.FAILED_INSERT),
        ("primeiro", HistoryStatus.FAILED_INSERT),
    ]


def test_ut228_rewrite_toggled_off_during_a_rewrite(env: Env) -> None:
    env.rewriter.gate = threading.Event()
    env.rewriter.output = lambda text: text.capitalize() + "."
    env.delivery()

    env.submit(1, "primeiro texto", LLM_ON)
    assert env.rewriter.started.wait(5)
    env.submit(2, "segundo texto")  # committed after the toggle: llm.enabled is False
    env.rewriter.gate.set()
    outcomes = env.outcomes.wait(2)

    assert outcomes[0].text == "Primeiro texto."
    assert env.rewriter.calls == ["primeiro texto"]
    assert env.injector.calls == [("Primeiro texto.", "paste"), ("segundo texto", "paste")]


# --- Errors, callbacks, direct injection ------------------------------------------------------


def test_injector_exception_is_contained(env: Env, logs: list[str]) -> None:
    env.injector.errors = [RuntimeError("boom")]
    env.delivery()

    env.submit(1, "texto")
    env.submit(2, "depois")
    outcomes = env.outcomes.wait(2)

    assert outcomes[0].status is DeliveryStatus.FAILED_INSERT
    assert outcomes[0].message is Msg.INSERT_FAILED_CLIPBOARD
    assert outcomes[1].status is DeliveryStatus.INSERTED
    assert env.injector.clipboard == "texto"
    assert statuses(env.history)[1] == ("texto", HistoryStatus.FAILED_INSERT)
    assert any(
        line.startswith("worker_error") and "thread=delivery" in line and "RuntimeError" in line
        for line in logs
    )


def test_failing_outcome_callback_does_not_stop_the_thread(env: Env) -> None:
    seen: list[int] = []

    def flaky(outcome: DeliveryOutcome) -> None:
        seen.append(outcome.dictation_id)
        if outcome.dictation_id == 1:
            raise RuntimeError("consumer bug")
        env.outcomes(outcome)

    env.delivery(on_outcome=flaky)
    env.submit(1, "um")
    env.submit(2, "dois")
    env.outcomes.wait(1)

    assert seen == [1, 2]


def test_callbacks_run_on_the_delivery_thread(env: Env) -> None:
    env.delivery()

    env.submit(1, "texto")
    env.outcomes.wait(1)

    assert env.outcomes.threads == list(env.injector.threads)
    assert env.outcomes.threads[0] != threading.get_ident()


def test_inserting_is_reported_for_slow_injections_before_the_outcome(env: Env) -> None:
    env.injector.delay_s = 0.7
    env.delivery()

    env.submit(1, "texto longo", TYPE_MODE)
    env.outcomes.wait(1)

    assert env.inserting_ids == [1]
    assert env.outcomes.events == ["inserting:1", "outcome:1"]


def test_fast_injection_reports_no_inserting(env: Env) -> None:
    env.delivery()

    env.submit(1, "rápido")
    env.outcomes.wait(1)

    assert env.inserting_ids == []


def test_inject_text_runs_after_queued_dictations(env: Env) -> None:
    env.injector.gate = threading.Event()
    delivery = env.delivery()
    results: list[InjectResult] = []
    done = threading.Event()

    def on_result(result: InjectResult) -> None:
        results.append(result)
        done.set()

    env.submit(1, "ditado")
    assert env.injector.started.wait(5)
    delivery.inject_text("hum reenviado", TYPE_MODE, on_result)
    env.injector.gate.set()
    assert done.wait(5)

    assert env.injector.calls == [("ditado", "paste"), ("hum reenviado", "type")]
    assert env.injector.settings[1] is TYPE_MODE.injection
    assert results == [env.injector.default]
    assert [e.text for e in env.history.entries()] == ["ditado"]


def test_non_text_outcome_is_no_speech(env: Env) -> None:
    delivery = env.delivery()
    d = Dictation(1, DictationMode.HOLD, BASE, committed_at_ms=0)

    delivery.submit(d, AsrOutcome(1, "no_speech"))
    (outcome,) = env.outcomes.wait(1)

    assert outcome.status is DeliveryStatus.NO_SPEECH
    assert env.injector.calls == []


def test_default_factory_caches_one_local_rewriter(home: Path) -> None:
    delivery = DeliveryThread(
        injector=FakeInjector(), history=HistoryStore(home), on_outcome=lambda _o: None
    )
    factory: Callable[[LlmSettings], object] = delivery._local_rewriter
    first = factory(LlmSettings(enabled=True))

    assert factory(LlmSettings(enabled=True)) is first
    assert factory(LlmSettings(enabled=True, model="outro")) is not first
