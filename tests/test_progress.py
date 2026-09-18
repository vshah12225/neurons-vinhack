"""Progress reporting: the journal event, the executor emissions and the GUI bar.

The GUI answers "what is it doing right now?" with a progress bar, so the events
that drive it are part of the contract: a step-by-step fraction while the size
of the work is known, and "indeterminate" while it is not (planning, waiting on
the model).
"""

import threading
from pathlib import Path

import numpy as np
import pytest

from furti_ai.config import Settings
from furti_ai.context import SceneObservation
from furti_ai.executor import PlanExecutor
from furti_ai.gui import FurtiApp
from furti_ai.memory import MemoryManager
from furti_ai.models import ActionType
from furti_ai.planner import PlanStep, TaskPlan, TaskPlanner
from furti_ai.tasklog import TaskJournal


def progress_events(journal: TaskJournal) -> list[dict]:
    return [event.metadata for event in journal._events if event.kind == "PROGRESS"]


# ------------------------------------------------------------------ journal
def test_progress_event_carries_a_usable_fraction(tmp_path: Path):
    journal = TaskJournal("t", tmp_path / "reports")

    journal.progress(2.5, 5, "acting: click Export")

    payload = progress_events(journal)[0]
    assert payload["progress_current"] == 2.5
    assert payload["progress_total"] == 5
    assert payload["progress_label"] == "acting: click Export"
    assert payload["progress_state"] == "determinate"
    assert journal.last_message.startswith("Progress: 2.5/5")


def test_an_unknown_total_means_indeterminate(tmp_path: Path):
    journal = TaskJournal("t", tmp_path / "reports")

    journal.progress(0, 0, "planning the task with the model")

    payload = progress_events(journal)[0]
    assert payload["progress_state"] == "indeterminate"
    assert payload["progress_total"] == 0


def test_progress_is_not_printed_to_the_console(tmp_path: Path, capsys):
    journal = TaskJournal("t", tmp_path / "reports")

    journal.progress(1, 2, "halfway")
    journal.system("a real event")

    console = capsys.readouterr().out
    assert "halfway" not in console
    assert "a real event" in console
    # ...but the event is still in the journal (report + status window).
    assert progress_events(journal)


# ----------------------------------------------------------------- executor
class RecordingInput:
    def __init__(self):
        self.clicks = []

    def move_to(self, x, y):
        pass

    def click(self, x, y, button="left", clicks=1):
        self.clicks.append((x, y))

    def double_click(self, x, y):
        pass

    def right_click(self, x, y):
        pass

    def type_text(self, text):
        pass

    def press_key(self, key, presses=1):
        pass

    def scroll(self, clicks):
        pass

    def drag(self, *args, **kwargs):
        pass


class FakeContext:
    def __init__(self, observations):
        self.observations = observations
        self.calls = 0

    def observe(self, instruction, force_fresh=False):
        observation = self.observations[min(self.calls, len(self.observations) - 1)]
        self.calls += 1
        return observation


class SilentLLM:
    """Planner stub: answers with a canned plan and counts its calls."""

    def __init__(self, response: str = ""):
        self.response = response
        self.call_count = 0

    def chat_text(self, system, user, purpose=""):
        self.call_count += 1
        return self.response

    def chat_vision(self, system, user, image_b64, purpose=""):
        return self.chat_text(system, user, purpose)


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(workspace=tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    return settings


def test_the_executor_reports_step_by_step_progress(tmp_path: Path):
    settings = make_settings(tmp_path)
    scene = SceneObservation(frame=np.full((480, 640, 3), 200, dtype=np.uint8))
    journal = TaskJournal("t", tmp_path / "reports")
    context = FakeContext([scene])
    planner = TaskPlanner(SilentLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings,
        journal,
        None,
        RecordingInput(),
        MemoryManager(settings.memory_file),
        planner,
        context,
        threading.Event(),
        "t",
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            PlanStep(index=1, description="Press escape", action=ActionType.KEY_PRESS,
                     params={"key": "esc"}),
            PlanStep(index=2, description="Press enter", action=ActionType.KEY_PRESS,
                     params={"key": "enter"}),
        ],
    )

    report = executor.execute("do two things", plan)

    totals = {payload["progress_total"] for payload in progress_events(journal)}
    assert totals == {2}
    currents = [payload["progress_current"] for payload in progress_events(journal)]
    # The bar only ever moves forward, and it ends on the last step.
    assert currents == sorted(currents)
    assert currents[0] == 0
    assert max(currents) == 2
    assert len(currents) > len(plan.steps)  # sub-phases, not just step starts
    assert report.results  # the run itself is unaffected by the telemetry


def test_progress_is_skipped_when_the_size_is_unknown(tmp_path: Path):
    """`_step_progress` is a no-op outside a driven run (e.g. direct step calls)."""
    executor = PlanExecutor.__new__(PlanExecutor)
    journal = TaskJournal("t", tmp_path / "reports")
    executor._journal = journal

    executor._step_progress(0.5, "acting")

    assert progress_events(journal) == []


# ------------------------------------------------------------------- GUI bar
class FakeBar:
    """Stand-in for ``ttk.Progressbar`` (records mode and value changes)."""

    def __init__(self):
        self.mode = "determinate"
        self.value = 0.0
        self.starts = 0
        self.stops = 0

    def cget(self, key):
        return self.mode if key == "mode" else self.value

    def configure(self, **kwargs):
        if "mode" in kwargs:
            self.mode = kwargs["mode"]
        if "value" in kwargs:
            self.value = kwargs["value"]

    def start(self, interval=None):
        self.starts += 1
        self.mode = "indeterminate"

    def stop(self):
        self.stops += 1


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeApp:
    """The pieces of ``FurtiApp`` that draw the progress bar.

    The real ``FurtiApp`` methods are bound against these fakes, so the tests
    exercise production code without opening a Tk window.
    """

    def __init__(self, stopped=False):
        self.progress_bar = FakeBar()
        self.progress_var = FakeVar()
        self.progress_detail_var = FakeVar()
        self._stop_event = threading.Event()
        self.lines: list[tuple[str, str]] = []
        if stopped:
            self._stop_event.set()

    def _set_progress(self, current, total, label=""):
        FurtiApp._set_progress(self, current, total, label)

    def _reset_progress(self, label):
        FurtiApp._reset_progress(self, label)

    def _finish_progress(self, success):
        FurtiApp._finish_progress(self, success)

    def _apply_journal_event(self, snapshot):
        FurtiApp._apply_journal_event(self, snapshot)

    def _append_log(self, kind, message):
        self.lines.append((kind, message))

    def _set_signal(self, *args):
        pass


def test_a_known_total_draws_a_percentage():
    app = FakeApp()

    app._set_progress(2.5, 5, "acting: click Export")

    assert app.progress_bar.mode == "determinate"
    assert app.progress_bar.value == pytest.approx(50.0)
    assert app.progress_var.get() == "Step 3 of 5 (50%)"
    assert app.progress_detail_var.get() == "acting: click Export"


def test_a_finished_bar_reads_as_complete():
    app = FakeApp()

    app._set_progress(5, 5, "finished")

    assert app.progress_bar.value == pytest.approx(100.0)
    assert app.progress_var.get() == "All 5 step(s) done (100%)"


def test_an_unknown_total_animates_instead_of_pretending():
    app = FakeApp()

    app._set_progress(0, 0, "planning the task with the model")

    assert app.progress_bar.mode == "indeterminate"
    assert app.progress_bar.starts == 1
    assert app.progress_var.get() == "planning the task with the model"


def test_switching_back_to_a_known_total_stops_the_animation():
    app = FakeApp()
    app._set_progress(0, 0, "planning")

    app._set_progress(1, 4, "acting")

    assert app.progress_bar.stops == 1
    assert app.progress_bar.mode == "determinate"
    assert app.progress_bar.value == pytest.approx(25.0)


def test_the_bar_never_draws_more_than_a_full_bar():
    app = FakeApp()

    app._set_progress(12, 5, "past the end")

    assert app.progress_bar.value == pytest.approx(100.0)


def test_the_finished_state_describes_how_the_run_ended():
    completed = FakeApp()
    completed._finish_progress(True)
    assert completed.progress_bar.value == pytest.approx(100.0)
    assert completed.progress_var.get() == "Task complete"

    stopped = FakeApp(stopped=True)
    stopped._finish_progress(False)
    assert stopped.progress_var.get() == "Stopped"

    failed = FakeApp()
    failed._finish_progress(False)
    assert "without completing" in failed.progress_var.get()


def test_the_bar_is_cleared_when_a_new_run_starts():
    app = FakeApp()
    app._set_progress(9, 10, "almost")

    app._reset_progress("Planning the task...")

    assert app.progress_bar.value == pytest.approx(0.0)
    assert app.progress_bar.mode == "determinate"
    assert app.progress_var.get() == "Planning the task..."


def test_progress_events_are_not_duplicated_into_the_log():
    """One log line per sub-phase would bury the real events."""
    app = FakeApp()

    app._apply_journal_event(
        {
            "event_kind": "PROGRESS",
            "progress_current": 1.0,
            "progress_total": 4,
            "progress_label": "acting: click Export",
        }
    )

    assert app.lines == []
    assert app.progress_bar.value == pytest.approx(25.0)


def test_a_real_event_still_reaches_the_log():
    app = FakeApp()

    app._apply_journal_event(
        {
            "event_kind": "CONFIRM",
            "message": "Step 1: click input dispatched successfully.",
            "timestamp": "12:00:00",
        }
    )

    assert [kind for kind, _ in app.lines] == ["CONFIRM"]


# ----------------------------------------------------------------- planner
def test_the_planner_reports_an_indeterminate_state_while_it_thinks(tmp_path: Path):
    settings = make_settings(tmp_path)
    journal = TaskJournal("t", tmp_path / "reports")

    class LLM:
        call_count = 0

        def chat_text(self, system, user, purpose=""):
            self.call_count += 1
            return (
                '{"goal": "g", "steps": [{"description": "Click Save", '
                '"action": "click", "target": "Save"}]}'
            )

    scene = SceneObservation(frame=np.zeros((10, 10, 3), dtype=np.uint8))
    planner = TaskPlanner(
        LLM(), settings, journal, FakeContext([scene]), threading.Event()
    )

    plan = planner.plan("click save")

    states = [payload["progress_state"] for payload in progress_events(journal)]
    assert states[0] == "indeterminate"
    assert states[-1] == "determinate"
    assert progress_events(journal)[-1]["progress_total"] == len(plan.steps)
