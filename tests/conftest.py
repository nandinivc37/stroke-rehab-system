import os
import sys
from pathlib import Path

import pytest

# Qt must render offscreen in automated tests (no display, no webcam).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Point the application at a fresh, empty database in a temp dir.
    The real data/rehab.db is never touched by tests.
    """

    import src.database.db as db

    path = tmp_path / "rehab_test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.initialize_database()
    return path


@pytest.fixture(autouse=True)
def non_blocking_message_boxes(monkeypatch):
    """
    Modal QMessageBox calls would block forever offscreen. Record them
    instead so tests can assert on what the user would have been told.
    """

    from PyQt6.QtWidgets import QMessageBox

    shown = []

    def record(kind, default):
        def show(_parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return default
        return show

    monkeypatch.setattr(QMessageBox, "warning",
                        record("warning", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "information",
                        record("information", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical",
                        record("critical", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question",
                        record("question", QMessageBox.StandardButton.Yes))
    return shown


@pytest.fixture
def scripted_camera(monkeypatch):
    """Replace camera start-up with scripted synthetic results."""

    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    script = {"results": [], "open": True}

    def fake_start(widget):
        if not script["open"]:
            widget.status_label.setText("Could not open camera")
            widget.camera = None
            return
        widget.camera = FakeCamera(len(script["results"]))
        widget.pose = ScriptedPose(script["results"])
        widget.analyzer.reset()
        widget.posture_monitor.reset()

    from src.ui.live_pose_widget import LivePoseWidget

    monkeypatch.setattr(LivePoseWidget, "start", fake_start)
    return script
