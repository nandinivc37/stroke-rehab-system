"""
Live pipeline and session persistence (schema 2):

  * capture/analysis on a worker thread, clean shutdown, diagnostics;
  * MediaPipe receives the unmirrored frame;
  * per-frame log, session status and v2 metrics persisted;
  * sessions survive save failures (reported) and application close;
  * the range-limited patient goes through the real session page,
    database and approval workflow end to end.

Scripted synthetic poses only: software behaviour, not webcam accuracy.
"""

import json
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import src.database.db as db
import src.ui.live_pose_widget as live
from src.adaptation.controller import SIMPLIFY, UNABLE_TO_ASSESS
from src.adaptation.policy import PolicyConfig, session_quality_issue
from src.adaptation.workflow import pending_proposals, review_proposal
from src.exercises.catalogue import get_exercise_config
from src.simulation.session_simulator import simulate_session
from src.simulation.synthetic_pose import FakeCamera, ScriptedPose
from tests.helpers import perform, prescribe, seed_patient

ABDUCTION = get_exercise_config("shoulder_abduction")


def wait_until(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------
# Worker thread
# ---------------------------------------------------------

def test_worker_thread_processes_frames_and_stops_cleanly(qapp):
    results = simulate_session("shoulder_abduction", "LEFT", [90, 90])
    widget = live.LivePoseWidget(ABDUCTION, "LEFT")
    widget.camera = FakeCamera(len(results), fps=30)
    widget.pose = ScriptedPose(results)

    widget.worker = live.CaptureWorker(widget)
    widget.worker.frame_ready.connect(widget._present)
    widget.worker.start()

    assert wait_until(lambda: widget.analyzer.frames_total == len(results))
    assert wait_until(lambda: "Reps: 2" in widget.status_label.text())

    stats = widget.get_stats()           # stops the worker first
    assert widget.worker is None
    assert stats["frames_total"] == len(results)
    assert stats["reps"] == 2
    assert stats["mean_fps"] == pytest.approx(30, rel=0.01)
    timing = stats["diagnostics"]["timing"]
    assert timing["inference"]["n"] == len(results)
    assert stats["diagnostics"]["threaded"] is True
    widget.stop()                        # idempotent after get_stats


def test_stop_while_running_joins_the_thread(qapp):
    results = simulate_session("shoulder_abduction", "LEFT", [90] * 20)
    widget = live.LivePoseWidget(ABDUCTION, "LEFT")
    widget.camera = FakeCamera(len(results), fps=30)
    widget.pose = ScriptedPose(results)
    widget.worker = worker = live.CaptureWorker(widget)
    worker.start()
    assert wait_until(lambda: widget.analyzer.frames_total > 5)
    widget.stop()
    assert worker.isFinished()
    assert widget.camera is None and widget.pose is None


def test_mediapipe_receives_the_unmirrored_frame(qapp):
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :10] = (255, 0, 0)          # asymmetric: left edge coloured

    class Camera(FakeCamera):
        def read(self):
            ok, _ = super().read()
            return ok, frame.copy() if ok else None

    received = []

    class RecordingPose(ScriptedPose):
        def process(self, rgb):
            received.append(rgb.copy())
            return super().process(rgb)

    widget = live.LivePoseWidget(ABDUCTION, "LEFT")
    widget.camera = Camera(1, fps=30)
    widget.pose = RecordingPose(
        simulate_session("shoulder_abduction", "LEFT", [10])[:1])
    widget.update_frame()
    assert np.array_equal(received[0], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_camera_loss_marks_the_session(qapp):
    widget = live.LivePoseWidget(ABDUCTION, "LEFT")
    widget.camera = FakeCamera(0, fps=30)
    widget.pose = ScriptedPose([])
    for _ in range(live.MAX_CONSECUTIVE_READ_FAILURES):
        widget.update_frame()
    assert "not delivering frames" in widget.status_label.text()
    assert widget.get_stats()["session_status"] == live.SESSION_CAMERA_LOST


def test_pose_settings_are_shared_and_overridable(monkeypatch):
    from src.pose_estimation import config
    monkeypatch.delenv("REHAB_POSE_COMPLEXITY", raising=False)
    assert config.pose_settings()["model_complexity"] == \
        config.DEFAULT_MODEL_COMPLEXITY
    monkeypatch.setenv("REHAB_POSE_COMPLEXITY", "2")
    assert config.pose_settings()["model_complexity"] == 2
    monkeypatch.setenv("REHAB_POSE_COMPLEXITY", "5")
    with pytest.raises(ValueError):
        config.pose_settings()


# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------

@pytest.fixture
def abduction(temp_db):
    patient_id, assessment_id = seed_patient()
    rec_id = prescribe(patient_id, assessment_id, "Shoulder Abduction",
                       "shoulder_abduction", "LEFT",
                       {"target_reps": 3, "target_range": 80})
    return patient_id, rec_id


def test_session_v2_metrics_and_frame_log_are_persisted(
        qapp, abduction, scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction
    page = RehabPage()
    perform(page, scripted_camera, [90, 90, 90])

    [row] = db.get_sessions(patient_id)
    assert row["metrics_version"] == 2
    assert row["session_status"] == "COMPLETE"
    assert row["reps"] == 3
    assert row["peak_rom"] == pytest.approx(90, abs=1)
    assert row["dtw_similarity"] is not None
    assert row["dtw_mean_error_deg"] is not None
    diagnostics = json.loads(row["diagnostics_json"])
    assert "timing" in diagnostics

    frames = db.get_session_frames(row["session_id"])
    assert len(frames) == row["frames_total"]
    assert {f["status"] for f in frames} == {"ok"}
    assert frames[1]["t_s"] == pytest.approx(1 / 30, abs=1e-3)
    assert max(f["angle"] for f in frames) == pytest.approx(90, abs=1)


def test_deleting_a_patient_removes_their_frame_logs(qapp, abduction,
                                                     scripted_camera,
                                                     tmp_path):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction
    perform(RehabPage(), scripted_camera, [90])
    [row] = db.get_sessions(patient_id)
    assert db.get_session_frames(row["session_id"])
    db.delete_patient(patient_id, backup_dir=tmp_path / "b")
    assert db.get_session_frames(row["session_id"]) == []


def test_save_failure_is_reported_not_crashed(qapp, abduction,
                                              scripted_camera, monkeypatch,
                                              non_blocking_message_boxes):
    from src.ui.pages import rehab_page as module

    def broken(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "save_session_summary", broken)
    page = module.RehabPage()
    out = perform(page, scripted_camera, [90])
    assert out is None
    assert "could not be saved" in page.session_status.text()
    assert any(kind == "critical" for kind, *_ in non_blocking_message_boxes)
    assert page.start_button.isEnabled()


def test_closing_the_app_mid_session_keeps_an_incomplete_session(
        qapp, abduction, scripted_camera):
    from src.ui.main_window import MainWindow

    patient_id, _ = abduction
    window = MainWindow()
    page = window.rehab_page
    page.refresh()
    scripted_camera["results"] = simulate_session(
        "shoulder_abduction", "LEFT", [90, 90])
    page.start_exercise()
    for _ in range(len(scripted_camera["results"]) // 2):
        page.live_widget.update_frame()

    window.close()

    [row] = db.get_sessions(patient_id)
    assert row["session_status"] == "INTERRUPTED_APP_CLOSED"
    assert page.live_widget is None
    issue = session_quality_issue(row, PolicyConfig())
    assert issue == "session incomplete (INTERRUPTED_APP_CLOSED)"


def test_incomplete_session_cannot_drive_adaptation(qapp, abduction,
                                                    scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction
    page = RehabPage()
    perform(page, scripted_camera, [90, 90, 90])
    scripted_camera["results"] = simulate_session(
        "shoulder_abduction", "LEFT", [90, 90, 90])
    page.start_exercise()
    for _ in scripted_camera["results"]:
        page.live_widget.update_frame()
    page.live_widget.camera_lost = True
    page.stop_exercise()
    assert page.last_adaptation["action"] == UNABLE_TO_ASSESS
    assert "CAMERA_LOST" in page.last_adaptation["rationale"]
    assert pending_proposals(patient_id) == []


# ---------------------------------------------------------
# Range-limited patient, end to end (audit P0-3)
# ---------------------------------------------------------

def test_range_limited_patient_full_loop(qapp, temp_db, scripted_camera):
    """
    A patient whose shoulder elevation peaks at ~74 deg against an 80 deg
    target: no repetition can be counted. After two such assessable
    sessions the system proposes a LOWER RANGE target (not fewer
    repetitions); the therapist approves; the next session runs at
    75 deg and the same movement now counts.
    """
    from src.ui.pages.rehab_page import RehabPage

    patient_id, assessment_id = seed_patient()
    original = prescribe(patient_id, assessment_id, "Shoulder Abduction",
                         "shoulder_abduction", "LEFT",
                         {"target_reps": 6, "target_range": 80})
    page = RehabPage()

    first = perform(page, scripted_camera, [76, 76, 76])
    assert first["action"] == "REPEAT"
    second = perform(page, scripted_camera, [76, 76, 76])
    assert second["action"] == SIMPLIFY
    assert second["proposed_parameters"] == {"target_reps": 6,
                                             "target_range": 75.0}
    # Not applied before approval.
    assert page.current_parameters["target_range"] == 80.0

    [proposal] = pending_proposals(patient_id)
    review_proposal(proposal["recommendation_id"], True, "Therapist A")
    page.refresh()
    assert page.current_parameters == {"target_reps": 6,
                                       "target_range": 75.0}

    perform(page, scripted_camera, [76, 76, 76])
    last = db.get_sessions(patient_id)[-1]
    assert json.loads(last["parameters_json"])["target_range"] == 75.0
    assert last["prescription_id"] == proposal["recommendation_id"]
    assert last["prescription_id"] != original
    assert last["reps"] == 3


def test_rejected_proposal_keeps_the_previous_settings(qapp, temp_db,
                                                       scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, assessment_id = seed_patient()
    original = prescribe(patient_id, assessment_id, "Shoulder Abduction",
                         "shoulder_abduction", "LEFT",
                         {"target_reps": 6, "target_range": 80})
    page = RehabPage()
    perform(page, scripted_camera, [76, 76, 76])
    perform(page, scripted_camera, [76, 76, 76])
    [proposal] = pending_proposals(patient_id)
    review_proposal(proposal["recommendation_id"], False, "Therapist A",
                    "keep target")
    page.refresh()
    assert page.current_prescription_id == original
    assert page.current_parameters["target_range"] == 80.0
    perform(page, scripted_camera, [76, 76, 76])
    assert db.get_sessions(patient_id)[-1]["prescription_id"] == original
