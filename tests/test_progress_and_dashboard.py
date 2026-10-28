"""WS6: Progress page, history helpers and dashboard counts."""

import json
import sqlite3

import src.database.db as db
from src.progress.history import (
    MISSING,
    group_key,
    group_sessions,
    posture_text,
    series,
    table_row,
)
from tests.helpers import perform, prescribe, seed_patient


def legacy_rows(path, patient_id):
    """Insert rows exactly as the original application wrote them."""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO sessions(patient_id, exercise, reps, max_angle, "
        "min_angle, session_date) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (patient_id, "Shoulder Flexion", 4, 172.9, 0.1,
             "2026-09-01 10:00:00"),
            (patient_id, "Elbow Flexion", 0, None, None,
             "2026-09-02 10:00:00"),
        ],
    )
    conn.commit()
    conn.close()


def v1_row(**overrides):
    row = {
        "session_id": 9, "session_date": "2026-10-01 10:00:00",
        "metrics_version": 1, "exercise": "Elbow Flexion",
        "exercise_id": "elbow_flexion", "affected_side": "LEFT",
        "reps": 5, "target_reps": 10, "max_angle": None,
        "trajectory_similarity": None, "valid_segments": 0,
        "frames_total": 100, "frames_measured": 0, "mean_fps": None,
        "parameters_json": json.dumps({"target_reps": 10}),
        "experimental_posture_json": None,
    }
    row.update(overrides)
    return row


def test_missing_values_render_as_dash_not_zero():
    cells = table_row(v1_row())
    assert cells[2] == "5 / 10"
    assert cells[3] == MISSING          # peak
    assert cells[4] == MISSING          # movement score
    assert cells[5] == MISSING          # DTW shape similarity
    assert cells[6] == "0"              # zero scored movements is real
    assert cells[7] == "0%"             # arm tracked in 0% of frames
    assert cells[8] == MISSING          # fps unknown
    assert cells[9] == MISSING          # status not recorded (pre-v2)


def test_legacy_and_current_sessions_are_grouped_separately():
    legacy = {"metrics_version": None, "exercise": "Elbow Flexion"}
    current = v1_row()
    groups = group_sessions([legacy, current])
    keys = list(groups)
    assert [k.legacy for k in keys] == [False, True]
    assert "legacy" in group_key(legacy).label()
    assert "(Left)" in group_key(current).label()


def test_series_skips_missing_values():
    rows = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
    assert series(rows, "x") == ([1, 3], [1.0, 3.0])


def test_posture_text_is_labelled_experimental():
    data = {"baseline_established": True, "measures": {
        "trunk_rotation": {"p95_abs_deviation": 4.2, "unit": "degrees"}}}
    text = posture_text({"experimental_posture_json": json.dumps(data)})
    assert text.startswith("EXPERIMENTAL")
    assert "not validated" in text
    assert posture_text({"experimental_posture_json": None}) is None


def test_progress_page_shows_history_and_review(
    qapp, temp_db, scripted_camera
):
    from src.ui.pages.progress_page import ProgressPage
    from src.ui.pages.rehab_page import RehabPage

    patient_id, assessment_id = seed_patient()
    legacy_rows(temp_db, patient_id)
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT",
              {"target_reps": 3, "target_range": 80})

    rehab = RehabPage()
    perform(rehab, scripted_camera, [84, 84, 84])
    perform(rehab, scripted_camera, [84, 84, 84])

    page = ProgressPage()
    labels = [page.group_combo.itemText(i)
              for i in range(page.group_combo.count())]
    assert labels[0].startswith("Shoulder Abduction (Left) — 2 session")
    assert sum("legacy" in label for label in labels) == 2

    # Current group: table has both sessions, charts have 3 panels.
    assert page.table.rowCount() == 2
    assert len(page.figure.axes) == 3

    # Legacy elbow row: NULL peak shown as missing, not 0.
    legacy_index = next(i for i, l in enumerate(labels)
                        if l.startswith("Elbow Flexion"))
    page.group_combo.setCurrentIndex(legacy_index)
    assert page.table.item(0, 3).text() == MISSING
    assert len(page.figure.axes) == 2
    assert "Legacy" in page.chart_note.text()

    # Pending proposal visible and reviewable from the page.
    assert page.approve_button.isEnabled()
    assert "INCREASE" in page.proposal_label.text()

    page.reviewer_input.setText("")
    page.review_current(True)
    assert db.get_active_prescription(patient_id)["source"] == "ASSESSMENT"

    page.reviewer_input.setText("Therapist A")
    page.review_current(True)
    active = db.get_active_prescription(patient_id)
    assert active["source"] == "ADAPTATION"
    assert active["parameters"]["target_reps"] == 5
    assert not page.approve_button.isEnabled()
    assert page.decision_table.rowCount() == 2


def test_dashboard_counts_from_database(qapp, temp_db):
    from src.ui.pages.dashboard_page import DashboardPage

    page = DashboardPage()
    assert page.stat_labels["patients"].text() == "0"
    assert page.stat_labels["last_session"].text() == "—"

    patient_id, _ = seed_patient()
    legacy_rows(temp_db, patient_id)
    page.refresh()

    counts = db.get_dashboard_counts()
    assert counts["patients"] == 1
    assert counts["sessions"] == 2
    assert counts["pending_adaptations"] == 0
    assert page.stat_labels["patients"].text() == "1"
    assert page.stat_labels["last_session"].text() == "2026-09-02 10:00:00"
