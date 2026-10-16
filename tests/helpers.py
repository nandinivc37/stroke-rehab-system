"""Shared helpers for UI/database workflow tests (synthetic data only)."""

import src.database.db as db
from src.simulation.session_simulator import simulate_session


def seed_patient(side="LEFT", joint="Shoulder"):
    patient_id = db.add_patient("Synthetic Patient", 60, "Other", "T", "")
    db.add_affected_area(patient_id, side, "Upper Limb", joint, "")
    assessment_id = db.save_clinical_assessment(
        patient_id, "Stroke", "Reduced ROM", 40, "None", ""
    )
    return patient_id, assessment_id


def prescribe(patient_id, assessment_id, exercise, exercise_id, side,
              parameters, approve=True):
    rec_id = db.save_recommendation(
        patient_id, assessment_id, exercise, "Easy", "test"
    )
    if exercise_id is not None:
        db.set_recommendation_prescription(
            rec_id, exercise, exercise_id, side, parameters, "ASSESSMENT"
        )
    if approve:
        db.update_recommendation_status(rec_id, "APPROVED")
    return rec_id


def perform(page, script, peaks, exercise="shoulder_abduction", side="LEFT"):
    script["results"] = simulate_session(exercise, side, peaks)
    page.start_exercise()
    for _ in script["results"]:
        page.live_widget.update_frame()

    page.stop_exercise()
    return page.last_adaptation
