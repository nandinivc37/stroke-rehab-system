"""WS3: therapist approval records the exercise actually prescribed."""

import src.database.db as db
from src.exercises.catalogue import get_exercise_config


def test_therapist_can_override_rule_based_suggestion(qapp, temp_db):
    from src.ui.pages.assessment_page import AssessmentPage

    patient_id = db.add_patient("Synthetic", 60, "Other", "T", "")
    db.add_affected_area(patient_id, "RIGHT", "Upper Limb", "Shoulder", "")
    assessment_id = db.save_clinical_assessment(
        patient_id, "Stroke", "Reduced ROM", 40, "None", ""
    )
    rec_id = db.save_recommendation(
        patient_id, assessment_id, "Shoulder Flexion", "Easy", "rule-based"
    )

    page = AssessmentPage()
    page.current_recommendation_id = rec_id
    page.current_affected_side = "RIGHT"
    page.exercise_choice.setEnabled(True)
    page.exercise_choice.setCurrentIndex(
        page.exercise_choice.findData("shoulder_abduction")
    )

    page.approve_recommendation()

    prescription = db.get_active_prescription(patient_id)
    config = get_exercise_config("shoulder_abduction")
    assert prescription["recommendation_id"] == rec_id
    assert prescription["exercise"] == "Shoulder Abduction"
    assert prescription["exercise_id"] == "shoulder_abduction"
    assert prescription["affected_side"] == "RIGHT"
    assert prescription["source"] == "ASSESSMENT"
    assert prescription["parameters"] == config.default_parameters()


def test_choice_lists_only_implemented_exercises(qapp, temp_db):
    from src.exercises.catalogue import implemented_exercises
    from src.ui.pages.assessment_page import AssessmentPage

    page = AssessmentPage()
    listed = [
        page.exercise_choice.itemData(i)
        for i in range(page.exercise_choice.count())
    ]
    assert listed == [c.exercise_id for c in implemented_exercises()]
