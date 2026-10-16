"""
Legacy affected-area body region "Arm" (stored before the current
registration options) is read as "Upper Limb" for upper-limb joints.

Temporary databases only; stored rows are never rewritten.
"""

import pytest

import src.database.db as db
from src.assessment.recommendation import recommend_exercise


def legacy_patient(body_region="Arm", joint="Elbow", side="LEFT"):
    """A patient whose affected area was stored with a legacy value,
    inserted directly as it exists in the real database."""
    patient_id = db.add_patient("Legacy Patient", 70, "Female", "T", "")
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO affected_areas(patient_id, side, body_region, joint, "
        "notes) VALUES (?, ?, ?, ?, ?)",
        (patient_id, side, body_region, joint,
         "Affected left elbow for rehabilitation"),
    )
    conn.commit()
    conn.close()
    return patient_id


def stored_area(patient_id):
    conn = db.get_connection()
    row = conn.execute(
        "SELECT side, body_region, joint, notes FROM affected_areas "
        "WHERE patient_id = ?", (patient_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------
# Normalization rule
# ---------------------------------------------------------

@pytest.mark.parametrize("region, joint, expected", [
    ("Arm", "Elbow", "Upper Limb"),
    ("Arm", "Shoulder", "Upper Limb"),
    ("Arm", "Wrist", "Upper Limb"),
    (" arm ", "Elbow", "Upper Limb"),
    ("Arm", "Hip", "Arm"),            # contradictory: left as stored
    ("Arm", "Other", "Arm"),
    ("Arm", None, "Arm"),
    ("Upper Limb", "Elbow", "Upper Limb"),
    ("Trunk", "Hip", "Trunk"),
    ("Lower Limb", "Knee", "Lower Limb"),
    (None, "Elbow", None),
])
def test_normalize_body_region(region, joint, expected):
    assert db.normalize_body_region(region, joint) == expected


# ---------------------------------------------------------
# Read boundary: normalized view, stored row untouched
# ---------------------------------------------------------

def test_legacy_area_is_read_as_upper_limb_without_rewriting(temp_db):
    patient_id = legacy_patient()
    [(_, side, region, joint, notes)] = db.get_affected_areas(patient_id)
    assert (side, region, joint) == ("LEFT", "Upper Limb", "Elbow")
    assert notes == "Affected left elbow for rehabilitation"
    # The database still holds exactly what was stored.
    assert stored_area(patient_id) == (
        "LEFT", "Arm", "Elbow", "Affected left elbow for rehabilitation")


def test_other_patients_are_unaffected(temp_db):
    legacy = legacy_patient()
    trunk = legacy_patient("Trunk", "Hip", "RIGHT")
    upper = legacy_patient("Upper Limb", "Shoulder")
    assert db.get_affected_areas(trunk)[0][1:4] == ("RIGHT", "Trunk", "Hip")
    assert db.get_affected_areas(upper)[0][2] == "Upper Limb"
    assert db.get_affected_areas(legacy)[0][2] == "Upper Limb"


# ---------------------------------------------------------
# Recommendations
# ---------------------------------------------------------

def recommend_for(patient_id, impairment="Reduced strength"):
    _, side, region, joint, _ = db.get_affected_areas(patient_id)[0]
    return recommend_exercise("Stroke", side, region, joint, impairment,
                              40.0, "No major restriction")


def test_legacy_arm_elbow_now_gets_elbow_flexion(temp_db):
    result = recommend_for(legacy_patient())
    assert result["success"] is True
    assert result["exercise_id"] == "elbow_flexion"


def test_raw_legacy_value_alone_still_does_not_match():
    # The engine itself is unchanged: only normalized input matches.
    raw = recommend_exercise("Stroke", "LEFT", "Arm", "Elbow",
                             "Reduced strength")
    assert raw["success"] is False


@pytest.mark.parametrize("region, joint", [
    ("Arm", "Hip"),
    ("Trunk", "Hip"),
    ("Upper Limb", "Wrist"),
    ("Lower Limb", "Knee"),
    ("Full Body", "Other"),
])
def test_unsupported_areas_still_get_no_recommendation(temp_db, region,
                                                       joint):
    result = recommend_for(legacy_patient(region, joint))
    assert result["success"] is False
    assert "No suitable exercise" in result["reason"]


def test_assessment_page_recommends_for_legacy_patient(qapp, temp_db):
    from src.ui.pages.assessment_page import AssessmentPage

    patient_id = legacy_patient()
    page = AssessmentPage()
    page.patient_input.setCurrentIndex(page.patient_input.findData(patient_id))
    assert "Upper Limb" in page.area_label.text()
    page.diagnosis_input.setCurrentText("Stroke")
    page.impairment_input.setCurrentText("Reduced strength")
    page.rom_input.setValue(40.0)
    page.restrictions_input.setCurrentText("No major restriction")

    page.save_assessment()

    assert page.exercise_label.text() == "Elbow Flexion"
    assert page.current_recommendation_id is not None
    assessment = db.get_current_assessment(patient_id)
    assert (assessment["diagnosis"], assessment["impairment"]) == (
        "Stroke", "Reduced strength")
    assert stored_area(patient_id)[1] == "Arm"       # still not rewritten


def test_history_is_preserved(qapp, temp_db, scripted_camera):
    """Sessions recorded before reading the normalized area remain."""
    from tests.helpers import perform, prescribe
    from src.ui.pages.rehab_page import RehabPage

    patient_id = legacy_patient("Arm", "Shoulder")
    assessment_id = db.save_clinical_assessment(
        patient_id, "Stroke", "Reduced strength", 40, "None", "")
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT", {"target_reps": 3})
    page = RehabPage()
    page.patient_combo.setCurrentIndex(page.patient_combo.findData(patient_id))
    perform(page, scripted_camera, [84, 84, 84])
    before = db.get_patient_record_counts(patient_id)

    assert recommend_for(patient_id)["exercise_id"] == "shoulder_flexion"
    assert db.get_patient_record_counts(patient_id) == before
    assert db.get_current_assessment(patient_id)["assessment_id"] == \
        assessment_id


# ---------------------------------------------------------
# New registrations use current values
# ---------------------------------------------------------

def test_registration_offers_only_current_body_regions(qapp):
    from src.ui.pages.patients_page import PatientDialog

    dialog = PatientDialog()
    options = [dialog.body_region_input.itemText(i)
               for i in range(dialog.body_region_input.count())]
    assert options == ["Upper Limb", "Lower Limb", "Trunk", "Full Body"]
    assert not dialog.body_region_input.isEditable()


def test_new_area_is_stored_with_current_value(temp_db):
    patient_id = db.add_patient("New", 50, "Male", "T", "")
    db.add_affected_area(patient_id, "LEFT", "Arm", "Elbow", "")
    db.add_affected_area(patient_id, "LEFT", "Upper Limb", "Shoulder", "")
    conn = db.get_connection()
    regions = [r for (r,) in conn.execute(
        "SELECT body_region FROM affected_areas WHERE patient_id = ?",
        (patient_id,))]
    conn.close()
    assert regions == ["Upper Limb", "Upper Limb"]
