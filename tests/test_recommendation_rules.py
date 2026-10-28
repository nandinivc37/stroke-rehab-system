"""
Objective 4: assessment-based recommendation rules (rules_v2) and
therapist approval of the exercise AND its starting parameters.

These tests show that the rules behave as specified. They do not show
that the rules are clinically appropriate (physiotherapist review).
"""

import pytest

import src.database.db as db
from src.assessment.recommendation import (
    RESTRICTION_LIMITED_RANGE,
    RESTRICTION_NONE,
    RESTRICTION_PAIN,
    RESTRICTION_THERAPIST,
    START_BASELINE,
    START_DEFAULT,
    START_LOWEST,
    exclusion_reason,
    override_note,
    recommend_exercise,
    starting_parameters,
)
from src.exercises.catalogue import get_exercise_config

FLEXION = get_exercise_config("shoulder_flexion")
ELBOW = get_exercise_config("elbow_flexion")
REACH = get_exercise_config("forward_reach")


def recommend(joint="Shoulder", impairment="Reduced ROM", baseline=120.0,
              restriction=RESTRICTION_NONE, diagnosis="Stroke"):
    return recommend_exercise(diagnosis, "LEFT", "Upper Limb", joint,
                              impairment, baseline, restriction)


# ---------------------------------------------------------
# Eligibility (R1-R4)
# ---------------------------------------------------------

def test_unsupported_diagnosis_gets_no_recommendation():
    result = recommend(diagnosis="Other")
    assert result["success"] is False
    assert "Diagnosis is not currently supported" in result["reason"]


@pytest.mark.parametrize("impairment", ["Movement compensation", "Other"])
def test_impairment_without_a_matching_exercise(impairment):
    result = recommend(impairment=impairment)
    assert result["success"] is False
    for name in ("Shoulder Flexion", "Shoulder Abduction", "Forward Reach"):
        assert "does not address" in result["excluded"][name]


def test_joint_selects_the_exercise_family():
    assert recommend(joint="Elbow")["exercise_id"] == "elbow_flexion"
    shoulder = recommend(joint="Shoulder")
    assert shoulder["exercise_id"] == "shoulder_flexion"
    assert shoulder["alternatives"] == ["Shoulder Abduction", "Forward Reach"]


def test_baseline_below_every_supported_target_excludes_the_exercise():
    # 45 deg cannot reach the lowest countable target of 50 deg.
    result = recommend(baseline=45.0)
    assert "shoulder_flexion" != result.get("exercise_id")
    assert "below the lowest range target" in \
        result["excluded"]["Shoulder Flexion"]
    # Forward reach is measured in % of arm length, not degrees.
    assert result["exercise_id"] == "forward_reach"


def test_elbow_baseline_below_minimum_gives_no_recommendation():
    result = recommend(joint="Elbow", baseline=50.0)
    assert result["success"] is False
    assert "60" in result["excluded"]["Elbow Flexion"]


def test_side_and_area_rules_are_explicit():
    assert exclusion_reason(FLEXION, "Stroke", "LEFT", "Lower Limb",
                            "Shoulder", "Reduced ROM") is not None
    assert exclusion_reason(FLEXION, "Stroke", "BOTH", "Upper Limb",
                            "Shoulder", "Reduced ROM") is not None
    assert exclusion_reason(FLEXION, "Stroke", "RIGHT", "Upper Limb",
                            "Shoulder", "Reduced ROM", 90) is None


# ---------------------------------------------------------
# Starting parameters (R5)
# ---------------------------------------------------------

def test_baseline_rom_caps_the_starting_range_target():
    params, start, note = starting_parameters(FLEXION, 72.0)
    assert params == {"target_reps": 10, "target_range": 70.0}
    assert start == START_BASELINE and "72" in note


@pytest.mark.parametrize("baseline", [80.0, 105.0, 170.0])
def test_baseline_at_or_above_default_keeps_the_default(baseline):
    params, start, _ = starting_parameters(FLEXION, baseline)
    assert params == FLEXION.default_parameters()
    assert start == START_DEFAULT


def test_baseline_exactly_at_lowest_target():
    params, _, _ = starting_parameters(FLEXION, 50.0)
    assert params["target_range"] == 50.0


@pytest.mark.parametrize("restriction", [RESTRICTION_PAIN,
                                         RESTRICTION_THERAPIST])
def test_uninterpretable_restrictions_start_at_lower_bounds(restriction):
    result = recommend(baseline=120.0, restriction=restriction)
    assert result["difficulty"] == START_LOWEST
    assert result["parameters"] == {"target_reps": 3, "target_range": 50.0}
    assert restriction in result["reason"]


def test_limited_range_uses_the_baseline_cap():
    result = recommend(baseline=66.0, restriction=RESTRICTION_LIMITED_RANGE)
    assert result["parameters"]["target_range"] == 65.0


def test_degrees_baseline_does_not_change_reach_targets():
    params, start, note = starting_parameters(REACH, 40.0)
    assert params == REACH.default_parameters()
    assert "does not apply" in note


def test_recommendation_is_deterministic():
    assert recommend(baseline=72.0) == recommend(baseline=72.0)


# ---------------------------------------------------------
# Override record
# ---------------------------------------------------------

def test_override_note_records_exercise_and_parameter_changes():
    assert override_note("Shoulder Flexion", {"target_reps": 10,
                         "target_range": 70.0}, FLEXION,
                         {"target_reps": 10, "target_range": 70.0}) == ""
    note = override_note("Shoulder Flexion",
                         {"target_reps": 10, "target_range": 70.0},
                         FLEXION, {"target_reps": 6, "target_range": 70.0})
    assert "target repetitions 10 -> 6" in note
    note = override_note("Shoulder Flexion", {}, ELBOW,
                         ELBOW.default_parameters())
    assert "prescribed Elbow Flexion instead of the suggested Shoulder" \
        in note


# ---------------------------------------------------------
# Assessment page: suggestion -> therapist edits -> approved prescription
# ---------------------------------------------------------

def fill_assessment(page, patient_id, baseline, restriction=RESTRICTION_NONE):
    page.patient_input.setCurrentIndex(page.patient_input.findData(patient_id))
    page.diagnosis_input.setCurrentText("Stroke")
    page.impairment_input.setCurrentText("Reduced ROM")
    page.rom_input.setValue(baseline)
    page.restrictions_input.setCurrentText(restriction)
    page.save_assessment()


@pytest.fixture
def patient(temp_db):
    patient_id = db.add_patient("Synthetic", 60, "Other", "T", "")
    db.add_affected_area(patient_id, "LEFT", "Upper Limb", "Shoulder", "")
    return patient_id


def test_suggested_parameters_reach_the_prescription(qapp, patient):
    from src.ui.pages.assessment_page import AssessmentPage

    page = AssessmentPage()
    fill_assessment(page, patient, 72.0)
    assert page.parameter_inputs["target_range"].value() == 70.0
    page.approve_recommendation()

    prescription = db.get_active_prescription(patient)
    assert prescription["exercise_id"] == "shoulder_flexion"
    assert prescription["parameters"] == {"target_range": 70.0,
                                          "target_reps": 10}
    assert "Therapist override" not in prescription["reason"]


def test_therapist_edited_parameters_are_stored_with_the_reason(qapp,
                                                                patient):
    from src.ui.pages.assessment_page import AssessmentPage

    page = AssessmentPage()
    fill_assessment(page, patient, 120.0)
    page.parameter_inputs["target_reps"].setValue(6)
    page.parameter_inputs["target_range"].setValue(500.0)   # out of bounds
    page.approve_recommendation()

    prescription = db.get_active_prescription(patient)
    assert prescription["parameters"] == {"target_reps": 6,
                                          "target_range": 150.0}
    assert "Therapist override" in prescription["reason"]
    assert prescription["difficulty"] == "Therapist-set start"


def test_choosing_another_exercise_recomputes_and_records(qapp, patient):
    from src.ui.pages.assessment_page import AssessmentPage

    page = AssessmentPage()
    fill_assessment(page, patient, 72.0)
    page.exercise_choice.setCurrentIndex(
        page.exercise_choice.findData("shoulder_abduction"))
    assert page.parameter_inputs["target_range"].value() == 70.0
    page.approve_recommendation()

    prescription = db.get_active_prescription(patient)
    assert prescription["exercise"] == "Shoulder Abduction"
    assert ("prescribed Shoulder Abduction instead of the suggested "
            "Shoulder Flexion") in prescription["reason"]


def test_rejected_recommendation_prescribes_nothing(qapp, patient):
    from src.ui.pages.assessment_page import AssessmentPage

    page = AssessmentPage()
    fill_assessment(page, patient, 120.0)
    page.reject_recommendation()
    assert db.get_active_prescription(patient) is None


def test_no_eligible_exercise_offers_nothing_to_approve(qapp, temp_db):
    from src.ui.pages.assessment_page import AssessmentPage

    patient_id = db.add_patient("Synthetic", 60, "Other", "T", "")
    db.add_affected_area(patient_id, "LEFT", "Upper Limb", "Elbow", "")
    page = AssessmentPage()
    fill_assessment(page, patient_id, 40.0)
    assert page.current_recommendation_id is None
    assert not page.approve_button.isEnabled()
    assert page.parameter_inputs == {}
