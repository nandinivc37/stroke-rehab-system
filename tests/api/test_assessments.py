"""API tests for assessment creation and therapist-reviewed prescriptions."""

from src.database import db
from src.exercises.catalogue import get_exercise_config


def create_patient_with_shoulder(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Assessment Patient"}
    )
    assert patient.status_code == 201
    patient_id = patient.json()["patient_id"]
    area = api_client.post(
        f"/api/v1/patients/{patient_id}/affected-areas",
        json={
            "side": "LEFT",
            "body_region": "Upper Limb",
            "joint": "Shoulder",
            "notes": "Synthetic test record",
        },
    )
    assert area.status_code == 201
    return patient_id


def create_assessment(api_client, patient_id):
    response = api_client.post(
        f"/api/v1/patients/{patient_id}/assessments",
        json={
            "diagnosis": "Stroke",
            "impairment": "Reduced ROM",
            "baseline_rom": 120,
            "restrictions": "No major restriction",
            "clinical_notes": "Synthetic assessment for API test",
        },
    )
    assert response.status_code == 201
    return response.json()


def test_create_and_get_current_assessment(api_client):
    patient_id = create_patient_with_shoulder(api_client)
    assessment = create_assessment(api_client, patient_id)

    assert assessment["patient_id"] == patient_id
    assert assessment["diagnosis"] == "Stroke"

    current = api_client.get(
        f"/api/v1/patients/{patient_id}/assessments/current"
    )
    assert current.status_code == 200
    assert current.json()["assessment_id"] == assessment["assessment_id"]


def test_assessment_requires_affected_area(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic No-Area Patient"}
    ).json()
    response = api_client.post(
        f"/api/v1/patients/{patient['patient_id']}/assessments",
        json={"diagnosis": "Stroke", "impairment": "Reduced ROM"},
    )
    assert response.status_code == 422


def test_recommendation_requires_therapist_approval(api_client):
    patient_id = create_patient_with_shoulder(api_client)
    assessment = create_assessment(api_client, patient_id)

    generated = api_client.post(
        f"/api/v1/patients/{patient_id}/recommendations/generate",
        json={"assessment_id": assessment["assessment_id"]},
    )
    assert generated.status_code == 200
    recommendation = generated.json()
    assert recommendation["status"] == "PENDING"
    recommendation_id = recommendation["recommendation_id"]
    assert db.get_active_prescription(patient_id) is None

    config = get_exercise_config(recommendation["exercise_id"])
    approved = api_client.post(
        f"/api/v1/recommendations/{recommendation_id}/approve",
        json={
            "reviewer": "Synthetic Therapist",
            "exercise_id": config.exercise_id,
            "parameters": config.default_parameters(),
            "note": "Synthetic approval test",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert approved.json()["reviewed_by"] == "test-therapist"
    assert approved.json()["review_note"] == "Synthetic approval test"

    prescription = db.get_active_prescription(patient_id)
    assert prescription["recommendation_id"] == recommendation_id
    assert prescription["exercise_id"] == config.exercise_id


def test_rejected_recommendation_does_not_create_prescription(api_client):
    patient_id = create_patient_with_shoulder(api_client)
    assessment = create_assessment(api_client, patient_id)
    generated = api_client.post(
        f"/api/v1/patients/{patient_id}/recommendations/generate",
        json={"assessment_id": assessment["assessment_id"]},
    ).json()

    rejected = api_client.post(
        f"/api/v1/recommendations/{generated['recommendation_id']}/reject",
        json={"reviewer": "Synthetic Therapist", "note": "Not appropriate"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"
    assert rejected.json()["reviewed_by"] == "test-therapist"
    assert db.get_active_prescription(patient_id) is None


def test_recommendation_cannot_be_approved_twice(api_client):
    patient_id = create_patient_with_shoulder(api_client)
    assessment = create_assessment(api_client, patient_id)
    generated = api_client.post(
        f"/api/v1/patients/{patient_id}/recommendations/generate",
        json={"assessment_id": assessment["assessment_id"]},
    ).json()
    config = get_exercise_config(generated["exercise_id"])
    payload = {
        "reviewer": "Synthetic Therapist",
        "exercise_id": config.exercise_id,
        "parameters": config.default_parameters(),
    }
    endpoint = f"/api/v1/recommendations/{generated['recommendation_id']}/approve"
    assert api_client.post(endpoint, json=payload).status_code == 200
    assert api_client.post(endpoint, json=payload).status_code == 409
