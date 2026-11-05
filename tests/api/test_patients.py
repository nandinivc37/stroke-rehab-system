"""Tests for patient and affected-area API endpoints."""


def test_create_and_get_patient(api_client):
    created = api_client.post(
        "/api/v1/patients",
        json={
            "name": "Synthetic Test Patient",
            "age": 58,
            "gender": "Other",
            "therapist": "Test Therapist",
            "notes": "Synthetic test data only",
        },
    )

    assert created.status_code == 201
    patient = created.json()
    assert patient["patient_id"] > 0
    assert patient["name"] == "Synthetic Test Patient"

    fetched = api_client.get(f"/api/v1/patients/{patient['patient_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == patient


def test_list_patients(api_client):
    api_client.post("/api/v1/patients", json={"name": "Patient One"})
    response = api_client.get("/api/v1/patients")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "Patient One"


def test_blank_patient_name_is_rejected(api_client):
    response = api_client.post("/api/v1/patients", json={"name": "   "})

    assert response.status_code == 422


def test_missing_patient_returns_404(api_client):
    response = api_client.get("/api/v1/patients/987654")

    assert response.status_code == 404


def test_create_and_list_affected_areas(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Area Patient"}
    ).json()
    patient_id = patient["patient_id"]

    created = api_client.post(
        f"/api/v1/patients/{patient_id}/affected-areas",
        json={
            "side": "LEFT",
            "body_region": "Arm",
            "joint": "Shoulder",
            "notes": "Synthetic affected-area test",
        },
    )

    assert created.status_code == 201
    assert created.json()["body_region"] == "Upper Limb"

    listed = api_client.get(f"/api/v1/patients/{patient_id}/affected-areas")
    assert listed.status_code == 200
    assert listed.json()[0]["side"] == "LEFT"


def test_affected_areas_for_missing_patient_returns_404(api_client):
    response = api_client.get("/api/v1/patients/987654/affected-areas")

    assert response.status_code == 404
