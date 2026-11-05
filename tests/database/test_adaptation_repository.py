"""Unit tests for SQLAlchemy adaptive decision and review persistence."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    AdaptationDecision,
    Base,
    ClinicalAssessment,
    ExerciseRecommendation,
    Patient,
)
from src.database.repositories import adaptation_repository


@pytest.fixture
def database_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    patient = Patient(name="Synthetic Rehab Patient")
    session.add(patient)
    session.flush()
    assessment = ClinicalAssessment(
        patient_id=patient.patient_id,
        diagnosis="Stroke",
        impairment="Reduced shoulder flexion",
    )
    session.add(assessment)
    session.commit()
    yield session, patient.patient_id, assessment.assessment_id
    session.close()
    engine.dispose()


def _record(session, patient_id, assessment_id, previous_id=11, target=8):
    return adaptation_repository.record_decision(
        session,
        patient_id=patient_id,
        exercise_id="shoulder_flexion",
        affected_side="LEFT",
        session_id=None,
        policy_version="test-policy",
        decision="PROGRESS",
        rationale="Synthetic test decision",
        inputs={"controller": {"action": "PROGRESS"}},
        previous_parameters={"target_reps": 5},
        proposed_parameters={"target_reps": target},
        previous_recommendation_id=previous_id,
        proposal={
            "assessment_id": assessment_id,
            "exercise": "Shoulder Flexion",
            "difficulty": "Level 1",
            "reason": "PROGRESS: Synthetic test decision",
        },
    )


def test_decision_and_pending_proposal_are_recorded_atomically(database_session):
    session, patient_id, assessment_id = database_session
    decision_id, proposal_id = _record(session, patient_id, assessment_id)

    decision = session.get(AdaptationDecision, decision_id)
    proposal = session.get(ExerciseRecommendation, proposal_id)
    assert decision.proposed_recommendation_id == proposal_id
    assert proposal.status == "PENDING"
    assert proposal.source == "ADAPTATION"
    assert proposal.adaptation_decision_id == decision_id
    assert proposal.parameters_json == '{"target_reps": 8}'


def test_new_proposal_supersedes_older_pending_proposal(database_session):
    session, patient_id, assessment_id = database_session
    _, first_id = _record(session, patient_id, assessment_id, target=8)
    _, second_id = _record(session, patient_id, assessment_id, target=9)

    assert session.get(ExerciseRecommendation, first_id).status == "SUPERSEDED"
    assert session.get(ExerciseRecommendation, second_id).status == "PENDING"


def test_therapist_review_is_single_transition(database_session):
    session, patient_id, assessment_id = database_session
    _, proposal_id = _record(session, patient_id, assessment_id)

    result = adaptation_repository.review_proposal(
        session,
        proposal_id,
        status="APPROVED",
        reviewer="Synthetic Therapist",
        note="Reviewed in automated test",
    )
    assert result == "APPROVED"
    proposal = session.get(ExerciseRecommendation, proposal_id)
    assert proposal.status == "APPROVED"
    assert proposal.reviewed_by == "Synthetic Therapist"
    assert proposal.approved_at is not None

    with pytest.raises(RuntimeError, match="no longer pending"):
        adaptation_repository.review_proposal(
            session,
            proposal_id,
            status="REJECTED",
            reviewer="Synthetic Therapist",
        )


def test_pending_and_decision_history_keep_review_context(database_session):
    session, patient_id, assessment_id = database_session
    decision_id, proposal_id = _record(session, patient_id, assessment_id)

    pending = adaptation_repository.list_pending_proposals(session, patient_id)
    assert len(pending) == 1
    assert pending[0]["recommendation_id"] == proposal_id
    assert pending[0]["decision"] == "PROGRESS"

    history = adaptation_repository.list_decision_history(session, patient_id)
    assert len(history) == 1
    assert history[0]["decision_id"] == decision_id
    assert history[0]["action"] == "PROGRESS"
    assert history[0]["proposal_status"] == "PENDING"
