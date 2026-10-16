"""
Adaptive loop persistence:

    session saved -> adaptive controller (controller.decide: attempt gate,
                     attempt check, policy.evaluate, safety guard)
                  -> record decision, action and patient feedback
                  -> (if parameters would change) PENDING prescription
    therapist approves -> APPROVED prescription
    next session -> rehab page loads the latest APPROVED prescription

Every evaluation is stored in adaptation_decisions with its inputs,
previous parameters, decision, rationale and proposed parameters,
including MAINTAIN / INSUFFICIENT_DATA outcomes.
"""

import json

from src.adaptation.controller import (
    ControllerConfig,
    compensation_from_session,
    decide,
)
from src.adaptation.policy import POLICY_VERSION, PolicyConfig
from src.database import db
from src.exercises.catalogue import get_exercise_config

STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_SUPERSEDED = "SUPERSEDED"


class AdaptationError(RuntimeError):
    pass


def propose_after_session(patient_id, session_id, policy=None):
    """
    Run the adaptive controller after a saved session (one attempt).

    Returns a dict: action, feedback, decision, rationale, decision_id,
    proposal_id (None unless a parameter change is proposed),
    previous/proposed parameters and next_attempt_parameters.
    Returns None when the session has no supported active prescription.
    """

    prescription = db.get_active_prescription(patient_id)

    if prescription is None:
        return None

    exercise = get_exercise_config(
        prescription.get("exercise_id") or prescription["exercise"]
    )

    if exercise is None or not exercise.implemented:
        return None

    session = next(
        (s for s in db.get_sessions(patient_id)
         if s["session_id"] == session_id),
        None,
    )

    if session is None or session.get("exercise_id") != exercise.exercise_id:
        return None

    side = session["affected_side"]
    history = db.get_sessions(patient_id, exercise.exercise_id, side)

    # Compensation columns are recorded as context only (all NULL while
    # no validated classifier exists); the controller never uses them.
    result = decide(
        exercise, prescription["parameters"], session, history,
        compensation=compensation_from_session(session),
        config=ControllerConfig(policy=policy or PolicyConfig()),
    )
    inputs = dict(result.policy_inputs)
    inputs["controller"] = {
        k: v for k, v in result.to_dict().items() if k != "policy_inputs"
    }

    conn = db.get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO adaptation_decisions(
                patient_id, exercise_id, affected_side, session_id,
                policy_version, decision, rationale, inputs_json,
                previous_parameters_json, proposed_parameters_json,
                previous_recommendation_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            exercise.exercise_id,
            side,
            session_id,
            POLICY_VERSION,
            result.decision,
            result.reason,
            json.dumps(inputs, sort_keys=True, default=str),
            json.dumps(result.previous_parameters, sort_keys=True),
            json.dumps(result.proposed_parameters, sort_keys=True),
            prescription["recommendation_id"],
        ))
        decision_id = cursor.lastrowid

        proposal_id = None

        if result.requires_approval:
            # Only one open proposal per patient: older ones are
            # superseded by the newer evaluation.
            conn.execute("""
                UPDATE exercise_recommendations
                SET status = ?
                WHERE patient_id = ?
                  AND source = 'ADAPTATION'
                  AND status = ?
            """, (STATUS_SUPERSEDED, patient_id, STATUS_PENDING))

            cursor = conn.execute("""
                INSERT INTO exercise_recommendations(
                    patient_id, assessment_id, exercise, difficulty,
                    reason, status, exercise_id, affected_side,
                    parameters_json, source, adaptation_decision_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ADAPTATION', ?)
            """, (
                patient_id,
                prescription["assessment_id"],
                exercise.display_name,
                prescription["difficulty"],
                f"{result.decision}: {result.reason}",
                STATUS_PENDING,
                exercise.exercise_id,
                side,
                json.dumps(result.proposed_parameters, sort_keys=True),
                decision_id,
            ))
            proposal_id = cursor.lastrowid

            conn.execute("""
                UPDATE adaptation_decisions
                SET proposed_recommendation_id = ?
                WHERE decision_id = ?
            """, (proposal_id, decision_id))

        conn.commit()
    finally:
        conn.close()

    return {
        "action": result.action,
        "feedback": result.feedback,
        "decision": result.decision,
        "rationale": result.reason,
        "decision_id": decision_id,
        "proposal_id": proposal_id,
        "previous_parameters": result.previous_parameters,
        "proposed_parameters": result.proposed_parameters,
        "next_attempt_parameters": result.next_attempt_parameters,
    }


def pending_proposals(patient_id):
    conn = db.get_connection()
    try:
        cursor = conn.execute("""
            SELECT r.recommendation_id, r.exercise, r.exercise_id,
                   r.affected_side, r.parameters_json, r.reason,
                   r.created_at, d.decision, d.previous_parameters_json,
                   d.previous_recommendation_id
            FROM exercise_recommendations r
            JOIN adaptation_decisions d
              ON d.decision_id = r.adaptation_decision_id
            WHERE r.patient_id = ?
              AND r.source = 'ADAPTATION'
              AND r.status = ?
            ORDER BY r.recommendation_id DESC
        """, (patient_id, STATUS_PENDING))
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


def decision_history(patient_id):
    conn = db.get_connection()
    try:
        cursor = conn.execute("""
            SELECT d.*, r.status AS proposal_status, r.reviewed_by
            FROM adaptation_decisions d
            LEFT JOIN exercise_recommendations r
              ON r.recommendation_id = d.proposed_recommendation_id
            WHERE d.patient_id = ?
            ORDER BY d.decision_id DESC
        """, (patient_id,))
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()

    # Controller action and patient feedback (None for decisions logged
    # before the controller existed).
    for row in rows:
        try:
            controller = json.loads(row["inputs_json"] or "{}").get(
                "controller") or {}
        except (TypeError, ValueError, AttributeError):
            controller = {}
        row["action"] = controller.get("action")
        row["feedback"] = controller.get("feedback")
    return rows


def review_proposal(proposal_id, approve, reviewer, note=""):
    """
    Approve or reject a PENDING adaptation proposal.

    Approval is refused when the prescription the proposal was based on
    is no longer the active one (e.g. the therapist assigned a different
    exercise in the meantime).
    """

    if not reviewer or not reviewer.strip():
        raise AdaptationError("A reviewer name is required.")

    conn = db.get_connection()
    try:
        row = conn.execute("""
            SELECT r.patient_id, r.status, d.previous_recommendation_id
            FROM exercise_recommendations r
            JOIN adaptation_decisions d
              ON d.decision_id = r.adaptation_decision_id
            WHERE r.recommendation_id = ?
              AND r.source = 'ADAPTATION'
        """, (proposal_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        raise AdaptationError("Adaptation proposal not found.")

    patient_id, status, based_on = row

    if status != STATUS_PENDING:
        raise AdaptationError(f"Proposal is already {status.lower()}.")

    if approve:
        active = db.get_active_prescription(patient_id)
        if active is None or active["recommendation_id"] != based_on:
            raise AdaptationError(
                "The prescription this proposal was based on is no longer "
                "active. Reject it; a new proposal will follow the next "
                "session."
            )

    new_status = STATUS_APPROVED if approve else STATUS_REJECTED

    conn = db.get_connection()
    try:
        conn.execute("""
            UPDATE exercise_recommendations
            SET status = ?,
                approved_at = CASE WHEN ? = 'APPROVED'
                                   THEN CURRENT_TIMESTAMP
                                   ELSE approved_at END,
                reviewed_by = ?,
                review_note = ?
            WHERE recommendation_id = ?
        """, (new_status, new_status, reviewer.strip(), note, proposal_id))
        conn.commit()
    finally:
        conn.close()

    return new_status


__all__ = [
    "POLICY_VERSION",
    "AdaptationError",
    "decision_history",
    "pending_proposals",
    "propose_after_session",
    "review_proposal",
]
