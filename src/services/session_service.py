"""Process-local session manager for landmark-based NARS analysis.

This first API implementation deliberately receives MediaPipe landmarks,
not camera frames. Camera access and pose inference remain on the client.
Session analyzer state is process-local; use one API worker for this
prototype. A shared state store is required before multi-worker deployment.
"""

from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

from src.database import db
from src.database.repositories import (
    session_repository,
    live_session_repository,
    patient_repository,
    assessment_repository,
)
from src.exercises.analyzer import ExerciseAnalyzer
from src.exercises.catalogue import get_exercise_config
from src.services.patient_service import get_patient


class SessionNotFoundError(LookupError):
    pass


class SessionConflictError(RuntimeError):
    pass


class SessionValidationError(ValueError):
    pass


class _SessionState:
    def __init__(self, session_id, patient_id, prescription, config, analyzer):
        self.session_id = session_id
        self.patient_id = patient_id
        self.prescription = prescription
        self.config = config
        self.analyzer = analyzer
        self.last_timestamp = None
        self.frame_count = 0


class SessionManager:
    """Owns analyzer instances and serializes updates per session."""

    def __init__(self):
        self._sessions = {}
        self._lock = RLock()

    def start(self, patient_id, database_session=None):
        if database_session is None:
            get_patient(patient_id)
            prescription = db.get_active_prescription(patient_id)
            areas = None
        else:
            get_patient(patient_id, session=database_session)
            approved = assessment_repository.list_recommendations(
                database_session, patient_id, status="APPROVED"
            )
            prescription = approved[0] if approved else None
            areas = patient_repository.list_affected_areas(
                database_session, patient_id
            )
        if prescription is None:
            raise SessionValidationError(
                "The patient does not have an active therapist-approved prescription."
            )

        config = get_exercise_config(
            prescription.get("exercise_id") or prescription.get("exercise")
        )
        if config is None or not config.implemented:
            raise SessionValidationError(
                "The prescribed exercise is not currently implemented."
            )

        side = (prescription.get("affected_side") or "").upper()
        if not side:
            if database_session is None:
                legacy_areas = db.get_affected_areas(patient_id)
                side = legacy_areas[0][1].upper() if legacy_areas else ""
            else:
                side = areas[0].side.upper() if areas else ""
        if side not in config.sides:
            raise SessionValidationError(
                "The active prescription has no supported affected side."
            )

        parameters = prescription.get("parameters") or None
        analyzer = ExerciseAnalyzer(config, side, parameters)
        session_id = str(uuid4())
        state = _SessionState(
            session_id, patient_id, prescription, config, analyzer
        )
        if database_session is not None:
            live_session_repository.create_live_session(
                database_session,
                session_id=session_id,
                patient_id=patient_id,
                prescription_id=prescription["recommendation_id"],
                exercise_id=config.exercise_id,
                affected_side=side,
            )
        with self._lock:
            self._sessions[session_id] = state
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "exercise_id": config.exercise_id,
            "exercise": config.display_name,
            "affected_side": side,
            "prescription_id": prescription["recommendation_id"],
            "status": "ACTIVE",
        }

    @staticmethod
    def _pose_results(frame):
        if frame.pose_landmarks is None:
            return None
        pose = SimpleNamespace(
            landmark=[SimpleNamespace(**point.model_dump())
                      for point in frame.pose_landmarks]
        )
        world = None
        if frame.pose_world_landmarks is not None:
            world = SimpleNamespace(
                landmark=[SimpleNamespace(**point.model_dump())
                          for point in frame.pose_world_landmarks]
            )
        return SimpleNamespace(
            pose_landmarks=pose,
            pose_world_landmarks=world,
        )

    def has_session(self, session_id):
        """Return whether an in-memory analyzer is active for this session."""
        with self._lock:
            return session_id in self._sessions

    def add_frame(self, session_id, frame):
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError("Active session was not found.")
            if (
                state.last_timestamp is not None
                and frame.timestamp_s <= state.last_timestamp
            ):
                raise SessionValidationError(
                    "Frame timestamps must increase strictly within a session."
                )

            if frame.image_width and frame.image_height:
                state.analyzer.set_image_size(
                    frame.image_width, frame.image_height
                )

            results = self._pose_results(frame)
            if results is None:
                analysis = state.analyzer.update(
                    None, timestamp=frame.timestamp_s
                )
            else:
                analysis = state.analyzer.update(
                    results, timestamp=frame.timestamp_s
                )
            state.last_timestamp = frame.timestamp_s
            state.frame_count += 1
            return {
                "status": analysis.status,
                "angle": analysis.angle,
                "plane": analysis.plane,
                "reps": analysis.reps,
                "movement_active": analysis.movement_active,
            }

    def complete(self, session_id, database_session=None):
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError("Active session was not found.")
            if state.frame_count == 0:
                raise SessionValidationError(
                    "A session with no received frames cannot be saved."
                )

            summary = state.analyzer.summary()
            summary["session_status"] = db.SESSION_STATUS_COMPLETE
            if database_session is None:
                session_db_id = db.save_session_summary(
                    patient_id=state.patient_id,
                    exercise_name=state.config.display_name,
                    stats=summary,
                    prescription_id=state.prescription["recommendation_id"],
                    session_status=db.SESSION_STATUS_COMPLETE,
                )
            else:
                session_db_id = session_repository.save_session_summary(
                    database_session,
                    patient_id=state.patient_id,
                    exercise_name=state.config.display_name,
                    stats=summary,
                    prescription_id=state.prescription["recommendation_id"],
                    session_status=db.SESSION_STATUS_COMPLETE,
                )
            if database_session is not None:
                live_session_repository.finish_live_session(
                    database_session, session_id, "COMPLETED",
                    frames_received=state.frame_count,
                )
            del self._sessions[session_id]

            return {
                "session_id": session_db_id,
                "patient_id": state.patient_id,
                "exercise_id": state.config.exercise_id,
                "exercise": state.config.display_name,
                "status": db.SESSION_STATUS_COMPLETE,
                "summary": summary,
            }

    def cancel(self, session_id, database_session=None):
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError("Active session was not found.")
            if database_session is not None:
                live_session_repository.finish_live_session(
                    database_session, session_id, "CANCELLED",
                    frames_received=state.frame_count,
                )
            del self._sessions[session_id]
            return {"session_id": session_id, "status": "CANCELLED"}


session_manager = SessionManager()
