"""Authentication, database-session, and role-based authorization dependencies."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.services.auth_service import AuthenticationError, decode_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_db_session(request: Request):
    """Yield a request-scoped SQLAlchemy session from the app-owned factory."""
    factory = getattr(request.app.state, "database_session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session factory is not initialized.",
        )
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_db_session),
):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(session, credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_roles(*allowed_roles):
    def dependency(user=Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your role is not permitted to perform this action.",
            )
        return user
    return dependency



def assert_patient_access(session: Session, patient_id: int, user):
    """Raise a non-disclosing 404 unless the user is assigned to the patient."""
    if user["role"] == "admin":
        return
    from src.database.repositories import assignment_repository

    if not assignment_repository.therapist_has_patient(
        session, patient_id=patient_id, therapist_user_id=user["user_id"]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient was not found.",
        )


def require_patient_access(
    patient_id: int,
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    """FastAPI dependency for routes with patient_id as a path parameter."""
    assert_patient_access(session, patient_id, user)
    return user


def require_session_access(
    session_id: str,
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    """Resolve session ownership before exposing or mutating a live session."""
    from src.database.repositories import live_session_repository

    live = live_session_repository.get_live_session(session, session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="Session was not found.")
    if user["role"] != "admin":
        from src.database.repositories import assignment_repository

        if not assignment_repository.therapist_has_patient(
            session, patient_id=live.patient_id, therapist_user_id=user["user_id"]
        ):
            raise HTTPException(status_code=404, detail="Session was not found.")
    return user



def require_proposal_access(
    proposal_id: int,
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    """Require access to the patient attached to an adaptive proposal."""
    if user["role"] == "admin":
        return user
    from src.database.repositories import adaptation_repository, assignment_repository

    proposal = adaptation_repository.get_adaptation_proposal(session, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Adaptation proposal was not found.")
    if user["role"] != "admin" and not assignment_repository.therapist_has_patient(
        session, patient_id=proposal["patient_id"], therapist_user_id=user["user_id"]
    ):
        raise HTTPException(status_code=404, detail="Adaptation proposal was not found.")
    return user



def require_recommendation_access(
    recommendation_id: int,
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    """Require access to the patient attached to a recommendation."""
    if user["role"] == "admin":
        return user
    from src.database.repositories import assessment_repository, assignment_repository

    recommendation = assessment_repository.get_recommendation(session, recommendation_id)
    if recommendation is None:
        raise HTTPException(status_code=404, detail="Recommendation was not found.")
    if user["role"] != "admin" and not assignment_repository.therapist_has_patient(
        session, patient_id=recommendation["patient_id"], therapist_user_id=user["user_id"]
    ):
        raise HTTPException(status_code=404, detail="Recommendation was not found.")
    return user
