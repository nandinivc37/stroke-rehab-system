"""Login and administrator-managed API user provisioning."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_current_user, get_db_session, require_roles
from src.schemas.auth import LoginRequest, TokenRead, UserCreate, UserRead
from src.services import auth_service

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenRead)
def login(payload: LoginRequest, session: Session = Depends(get_db_session)):
    try:
        user = auth_service.authenticate(session, payload.username, payload.password)
        return auth_service.issue_token(user)
    except auth_service.AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.get("/me", response_model=UserRead)
def read_current_user(user=Depends(get_current_user)):
    """Return the authenticated account, using its current database role."""
    return user


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("admin"))],
)
def create_user(
    payload: UserCreate,
    session: Session = Depends(get_db_session),
):
    try:
        return auth_service.create_user(
            session, payload.username, payload.password, payload.role
        )
    except auth_service.UserConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
