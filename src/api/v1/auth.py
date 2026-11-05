"""Login and administrator-managed API user provisioning."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import require_roles
from src.schemas.auth import LoginRequest, TokenRead, UserCreate, UserRead
from src.services import auth_service

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenRead)
def login(payload: LoginRequest):
    try:
        user = auth_service.authenticate(payload.username, payload.password)
        return auth_service.issue_token(user)
    except auth_service.AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("admin"))],
)
def create_user(payload: UserCreate):
    try:
        return auth_service.create_user(
            payload.username, payload.password, payload.role
        )
    except auth_service.UserConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
