"""Transport schemas for NARS API authentication."""

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class TokenRead(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=12, max_length=256)
    role: str = Field(pattern="^(therapist|admin)$")

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value):
        value = value.strip().lower()
        if not value:
            raise ValueError("Username cannot be blank.")
        return value


class UserRead(BaseModel):
    user_id: int
    username: str
    role: str
    is_active: bool
