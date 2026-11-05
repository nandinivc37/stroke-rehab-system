"""Patient API schemas. These describe transport data, not clinical validation."""

from pydantic import BaseModel, ConfigDict, Field


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    age: int | None = Field(default=None, ge=0, le=120)
    gender: str | None = Field(default=None, max_length=80)
    therapist: str | None = Field(default=None, max_length=200)
    notes: str | None = None


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: int
    name: str
    age: int | None = None
    gender: str | None = None
    therapist: str | None = None
    notes: str | None = None


class AffectedAreaCreate(BaseModel):
    side: str = Field(min_length=1, max_length=30)
    body_region: str = Field(min_length=1, max_length=80)
    joint: str | None = Field(default=None, max_length=80)
    notes: str | None = None


class AffectedAreaRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    area_id: int
    side: str
    body_region: str
    joint: str | None = None
    notes: str | None = None
