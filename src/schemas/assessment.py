"""Clinical assessment and therapist-review API schemas."""

from pydantic import BaseModel, Field


class AssessmentCreate(BaseModel):
    diagnosis: str = Field(min_length=1, max_length=120)
    impairment: str = Field(min_length=1, max_length=120)
    baseline_rom: float | None = Field(default=None, ge=0, le=360)
    restrictions: str | None = Field(default=None, max_length=160)
    clinical_notes: str | None = None


class AssessmentRead(BaseModel):
    assessment_id: int
    patient_id: int
    diagnosis: str
    impairment: str
    baseline_rom: float | None = None
    restrictions: str | None = None
    clinical_notes: str | None = None
    assessment_date: str | None = None


class RecommendationReview(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    exercise_id: str | None = Field(default=None, max_length=100)
    parameters: dict[str, float | int] | None = None
    note: str | None = None


class RecommendationRead(BaseModel):
    recommendation_id: int
    patient_id: int
    assessment_id: int
    exercise: str
    exercise_id: str | None = None
    difficulty: str | None = None
    reason: str | None = None
    status: str
    parameters: dict = Field(default_factory=dict)
    rules_version: str | None = None
