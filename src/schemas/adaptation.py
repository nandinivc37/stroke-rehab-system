"""Schemas for conservative adaptive decision-support endpoints."""

from pydantic import BaseModel, Field, field_validator


class AdaptationEvaluate(BaseModel):
    session_id: int = Field(gt=0)


class AdaptationReview(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=2000)

    @field_validator("reviewer")
    @classmethod
    def reviewer_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("Reviewer name cannot be blank.")
        return value.strip()


class AdaptationReviewRead(BaseModel):
    proposal_id: int
    status: str
    reviewer: str
