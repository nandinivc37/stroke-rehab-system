"""Schemas for landmark-based live rehabilitation sessions.

The API accepts pose landmarks rather than raw video frames. This keeps
camera capture and MediaPipe inference on the patient-side client.
"""

from pydantic import BaseModel, Field, model_validator


class PoseLandmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    presence: float | None = Field(default=None, ge=0.0, le=1.0)


class PoseFrameCreate(BaseModel):
    timestamp_s: float = Field(ge=0.0)
    pose_landmarks: list[PoseLandmark] | None = None
    pose_world_landmarks: list[PoseLandmark] | None = None
    image_width: int | None = Field(default=None, gt=0, le=16384)
    image_height: int | None = Field(default=None, gt=0, le=16384)

    @model_validator(mode="after")
    def validate_landmark_count(self):
        for field_name in ("pose_landmarks", "pose_world_landmarks"):
            points = getattr(self, field_name)
            if points is not None and len(points) != 33:
                raise ValueError(f"{field_name} must contain exactly 33 landmarks.")
        return self


class SessionStart(BaseModel):
    patient_id: int = Field(gt=0)


class SessionStarted(BaseModel):
    session_id: str
    patient_id: int
    exercise_id: str
    exercise: str
    affected_side: str
    prescription_id: int
    status: str


class FrameAnalysisRead(BaseModel):
    status: str
    angle: float | None = None
    plane: float | None = None
    reps: int
    movement_active: bool


class SessionCompleteRead(BaseModel):
    session_id: int
    patient_id: int
    exercise_id: str
    exercise: str
    status: str
    summary: dict
