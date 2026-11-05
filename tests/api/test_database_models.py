"""Tests for NARS ORM metadata and initial relational schema."""

from sqlalchemy import create_engine, inspect

from src.database.sqlalchemy_support import Base
from src.database import models  # noqa: F401 - register mappings


def test_orm_metadata_creates_all_core_nars_tables(tmp_path):
    engine = create_engine("sqlite:///" + (tmp_path / "schema.db").as_posix())
    Base.metadata.create_all(engine)
    inspector = inspect(engine)

    expected = {
        "patients",
        "affected_areas",
        "clinical_assessments",
        "exercise_recommendations",
        "sessions",
        "session_frames",
        "adaptation_decisions",
        "api_users",
    }
    assert expected.issubset(set(inspector.get_table_names()))
    engine.dispose()


def test_historical_session_metrics_remain_nullable():
    columns = {
        column.name: column
        for column in Base.metadata.tables["sessions"].columns
    }
    for name in (
        "metrics_version",
        "peak_rom",
        "dtw_similarity",
        "dtw_mean_error_deg",
        "trajectory_similarity",
        "experimental_posture_json",
        "compensation_json",
    ):
        assert columns[name].nullable is True


def test_session_frame_log_keeps_composite_primary_key():
    table = Base.metadata.tables["session_frames"]
    primary_key = {column.name for column in table.primary_key.columns}
    assert primary_key == {"session_id", "frame_index"}



def test_api_users_enforce_roles_and_case_insensitive_usernames():
    table = Base.metadata.tables["api_users"]
    assert any(
        constraint.name == "ck_api_users_role"
        for constraint in table.constraints
    )
    assert "uq_api_users_username_lower" in {
        index.name for index in table.indexes
    }
