"""Conservative, verified copy of NARS SQLite records into PostgreSQL.

Default mode is a read-only dry run. Applying requires --apply and a target
whose dialect is PostgreSQL and whose NARS tables are empty. The source is
opened read-only and snapshotted through SQLite's backup API before copying.
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, MetaData, func, inspect, select, text

from src.database.migrations import SCHEMA_VERSION, schema_version
from src.database.sqlalchemy_support import create_database_engine
from src.paths import DB_BACKUP_DIR, DB_PATH


TABLE_ORDER = (
    "patients",
    "affected_areas",
    "clinical_assessments",
    "exercise_recommendations",
    "sessions",
    "session_frames",
    "adaptation_decisions",
    "api_users",
)

REQUIRED_TABLES = set(TABLE_ORDER) - {"api_users"}


class DataMigrationError(RuntimeError):
    """Raised when the source or target is unsafe to migrate."""


def _readonly_connection(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _source_tables(conn):
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        if not row[0].startswith("sqlite_")
    }


def _row_counts(conn):
    return {
        table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in sorted(_source_tables(conn))
    }


def _validate_source(conn):
    version = schema_version(conn)
    if version < SCHEMA_VERSION:
        raise DataMigrationError(
            f"SQLite schema version is {version}; expected at least "
            f"{SCHEMA_VERSION}. Run the existing SQLite schema migration "
            "and review its verified backup before exporting."
        )
    source_tables = _source_tables(conn)
    missing = REQUIRED_TABLES - source_tables
    unsupported = source_tables - set(TABLE_ORDER)
    if unsupported:
        raise DataMigrationError(
            "Source contains tables that are not mapped for migration: "
            + ", ".join(sorted(unsupported))
        )
    if missing:
        raise DataMigrationError(
            "Required source tables are missing: " + ", ".join(sorted(missing))
        )
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise DataMigrationError(
            f"SQLite foreign_key_check found {len(violations)} violation(s). "
            "Resolve them before migration."
        )


def _target_tables(engine):
    return set(inspect(engine).get_table_names())


def _load_target_metadata(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine, only=list(TABLE_ORDER))
    return metadata


def _validate_schema_compatibility(source_conn, target_metadata):
    problems = []
    source_tables = _source_tables(source_conn)
    for name in TABLE_ORDER:
        if name not in source_tables:
            continue
        source_columns = {
            row[1] for row in source_conn.execute(f'PRAGMA table_info("{name}")')
        }
        target_columns = set(target_metadata.tables[name].columns.keys())
        missing_in_target = source_columns - target_columns
        missing_in_source = target_columns - source_columns
        if missing_in_target or missing_in_source:
            details = []
            if missing_in_target:
                details.append(
                    "target missing source columns: "
                    + ", ".join(sorted(missing_in_target))
                )
            if missing_in_source:
                details.append(
                    "source missing target columns: "
                    + ", ".join(sorted(missing_in_source))
                )
            problems.append(f"{name}: " + "; ".join(details))
    if problems:
        raise DataMigrationError(
            "Source and target schemas do not match exactly. No columns will "
            "be silently discarded or fabricated:\n- " + "\n- ".join(problems)
        )


def _validate_target_empty(engine, metadata):
    nonempty = []
    with engine.connect() as connection:
        for name in TABLE_ORDER:
            table = metadata.tables[name]
            count = connection.execute(
                select(func.count()).select_from(table)
            ).scalar_one()
            if count:
                nonempty.append(f"{name} ({count} rows)")
    if nonempty:
        raise DataMigrationError(
            "Target NARS tables must be empty before import: "
            + ", ".join(nonempty)
        )


def build_migration_plan(source_path, target_engine):
    """Inspect both schemas and counts without modifying either database."""
    with _readonly_connection(source_path) as source:
        _validate_source(source)
        missing_target = set(TABLE_ORDER) - _target_tables(target_engine)
        if missing_target:
            raise DataMigrationError(
                "Target schema is incomplete. Apply the reviewed Alembic "
                "revision first; missing: " + ", ".join(sorted(missing_target))
            )
        metadata = _load_target_metadata(target_engine)
        _validate_schema_compatibility(source, metadata)
        _validate_target_empty(target_engine, metadata)
        return {
            "source": str(Path(source_path).resolve()),
            "target_dialect": target_engine.dialect.name,
            "sqlite_schema_version": schema_version(source),
            "row_counts": _row_counts(source),
            "tables_to_copy": [name for name in TABLE_ORDER if name in _source_tables(source)],
            "target_is_empty": True,
            "foreign_key_violations": 0,
        }


def _snapshot_source(source_path, backup_dir):
    """Create and verify a consistent SQLite snapshot using SQLite backup API."""
    source_path = Path(source_path).resolve()
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot = backup_dir / f"{source_path.stem}_pre_postgres_{stamp}.db"

    source = _readonly_connection(source_path)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()

    with sqlite3.connect(snapshot) as check:
        result = check.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            snapshot.unlink(missing_ok=True)
            raise DataMigrationError("SQLite migration snapshot failed integrity_check.")
        counts = _row_counts(check)
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return snapshot, counts, digest


def _coerce_datetime(value):
    if value is None or isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    raise DataMigrationError(
        f"Cannot safely parse stored timestamp {raw!r}; refusing to alter it."
    )


def _prepare_row(row, table):
    values = dict(row)
    for column in table.columns:
        if column.name in values and isinstance(column.type, __import__(
            "sqlalchemy"
        ).DateTime):
            values[column.name] = _coerce_datetime(values[column.name])
    return values


def _reset_postgres_sequences(connection, metadata):
    for name in TABLE_ORDER:
        table = metadata.tables[name]
        primary_key = list(table.primary_key.columns)
        if len(primary_key) != 1:
            continue
        column = primary_key[0]
        if not isinstance(column.type, Integer):
            continue
        # Names are restricted to the fixed TABLE_ORDER allowlist.
        connection.execute(
            text(
                "SELECT setval(pg_get_serial_sequence(:table_name, :column_name), "
                f"COALESCE(MAX({column.name}), 1), MAX({column.name}) IS NOT NULL) "
                f'FROM "{name}"'
            ),
            {"table_name": name, "column_name": column.name},
        )


def apply_migration(source_path, target_engine, backup_dir):
    """Copy all NARS rows in one target transaction, preserving primary keys."""
    if target_engine.dialect.name != "postgresql":
        raise DataMigrationError(
            "Applying a migration is permitted only to PostgreSQL. "
            "SQLite target engines are supported only by isolated tests."
        )

    plan = build_migration_plan(source_path, target_engine)
    snapshot, snapshot_counts, snapshot_hash = _snapshot_source(
        source_path, backup_dir
    )

    try:
        with _readonly_connection(snapshot) as source:
            metadata = _load_target_metadata(target_engine)
            with target_engine.begin() as connection:
                source_tables = _source_tables(source)
                for name in TABLE_ORDER:
                    if name not in source_tables:
                        continue
                    source_rows = source.execute(
                        f'SELECT * FROM "{name}" ORDER BY '
                        + ", ".join(
                            f'"{row[1]}"'
                            for row in source.execute(
                                f'PRAGMA table_info("{name}")'
                            )
                            if row[5] > 0
                        )
                    ).fetchall()
                    table = metadata.tables[name]
                    for offset in range(0, len(source_rows), 500):
                        batch = [
                            _prepare_row(row, table)
                            for row in source_rows[offset:offset + 500]
                        ]
                        if batch:
                            connection.execute(table.insert(), batch)

                _reset_postgres_sequences(connection, metadata)
                source_counts = {
                    name: (
                        source.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                        if name in source_tables else 0
                    )
                    for name in TABLE_ORDER
                }
                target_counts = {
                    name: connection.execute(
                        select(func.count()).select_from(metadata.tables[name])
                    ).scalar_one()
                    for name in TABLE_ORDER
                }
                verification = {
                    "source_counts": source_counts,
                    "target_counts": target_counts,
                    "matches": source_counts == target_counts,
                }
                if not verification["matches"]:
                    raise DataMigrationError(
                        "Post-import row-count verification failed; "
                        "the PostgreSQL transaction will be rolled back. "
                        + json.dumps(verification, sort_keys=True)
                    )

        return {
            "plan": plan,
            "snapshot_path": str(snapshot),
            "snapshot_sha256": snapshot_hash,
            "snapshot_row_counts": snapshot_counts,
            "verification": verification,
        }
    except Exception:
        # Keep the verified snapshot for diagnosis and recovery.
        raise


def verify_migration(source_path, target_engine):
    """Compare per-table row counts after import; no source writes occur."""
    with _readonly_connection(source_path) as source:
        source_counts = {
            name: (
                source.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                if name in _source_tables(source) else 0
            )
            for name in TABLE_ORDER
        }
    metadata = _load_target_metadata(target_engine)
    target_counts = {}
    with target_engine.connect() as connection:
        for name in TABLE_ORDER:
            table = metadata.tables[name]
            target_counts[name] = connection.execute(
                select(func.count()).select_from(table)
            ).scalar_one()
    return {
        "source_counts": source_counts,
        "target_counts": target_counts,
        "matches": source_counts == target_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DB_PATH)
    parser.add_argument("--backup-dir", type=Path, default=DB_BACKUP_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy data. Without this flag, only a dry run is performed.",
    )
    args = parser.parse_args()

    engine = create_database_engine()
    try:
        if args.apply:
            result = apply_migration(args.source, engine, args.backup_dir)
            print(json.dumps(result, indent=2, default=str))
        else:
            if engine.dialect.name != "postgresql":
                raise DataMigrationError(
                    "Set NARS_DATABASE_URL to a PostgreSQL URL for the dry run."
                )
            result = build_migration_plan(args.source, engine)
            print("DRY RUN ONLY — no database was modified.")
            print(json.dumps(result, indent=2, default=str))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
