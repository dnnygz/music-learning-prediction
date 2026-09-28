#!/usr/bin/env python3
"""Create a validated event-grain Parquet dataset from the raw Yousician JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.common import RAW_COLUMNS, stream_json_array


COUNT_COLUMNS = (
    "notes_successful",
    "notes_evaluated",
    "chords_successful",
    "chords_evaluated",
)
NUMERIC_COLUMNS = COUNT_COLUMNS + (
    "days_since_signup",
    "difficulty_level",
    "session_index",
    "time_playing",
    "exercise_part_index",
)
STRING_COLUMNS = ("user_id", "song_id", "exercise_id", "song_type", "play_mode", "exit_status")


def _completed(value: Any) -> bool:
    return str(value).strip().lower() in {"t", "true", "1", "yes"}


def _abandoned(value: Any) -> bool:
    return str(value).strip().lower() in {"quit", "abandoned", "cancelled", "canceled"}


def load_raw_events(path: Path) -> pd.DataFrame:
    """Read the raw array while discarding the nested event payload."""
    records = []
    for source_row_number, raw in enumerate(stream_json_array(path)):
        record = {column: raw.get(column) for column in RAW_COLUMNS}
        record["source_row_number"] = source_row_number
        records.append(record)
    return pd.DataFrame.from_records(records, columns=(*RAW_COLUMNS, "source_row_number"))


def clean_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and add transparent derived event measures."""
    events = raw.copy()

    if "source_row_number" not in events:
        events["source_row_number"] = range(len(events))

    for column in STRING_COLUMNS:
        events[column] = events[column].astype("string").str.strip()
        events.loc[events[column] == "", column] = pd.NA

    events["events_data"] = events["events_data"].astype("string")
    events["events_data_hash"] = events["events_data"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        if pd.notna(value)
        else pd.NA,
        na_action=None,
    ).astype("string")

    for column in NUMERIC_COLUMNS:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    events["source_row_number"] = pd.to_numeric(
        events["source_row_number"], errors="raise"
    ).astype("int64")

    for column in COUNT_COLUMNS:
        events[column] = events[column].astype("Int64")

    events["completed"] = events["is_played_in_full"].map(_completed).astype("boolean")
    events["abandoned"] = events["exit_status"].map(_abandoned).astype("boolean")
    events["day_index"] = events["days_since_signup"].floordiv(1).astype("Int64")
    events["total_successful"] = events["notes_successful"] + events["chords_successful"]
    events["total_evaluated"] = events["notes_evaluated"] + events["chords_evaluated"]

    events["success_rate_notes"] = events["notes_successful"].div(
        events["notes_evaluated"].where(events["notes_evaluated"] > 0)
    )
    events["success_rate_chords"] = events["chords_successful"].div(
        events["chords_evaluated"].where(events["chords_evaluated"] > 0)
    )
    events["success_rate_total"] = events["total_successful"].div(
        events["total_evaluated"].where(events["total_evaluated"] > 0)
    )

    return events.drop(columns=["is_played_in_full"])


def audit_events(events: pd.DataFrame) -> dict[str, Any]:
    """Return machine-readable integrity and grain checks."""
    duplicate_key = [
        "user_id",
        "song_id",
        "exercise_id",
        "days_since_signup",
        "session_index",
        "exercise_part_index",
        "play_mode",
        "exit_status",
        "notes_evaluated",
        "notes_successful",
        "chords_evaluated",
        "chords_successful",
        "time_playing",
        "completed",
    ]
    students_per_exercise = events.groupby("exercise_id", observed=True)["user_id"].nunique()
    events_per_exercise = events.groupby("exercise_id", observed=True).size()
    raw_content_key = [*duplicate_key, "events_data_hash"]
    duplicate_mask = events.duplicated(subset=duplicate_key, keep=False)
    raw_duplicate_mask = events.duplicated(subset=raw_content_key, keep=False)
    exact_duplicate_count = int(events.duplicated(subset=raw_content_key).sum())
    composite_duplicate_count = int(events.duplicated(subset=duplicate_key).sum())

    checks = {
        "missing_user_id": int(events["user_id"].isna().sum()),
        "negative_days": int((events["days_since_signup"] < 0).sum()),
        "days_at_or_above_31": int((events["days_since_signup"] >= 31).sum()),
        "negative_counts": int((events[list(COUNT_COLUMNS)] < 0).sum().sum()),
        "notes_success_exceeds_evaluated": int(
            (events["notes_successful"] > events["notes_evaluated"]).sum()
        ),
        "chords_success_exceeds_evaluated": int(
            (events["chords_successful"] > events["chords_evaluated"]).sum()
        ),
    }
    return {
        "grain": "one raw platform interaction",
        "rows": int(len(events)),
        "columns": int(len(events.columns)),
        "users": int(events["user_id"].nunique(dropna=True)),
        "songs": int(events["song_id"].nunique(dropna=True)),
        "exercises": int(events["exercise_id"].nunique(dropna=True)),
        "day_min": float(events["days_since_signup"].min()),
        "day_max": float(events["days_since_signup"].max()),
        "exact_duplicate_rows": exact_duplicate_count,
        "exact_duplicate_rate": exact_duplicate_count / len(events),
        "duplicate_composite_keys": composite_duplicate_count,
        "duplicate_composite_key_rate": composite_duplicate_count / len(events),
        "students_affected_by_duplicate_keys": int(
            events.loc[duplicate_mask, "user_id"].nunique()
        ),
        "students_affected_by_exact_raw_duplicates": int(
            events.loc[raw_duplicate_mask, "user_id"].nunique()
        ),
        "zero_total_evaluated_rows": int((events["total_evaluated"] == 0).sum()),
        "checks": checks,
        "exercise_coverage": {
            "single_student_exercises": int((students_per_exercise == 1).sum()),
            "median_students_per_exercise": float(students_per_exercise.median()),
            "max_students_per_exercise": int(students_per_exercise.max()),
            "median_events_per_exercise": float(events_per_exercise.median()),
            "max_events_per_exercise": int(events_per_exercise.max()),
        },
        "null_counts": {column: int(value) for column, value in events.isna().sum().items()},
    }


def assert_critical_checks(audit: dict[str, Any]) -> None:
    """Stop the pipeline when stable source invariants fail."""
    failures = {name: value for name, value in audit["checks"].items() if value != 0}
    if failures:
        raise ValueError(f"Critical event integrity checks failed: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/yousician_ukulele.json"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/events_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = clean_events(load_raw_events(args.input))
    audit = audit_events(events)
    assert_critical_checks(audit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.output, index=False)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(events):,} events to {args.output}")
    print(f"Wrote audit to {args.audit_output}")


if __name__ == "__main__":
    main()
