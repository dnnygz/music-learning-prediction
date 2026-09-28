#!/usr/bin/env python3
"""Build the binomial modeling table at student-exercise-day-context grain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


GROUP_COLUMNS = [
    "user_id",
    "exercise_id",
    "song_id",
    "day_index",
    "difficulty_level",
    "play_mode",
    "song_type",
    "response_type",
]


def build_modeling_table(events: pd.DataFrame) -> pd.DataFrame:
    """Return evaluated note/chord counts without averaging event-level rates."""
    required = {
        "user_id",
        "exercise_id",
        "song_id",
        "day_index",
        "difficulty_level",
        "play_mode",
        "song_type",
        "source_row_number",
        "notes_successful",
        "notes_evaluated",
        "chords_successful",
        "chords_evaluated",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")

    id_columns = [
        "user_id",
        "exercise_id",
        "song_id",
        "day_index",
        "difficulty_level",
        "play_mode",
        "song_type",
        "source_row_number",
    ]
    notes = events[id_columns + ["notes_successful", "notes_evaluated"]].rename(
        columns={"notes_successful": "successful", "notes_evaluated": "evaluated"}
    )
    notes["response_type"] = "note"
    chords = events[id_columns + ["chords_successful", "chords_evaluated"]].rename(
        columns={"chords_successful": "successful", "chords_evaluated": "evaluated"}
    )
    chords["response_type"] = "chord"

    long = pd.concat([notes, chords], ignore_index=True)
    long = long.loc[long["evaluated"] > 0].copy()
    table = (
        long.groupby(GROUP_COLUMNS, observed=True, dropna=False)
        .agg(
            successful=("successful", "sum"),
            evaluated=("evaluated", "sum"),
            source_event_rows=("source_row_number", "nunique"),
        )
        .reset_index()
    )
    table["success_rate"] = table["successful"] / table["evaluated"]
    table = table.sort_values(GROUP_COLUMNS, kind="stable").reset_index(drop=True)
    return table


def audit_modeling_table(events: pd.DataFrame, table: pd.DataFrame) -> dict[str, Any]:
    """Reconcile the aggregate to event-level numerators and denominators."""
    expected = {
        "note": {
            "successful": int(events.loc[events["notes_evaluated"] > 0, "notes_successful"].sum()),
            "evaluated": int(events["notes_evaluated"].sum()),
        },
        "chord": {
            "successful": int(events.loc[events["chords_evaluated"] > 0, "chords_successful"].sum()),
            "evaluated": int(events["chords_evaluated"].sum()),
        },
    }
    actual_by_type = table.groupby("response_type", observed=True)[
        ["successful", "evaluated"]
    ].sum()
    reconciliation = {}
    for response_type in ("note", "chord"):
        actual = {
            measure: int(actual_by_type.loc[response_type, measure])
            for measure in ("successful", "evaluated")
        }
        reconciliation[response_type] = {
            "source": expected[response_type],
            "aggregate": actual,
            "matches": actual == expected[response_type],
        }

    null_counts = {column: int(table[column].isna().sum()) for column in GROUP_COLUMNS}
    duplicate_keys = int(table.duplicated(GROUP_COLUMNS).sum())
    invalid_binomial_rows = int(
        ((table["evaluated"] <= 0) | (table["successful"] < 0) | (table["successful"] > table["evaluated"])).sum()
    )
    return {
        "grain": "one row per student-exercise-day-context-response type",
        "rows": int(len(table)),
        "users": int(table["user_id"].nunique()),
        "exercises": int(table["exercise_id"].nunique()),
        "days": int(table["day_index"].nunique()),
        "day_min": int(table["day_index"].min()),
        "day_max": int(table["day_index"].max()),
        "response_type_rows": {
            str(key): int(value)
            for key, value in table["response_type"].value_counts().items()
        },
        "duplicate_grain_keys": duplicate_keys,
        "invalid_binomial_rows": invalid_binomial_rows,
        "null_group_keys": null_counts,
        "reconciliation": reconciliation,
        "exact_raw_duplicates_retained": True,
    }


def assert_modeling_table(audit: dict[str, Any]) -> None:
    failures = []
    if audit["duplicate_grain_keys"]:
        failures.append("duplicate grain keys")
    if audit["invalid_binomial_rows"]:
        failures.append("invalid binomial counts")
    if any(audit["null_group_keys"].values()):
        failures.append("null grouping keys")
    if not all(item["matches"] for item in audit["reconciliation"].values()):
        failures.append("source-to-aggregate reconciliation")
    if failures:
        raise ValueError(f"Modeling table checks failed: {', '.join(failures)}")


def render_report(audit: dict[str, Any]) -> str:
    note = audit["reconciliation"]["note"]["aggregate"]
    chord = audit["reconciliation"]["chord"]["aggregate"]
    return f"""# Student–exercise–day modeling table audit

## Grain

One row represents one student, exercise, integer day, song, declared
difficulty, play mode, song type, and response type (`note` or `chord`). Context
is part of the key because play mode can vary within a student–exercise–day.

Only rows with a positive evaluated denominator are emitted. Event-level rates
are never averaged; successful and evaluated counts are summed separately.

## Profile

- Rows: {audit['rows']:,}.
- Students: {audit['users']:,}.
- Exercises: {audit['exercises']:,}.
- Day range: {audit['day_min']}–{audit['day_max']}.
- Note rows: {audit['response_type_rows'].get('note', 0):,}.
- Chord rows: {audit['response_type_rows'].get('chord', 0):,}.

## Reconciliation

| Response type | Successful | Evaluated | Matches event source |
|---|---:|---:|---:|
| Notes | {note['successful']:,} | {note['evaluated']:,} | Yes |
| Chords | {chord['successful']:,} | {chord['evaluated']:,} | Yes |

## Integrity checks

- Duplicate grain keys: {audit['duplicate_grain_keys']:,}.
- Invalid binomial rows: {audit['invalid_binomial_rows']:,}.
- Null grouping keys: {sum(audit['null_group_keys'].values()):,}.
- Exact raw duplicates remain included consistently with the principal data
  pipeline; their removal belongs in a sensitivity analysis.

This table is suitable as the input layer for the hierarchical binomial model.
It is not yet a student-level prediction dataset or a finalized trajectory
target.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/modeling_table_audit.json"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/data_quality/modeling_table_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_parquet(args.events)
    table = build_modeling_table(events)
    audit = audit_modeling_table(events, table)
    assert_modeling_table(audit)
    for path in (args.output, args.audit_output, args.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_report(audit), encoding="utf-8")
    print(f"Wrote {len(table):,} modeling rows to {args.output}")
    print(f"Wrote audit report to {args.report_output}")


if __name__ == "__main__":
    main()
