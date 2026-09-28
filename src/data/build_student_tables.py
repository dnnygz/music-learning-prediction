#!/usr/bin/env python3
"""Build a complete student coverage table from cleaned events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.common import FUTURE_END, FUTURE_START, INITIAL_END, INITIAL_START


def _window_summary(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grouped = events.groupby("user_id", observed=True)
    return grouped.agg(
        **{
            f"{prefix}_days_active": ("day_index", "nunique"),
            f"{prefix}_sessions": ("session_index", "nunique"),
            f"{prefix}_events": ("user_id", "size"),
            f"{prefix}_notes_evaluated": ("notes_evaluated", "sum"),
            f"{prefix}_chords_evaluated": ("chords_evaluated", "sum"),
            f"{prefix}_total_evaluated": ("total_evaluated", "sum"),
        }
    )


def build_student_coverage(
    events: pd.DataFrame,
    min_initial_active_days: int,
    min_initial_evaluated: int,
    min_future_active_days: int,
    min_future_evaluated: int,
) -> pd.DataFrame:
    """Keep every student and expose why future trajectory is not observable."""
    initial_mask = events["days_since_signup"].between(INITIAL_START, INITIAL_END, inclusive="left")
    future_mask = events["days_since_signup"].between(FUTURE_START, FUTURE_END, inclusive="left")

    students = pd.DataFrame(index=pd.Index(sorted(events["user_id"].dropna().unique()), name="user_id"))
    initial = _window_summary(events.loc[initial_mask], "initial")
    future = _window_summary(events.loc[future_mask], "future")
    coverage = students.join(initial, how="left").join(future, how="left")

    count_columns = [column for column in coverage if column.endswith(("_active", "_sessions", "_events", "_evaluated"))]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype("int64")
    coverage["initial_eligible"] = (
        (coverage["initial_days_active"] >= min_initial_active_days)
        & (coverage["initial_total_evaluated"] >= min_initial_evaluated)
    )
    coverage["future_active"] = coverage["future_events"] > 0
    coverage["future_has_evaluations"] = coverage["future_total_evaluated"] > 0
    coverage["future_observable"] = (
        (coverage["future_days_active"] >= min_future_active_days)
        & (coverage["future_total_evaluated"] >= min_future_evaluated)
    )

    no_activity = coverage["future_events"] == 0
    no_evaluations = (coverage["future_events"] > 0) & (coverage["future_total_evaluated"] == 0)
    too_few_days = coverage["future_has_evaluations"] & (
        coverage["future_days_active"] < min_future_active_days
    )
    too_few_evaluations = (
        coverage["future_has_evaluations"]
        & (coverage["future_days_active"] >= min_future_active_days)
    ) & (coverage["future_total_evaluated"] < min_future_evaluated)

    coverage["reason_not_observable"] = "observable"
    coverage.loc[no_activity, "reason_not_observable"] = "no_future_activity"
    coverage.loc[no_evaluations, "reason_not_observable"] = "no_future_evaluations"
    coverage.loc[too_few_days, "reason_not_observable"] = "too_few_future_active_days"
    coverage.loc[too_few_evaluations, "reason_not_observable"] = "too_few_future_evaluations"

    no_initial_activity = coverage["initial_events"] == 0
    no_initial_evaluations = (coverage["initial_events"] > 0) & (
        coverage["initial_total_evaluated"] == 0
    )
    too_few_initial_days = (coverage["initial_total_evaluated"] > 0) & (
        coverage["initial_days_active"] < min_initial_active_days
    )
    too_few_initial_evaluations = (
        (coverage["initial_total_evaluated"] > 0)
        & (coverage["initial_days_active"] >= min_initial_active_days)
    ) & (coverage["initial_total_evaluated"] < min_initial_evaluated)
    coverage["reason_not_initial_eligible"] = "eligible"
    coverage.loc[no_initial_activity, "reason_not_initial_eligible"] = "no_initial_activity"
    coverage.loc[no_initial_evaluations, "reason_not_initial_eligible"] = "no_initial_evaluations"
    coverage.loc[too_few_initial_days, "reason_not_initial_eligible"] = "too_few_initial_active_days"
    coverage.loc[
        too_few_initial_evaluations, "reason_not_initial_eligible"
    ] = "too_few_initial_evaluations"
    return coverage.reset_index()


def coverage_audit(
    coverage: pd.DataFrame,
    min_initial_active_days: int,
    min_initial_evaluated: int,
    min_future_active_days: int,
    min_future_evaluated: int,
) -> dict[str, Any]:
    reasons = coverage["reason_not_observable"].value_counts(dropna=False)
    initial_reasons = coverage["reason_not_initial_eligible"].value_counts(dropna=False)
    return {
        "grain": "one row per raw student",
        "students": int(len(coverage)),
        "unique_students": int(coverage["user_id"].nunique()),
        "duplicate_user_ids": int(coverage["user_id"].duplicated().sum()),
        "initial_window": f"[{INITIAL_START}, {INITIAL_END})",
        "future_window": f"[{FUTURE_START}, {FUTURE_END})",
        "observability_thresholds": {
            "min_initial_active_days": min_initial_active_days,
            "min_initial_evaluated": min_initial_evaluated,
            "min_future_active_days": min_future_active_days,
            "min_future_evaluated": min_future_evaluated,
        },
        "initial_eligible_students": int(coverage["initial_eligible"].sum()),
        "future_active_students": int(coverage["future_active"].sum()),
        "future_observable_students": int(coverage["future_observable"].sum()),
        "analysis_cohort_students": int(
            (coverage["initial_eligible"] & coverage["future_observable"]).sum()
        ),
        "initial_eligibility_reasons": {
            str(key): int(value) for key, value in initial_reasons.items()
        },
        "future_observability_reasons": {
            str(key): int(value) for key, value in reasons.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/student_coverage.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/student_coverage_audit.json"))
    parser.add_argument("--min-initial-active-days", type=int, default=1)
    parser.add_argument("--min-initial-evaluated", type=int, default=1)
    parser.add_argument("--min-future-active-days", type=int, default=3)
    parser.add_argument("--min-future-evaluated", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = (
        args.min_initial_active_days,
        args.min_initial_evaluated,
        args.min_future_active_days,
        args.min_future_evaluated,
    )
    if any(value < 1 for value in thresholds):
        raise ValueError("Observability thresholds must be positive integers")

    events = pd.read_parquet(args.input)
    coverage = build_student_coverage(
        events,
        min_initial_active_days=args.min_initial_active_days,
        min_initial_evaluated=args.min_initial_evaluated,
        min_future_active_days=args.min_future_active_days,
        min_future_evaluated=args.min_future_evaluated,
    )
    audit = coverage_audit(
        coverage,
        min_initial_active_days=args.min_initial_active_days,
        min_initial_evaluated=args.min_initial_evaluated,
        min_future_active_days=args.min_future_active_days,
        min_future_evaluated=args.min_future_evaluated,
    )

    if audit["students"] != audit["unique_students"]:
        raise ValueError("Student coverage does not have one row per user")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(args.output, index=False)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(coverage):,} students to {args.output}")
    print(f"Wrote audit to {args.audit_output}")


if __name__ == "__main__":
    main()
