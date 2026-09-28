#!/usr/bin/env python3
"""Audit cohort exclusions, duplicate structure, and observability sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_student_tables import build_student_coverage


TABULAR_DUPLICATE_KEY = [
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


def audit_duplicates(events: pd.DataFrame) -> dict[str, Any]:
    raw_duplicate_key = [*TABULAR_DUPLICATE_KEY, "events_data_hash"]
    duplicate_rows = events.loc[
        events.duplicated(TABULAR_DUPLICATE_KEY, keep=False)
    ].copy()
    groups = duplicate_rows.groupby(TABULAR_DUPLICATE_KEY, dropna=False, observed=True)
    group_sizes = groups.size()
    source_spans = groups["source_row_number"].agg(lambda values: int(values.max() - values.min()))
    hash_counts = groups["events_data_hash"].nunique(dropna=False)
    raw_duplicate_rows = events.loc[events.duplicated(raw_duplicate_key, keep=False)].copy()
    raw_groups = raw_duplicate_rows.groupby(raw_duplicate_key, dropna=False, observed=True).size()
    later_raw_duplicate_mask = events.duplicated(raw_duplicate_key, keep="first")

    attempt_key = ["user_id", "exercise_id", "days_since_signup", "exercise_part_index"]
    same_time = events.loc[events.duplicated(attempt_key, keep=False)].copy()
    same_time_groups = same_time.groupby(attempt_key, dropna=False, observed=True)
    same_time_summary = same_time_groups.agg(
        rows=("source_row_number", "size"),
        distinct_outcomes=("success_rate_total", "nunique"),
        distinct_time_playing=("time_playing", "nunique"),
        distinct_exit_status=("exit_status", "nunique"),
    )
    varying_same_time = same_time_summary[
        (same_time_summary["distinct_outcomes"] > 1)
        | (same_time_summary["distinct_time_playing"] > 1)
        | (same_time_summary["distinct_exit_status"] > 1)
    ]

    duplicate_by_day = duplicate_rows.groupby("day_index", observed=True).size()
    duplicate_by_user = duplicate_rows.groupby("user_id", observed=True).size().sort_values(
        ascending=False
    )
    return {
        "tabular_duplicate_groups": int(len(group_sizes)),
        "rows_in_tabular_duplicate_groups": int(len(duplicate_rows)),
        "later_tabular_duplicate_occurrences": int((group_sizes - 1).sum()),
        "tabular_groups_with_distinct_events_data": int((hash_counts > 1).sum()),
        "exact_raw_duplicate_groups": int(len(raw_groups)),
        "rows_in_exact_raw_duplicate_groups": int(len(raw_duplicate_rows)),
        "later_exact_raw_duplicate_occurrences": int((raw_groups - 1).sum()),
        "users_affected_by_exact_raw_duplicates": int(
            raw_duplicate_rows["user_id"].nunique()
        ),
        "evaluated_elements_in_later_exact_raw_duplicates": int(
            events.loc[later_raw_duplicate_mask, "total_evaluated"].sum()
        ),
        "share_of_evaluated_elements_in_later_exact_raw_duplicates": float(
            events.loc[later_raw_duplicate_mask, "total_evaluated"].sum()
            / events["total_evaluated"].sum()
        ),
        "group_size_distribution": {
            str(int(size)): int(count)
            for size, count in group_sizes.value_counts().sort_index().items()
        },
        "adjacent_or_nearby_groups_source_span_le_5": int((source_spans <= 5).sum()),
        "groups_source_span_over_100": int((source_spans > 100).sum()),
        "affected_users": int(duplicate_rows["user_id"].nunique()),
        "top_10_users_by_duplicate_group_rows": {
            str(user): int(count) for user, count in duplicate_by_user.head(10).items()
        },
        "duplicate_group_rows_by_day": {
            str(int(day)): int(count) for day, count in duplicate_by_day.items()
        },
        "same_user_exercise_exact_time_groups": int(len(same_time_summary)),
        "same_time_groups_with_varying_observed_values": int(len(varying_same_time)),
    }


def sensitivity_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for initial_days in (1, 2, 3):
        for initial_evaluated in (1, 50, 100, 500):
            for future_days in (1, 2, 3, 5, 7):
                for future_evaluated in (50, 100, 200, 500):
                    coverage = build_student_coverage(
                        events,
                        min_initial_active_days=initial_days,
                        min_initial_evaluated=initial_evaluated,
                        min_future_active_days=future_days,
                        min_future_evaluated=future_evaluated,
                    )
                    initial_eligible = coverage["initial_eligible"]
                    future_observable = coverage["future_observable"]
                    rows.append(
                        {
                            "min_initial_active_days": initial_days,
                            "min_initial_evaluated": initial_evaluated,
                            "min_future_active_days": future_days,
                            "min_future_evaluated": future_evaluated,
                            "initial_eligible": int(initial_eligible.sum()),
                            "future_observable_all_users": int(future_observable.sum()),
                            "analysis_cohort": int((initial_eligible & future_observable).sum()),
                            "future_not_observable_among_initial_eligible": int(
                                (initial_eligible & ~future_observable).sum()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def exclusion_audit(events: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    first_day = events.groupby("user_id", observed=True)["days_since_signup"].min()
    last_day = events.groupby("user_id", observed=True)["days_since_signup"].max()
    indexed = coverage.set_index("user_id")
    not_initial = indexed.loc[~indexed["initial_eligible"]]
    initial = indexed["initial_eligible"]
    future = indexed["future_observable"]

    return {
        "initial_eligibility_reasons": {
            str(key): int(value)
            for key, value in indexed["reason_not_initial_eligible"].value_counts().items()
        },
        "future_observability_reasons": {
            str(key): int(value)
            for key, value in indexed["reason_not_observable"].value_counts().items()
        },
        "initial_future_cross_tab": {
            "initial_eligible_and_future_observable": int((initial & future).sum()),
            "initial_eligible_not_future_observable": int((initial & ~future).sum()),
            "not_initial_eligible_but_future_observable": int((~initial & future).sum()),
            "neither": int((~initial & ~future).sum()),
        },
        "not_initial_eligible_first_day": {
            "min": float(first_day.loc[not_initial.index].min()),
            "median": float(first_day.loc[not_initial.index].median()),
            "max": float(first_day.loc[not_initial.index].max()),
        },
        "not_initial_eligible_last_day": {
            "min": float(last_day.loc[not_initial.index].min()),
            "median": float(last_day.loc[not_initial.index].median()),
            "max": float(last_day.loc[not_initial.index].max()),
        },
    }


def render_report(
    exclusions: dict[str, Any], duplicates: dict[str, Any], sensitivity: pd.DataFrame
) -> str:
    selected = sensitivity[
        (sensitivity["min_initial_active_days"] == 1)
        & (sensitivity["min_initial_evaluated"] == 1)
        & sensitivity["min_future_active_days"].isin([1, 2, 3, 5, 7])
        & sensitivity["min_future_evaluated"].isin([50, 100, 200, 500])
    ]
    pivot = selected.pivot(
        index="min_future_active_days",
        columns="min_future_evaluated",
        values="analysis_cohort",
    )
    table_lines = ["| Future active days | 50 | 100 | 200 | 500 |", "|---:|---:|---:|---:|---:|"]
    for active_days, values in pivot.iterrows():
        table_lines.append(
            f"| {active_days} | {values[50]} | {values[100]} | {values[200]} | {values[500]} |"
        )

    selected_initial = sensitivity[
        (sensitivity["min_future_active_days"] == 3)
        & (sensitivity["min_future_evaluated"] == 100)
    ]
    initial_pivot = selected_initial.pivot(
        index="min_initial_active_days",
        columns="min_initial_evaluated",
        values="analysis_cohort",
    )
    initial_table_lines = [
        "| Initial active days | 1 | 50 | 100 | 500 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for active_days, values in initial_pivot.iterrows():
        initial_table_lines.append(
            f"| {active_days} | {values[1]} | {values[50]} | {values[100]} | {values[500]} |"
        )

    cross = exclusions["initial_future_cross_tab"]
    return f"""# Cohort and duplicate semantics audit

## Eligibility and observability

- Initial eligibility reasons: {exclusions['initial_eligibility_reasons']}.
- Future observability reasons: {exclusions['future_observability_reasons']}.
- Eligible initially and observable later: {cross['initial_eligible_and_future_observable']}.
- Eligible initially but not observable later: {cross['initial_eligible_not_future_observable']}.
- Not initially eligible but observable later: {cross['not_initial_eligible_but_future_observable']}.
- Neither: {cross['neither']}.

The students without initial eligibility have no events in `[0, 7)`. Their
first observed day ranges from
{exclusions['not_initial_eligible_first_day']['min']:.3f} to
{exclusions['not_initial_eligible_first_day']['max']:.3f}, with median
{exclusions['not_initial_eligible_first_day']['median']:.3f}. This is a cohort
timing limitation, not missing success denominators in the initial window.

## Duplicate structure

- Tabular duplicate groups when `events_data` is ignored:
  {duplicates['tabular_duplicate_groups']}.
- Those groups with different nested `events_data`:
  {duplicates['tabular_groups_with_distinct_events_data']}.
- Later tabular duplicate occurrences:
  {duplicates['later_tabular_duplicate_occurrences']}.
- Exact raw duplicate groups including `events_data`:
  {duplicates['exact_raw_duplicate_groups']}.
- Later exact raw duplicate occurrences:
  {duplicates['later_exact_raw_duplicate_occurrences']}.
- Users affected by exact raw duplicates:
  {duplicates['users_affected_by_exact_raw_duplicates']}.
- Share of all evaluated elements carried by later exact raw duplicates:
  {100 * duplicates['share_of_evaluated_elements_in_later_exact_raw_duplicates']:.3f}%.
- Affected students: {duplicates['affected_users']}.
- Group sizes: {duplicates['group_size_distribution']}.
- Groups located within five source rows: {duplicates['adjacent_or_nearby_groups_source_span_le_5']}.
- Groups separated by more than 100 source rows: {duplicates['groups_source_span_over_100']}.
- Same user/exercise/exact-time groups with varying outcomes or statuses:
  {duplicates['same_time_groups_with_varying_observed_values']}.

Exact raw duplicates remain unresolved. Tabular matches with different nested
event payloads are not duplicates and must not be removed. Source-row proximity
is evidence about file layout, not proof that a record is erroneous. Near
matches with different outcomes are treated as repeated attempts.

## Threshold sensitivity

The table reports the final analysis cohort among initially eligible students
under initial thresholds of one active day and one evaluated element.

{chr(10).join(table_lines)}

Holding the future threshold at three active days and 100 evaluated elements,
initial-threshold sensitivity is:

{chr(10).join(initial_table_lines)}

The full grid is stored in `data/interim/threshold_sensitivity.parquet` and also
varies initial eligibility thresholds. Observability must be frozen after
examining trajectory reliability, not merely by maximizing cohort size.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--coverage", type=Path, default=Path("data/interim/student_coverage.parquet"))
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=Path("data/interim/threshold_sensitivity.parquet"),
    )
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/semantics_audit.json"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/semantics_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_parquet(args.events)
    coverage = pd.read_parquet(args.coverage)
    duplicates = audit_duplicates(events)
    exclusions = exclusion_audit(events, coverage)
    sensitivity = sensitivity_table(events)
    audit = {"exclusions": exclusions, "duplicates": duplicates}

    args.sensitivity_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_parquet(args.sensitivity_output, index=False)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(
        render_report(exclusions, duplicates, sensitivity), encoding="utf-8"
    )
    print(f"Wrote sensitivity grid to {args.sensitivity_output}")
    print(f"Wrote semantics audit to {args.audit_output}")
    print(f"Wrote semantics report to {args.report_output}")


if __name__ == "__main__":
    main()
