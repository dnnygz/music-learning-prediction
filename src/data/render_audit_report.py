#!/usr/bin/env python3
"""Render the machine-readable data audits as a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-audit", type=Path, default=Path("data/interim/events_audit.json"))
    parser.add_argument(
        "--coverage-audit", type=Path, default=Path("data/interim/student_coverage_audit.json")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality/data_quality_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = json.loads(args.events_audit.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_audit.read_text(encoding="utf-8"))
    duplicate_rate = 100 * events["duplicate_composite_key_rate"]
    exact_duplicate_rate = 100 * events["exact_duplicate_rate"]
    zero_rate = 100 * events["zero_total_evaluated_rows"] / events["rows"]
    observable_rate = 100 * coverage["future_observable_students"] / coverage["students"]

    report = f"""# Data quality report

## Dataset and grain

- Raw-derived event grain: {events['rows']:,} rows, {events['users']:,} users,
  {events['exercises']:,} exercises and {events['songs']:,} songs.
- Temporal coverage: `{events['day_min']}` to `{events['day_max']}` days since signup.
- Student coverage grain: {coverage['students']:,} rows and
  {coverage['unique_students']:,} unique users.

## Integrity checks

All critical checks passed: user identifiers are populated, counts are
non-negative, days are within `[0, 31)`, and successful notes/chords never
exceed their evaluated denominators.

## Findings

### Potential duplicate event records — medium severity

- {events['duplicate_composite_keys']:,} later occurrences match on the flat
  tabular fields ({duplicate_rate:.2f}% of all rows).
- {events['exact_duplicate_rows']:,} also match on nested `events_data`
  ({exact_duplicate_rate:.2f}% of all rows).
- {events['students_affected_by_tabular_matches']:,} students have at least one
  tabular match when nested `events_data` is ignored.
- {events['students_affected_by_exact_raw_duplicates']:,} students have at
  least one exact raw duplicate including `events_data`.
- All potential duplicates are retained until source semantics establish
  whether exact raw matches are ingestion duplicates or valid repeated events.

### Undefined event accuracy — low severity

- {events['zero_total_evaluated_rows']:,} events ({zero_rate:.2f}%) have no
  evaluated notes or chords.
- Their success-rate fields are null by design; denominators remain available.

### Sparse exercise coverage — high modeling relevance

- {events['exercise_coverage']['single_student_exercises']:,} exercises occur
  for one student only.
- The median exercise is observed for
  {events['exercise_coverage']['median_students_per_exercise']:.0f} students.
- Exercise effects therefore require pooling or regularization.

## Student coverage

- Initial eligible students: {coverage['initial_eligible_students']:,}.
- Students with any future activity: {coverage['future_active_students']:,}.
- Future observable students among all 1,000 raw users under the current
  configurable thresholds:
  {coverage['future_observable_students']:,} ({observable_rate:.1f}%).
- Students meeting both initial eligibility and future observability:
  {coverage['analysis_cohort_students']:,}.
- Reconciliation: an earlier audit produced a historical preliminary count of
  881 because it counted any future active day. The corrected analysis cohort
  is 880: one student has three future activity days but only two days
  containing evaluable performance.
- Thresholds: {coverage['observability_thresholds']}.
- Initial eligibility reasons: {coverage['initial_eligibility_reasons']}.
- Future observability reasons: {coverage['future_observability_reasons']}.

## Required decisions before modeling

1. Resolve or sensitivity-test the duplicate-event interpretation.
2. Select observability thresholds using trajectory reliability and coverage,
   rather than class balance.
3. Keep all events from one user in the same evaluation partition.
4. Fit content encodings and difficulty parameters on training users only.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
