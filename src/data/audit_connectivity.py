#!/usr/bin/env python3
"""Audit student-exercise graph connectivity before hierarchical modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def graph_components(edges: pd.DataFrame) -> dict[str, Any]:
    union_find = UnionFind()
    for user_id, exercise_id in edges[["user_id", "exercise_id"]].itertuples(index=False):
        union_find.union(f"u:{user_id}", f"e:{exercise_id}")
    components: dict[str, dict[str, int]] = {}
    for node in union_find.parent:
        root = union_find.find(node)
        counts = components.setdefault(root, {"users": 0, "exercises": 0})
        counts["users" if node.startswith("u:") else "exercises"] += 1
    ordered = sorted(components.values(), key=lambda item: sum(item.values()), reverse=True)
    return {
        "components": len(ordered),
        "largest_component_users": ordered[0]["users"],
        "largest_component_exercises": ordered[0]["exercises"],
        "other_components": ordered[1:20],
    }


def cold_start_simulation(edges: pd.DataFrame, events: pd.DataFrame, repeats: int = 50) -> dict[str, float]:
    users = np.array(sorted(edges["user_id"].unique()))
    rng = np.random.default_rng(20260928)
    event_pairs = events[["user_id", "exercise_id"]]
    unseen_exercise_rates, unseen_event_rates = [], []
    for _ in range(repeats):
        shuffled = rng.permutation(users)
        train_users = set(shuffled[: int(0.8 * len(shuffled))])
        test_users = set(shuffled[int(0.8 * len(shuffled)) :])
        train_exercises = set(edges.loc[edges["user_id"].isin(train_users), "exercise_id"])
        test_exercises = set(edges.loc[edges["user_id"].isin(test_users), "exercise_id"])
        unseen = test_exercises - train_exercises
        test_events = event_pairs.loc[event_pairs["user_id"].isin(test_users)]
        unseen_exercise_rates.append(len(unseen) / len(test_exercises))
        unseen_event_rates.append(test_events["exercise_id"].isin(unseen).mean())
    return {
        "median_unseen_test_exercise_rate": float(np.median(unseen_exercise_rates)),
        "p90_unseen_test_exercise_rate": float(np.quantile(unseen_exercise_rates, 0.9)),
        "median_test_event_rate_on_unseen_exercises": float(np.median(unseen_event_rates)),
        "p90_test_event_rate_on_unseen_exercises": float(np.quantile(unseen_event_rates, 0.9)),
    }


def audit(events: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    evaluated = events.loc[events["total_evaluated"] > 0].copy()
    edges = evaluated[["user_id", "exercise_id"]].drop_duplicates()
    exercise = evaluated.groupby("exercise_id", observed=True).agg(
        students=("user_id", "nunique"), events=("source_row_number", "size"),
        evaluated=("total_evaluated", "sum"), successful=("total_successful", "sum"),
        songs=("song_id", "nunique"), difficulty_levels=("difficulty_level", "nunique"),
        median_difficulty=("difficulty_level", "median"),
    )
    exercise["empirical_success_rate"] = exercise["successful"] / exercise["evaluated"]
    student = evaluated.groupby("user_id", observed=True).agg(
        exercises=("exercise_id", "nunique"), events=("source_row_number", "size"),
        evaluated=("total_evaluated", "sum"), active_days=("day_index", "nunique"),
    )
    counts = exercise["students"]
    summary = {
        "users": int(evaluated["user_id"].nunique()),
        "exercises": int(evaluated["exercise_id"].nunique()),
        "unique_student_exercise_edges": int(len(edges)),
        "exercise_support": {
            "single_student": int((counts == 1).sum()),
            "two_to_four_students": int(counts.between(2, 4).sum()),
            "at_least_5_students": int((counts >= 5).sum()),
            "at_least_10_students": int((counts >= 10).sum()),
            "at_least_20_students": int((counts >= 20).sum()),
            "median_students": float(counts.median()),
            "p90_students": float(counts.quantile(0.9)),
            "max_students": int(counts.max()),
        },
        "student_support": {
            "median_exercises": float(student["exercises"].median()),
            "p10_exercises": float(student["exercises"].quantile(0.1)),
            "p90_exercises": float(student["exercises"].quantile(0.9)),
            "median_events": float(student["events"].median()),
        },
        "graph": graph_components(edges),
        "cold_start_80_20_user_split": cold_start_simulation(edges, evaluated),
    }
    return summary, exercise.reset_index(), student.reset_index()


def render_report(summary: dict[str, Any]) -> str:
    support, student = summary["exercise_support"], summary["student_support"]
    graph, cold = summary["graph"], summary["cold_start_80_20_user_split"]
    return f"""# Student–exercise connectivity audit

## Scope

This audit uses events with at least one evaluated note or chord. It determines
whether exercise difficulty can be identified separately from student
performance before fitting a hierarchical binomial model.

## Graph summary

- Students: {summary['users']:,}.
- Exercises: {summary['exercises']:,}.
- Unique student–exercise edges: {summary['unique_student_exercise_edges']:,}.
- Connected components: {graph['components']:,}.
- Largest component: {graph['largest_component_users']:,} students and
  {graph['largest_component_exercises']:,} exercises.

## Exercise support

| Student coverage per exercise | Exercises |
|---|---:|
| Exactly 1 student | {support['single_student']:,} |
| 2–4 students | {support['two_to_four_students']:,} |
| At least 5 students | {support['at_least_5_students']:,} |
| At least 10 students | {support['at_least_10_students']:,} |
| At least 20 students | {support['at_least_20_students']:,} |

The median exercise is observed for {support['median_students']:.0f} students;
the 90th percentile is {support['p90_students']:.0f}, and the maximum is
{support['max_students']:,}.

## Student support

- Median distinct exercises per student: {student['median_exercises']:.0f}.
- 10th–90th percentile: {student['p10_exercises']:.0f}–{student['p90_exercises']:.0f}.
- Median evaluated event rows per student: {student['median_events']:.0f}.

## Exercises unseen under student-level evaluation splits

Across 50 deterministic 80/20 user splits:

- Median share of distinct test exercises unseen in training:
  {cold['median_unseen_test_exercise_rate']:.2%}.
- 90th percentile: {cold['p90_unseen_test_exercise_rate']:.2%}.
- Median share of test event rows belonging to unseen exercises:
  {cold['median_test_event_rate_on_unseen_exercises']:.2%}.
- 90th percentile: {cold['p90_test_event_rate_on_unseen_exercises']:.2%}.

## Modeling implications

1. Exercise effects require partial pooling; rare exercises cannot receive
   unrestricted fixed-effect estimates.
2. Component structure determines whether effects are globally comparable.
3. Unseen exercises need a fallback based on declared difficulty and content
   features, not a globally calculated frequency encoding.
4. Start with student and exercise random intercepts plus declared difficulty,
   then add context through ablations.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--exercise-output", type=Path, default=Path("data/interim/exercise_connectivity.parquet"))
    parser.add_argument("--student-output", type=Path, default=Path("data/interim/student_connectivity.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/connectivity_audit.json"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/data_quality/connectivity_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, exercise, student = audit(pd.read_parquet(args.events))
    for path in (args.exercise_output, args.student_output, args.audit_output, args.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    exercise.to_parquet(args.exercise_output, index=False)
    student.to_parquet(args.student_output, index=False)
    args.audit_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
