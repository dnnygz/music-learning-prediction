"""Focused tests for stable data-pipeline invariants."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_student_tables import build_student_coverage
from src.data.clean_events import audit_events, clean_events
from src.data.build_modeling_table import (
    assert_modeling_table,
    audit_modeling_table,
    build_modeling_table,
)
from src.data.common import RAW_COLUMNS


def raw_event(user_id: str, day: float, evaluated: int = 10) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in RAW_COLUMNS}
    row.update(
        {
            "user_id": user_id,
            "song_id": "song-1",
            "exercise_id": "exercise-1",
            "days_since_signup": day,
            "song_type": "lesson",
            "difficulty_level": 1,
            "session_index": int(day) + 1,
            "play_mode": "practice",
            "chords_evaluated": 0,
            "chords_successful": 0,
            "notes_evaluated": evaluated,
            "notes_successful": evaluated - 1 if evaluated else 0,
            "time_playing": 60,
            "is_played_in_full": "true",
            "exit_status": "completed",
            "exercise_part_index": 0,
            "events_data": '{"data":{"timing_offset":[-4],"pitch_offset":[0]}}',
        }
    )
    return row


class DataPipelineTests(unittest.TestCase):
    def test_clean_events_preserves_denominators_and_null_rate(self) -> None:
        raw = pd.DataFrame.from_records(
            [raw_event("u1", 1.0, evaluated=10), raw_event("u2", 2.0, evaluated=0)],
            columns=RAW_COLUMNS,
        )
        events = clean_events(raw)

        self.assertEqual(events.loc[0, "total_evaluated"], 10)
        self.assertAlmostEqual(events.loc[0, "success_rate_total"], 0.9)
        self.assertTrue(pd.isna(events.loc[1, "success_rate_total"]))
        self.assertEqual(len(events.loc[0, "events_data_hash"]), 64)
        self.assertEqual(audit_events(events)["checks"]["negative_counts"], 0)

    def test_coverage_keeps_students_without_future_activity(self) -> None:
        raw = pd.DataFrame.from_records(
            [
                raw_event("u1", 1.0),
                raw_event("u1", 8.0),
                raw_event("u1", 9.0),
                raw_event("u1", 10.0),
                raw_event("u2", 1.0),
            ],
            columns=RAW_COLUMNS,
        )
        events = clean_events(raw)
        coverage = build_student_coverage(
            events,
            min_initial_active_days=1,
            min_initial_evaluated=1,
            min_future_evaluated_days=3,
            min_future_evaluated=20,
        ).set_index("user_id")

        self.assertEqual(len(coverage), 2)
        self.assertTrue(coverage.loc["u1", "future_observable"])
        self.assertFalse(coverage.loc["u2", "future_observable"])
        self.assertEqual(coverage.loc["u2", "reason_not_observable"], "no_future_activity")

    def test_time_boundary_assigns_day_seven_to_future(self) -> None:
        raw = pd.DataFrame.from_records(
            [raw_event("u1", 6.999), raw_event("u1", 7.0)], columns=RAW_COLUMNS
        )
        coverage = build_student_coverage(
            clean_events(raw),
            min_initial_active_days=1,
            min_initial_evaluated=1,
            min_future_evaluated_days=1,
            min_future_evaluated=1,
        ).iloc[0]

        self.assertEqual(coverage["initial_events"], 1)
        self.assertEqual(coverage["future_events"], 1)

    def test_modeling_table_separates_response_types_and_reconciles(self) -> None:
        first = raw_event("u1", 1.2, evaluated=10)
        first["chords_evaluated"] = 4
        first["chords_successful"] = 3
        second = raw_event("u1", 1.8, evaluated=5)
        second["notes_successful"] = 4
        events = clean_events(pd.DataFrame.from_records([first, second], columns=RAW_COLUMNS))

        table = build_modeling_table(events)
        audit = audit_modeling_table(events, table)
        assert_modeling_table(audit)

        notes = table.loc[table["response_type"] == "note"].iloc[0]
        chords = table.loc[table["response_type"] == "chord"].iloc[0]
        self.assertEqual((notes["successful"], notes["evaluated"]), (13, 15))
        self.assertEqual((chords["successful"], chords["evaluated"]), (3, 4))
        self.assertEqual(audit["duplicate_grain_keys"], 0)


if __name__ == "__main__":
    unittest.main()
