# Data dictionary

## Conventions

- Event window for initial features: `[0, 7)` elapsed days since signup.
- Event window for future outcomes: `[7, 31)` elapsed days since signup.
- Rates are null when their denominator is zero.
- Raw count denominators are always retained.
- Every student remains present in the coverage table.
- Observability thresholds are configurable pipeline parameters.

## `events_clean.parquet`

Grain: one raw platform interaction. Exact semantic duplicates are currently
retained while their source meaning is investigated.

| Column | Type | Definition |
|---|---|---|
| `source_row_number` | integer | Zero-based position in the raw JSON array; lineage only. |
| `events_data` | string/null | Original nested event payload serialized as JSON; includes arrays such as timing offset, pitch offset, duration, pitches, strings, and rejection reason. Retained without aggregation. |
| `events_data_hash` | string/null | SHA-256 lineage fingerprint of `events_data`. |
| `user_id` | string | Student identifier. |
| `song_id` | string | Song identifier. |
| `exercise_id` | string | Exercise identifier. |
| `days_since_signup` | float | Continuous elapsed days since signup. |
| `day_index` | integer | Floor of `days_since_signup`. |
| `song_type` | string | Source-provided content type. |
| `difficulty_level` | numeric | Source-provided difficulty level. |
| `session_index` | numeric | Source-provided session sequence index. |
| `exercise_part_index` | numeric | Source-provided exercise-part index. |
| `play_mode` | string | Mode in which the content was played. |
| `notes_successful` | integer | Successfully evaluated notes. |
| `notes_evaluated` | integer | Evaluated notes; denominator for note accuracy. |
| `chords_successful` | integer | Successfully evaluated chords. |
| `chords_evaluated` | integer | Evaluated chords; denominator for chord accuracy. |
| `total_successful` | integer | Notes plus chords successful. |
| `total_evaluated` | integer | Notes plus chords evaluated. |
| `success_rate_notes` | float/null | `notes_successful / notes_evaluated`. |
| `success_rate_chords` | float/null | `chords_successful / chords_evaluated`. |
| `success_rate_total` | float/null | `total_successful / total_evaluated`. |
| `time_playing` | numeric | Source-provided playing time; source unit retained. |
| `completed` | boolean | Normalized full-play indicator. |
| `abandoned` | boolean | Exit status indicates quit, abandoned, or cancelled. |
| `exit_status` | string | Original exit-status category. |

## `student_coverage.parquet`

Grain: exactly one row per student appearing in the raw JSON.

| Column group | Definition |
|---|---|
| `initial_*` | Counts calculated from events in `[0, 7)`. |
| `future_*` | Counts calculated from events in `[7, 31)`. |
| `*_days_active` | Distinct integer day indices with events. |
| `*_sessions` | Distinct source session indices. |
| `*_events` | Number of event rows, including unresolved exact duplicates. |
| `*_notes_evaluated` | Sum of evaluated notes. |
| `*_chords_evaluated` | Sum of evaluated chords. |
| `*_total_evaluated` | Notes plus chords evaluated. |
| `initial_eligible` | Meets configurable initial-day and evaluated-count thresholds. |
| `reason_not_initial_eligible` | Mutually exclusive eligibility reason. |
| `future_active` | Has at least one future event. |
| `future_has_evaluations` | Has at least one future evaluated note or chord. |
| `future_observable` | Meets configurable future-day and evaluated-count thresholds. |
| `reason_not_observable` | Mutually exclusive future-observability reason. |

Default thresholds are provisional:

- Initial: at least 1 active day and 1 evaluated element.
- Future: at least 3 active days and 100 evaluated elements.

They must be frozen only after sensitivity and trajectory-reliability analysis.
