# Data quality report

## Dataset and grain

- Raw-derived event grain: 415,461 rows, 1,000 users,
  971 exercises and 517 songs.
- Temporal coverage: `0.001181` to `30.999954` days since signup.
- Student coverage grain: 1,000 rows and
  1,000 unique users.

## Integrity checks

All critical checks passed: user identifiers are populated, counts are
non-negative, days are within `[0, 31)`, and successful notes/chords never
exceed their evaluated denominators.

## Findings

### Potential duplicate event records — medium severity

- 6,647 later occurrences match on the flat
  tabular fields (1.60% of all rows).
- 478 also match on nested `events_data`
  (0.12% of all rows).
- 498 students have at least one
  tabular match when nested `events_data` is ignored.
- 165 students have at
  least one exact raw duplicate including `events_data`.
- All potential duplicates are retained until source semantics establish
  whether exact raw matches are ingestion duplicates or valid repeated events.

### Undefined event accuracy — low severity

- 3,289 events (0.79%) have no
  evaluated notes or chords.
- Their success-rate fields are null by design; denominators remain available.

### Sparse exercise coverage — high modeling relevance

- 174 exercises occur
  for one student only.
- The median exercise is observed for
  5 students.
- Exercise effects therefore require pooling or regularization.

## Student coverage

- Initial eligible students: 945.
- Students with any future activity: 988.
- Future observable students among all 1,000 raw users under the current
  configurable thresholds:
  935 (93.5%).
- Students meeting both initial eligibility and future observability:
  880.
- Reconciliation: an earlier audit produced a historical preliminary count of
  881 because it counted any future active day. The corrected analysis cohort
  is 880: one student has three future activity days but only two days
  containing evaluable performance.
- Thresholds: {'min_future_evaluated': 100, 'min_future_evaluated_days': 3, 'min_initial_active_days': 1, 'min_initial_evaluated': 1}.
- Initial eligibility reasons: {'eligible': 945, 'no_initial_activity': 55}.
- Future observability reasons: {'no_future_activity': 12, 'observable': 935, 'too_few_future_evaluated_days': 53}.

## Required decisions before modeling

1. Resolve or sensitivity-test the duplicate-event interpretation.
2. Select observability thresholds using trajectory reliability and coverage,
   rather than class balance.
3. Keep all events from one user in the same evaluation partition.
4. Fit content encodings and difficulty parameters on training users only.
