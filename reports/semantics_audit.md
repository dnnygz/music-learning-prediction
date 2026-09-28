# Cohort and duplicate semantics audit

## Eligibility and observability

- Initial eligibility reasons: {'eligible': 945, 'no_initial_activity': 55}.
- Future observability reasons: {'observable': 936, 'too_few_future_active_days': 52, 'no_future_activity': 12}.
- Eligible initially and observable later: 881.
- Eligible initially but not observable later: 64.
- Not initially eligible but observable later: 55.
- Neither: 0.

The students without initial eligibility have no events in `[0, 7)`. Their
first observed day ranges from
7.017 to
25.730, with median
10.894. This is a cohort
timing limitation, not missing success denominators in the initial window.

## Duplicate structure

- Tabular duplicate groups when `events_data` is ignored:
  4657.
- Those groups with different nested `events_data`:
  4471.
- Later tabular duplicate occurrences:
  6647.
- Exact raw duplicate groups including `events_data`:
  323.
- Later exact raw duplicate occurrences:
  478.
- Users affected by exact raw duplicates:
  165.
- Share of all evaluated elements carried by later exact raw duplicates:
  0.003%.
- Affected students: 498.
- Group sizes: {'2': 3504, '3': 733, '4': 240, '5': 82, '6': 46, '7': 21, '8': 7, '9': 12, '10': 6, '11': 2, '12': 2, '16': 1, '18': 1}.
- Groups located within five source rows: 2614.
- Groups separated by more than 100 source rows: 0.
- Same user/exercise/exact-time groups with varying outcomes or statuses:
  37358.

Exact raw duplicates remain unresolved. Tabular matches with different nested
event payloads are not duplicates and must not be removed. Source-row proximity
is evidence about file layout, not proof that a record is erroneous. Near
matches with different outcomes are treated as repeated attempts.

## Threshold sensitivity

The table reports the final analysis cohort among initially eligible students
under initial thresholds of one active day and one evaluated element.

| Minimum future active days ↓ / Minimum evaluated elements → | 50 elements | 100 elements | 200 elements | 500 elements |
|---|---:|---:|---:|---:|
| 1 day | 931 | 930 | 927 | 922 |
| 2 days | 905 | 905 | 903 | 902 |
| 3 days | 881 | 881 | 880 | 880 |
| 5 days | 805 | 805 | 805 | 805 |
| 7 days | 665 | 665 | 665 | 665 |

Holding the future threshold at three active days and 100 evaluated elements,
initial-threshold sensitivity is:

| Minimum initial active days ↓ / Minimum evaluated elements → | 1 element | 50 elements | 100 elements | 500 elements |
|---|---:|---:|---:|---:|
| 1 day | 881 | 871 | 864 | 830 |
| 2 days | 836 | 834 | 834 | 816 |
| 3 days | 764 | 764 | 764 | 760 |

The full grid is stored in `data/interim/threshold_sensitivity.parquet` and also
varies initial eligibility thresholds. Observability must be frozen after
examining trajectory reliability, not merely by maximizing cohort size.
