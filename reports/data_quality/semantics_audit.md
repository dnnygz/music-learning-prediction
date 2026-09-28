# Cohort and duplicate semantics audit

## Eligibility and observability

- Initial eligibility reasons: {'eligible': 945, 'no_initial_activity': 55}.
- Future observability reasons: {'observable': 935, 'too_few_future_evaluated_days': 53, 'no_future_activity': 12}.
- Eligible initially and observable later: 880.
- Eligible initially but not observable later: 65.
- Not initially eligible but observable later: 55.
- Neither: 0.

Reconciliation note: an earlier audit produced a historical preliminary count
of 881 because it counted any future active day. The corrected analysis cohort
is 880: one student has three future activity days but only two days containing
evaluable performance.

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
- Users affected by any tabular match when `events_data` is ignored:
  498.
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

Each cell is the number of students who meet or exceed both the row's day count
and the column's evaluated-element count. Future days count only days containing
at least one evaluated note or chord.

| Future evaluated active days ↓ / Evaluated elements → | 50 elements | 100 elements | 200 elements | 500 elements |
| ------------------------------------------------------ | ----------: | -----------: | -----------: | -----------: |
| 1 day | 931 | 930 | 927 | 922 |
| 2 days | 905 | 905 | 903 | 902 |
| 3 days | 880 | 880 | 879 | 879 |
| 5 days | 805 | 805 | 805 | 805 |
| 7 days | 665 | 665 | 665 | 665 |

Holding the future threshold at three evaluated days and 100 evaluated elements,
initial-threshold sensitivity is:

| Initial active days ↓ / Evaluated elements → | 1 element | 50 elements | 100 elements | 500 elements |
| ------------------------------------------------ | --------: | ----------: | -----------: | -----------: |
| 1 day | 880 | 870 | 863 | 829 |
| 2 days | 835 | 833 | 833 | 815 |
| 3 days | 763 | 763 | 763 | 759 |

The full grid is stored in `data/interim/threshold_sensitivity.parquet` and also
varies initial eligibility thresholds. Observability must be frozen after
examining trajectory reliability, not merely by maximizing cohort size.
