# Student–exercise–day modeling table audit

## Grain

One row represents one student, exercise, integer day, song, declared
difficulty, play mode, song type, and response type (`note` or `chord`). Context
is part of the key because play mode can vary within a student–exercise–day.

Only rows with a positive evaluated denominator are emitted. Event-level rates
are never averaged; successful and evaluated counts are summed separately.

## Profile

- Rows: 100,184.
- Students: 1,000.
- Exercises: 969.
- Day range: 0–30.
- Note rows: 50,098.
- Chord rows: 50,086.

## Reconciliation

| Response type | Successful | Evaluated | Matches event source |
|---|---:|---:|---:|
| Notes | 21,116,851 | 25,926,994 | Yes |
| Chords | 19,071,577 | 22,912,723 | Yes |

## Integrity checks

- Duplicate grain keys: 0.
- Invalid binomial rows: 0.
- Null grouping keys: 0.
- Exact raw duplicates remain included consistently with the principal data
  pipeline; their removal belongs in a sensitivity analysis.

This table is suitable as the input layer for the hierarchical binomial model.
It is not yet a student-level prediction dataset or a finalized trajectory
target.
