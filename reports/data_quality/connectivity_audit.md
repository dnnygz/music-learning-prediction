# Student–exercise connectivity audit

## Scope

This audit uses events with at least one evaluated note or chord. It determines
whether exercise difficulty can be identified separately from student
performance before fitting a hierarchical binomial model.

## Graph summary

- Students: 1,000.
- Exercises: 969.
- Unique student–exercise edges: 49,797.
- Connected components: 1.
- Largest component: 1,000 students and
  969 exercises.

## Exercise support

| Student coverage per exercise | Exercises |
|---|---:|
| Exactly 1 student | 175 |
| 2–4 students | 270 |
| At least 5 students | 524 |
| At least 10 students | 360 |
| At least 20 students | 257 |

The median exercise is observed for 5 students;
the 90th percentile is 100, and the maximum is
985.

## Student support

- Median distinct exercises per student: 45.
- 10th–90th percentile: 24–85.
- Median evaluated event rows per student: 338.

## Exercises unseen under student-level evaluation splits

Across 50 deterministic 80/20 user splits:

- Median share of distinct test exercises unseen in training:
  5.82%.
- 90th percentile: 7.95%.
- Median share of test event rows belonging to unseen exercises:
  0.15%.
- 90th percentile: 0.24%.

## Modeling implications

1. Exercise effects require partial pooling; rare exercises cannot receive
   unrestricted fixed-effect estimates.
2. Component structure determines whether effects are globally comparable.
3. Unseen exercises need a fallback based on declared difficulty and content
   features, not a globally calculated frequency encoding.
4. Start with student and exercise random intercepts plus declared difficulty,
   then add context through ablations.
