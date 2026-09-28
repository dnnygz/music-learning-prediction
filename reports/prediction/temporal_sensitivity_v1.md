# Temporal sensitivity of adjusted-trajectory prediction

## Question

Is the failure of early trajectory prediction caused by insufficient observation
time, or does the platform lack enough early behavioral signal to identify later
individual performance dynamics?

## Controlled design

- Histories compared: `[0, 7)`, `[0, 14)`, and `[0, 21)`.
- Common target window: `[21, 31)`.
- Initial eligibility: at least one active day and one evaluated element.
- Target observability: at least three evaluated days and 100 evaluated elements.
- Main comparison cohort: intersection of all initial cohorts and target-observable
  students.
- Common students: 505.
- Students with an estimable note target: 484.
- Students with an estimable chord target: 487.
- Identical user folds are used for every history window.

The common future window prevents temporal overlap. The common cohort prevents a
longer-history model from appearing better merely because its population changed.

## Cohort accounting

| history_days | initial_eligible | target_observable | eligible_and_target_observable | common_analysis_cohort |
| --- | --- | --- | --- | --- |
| 7 | 945 | 546 | 505 | 505 |
| 14 | 980 | 546 | 527 | 505 |
| 21 | 996 | 546 | 542 | 505 |

Eligibility increases with a longer observation window because the windows are
nested: a student eligible at day 7 remains eligible at days 14 and 21. The main
sample reduction comes from requiring a trajectory estimable in the shorter
future window `[21, 31)`, not from losing initial history.

## Continuous adjusted-slope prediction

| history_days | response_type | model | rmse_posterior_mean | r2_posterior_mean | correlation_posterior_mean |
| --- | --- | --- | --- | --- | --- |
| 0 | chord | constant_baseline | 0.0092 | 0.0000 | nan |
| 0 | note | constant_baseline | 0.0210 | 0.0000 | nan |
| 7 | chord | ridge | 0.0093 | -0.0252 | 0.0390 |
| 7 | chord | random_forest | 0.0095 | -0.0520 | 0.0240 |
| 7 | note | ridge | 0.0214 | -0.0366 | 0.0022 |
| 7 | note | random_forest | 0.0214 | -0.0389 | 0.0520 |
| 14 | chord | ridge | 0.0093 | -0.0264 | 0.0242 |
| 14 | chord | random_forest | 0.0094 | -0.0316 | 0.0383 |
| 14 | note | ridge | 0.0216 | -0.0537 | -0.0206 |
| 14 | note | random_forest | 0.0214 | -0.0393 | 0.0625 |
| 21 | chord | ridge | 0.0094 | -0.0440 | -0.0232 |
| 21 | chord | random_forest | 0.0094 | -0.0452 | 0.0211 |
| 21 | note | ridge | 0.0214 | -0.0358 | 0.0278 |
| 21 | note | random_forest | 0.0214 | -0.0407 | 0.0462 |

## Probability of a relevant positive slope

The operational probability target is $P(\lambda_i > 0.02)$.

| history_days | response_type | model | expected_log_loss | expected_brier | correlation_probability |
| --- | --- | --- | --- | --- | --- |
| 0 | chord | constant_baseline | 0.6787 | 0.2428 | nan |
| 0 | note | constant_baseline | 0.6885 | 0.2477 | nan |
| 7 | chord | ridge | 0.6800 | 0.2434 | 0.0042 |
| 7 | chord | random_forest | 0.6789 | 0.2429 | 0.1007 |
| 7 | note | ridge | 0.6920 | 0.2494 | -0.0340 |
| 7 | note | random_forest | 0.6931 | 0.2499 | -0.0143 |
| 14 | chord | ridge | 0.6799 | 0.2434 | -0.0283 |
| 14 | chord | random_forest | 0.6796 | 0.2432 | 0.0436 |
| 14 | note | ridge | 0.6933 | 0.2501 | -0.0468 |
| 14 | note | random_forest | 0.6925 | 0.2497 | -0.0049 |
| 21 | chord | ridge | 0.6802 | 0.2435 | -0.0449 |
| 21 | chord | random_forest | 0.6799 | 0.2434 | 0.0411 |
| 21 | note | ridge | 0.6922 | 0.2494 | -0.0089 |
| 21 | note | random_forest | 0.6919 | 0.2494 | 0.0192 |

## Interpretation

Before running the experiment, three possible result patterns were defined:

- **Case A:** prediction is weak with 7 days but improves materially with both
  14 and 21 days; insufficient initial observation time is the likely limitation.
- **Case B:** results remain similarly weak with 7, 14, and 21 days; extending
  the initial observation window does not recover sufficient predictive signal.
- **Case C:** prediction becomes useful only with 21 days; a later prediction
  point may be justified even though 7 days is insufficient.

The result matches **Case B**: extending the initial history does not recover
useful predictive signal. Every continuous model has negative out-of-fold
$R^2$, so each performs worse than predicting the response-specific cohort mean.
All probabilistic models also have slightly worse expected log-loss than their
constant baselines. Correlations remain close to zero and do not improve
consistently from 7 to 14 or 21 days.

Therefore, under the current features, target, and common-cohort design, the
failure of trajectory prediction cannot be attributed only to using seven days
of initial observation. Waiting until day 14 or day 21 does not materially improve
identification of later individual adjusted-performance dynamics.

The `[21, 31)` slope target is estimated with mean-field ADVI. Its uncertainty is
propagated in the continuous two-stage models, but no window-specific NUTS
validation was performed; conclusions should emphasize relative predictive
performance rather than exact posterior coverage.

The fixed target contains only ten elapsed days, compared with 24 days in the
original `[7, 31)` analysis. This is necessary to compare all histories without
overlap, but can make the common trajectory noisier. The experiment answers
whether longer histories improve prediction of this same later target; it does
not prove that every possible longer-horizon trajectory target is unpredictable.
