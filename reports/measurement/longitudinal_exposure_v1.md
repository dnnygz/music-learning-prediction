# Lagged exposure and longitudinal platform performance

## Executive summary

This experiment compares only M0 and M1. It asks whether prior-day evaluated
exposure is associated with subsequent platform-recorded performance after
adjusting for student level, exercise, declared difficulty, population time,
and student-specific temporal slopes. Results are associational, not causal.

M0 supports a positive average temporal trend and non-zero heterogeneity in
student slopes. In M1, both between-student and within-student exposure
coefficients are positive and their approximate ADVI intervals exclude zero.
However, M1 performs materially worse than M0 in both out-of-sample designs and
systematically overpredicts success. The exposure association estimated in the
training period therefore does not generalize as a stable predictive association
to days 21--30.

## Design

- Training: `[0, 21)`.
- Primary evaluation: rolling-origin `[21, 31)`; the prediction for day `t` uses
  observed exposure from day `t-1` and no information from day `t` or later.
- Sensitivity: frozen day 21; outcomes on day 21 use exposure from day 20.
- The student-specific exposure mean is calculated only over training calendar
  days 0--20, including zero-exposure days, and remains frozen in both evaluations.
- Notes and chords are modeled separately with hierarchical beta-binomial models.
- Inference: 20,000 ADVI iterations and
  1,000 posterior draws.

Exposure is defined as:

$$ExposureRaw_{i,t-1}=\log(1+EvaluatedElements_{i,t-1})$$

$$BetweenExposure_i=mean_{t\in[0,21)}(ExposureRaw_{i,t-1})$$

$$WithinExposure_{i,t}=ExposureRaw_{i,t-1}-BetweenExposure_i$$

## Out-of-sample comparison

| evaluation | response_type | model | joint_user_lpd | delta_lpd_vs_M0 | binary_log_loss | delta_log_loss_vs_M0 | brier_score | temporal_calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_21_31 | note | M0 | -52353.9957 | 0.0000 | 0.4663 | -0.0000 | 0.1474 | 0.0092 |
| frozen_day_21 | note | M0 | -6296.5127 | 0.0000 | 0.4699 | -0.0000 | 0.1498 | 0.0189 |
| rolling_21_31 | note | M1 | -54286.3968 | -1932.4011 | 0.4857 | -0.0194 | 0.1499 | 0.0650 |
| frozen_day_21 | note | M1 | -6648.3648 | -351.8520 | 0.4811 | -0.0112 | 0.1507 | 0.0569 |
| rolling_21_31 | chord | M0 | -43653.4599 | 0.0000 | 0.4428 | -0.0000 | 0.1381 | 0.0231 |
| frozen_day_21 | chord | M0 | -5724.8703 | 0.0000 | 0.4469 | -0.0000 | 0.1407 | 0.0210 |
| rolling_21_31 | chord | M1 | -44921.8458 | -1268.3859 | 0.4529 | -0.0102 | 0.1381 | 0.0472 |
| frozen_day_21 | chord | M1 | -5993.5919 | -268.7217 | 0.4593 | -0.0124 | 0.1428 | 0.0545 |

Higher LPD and lower log-loss, Brier score, and temporal calibration error are
better. M1 should be considered an improvement only if gains are reasonably
consistent across these criteria and response types.

## Posterior parameters fitted on `[0, 21)`

| response_type | model | parameter | estimate | posterior_sd | hdi_3pct | hdi_97pct | probability_positive | interval_excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | M0 | beta_time_per_day | 0.0485 | 0.0010 | 0.0466 | 0.0504 | 1.0000 | True |
| note | M0 | sigma_student_slope_per_day | 0.0402 | 0.0007 | 0.0389 | 0.0415 | 1.0000 | True |
| note | M1 | beta_time_per_day | 0.0511 | 0.0009 | 0.0493 | 0.0528 | 1.0000 | True |
| note | M1 | sigma_student_slope_per_day | 0.0396 | 0.0006 | 0.0384 | 0.0408 | 1.0000 | True |
| note | M1 | beta_between_exposure | 0.1454 | 0.0027 | 0.1404 | 0.1502 | 1.0000 | True |
| note | M1 | beta_within_exposure | 0.0180 | 0.0009 | 0.0163 | 0.0198 | 1.0000 | True |
| chord | M0 | beta_time_per_day | 0.0390 | 0.0010 | 0.0372 | 0.0408 | 1.0000 | True |
| chord | M0 | sigma_student_slope_per_day | 0.0407 | 0.0007 | 0.0393 | 0.0420 | 1.0000 | True |
| chord | M1 | beta_time_per_day | 0.0395 | 0.0011 | 0.0376 | 0.0417 | 1.0000 | True |
| chord | M1 | sigma_student_slope_per_day | 0.0421 | 0.0008 | 0.0406 | 0.0436 | 1.0000 | True |
| chord | M1 | beta_between_exposure | 0.1568 | 0.0027 | 0.1518 | 0.1617 | 1.0000 | True |
| chord | M1 | beta_within_exposure | 0.0119 | 0.0010 | 0.0101 | 0.0139 | 1.0000 | True |

`beta_time_per_day` and `sigma_student_slope_per_day` are expressed in log-odds
per elapsed day. Exposure coefficients represent a one-unit change in lagged
`log1p(evaluated elements)`. An interval excluding zero is evidence of posterior
identifiability under mean-field ADVI, not evidence of a causal effect.

Adding exposure does not substantially reorganize the baseline temporal
structure. For notes, `beta_time_per_day` changes from 0.0485 to 0.0511 (+5.3%)
and `sigma_student_slope_per_day` from 0.0402 to 0.0396 (-1.4%). For chords,
the corresponding changes are 0.0390 to 0.0395 (+1.4%) and 0.0407 to 0.0421
(+3.4%).

## Rolling-origin calibration by day

| response_type | model | day_index | evaluated | predicted_probability | observed_rate | absolute_calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| note | M0 | 21 | 732731 | 0.7918 | 0.8107 | 0.0189 |
| note | M0 | 22 | 612465 | 0.8036 | 0.8231 | 0.0195 |
| note | M0 | 23 | 637489 | 0.8067 | 0.8114 | 0.0046 |
| note | M0 | 24 | 646735 | 0.8156 | 0.8242 | 0.0087 |
| note | M0 | 25 | 576815 | 0.8249 | 0.8241 | 0.0008 |
| note | M0 | 26 | 591340 | 0.8250 | 0.8179 | 0.0071 |
| note | M0 | 27 | 586603 | 0.8131 | 0.8153 | 0.0022 |
| note | M0 | 28 | 504388 | 0.8073 | 0.8134 | 0.0061 |
| note | M0 | 29 | 481182 | 0.8193 | 0.8003 | 0.0190 |
| note | M0 | 30 | 524984 | 0.8104 | 0.8131 | 0.0027 |
| note | M1 | 21 | 732731 | 0.8676 | 0.8107 | 0.0569 |
| note | M1 | 22 | 612465 | 0.8776 | 0.8231 | 0.0545 |
| note | M1 | 23 | 637489 | 0.8776 | 0.8114 | 0.0662 |
| note | M1 | 24 | 646735 | 0.8831 | 0.8242 | 0.0588 |
| note | M1 | 25 | 576815 | 0.8930 | 0.8241 | 0.0689 |
| note | M1 | 26 | 591340 | 0.8907 | 0.8179 | 0.0728 |
| note | M1 | 27 | 586603 | 0.8792 | 0.8153 | 0.0638 |
| note | M1 | 28 | 504388 | 0.8780 | 0.8134 | 0.0646 |
| note | M1 | 29 | 481182 | 0.8863 | 0.8003 | 0.0861 |
| note | M1 | 30 | 524984 | 0.8775 | 0.8131 | 0.0644 |
| chord | M0 | 21 | 740463 | 0.7978 | 0.8188 | 0.0210 |
| chord | M0 | 22 | 594485 | 0.8058 | 0.8241 | 0.0184 |
| chord | M0 | 23 | 565304 | 0.8118 | 0.8491 | 0.0372 |
| chord | M0 | 24 | 593601 | 0.8088 | 0.8515 | 0.0426 |
| chord | M0 | 25 | 509117 | 0.8189 | 0.8542 | 0.0352 |
| chord | M0 | 26 | 490157 | 0.8105 | 0.8223 | 0.0117 |
| chord | M0 | 27 | 524564 | 0.8189 | 0.8397 | 0.0207 |
| chord | M0 | 28 | 519857 | 0.8066 | 0.8209 | 0.0143 |
| chord | M0 | 29 | 454431 | 0.8017 | 0.8139 | 0.0123 |
| chord | M0 | 30 | 455984 | 0.7965 | 0.8081 | 0.0115 |
| chord | M1 | 21 | 740463 | 0.8733 | 0.8188 | 0.0545 |
| chord | M1 | 22 | 594485 | 0.8803 | 0.8241 | 0.0562 |
| chord | M1 | 23 | 565304 | 0.8839 | 0.8491 | 0.0348 |
| chord | M1 | 24 | 593601 | 0.8806 | 0.8515 | 0.0291 |
| chord | M1 | 25 | 509117 | 0.8851 | 0.8542 | 0.0309 |
| chord | M1 | 26 | 490157 | 0.8794 | 0.8223 | 0.0571 |
| chord | M1 | 27 | 524564 | 0.8836 | 0.8397 | 0.0439 |
| chord | M1 | 28 | 519857 | 0.8751 | 0.8209 | 0.0541 |
| chord | M1 | 29 | 454431 | 0.8699 | 0.8139 | 0.0560 |
| chord | M1 | 30 | 455984 | 0.8654 | 0.8081 | 0.0573 |

## Interpretation boundaries

M1 is not retained as an improvement over M0. In rolling-origin evaluation it
worsens joint LPD by 1,932 for notes and 1,268 for chords, increases log-loss,
and increases temporal calibration error from 0.0092 to 0.0650 for notes and
from 0.0231 to 0.0472 for chords. Frozen day-21 sensitivity leads to the same
decision. Positive training-period coefficients alone are insufficient evidence
to proceed to an exposure-by-time interaction.

Rolling-origin is a dynamic longitudinal evaluation, not a frozen intervention
model at day 21. For outcomes after day 21 it uses only exposure already observed
on the immediately preceding day. Frozen day-21 sensitivity answers the narrower
question using information available at the original cutoff.

The exposure coefficients describe conditional associations. Motivation,
previous musical experience, available time, and other unrecorded factors can
affect both practice and performance, so the estimates must not be interpreted
as effects of increasing practice.
