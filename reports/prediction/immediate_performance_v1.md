# Immediate performance prediction

## Question

Given the student, exercise, declared difficulty, practice context, and response
type, what is the probability that an evaluated element will be successful?

## Design

- Training window: `[0, 21)`.
- Temporal test window: `[21, 31)`.
- Notes and chords fitted separately.
- Models: global-rate baseline, basic hierarchical beta-binomial, and hierarchical
  beta-binomial with practice mode and song type.
- Exercise and student effects use partial pooling.
- Unseen students and exercises are integrated using population random-effect
  distributions rather than global frequency encodings.
- Inference: 20,000 ADVI iterations and
  1,000 posterior draws.

## Holdout coverage

| scenario | rows | students | exercises | evaluated_elements |
| --- | --- | --- | --- | --- |
| known_student_known_exercise | 21265 | 817 | 646 | 11176749 |
| known_student_new_exercise | 142 | 61 | 55 | 61474 |
| new_student_known_exercise | 213 | 4 | 93 | 104472 |

## Overall temporal holdout performance

| response_type | model | binary_log_loss | brier_score | expected_calibration_error | pearson_residual_sd | beta_binomial_lpd |
| --- | --- | --- | --- | --- | --- | --- |
| note | global_rate_baseline | 0.4780 | 0.1504 | 0.0015 | 7.8388 | nan |
| note | hierarchical_basic | 0.4575 | 0.1447 | 0.0144 | 1.2672 | -53240.1046 |
| note | hierarchical_context | 0.4576 | 0.1447 | 0.0144 | 1.2568 | -53100.1378 |
| chord | global_rate_baseline | 0.4549 | 0.1407 | 0.0023 | 7.8287 | nan |
| chord | hierarchical_basic | 0.4375 | 0.1365 | 0.0218 | 1.2574 | -44377.3984 |
| chord | hierarchical_context | 0.4394 | 0.1374 | 0.0330 | 1.2023 | -44046.6668 |

Lower log-loss, Brier score, and calibration error are better. Beta-binomial LPD
is only defined for the probabilistic hierarchical models; higher values are
better.

## Performance by prediction scenario

| response_type | scenario | rows | evaluated_elements | binary_log_loss | brier_score | expected_calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| note | known_student_known_exercise | 11013 | 5806391 | 0.4574 | 0.1447 | 0.0135 |
| note | known_student_new_exercise | 89 | 41317 | 0.5637 | 0.1886 | 0.0874 |
| note | new_student_known_exercise | 90 | 47024 | 0.3902 | 0.1138 | 0.1134 |
| chord | known_student_known_exercise | 10252 | 5370358 | 0.4399 | 0.1376 | 0.0319 |
| chord | known_student_new_exercise | 53 | 20157 | 0.4902 | 0.1567 | 0.1317 |
| chord | new_student_known_exercise | 123 | 57448 | 0.3746 | 0.1088 | 0.1172 |

The known-student/known-exercise scenario is the primary result. Cold-start
student and exercise estimates are descriptive because the temporal holdout
contains very few new students and relatively few unseen-exercise rows.

## Interpretation

The hierarchical model improves immediate mean-probability prediction over the
global-rate baseline. For notes, the basic model reduces log-loss from 0.4780 to
0.4575 and Brier score from 0.1504 to 0.1447. For chords, it reduces log-loss
from 0.4549 to 0.4375 and Brier score from 0.1407 to 0.1365. This is evidence that
student, exercise, difficulty, and time contain useful signal about current
platform-recorded performance.

Adding practice mode and song type improves beta-binomial LPD and slightly lowers
residual dispersion, but does not improve element-level mean probabilities. The
context model is virtually tied with the basic model for notes and is worse in
chord log-loss and Brier score. Therefore, the richer context specification is
not selected as the best immediate probability model from this experiment.

The global-rate baseline has very low aggregate calibration error by construction:
one constant probability closely matches the overall success rate. That does not
make it individually informative. Its worse log-loss and Brier scores show that
it cannot distinguish easier from harder current interactions.

This experiment predicts platform-recorded success under current conditions. It
does not predict whether a student will improve and does not identify a causal
effect of practice mode, song type, or difficulty. The result shows that the
hierarchical structure is useful for estimating immediate expected performance
even though early behavior did not predict later individual slopes.
