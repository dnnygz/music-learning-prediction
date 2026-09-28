# First-week behavior predicts future observability, while individual learning trajectories remain difficult to predict

## Conclusion

Behavioral signals recorded during `[0, 7)` contain useful out-of-sample information
about whether a student will have enough evaluated activity in `[7, 31)` to estimate
a subsequent performance trajectory. The strongest ranking model reached ROC-AUC
0.841 and average precision
0.305, compared with a non-observability
prevalence of 0.069 (4.4x lift).

This supports predictability of **future observability**, not a causal claim about
retention and not a claim that the model is ready for operational deployment.

## Operational outcome

`FutureObservable = 1` when a student has at least three future days containing
evaluated performance and at least 100 evaluated elements during `[7, 31)`.
It means that the platform collected enough longitudinal evidence to estimate
performance dynamics. It is an operational data-coverage definition, not an
intrinsic student attribute.

- Initially eligible students: 945.
- Future observable: 880.
- Future not observable: 65.
- Non-observability prevalence: 6.9%.

## Out-of-fold evaluation

Predictions use five-fold stratified cross-validation. All features come from
`[0, 7)` and the outcome comes from `[7, 31)`. No future activity is included in
the predictors. The table treats future non-observability as the positive class
because it is the rare operational case.

| Model | ROC-AUC | 95% bootstrap CI | Average precision | 95% bootstrap CI | Brier | Log-loss | Precision @ 0.5 | Recall @ 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prevalence baseline | 0.500 | [0.500, 0.500] | 0.069 | [0.069, 0.069] | 0.064 | 0.250 | 0.000 | 0.000 |
| Balanced logistic regression | 0.841 | [0.787, 0.886] | 0.305 | [0.234, 0.418] | 0.154 | 0.479 | 0.195 | 0.769 |
| Balanced random forest | 0.831 | [0.769, 0.881] | 0.302 | [0.232, 0.418] | 0.075 | 0.257 | 0.316 | 0.462 |

The balanced logistic model provides the best discrimination and recovers most
non-observable students at the default threshold, but class weighting makes its
probabilities overconfident: its Brier score and log-loss are worse than the
prevalence baseline. The random forest is less aggressive and better calibrated,
but misses more non-observable students. Threshold selection and probability
calibration must therefore depend on an explicit operational cost before use.

At the provisional 0.5 threshold, logistic regression identifies
50 of 65 non-observable
students, with 207 false positives. The random
forest identifies 30, with
65 false positives. These counts illustrate why
the threshold cannot be selected without defining the cost of missed and
unnecessary follow-up.

## Limitations

- Only 65 students belong to the minority class, so
  uncertainty remains material despite the observed ranking signal.
- The observability threshold is operational and should be included in later
  sensitivity analysis.
- Stratified cross-validation evaluates generalization within this 945-student
  cohort; it does not establish temporal or external generalization.
- Observability is not identical to learning, persistence, or platform retention.

## Answer to the observability sub-question

Yes, under the current operational definition and dataset, first-week behavioral
signals predict future observability better than a prevalence-only baseline.
The supported claim concerns ranking/discrimination. Calibrated risk estimation
and an operational decision threshold remain future work.

## Final answer to the original research question

The original question contained two related but distinct prediction tasks.

For future observability, first-week behavioral signals showed meaningful predictive information. Models trained exclusively on [0,7) activity substantially outperformed a prevalence baseline when identifying students who would fail to generate sufficient future evaluated activity.

For subsequent individual performance trajectory, however, increasingly complex formulations did not demonstrate reliable predictive improvement. This indicates that early behavioral signals contain information about continued measurable engagement, but not enough information to identify individual learning dynamics after controlling for exercise difficulty, content composition, and measurement uncertainty.

Therefore, the answer is asymmetric: early behavior predicts whether future learning evidence will be available, but does not reliably predict how an individual's measured performance trajectory will evolve.