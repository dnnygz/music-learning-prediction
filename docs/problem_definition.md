# Problem definition

## Research question

Can behavioral signals recorded during a student's first seven elapsed days predict future observability and, among observable students, the subsequent trajectory of platform-recorded performance?

This project measures behavior and performance inside the platform. It does not
claim to measure general musical learning outside the observed activities.

The question contains two related but distinct predictive tasks:

1. $X_{[0,7)} \rightarrow FutureObservable$ for all initially eligible students.
2. $X_{[0,7)} \rightarrow \lambda_i$ among students whose future performance
   trajectory can be estimated.

The first task asks whether the student will remain measurable in the platform.
The second asks whether early behavior predicts the direction or magnitude of a
later content-adjusted performance trajectory. Success in one task does not imply
success in the other.

## Units of analysis

- **Event:** one recorded interaction between a student and the platform.
- **Student:** the independent unit used for prediction and model evaluation.
- **Trajectory:** a longitudinal outcome estimated after the initial period.

The raw event count is not the predictive sample size. Model evaluation must
keep every event from one student in the same data partition.

## Time windows

`days_since_signup` is continuous, so the windows use half-open intervals:

- Initial features: `[0, 7)` elapsed days.
- Future outcomes: `[7, 31)` elapsed days.

This represents exactly seven elapsed days of initial observation and uses all
remaining activity in the dataset for the future outcome. Alternative windows,
including `[15, 31)`, belong in sensitivity analysis rather than the primary
definition.

## Outcomes

1. **Future observability:** whether enough future evidence exists to estimate
   a longitudinal trajectory. In the primary operational definition,
   `FutureObservable = 1` requires at least three future days with evaluated
   performance and at least 100 evaluated elements during `[7, 31)`.
2. **Future adjusted trajectory:** a continuous, partially pooled slope of
   platform-recorded performance during `[7, 31)`, adjusted for content
   difficulty, exercise, response type, and measurement uncertainty.

Future observability is an operational data-coverage outcome, not an intrinsic
property of the student. It can change if the minimum number of days, evaluated
elements, or future window changes. A suitable methods statement is:

> Future observability was operationalized as the availability of sufficient
> future evaluated interactions to estimate subsequent performance dynamics.

Positive/stable/negative trajectory labels are operational translations of the
continuous slope, not the primary statistical target.

## Results for the original question

### Future observability

The initial cohort contains 945 students: 880 are future observable and 65 are
not observable under the primary definition. Models were evaluated with
five-fold stratified out-of-fold predictions using only features from `[0, 7)`.
Future non-observability is treated as the positive class because it is the rare
operational case.

| Model | ROC-AUC | 95% bootstrap CI | Average precision | 95% bootstrap CI | Brier | Recall at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Prevalence baseline | 0.500 | [0.500, 0.500] | 0.069 | [0.069, 0.069] | 0.064 | 0.000 |
| Balanced logistic regression | **0.841** | **[0.787, 0.886]** | **0.305** | **[0.234, 0.418]** | 0.154 | **0.769** |
| Balanced random forest | 0.831 | [0.769, 0.881] | 0.302 | [0.232, 0.418] | 0.075 | 0.462 |

The early signals therefore contain useful information for ranking students by
risk of future non-observability. The balanced logistic probabilities are not
well calibrated: class weighting improves minority recall but produces worse
Brier score and log-loss than the prevalence baseline. The supported conclusion
is predictive discrimination, not deployment-ready probability estimation.

### Future adjusted trajectory

The future trajectory was evaluated with three formulations: probability of a
relevant positive slope, continuous slope prediction with propagated target
uncertainty, and a joint hierarchical model. Improvements over constant or
random-slope baselines were negligible or inconsistent. First-week behavioral
signals did not demonstrate reliable out-of-sample prediction of individual
future adjusted trajectories under the current dataset and specifications.

### Answer

The original question has an asymmetric answer:

> Behavioral signals recorded during the first seven elapsed days predict
> future observability better than a prevalence-only baseline. Among observable
> students, the same signals do not reliably predict the subsequent adjusted
> trajectory of platform-recorded performance.

This does not invalidate the hierarchical measurement model. It distinguishes
predicting whether sufficient future evidence will exist from predicting the
latent dynamics estimated from that evidence.

## Next experiments

### Experiment 2 — temporal sensitivity

Evaluate whether additional initial history changes the conclusions by comparing
features from `[0, 7)`, `[0, 14)`, and `[0, 21)`. For a fair comparison, the
primary design will hold the outcome window fixed at `[21, 31)` so predictor and
outcome periods never overlap. Results must also report how the longer initial
window changes cohort eligibility and outcome prevalence.

**Result:** Initial eligibility increases from 945 students at day 7 to 980 at
day 14 and 996 at day 21. With the common `[21, 31)` trajectory-observability
requirement, the per-window analysis cohorts contain 505, 527, and 542 students,
respectively. The controlled intersection contains 505 students; response-specific
targets are available for 484 students for notes and 487 for chords.

On this fixed cohort and outcome window, extending the history does not improve
trajectory prediction. Ridge and Random Forest models have negative out-of-fold
$R^2$ for every history length and response type. Expected probabilistic log-loss
for $P(\lambda_i > 0.02)$ is also slightly worse than the response-specific
constant baseline in every comparison. The evidence therefore matches Case B:
results remain similarly weak with 7, 14, and 21 days, so extending the initial
observation window does not recover sufficient predictive signal. The failure
cannot be explained solely by observing only seven initial days.
The common target spans only ten elapsed days, so this conclusion is specific to
the controlled `[21, 31)` outcome and does not cover every possible later or
longer trajectory definition.

### Experiment 3 — immediate performance prediction

Evaluate the closer-to-observation task:

$$
(student, exercise, difficulty, context) \rightarrow P(success)
$$

This experiment will reuse the beta-binomial performance specification while
using temporal or group-aware holdouts. It will distinguish prediction for known
students and exercises from cold-start evaluation; random event splitting is not
valid because it would leak student and exercise information.

## Data layers

1. `events_clean.parquet`: validated event-grain data derived from the raw JSON.
2. `student_coverage.parquet`: one row for every raw student, including explicit
   observability status and exclusion reason.
3. `student_exercise_day.parquet`: binomial numerator and denominator at
   student–exercise–day–context–response-type grain for the longitudinal
   measurement model.
4. `student_prediction.parquet`: first-week student features joined to estimated
   trajectory outcomes and their calibrated uncertainty.

The legacy CSV is retained only as an exploratory baseline and is not the source
of truth for the redesigned pipeline.

## Evidence and reproducibility

- Observability implementation: `src/models/prediction/predict_observability.py`.
- Observability report: `reports/prediction/observability_v1.md`.
- Observability artifacts: `data/model_outputs/observability_v1/`.
- Trajectory comparison: `reports/prediction/predictive_formulations_v1.md`.
