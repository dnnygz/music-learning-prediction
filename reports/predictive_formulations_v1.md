# Comparison of predictive formulations

## Question

Can behavioral signals from `[0, 7)` predict the subsequent content-adjusted
platform-performance slope estimated in `[7, 31)`?

Notes and chords are modeled separately. Every evaluation split is defined by
student. No future measurement is included among the predictors.

## Target construction

The full-cohort ADVI slope means are retained, while their posterior standard
deviations are inflated using the median NUTS/ADVI ratios measured on the fixed
25% validation cohort:

- Notes: 1.743×.
- Chords: 1.417×.

This is an uncertainty calibration approximation, not a full-population NUTS
posterior. Model A uses the provisional threshold `delta = 0.02` log-odds/day.

## Predictive formulations

The prediction problem was evaluated under three progressively complex formulations.

The first formulation treats the estimated individual trajectory as a derived
target:

$$X_{0-7}\rightarrow \hat{\lambda_i}$$

The second formulation incorporates uncertainty in the latent trajectory by
predicting the probability of a relevant positive trajectory:

$$X_{0-7}\rightarrow P(\lambda_i>\delta)$$

The third formulation avoids treating the estimated trajectory as an observed
target and integrates prediction directly into the hierarchical measurement
model:

$$X_{0-7}\rightarrow \lambda_i \rightarrow Y$$

These formulations represent increasing levels of statistical integration:
from a two-stage predictive approach, to uncertainty-aware prediction, to a
joint hierarchical latent-variable model.

## Model A — probability of a relevant positive slope

Five-fold out-of-fold results:

| Response | Model | Expected log-loss | Constant baseline | Expected Brier | Correlation |
|---|---|---:|---:|---:|---:|
| Chord | Ridge | 0.6869 | 0.6875 | 0.2467 | 0.136 |
| Chord | Random forest | 0.6878 | 0.6875 | 0.2473 | 0.116 |
| Note | Ridge | 0.6896 | 0.6903 | 0.2482 | 0.127 |
| Note | Random forest | 0.6920 | 0.6903 | 0.2494 | 0.105 |

The improvement over predicting the cohort-average probability is negligible.

## Model B — continuous slope with multiple imputation

Targets are repeatedly sampled from the calibrated Normal approximation to
each student's slope posterior. Predictions are averaged across imputations.

| Response | Model | RMSE slope mean | Constant RMSE | R² | Correlation |
|---|---|---:|---:|---:|---:|
| Chord | Ridge | 0.01506 | 0.01511 | 0.0070 | 0.137 |
| Chord | Random forest | 0.01509 | 0.01511 | 0.0032 | 0.122 |
| Note | Ridge | 0.02252 | 0.02228 | -0.0220 | 0.094 |
| Note | Random forest | 0.02226 | 0.02228 | 0.0020 | 0.142 |

The first-week features explain effectively none of the variation in posterior
slope means. Expected RMSE including target uncertainty is 0.029 for chords and
0.039 for notes, larger than the between-student predictive signal.

## Model C — joint hierarchical slope regression

Eight standardized early features enter the beta-binomial model through
interactions with time, so they predict the population mean of the student
slope while a residual random slope remains.

| Response | Model | Held-out joint LPD | Delta LPD | Log-loss | Residual SD |
|---|---|---:|---:|---:|---:|
| Note | Random-slope baseline | -27,008.7 | — | 0.4899 | 1.0751 |
| Note | Joint early features | -27,036.2 | -27.6 | 0.4902 | 1.0780 |
| Chord | Random-slope baseline | -20,760.8 | — | 0.4816 | 1.0307 |
| Chord | Joint early features | -20,749.0 | +11.7 | 0.4825 | 1.0442 |

The joint model does not produce a consistent improvement. The small chord LPD
gain is accompanied by worse element-level loss and residual dispersion. ADVI
intervals for individual early-feature coefficients must not be interpreted as
evidence because the overall held-out model does not improve and prior NUTS
validation showed that ADVI understates slope uncertainty.

## Decision
None of the evaluated formulations demonstrated reliable out-of-sample predictive capability for individual future adjusted trajectories using only behavioral signals recorded during the first seven elapsed days.

The three approaches addressed the same prediction problem under different assumptions:

1. A two-stage approach predicting the estimated individual trajectory.
2. A probabilistic formulation predicting the probability of a relevant positive trajectory.
3. A joint hierarchical model incorporating early behavioral features directly into the latent trajectory estimation process.

Across all formulations, improvements over simple baselines were negligible or inconsistent. The joint hierarchical model, despite providing a statistically integrated treatment of the latent trajectory and measurement process, did not produce a consistent improvement in held-out predictive performance.

These results suggest that the limiting factor is not model complexity, but the amount of predictive information contained in the first seven days of platform activity. Early behavioral signals capture aspects of initial engagement and observed performance; however, they are insufficient to reliably determine individual future performance trajectories after controlling for exercise difficulty, content composition, and measurement uncertainty.

Possible explanations include:

- Individual learning trajectories may depend on factors not represented in platform logs, such as previous musical experience, motivation, learning objectives, or external practice.
- Seven days may provide insufficient observations to estimate stable individual learning dynamics.
- Future practice behavior may introduce changes that cannot be inferred from the initial observation period alone.

Therefore, the project does not support claiming successful early prediction of individual learning trajectories with the available data. Instead, the results provide evidence about the limits of early behavioral prediction under this dataset and formulation.

A complementary analysis should evaluate whether first-week signals are more informative for predicting future observability or retention, since continuity-related outcomes may contain stronger behavioral signals than long-term individual performance trajectories.

Future improvements would require either additional information sources, a revised prediction horizon or alternative targets that better match the observable behavior captured by the platform.
