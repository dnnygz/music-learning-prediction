# Nonlinear beta-binomial bridge model

## Design

- User split: 790 train / 198 test.
- Linear baseline: standardized difficulty and day.
- Bridge: cubic B-splines with 5 degrees of freedom for difficulty and day.
- The likelihood and hierarchical student/exercise structure are otherwise identical.

## Held-out comparison

| response_type | model | joint_user_lpd | delta_lpd_vs_linear | binary_log_loss | brier_score | weighted_absolute_row_error | pearson_residual_sd | kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | linear | -29164.7371 | 0.0000 | 0.4940 | 0.1585 | 0.1327 | 1.0688 | 10.8025 |
| note | spline | -29196.1736 | -31.4365 | 0.5586 | 0.1872 | 0.2034 | 0.9784 | 10.8788 |
| chord | linear | -21607.6784 | 0.0000 | 0.4752 | 0.1502 | 0.1223 | 1.0435 | 10.2013 |
| chord | spline | -22355.5698 | -747.8914 | 0.8925 | 0.3397 | 0.4385 | 1.1267 | 10.1282 |

## Findings

- Note: spline changes LPD by -31.4 and log loss by +0.0646. Its largest supported difficulty calibration error is -0.316 at level 7.
- Chord: spline changes LPD by -747.9 and log loss by +0.4173. Its largest supported difficulty calibration error is -0.536 at level 6.

The bridge is accepted only if it improves mean calibration without sacrificing the beta-binomial residual behavior. No additional spline sizes are searched.
