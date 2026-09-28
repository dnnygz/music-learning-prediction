# Binomial versus beta-binomial dispersion diagnostic

## Design

- Measurement window: `[7, 31)`.
- User holdout: 790 train / 198 test.
- Both families use Model 4 predictors and identical zero-centered hierarchical priors.
- Notes and chords are fitted separately.
- Held-out likelihood integrates all rows from each student jointly.

## Comparison

| response_type | family | joint_user_log_predictive_density | delta_lpd_vs_binomial | binary_log_loss | brier_score | expected_calibration_error | pearson_residual_sd | absolute_residual_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | binomial | -95337.7868 | 0.0000 | 0.4714 | 0.1491 | 0.0279 | 2.1017 | 4.2833 |
| note | beta_binomial | -28916.7941 | 66420.9927 | 0.4887 | 0.1560 | 0.0811 | 1.0475 | 2.0017 |
| chord | binomial | -70205.2885 | 0.0000 | 0.4606 | 0.1443 | 0.0229 | 1.9499 | 3.8120 |
| chord | beta_binomial | -21525.0422 | 48680.2463 | 0.4747 | 0.1500 | 0.0713 | 1.0435 | 1.9619 |

## Findings

- Note: beta-binomial changes held-out joint LPD by 66,421.0; residual SD changes from 2.10 to 1.05. Estimated kappa is 9.27 and rho is 0.1020.
- Chord: beta-binomial changes held-out joint LPD by 48,680.2; residual SD changes from 1.95 to 1.04. Estimated kappa is 9.22 and rho is 0.1015.

Beta-binomial Pearson residuals use its posterior predictive variance, including parameter uncertainty. A residual SD closer to one indicates that the observation family accounts for more of the remaining count variability.

## Limitation

Inference uses mean-field ADVI. This is an efficient screening comparison; final uncertainty and the dispersion parameter should be checked with stronger variational diagnostics or NUTS before freezing the target.
