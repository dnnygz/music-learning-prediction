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
| note | binomial | -101557.7805 | 0.0000 | 0.4788 | 0.1522 | 0.0521 | 1.5384 | 3.0277 |
| note | beta_binomial | -29190.4537 | 72367.3268 | 0.4938 | 0.1584 | 0.0889 | 1.0705 | 2.0552 |
| chord | binomial | -77197.5302 | 0.0000 | 0.4655 | 0.1464 | 0.0416 | 1.4294 | 2.6977 |
| chord | beta_binomial | -21609.1553 | 55588.3749 | 0.4741 | 0.1498 | 0.0697 | 1.0426 | 1.9621 |

## Findings

- Note: beta-binomial changes held-out joint LPD by 72,367.3; residual SD changes from 1.54 to 1.07. Estimated kappa is 11.01 and rho is 0.0833.
- Chord: beta-binomial changes held-out joint LPD by 55,588.4; residual SD changes from 1.43 to 1.04. Estimated kappa is 10.26 and rho is 0.0888.

Beta-binomial Pearson residuals use its posterior predictive variance, including parameter uncertainty. A residual SD closer to one indicates that the observation family accounts for more of the remaining count variability.

## Limitation

Inference uses mean-field ADVI. This is an efficient screening comparison; final uncertainty and the dispersion parameter should be checked with stronger variational diagnostics or NUTS before freezing the target.
