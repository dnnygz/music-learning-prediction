# Incremental hierarchical measurement model

## Scope

- Measurement window: `[7, 31)`.
- User-level split: 790 train / 198 test.
- Notes and chords were fitted as separate binomial model sequences.
- Inference: mean-field ADVI through Bambi/PyMC; uncertainty is approximate.
- AIC/BIC are not reported because these are Bayesian hierarchical fits; WAIC and held-out log predictive density are used instead.

## Held-out comparison

| response_type | model | test_log_predictive_density | delta_test_lpd_vs_previous | test_binary_log_loss | test_brier_score | test_expected_calibration_error | train_waic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| note | model_0 | -184012.4903 |  | 0.4793 | 0.1509 | 0.0044 | 4131259.1563 |
| note | model_1 | -165652.4321 | 18360.0582 | 0.4733 | 0.1491 | 0.0058 | 5024250.5969 |
| note | model_2 | -142283.7046 | 23368.7275 | 0.4692 | 0.1480 | 0.0073 | 7482860.4445 |
| note | model_3 | -100661.9311 | 41621.7736 | 0.4706 | 0.1487 | 0.0235 | 5056291.2695 |
| note | model_4 | -95893.6499 | 4768.2812 | 0.4721 | 0.1494 | 0.0279 | 5511525.9175 |
| chord | model_0 | -146392.6174 |  | 0.4692 | 0.1467 | 0.0078 | 5246550.5865 |
| chord | model_1 | -138980.1383 | 7412.4791 | 0.4657 | 0.1457 | 0.0132 | 6558725.0952 |
| chord | model_2 | -114752.2691 | 24227.8692 | 0.4591 | 0.1436 | 0.0134 | 9391800.8767 |
| chord | model_3 | -74236.8969 | 40515.3722 | 0.4592 | 0.1439 | 0.0168 | 6206114.9115 |
| chord | model_4 | -71213.3448 | 3023.5521 | 0.4626 | 0.1451 | 0.0282 | 5947701.1319 |

The held-out log predictive density integrates all rows from a user jointly, preserving the shared latent student effect. Element-level log loss, Brier score, and calibration are complementary and can disagree with this distributional criterion.

## Incremental contribution

### Note
- `model_1` improved held-out joint user LPD by 18,360.1.
- `model_2` improved held-out joint user LPD by 23,368.7.
- `model_3` improved held-out joint user LPD by 41,621.8.
- `model_4` improved held-out joint user LPD by 4,768.3.
### Chord
- `model_1` improved held-out joint user LPD by 7,412.5.
- `model_2` improved held-out joint user LPD by 24,227.9.
- `model_3` improved held-out joint user LPD by 40,515.4.
- `model_4` improved held-out joint user LPD by 3,023.6.

## Residual diagnostics

- Note: lowest element-level log loss is `model_2` (0.4692); Model 4 residual SD is 7.47 and calibration error is 0.0279.
- Chord: lowest element-level log loss is `model_2` (0.4591); Model 4 residual SD is 7.37 and calibration error is 0.0282.

Residual standard deviations far above one indicate substantial extra-binomial dispersion remains. This supports evaluating a beta-binomial sensitivity model after the binomial baseline. WAIC from mean-field ADVI is retained as a diagnostic but is not used for model selection because variational posterior variance can be underestimated or distorted.

## Interpretation constraints

Model 4 has a student random intercept and a global linear time effect. It can export adjusted daily performance, but it does not yet estimate a distinct latent learning-rate parameter for each student. Probability-scale slopes vary with baseline performance through the logistic link; a later random-slope model is required before interpreting slope differences as individual learning trajectories.
