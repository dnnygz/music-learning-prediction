# Beta-binomial student random-slope experiment

## Design

- User split: 790 train / 198 test.
- Notes and chords fitted separately.
- Random slopes are reported as adjusted future-performance change in log-odds per day.
- `adjusted_trajectory_slope` includes the global slope plus the centered student deviation.

## Held-out model comparison

| response_type | model | joint_user_lpd | delta_lpd_vs_random_intercept | binary_log_loss | brier_score | pearson_residual_sd | kappa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| note | random_intercept | -29164.7371 | 0.0000 | 0.4940 | 0.1585 | 1.0688 | 10.8025 |
| note | random_slope | -29118.4630 | 46.2741 | 0.4956 | 0.1592 | 1.0650 | 11.2464 |
| chord | random_intercept | -21607.6784 | 0.0000 | 0.4752 | 0.1502 | 1.0435 | 10.2013 |
| chord | random_slope | -21565.0425 | 42.6359 | 0.4764 | 0.1508 | 1.0470 | 10.6927 |

## Population slope variation

| response_type | window | sigma_lambda_estimate | sigma_lambda_posterior_sd | sigma_lambda_hdi_3pct | sigma_lambda_hdi_97pct | global_slope_estimate | global_slope_sd | probability_sigma_lambda_gt_0_001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | 7_21 | 0.0567 | 0.0010 | 0.0548 | 0.0587 | 0.0325 | 0.0017 | 1.0000 |
| note | 7_31 | 0.0285 | 0.0005 | 0.0275 | 0.0294 | 0.0242 | 0.0008 | 1.0000 |
| chord | 7_21 | 0.0278 | 0.0014 | 0.0252 | 0.0304 | 0.0330 | 0.0017 | 1.0000 |
| chord | 7_31 | 0.0233 | 0.0006 | 0.0221 | 0.0245 | 0.0245 | 0.0009 | 1.0000 |

## Individual slope identifiability

| response_type | slope_class | students | percentage |
| --- | --- | --- | --- |
| chord | negative | 2 | 0.0021 |
| chord | positive | 262 | 0.2743 |
| chord | uncertain | 691 | 0.7236 |
| note | negative | 9 | 0.0093 |
| note | positive | 301 | 0.3122 |
| note | uncertain | 654 | 0.6784 |

## Temporal stability

| response_type | students_in_both_windows | pearson_correlation | spearman_correlation | sign_concordance | classification_concordance | median_absolute_slope_difference |
| --- | --- | --- | --- | --- | --- | --- |
| chord | 911 | 0.5969 | 0.5152 | 0.9473 | 0.7980 | 0.0100 |
| note | 925 | 0.7072 | 0.5624 | 0.8443 | 0.7676 | 0.0171 |

## Calibration error by held-out user volume

| response_type | model | segment_value | evaluated | calibration_error |
| --- | --- | --- | --- | --- |
| chord | random_intercept | Q1_low | 290806 | -0.0105 |
| chord | random_intercept | Q2 | 515962 | -0.0601 |
| chord | random_intercept | Q3 | 676731 | -0.0955 |
| chord | random_intercept | Q4_high | 997199 | -0.0778 |
| chord | random_slope | Q1_low | 290806 | -0.0106 |
| chord | random_slope | Q2 | 515962 | -0.0611 |
| chord | random_slope | Q3 | 676731 | -0.0975 |
| chord | random_slope | Q4_high | 997199 | -0.0803 |
| note | random_intercept | Q1_low | 495478 | -0.0174 |
| note | random_intercept | Q2 | 774205 | -0.0767 |
| note | random_intercept | Q3 | 848330 | -0.0998 |
| note | random_intercept | Q4_high | 1205322 | -0.1186 |
| note | random_slope | Q1_low | 495478 | -0.0180 |
| note | random_slope | Q2 | 774205 | -0.0799 |
| note | random_slope | Q3 | 848330 | -0.1027 |
| note | random_slope | Q4_high | 1205322 | -0.1211 |

## Interpretation

Population slope variation is non-zero, but most individual intervals overlap zero. The random-slope model improves joint held-out LPD only modestly and does not improve element-level log loss or volume-conditioned mean calibration.

Inference uses 20,000-iteration mean-field ADVI with 1,000 posterior draws. Mean-field ADVI can underestimate dependence and uncertainty, so the reported identifiable percentages may be optimistic and require a reduced-data NUTS check before target construction.

These slopes describe change in platform-recorded, content-adjusted future performance. They are not yet labeled as learning.
