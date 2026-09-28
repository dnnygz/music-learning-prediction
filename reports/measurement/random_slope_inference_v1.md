# Random-slope ADVI versus NUTS validation

## Design

- Deterministic user subsample: 25% (247 users).
- ADVI: 20,000 iterations, 1,000 draws.
- NUTS: 4 chains, 750 tune and 750 retained draws per chain.
- Both methods use identical beta-binomial random-slope specifications and data scaling.

## Population parameter comparison

| response_type | parameter | advi_estimate | nuts_estimate | advi_sd | nuts_sd | absolute_difference | difference_in_nuts_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chord | 1|exercise_id_sigma | 0.7262 | 0.7637 | 0.0124 | 0.0462 | 0.0375 | 0.8100 |
| chord | 1|user_id_sigma | 0.8408 | 0.8739 | 0.0087 | 0.0448 | 0.0331 | 0.7400 |
| chord | Intercept | 1.2128 | 1.1967 | 0.0083 | 0.0848 | 0.0161 | 0.1894 |
| chord | difficulty_z | -0.7065 | -0.6952 | 0.0111 | 0.0510 | 0.0114 | 0.2234 |
| chord | global_slope_per_day | 0.0251 | 0.0241 | 0.0012 | 0.0033 | 0.0010 | 0.3098 |
| chord | kappa | 10.2220 | 10.6117 | 0.1954 | 0.2449 | 0.3898 | 1.5915 |
| chord | sigma_lambda_per_day | 0.0209 | 0.0319 | 0.0013 | 0.0032 | 0.0109 | 3.4548 |
| note | 1|exercise_id_sigma | 0.6360 | 0.6621 | 0.0127 | 0.0368 | 0.0260 | 0.7065 |
| note | 1|user_id_sigma | 0.7142 | 0.7682 | 0.0094 | 0.0389 | 0.0540 | 1.3861 |
| note | Intercept | 1.1835 | 1.1533 | 0.0077 | 0.0641 | 0.0302 | 0.4712 |
| note | difficulty_z | -0.7219 | -0.7235 | 0.0089 | 0.0343 | 0.0017 | 0.0488 |
| note | global_slope_per_day | 0.0243 | 0.0219 | 0.0011 | 0.0029 | 0.0024 | 0.8366 |
| note | kappa | 11.0807 | 11.6191 | 0.1864 | 0.2417 | 0.5384 | 2.2276 |
| note | sigma_lambda_per_day | 0.0144 | 0.0304 | 0.0011 | 0.0035 | 0.0160 | 4.6360 |

## Individual slope comparison

| response_type | students | pearson_slope_correlation | spearman_slope_correlation | median_absolute_slope_difference | median_advi_sd | median_nuts_sd | median_sd_ratio_advi_over_nuts | classification_concordance | advi_identifiable_rate | nuts_identifiable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | 244 | 0.9518 | 0.9723 | 0.0071 | 0.0120 | 0.0217 | 0.5651 | 0.6516 | 0.5738 | 0.2418 |
| chord | 238 | 0.9803 | 0.9836 | 0.0045 | 0.0157 | 0.0236 | 0.7046 | 0.8824 | 0.3025 | 0.1849 |

## Classification counts

| response_type | method | slope_class | students | percentage |
| --- | --- | --- | --- | --- |
| chord | advi | positive | 72 | 0.3025 |
| chord | advi | uncertain | 166 | 0.6975 |
| chord | nuts | positive | 44 | 0.1849 |
| chord | nuts | uncertain | 194 | 0.8151 |
| note | advi | positive | 140 | 0.5738 |
| note | advi | uncertain | 104 | 0.4262 |
| note | nuts | negative | 2 | 0.0082 |
| note | nuts | positive | 57 | 0.2336 |
| note | nuts | uncertain | 185 | 0.7582 |

## NUTS diagnostics

| response_type | divergences | max_rhat | min_ess_bulk | min_ess_tail |
| --- | --- | --- | --- | --- |
| note | 0 | 1.0247 | 173.4310 | 310.6186 |
| chord | 0 | 1.0365 | 151.3991 | 232.9112 |

The objective is uncertainty validation, not target construction. `adjusted_trajectory_slope` remains a content-adjusted platform-performance slope.
