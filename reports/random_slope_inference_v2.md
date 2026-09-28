# Random-slope ADVI versus NUTS validation

## Design

- Deterministic user subsample: 25% (247 users).
- ADVI: 20,000 iterations, 1,000 draws.
- NUTS: 4 chains, 1,500 tune and 1,500 retained draws per chain.
- Both methods use identical beta-binomial random-slope specifications and data scaling.

## Population parameter comparison

| response_type | parameter | advi_estimate | nuts_estimate | advi_sd | nuts_sd | absolute_difference | difference_in_nuts_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chord | 1|exercise_id_sigma | 0.7262 | 0.7681 | 0.0124 | 0.0461 | 0.0419 | 0.9094 |
| chord | 1|user_id_sigma | 0.8408 | 0.8727 | 0.0087 | 0.0445 | 0.0320 | 0.7182 |
| chord | Intercept | 1.2128 | 1.2013 | 0.0083 | 0.0830 | 0.0115 | 0.1390 |
| chord | difficulty_z | -0.7065 | -0.6983 | 0.0111 | 0.0477 | 0.0083 | 0.1736 |
| chord | global_slope_per_day | 0.0251 | 0.0239 | 0.0012 | 0.0033 | 0.0012 | 0.3630 |
| chord | kappa | 10.2220 | 10.6074 | 0.1954 | 0.2418 | 0.3854 | 1.5942 |
| chord | sigma_lambda_per_day | 0.0209 | 0.0318 | 0.0013 | 0.0033 | 0.0109 | 3.3262 |
| note | 1|exercise_id_sigma | 0.6360 | 0.6651 | 0.0127 | 0.0374 | 0.0291 | 0.7777 |
| note | 1|user_id_sigma | 0.7142 | 0.7637 | 0.0094 | 0.0391 | 0.0495 | 1.2658 |
| note | Intercept | 1.1835 | 1.1541 | 0.0077 | 0.0701 | 0.0293 | 0.4185 |
| note | difficulty_z | -0.7219 | -0.7245 | 0.0089 | 0.0350 | 0.0027 | 0.0766 |
| note | global_slope_per_day | 0.0243 | 0.0219 | 0.0011 | 0.0030 | 0.0024 | 0.8140 |
| note | kappa | 11.0807 | 11.6256 | 0.1864 | 0.2459 | 0.5449 | 2.2155 |
| note | sigma_lambda_per_day | 0.0144 | 0.0305 | 0.0011 | 0.0034 | 0.0160 | 4.7272 |

## Individual slope comparison

| response_type | students | pearson_slope_correlation | spearman_slope_correlation | median_absolute_slope_difference | median_advi_sd | median_nuts_sd | median_sd_ratio_advi_over_nuts | classification_concordance | advi_identifiable_rate | nuts_identifiable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| note | 244 | 0.9524 | 0.9733 | 0.0070 | 0.0120 | 0.0218 | 0.5738 | 0.6434 | 0.5738 | 0.2336 |
| chord | 238 | 0.9810 | 0.9840 | 0.0047 | 0.0157 | 0.0235 | 0.7057 | 0.8697 | 0.3025 | 0.1807 |

## Classification counts

| response_type | method | slope_class | students | percentage |
| --- | --- | --- | --- | --- |
| chord | advi | positive | 72 | 0.3025 |
| chord | advi | uncertain | 166 | 0.6975 |
| chord | nuts | negative | 1 | 0.0042 |
| chord | nuts | positive | 42 | 0.1765 |
| chord | nuts | uncertain | 195 | 0.8193 |
| note | advi | positive | 140 | 0.5738 |
| note | advi | uncertain | 104 | 0.4262 |
| note | nuts | negative | 2 | 0.0082 |
| note | nuts | positive | 55 | 0.2254 |
| note | nuts | uncertain | 187 | 0.7664 |

## NUTS diagnostics

| response_type | divergences | max_rhat | min_ess_bulk | min_ess_tail |
| --- | --- | --- | --- | --- |
| note | 0 | 1.0137 | 329.4466 | 450.8063 |
| chord | 0 | 1.0277 | 198.9287 | 427.9032 |

## Findings

- Note: ADVI/NUTS slope Spearman is 0.973, but median ADVI uncertainty is only 57.4% of NUTS. The identifiable rate changes from 57.4% to 23.4%.
- Chord: ADVI/NUTS slope Spearman is 0.984, but median ADVI uncertainty is only 70.6% of NUTS. The identifiable rate changes from 30.3% to 18.1%.

ADVI is adequate for approximate ranking and point estimates, but its slope uncertainty is not calibrated. Downstream two-stage regression or classification must not treat the ADVI posterior standard deviation as ground truth. NUTS has no divergences, although remaining R-hat values above 1.01 mean this is a strong sensitivity validation rather than a final full-population posterior.

The objective is uncertainty validation, not target construction. `adjusted_trajectory_slope` remains a content-adjusted platform-performance slope.
