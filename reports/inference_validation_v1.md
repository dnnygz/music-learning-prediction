# Beta-binomial inference validation

## Design

- Deterministic user subsample: 25%.
- ADVI: 20,000 iterations and 1,000 posterior draws.
- NUTS: 2 chains, 500 tune and 500 retained draws per chain.
- Both methods use the same rows, scaling, likelihood, priors, and formula.

## Parameter comparison

| response_type | parameter | advi_estimate | nuts_estimate | absolute_difference | difference_in_nuts_sd |
| --- | --- | --- | --- | --- | --- |
| chord | 1|exercise_id_sigma | 0.7102 | 0.7435 | 0.0333 | 0.7203 |
| chord | 1|user_id_sigma | 0.8200 | 0.8525 | 0.0325 | 0.7918 |
| chord | Intercept | 1.2200 | 1.1958 | 0.0242 | 0.3069 |
| chord | day_z | 0.1681 | 0.1807 | 0.0126 | 0.9802 |
| chord | difficulty_z | -0.6947 | -0.6791 | 0.0156 | 0.3619 |
| chord | kappa | 9.9611 | 10.0125 | 0.0514 | 0.2284 |
| note | 1|exercise_id_sigma | 0.6346 | 0.6538 | 0.0192 | 0.5332 |
| note | 1|user_id_sigma | 0.7179 | 0.7377 | 0.0199 | 0.5511 |
| note | Intercept | 1.1852 | 1.1857 | 0.0005 | 0.0079 |
| note | day_z | 0.1514 | 0.1576 | 0.0062 | 0.5494 |
| note | difficulty_z | -0.7053 | -0.7195 | 0.0142 | 0.4132 |
| note | kappa | 11.1036 | 11.0808 | 0.0228 | 0.0965 |

## NUTS diagnostics

- Note: 0 divergences; maximum R-hat 1.028; minimum bulk ESS 89.
- Chord: 0 divergences; maximum R-hat 1.021; minimum bulk ESS 61.

Agreement is evaluated on the global coefficients, hierarchical standard deviations, and beta-binomial concentration. Group-level effects themselves are not compared one by one in this screening run.
