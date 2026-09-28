# Conditional calibration of the long-ADVI beta-binomial model

## Scope

This diagnostic uses the same 198 held-out students as the dispersion
comparison. Calibration error is defined as predicted probability minus the
observed evaluated-element rate, so negative values mean underprediction.

## By evidence volume

| Response | Q1 low | Q2 | Q3 | Q4 high |
|---|---:|---:|---:|---:|
| Notes | -0.017 | -0.078 | -0.100 | -0.117 |
| Chords | -0.010 | -0.058 | -0.094 | -0.076 |

The model is closest to calibrated for low-volume students and increasingly
underpredicts higher-volume students. This pattern matters directly for target
construction because uncertainty alone does not explain the systematic mean
bias.

## By declared difficulty

Calibration deteriorates across the well-populated middle and upper difficulty
levels. The largest supported gaps are:

- Notes, difficulty 7: predicted 0.517 versus observed 0.724, error -0.207
  over 90,070 evaluated elements.
- Chords, difficulty 8: predicted 0.510 versus observed 0.682, error -0.172
  over 9,559 evaluated elements.
- Chords, difficulty 6: predicted 0.653 versus observed 0.804, error -0.151
  over 210,346 evaluated elements.

The apparent reversals at difficulty 9–10 have very small support and should
not be treated as stable calibration evidence.

## By day

The largest absolute daily errors are:

- Notes, day 19: -0.132 over 186,638 evaluated elements.
- Chords, day 16: -0.123 over 102,202 evaluated elements.

The global linear time term is therefore not sufficient to calibrate every day.
A nonlinear global time effect should be considered before interpreting a
student-specific random slope.

## Decision

The beta-binomial observation family is strongly supported for dispersion, but
this Model 4 mean structure is not ready to define the final trajectory target.
Before freezing it, test at least a nonlinear global time effect and inspect
whether the difficulty relationship needs nonlinear or varying effects.

Detailed segment values are stored in
`data/model_outputs/dispersion_advi20k/conditional_calibration.csv`.
