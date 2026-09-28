# Problem definition

## Research question

Can behavioral signals recorded during a student's first seven elapsed days predict future observability and, among observable students, the subsequent trajectory of platform-recorded performance?

This project measures behavior and performance inside the platform. It does not
claim to measure general musical learning outside the observed activities.

## Units of analysis

- **Event:** one recorded interaction between a student and the platform.
- **Student:** the independent unit used for prediction and model evaluation.
- **Trajectory:** a longitudinal outcome estimated after the initial period.

The raw event count is not the predictive sample size. Model evaluation must
keep every event from one student in the same data partition.

## Time windows

`days_since_signup` is continuous, so the windows use half-open intervals:

- Initial features: `[0, 7)` elapsed days.
- Future outcomes: `[7, 31)` elapsed days.

This represents exactly seven elapsed days of initial observation and uses all
remaining activity in the dataset for the future outcome. Alternative windows,
including `[15, 31)`, belong in sensitivity analysis rather than the primary
definition.

## Outcomes

1. **Future observability:** whether enough future evidence exists to estimate
   a longitudinal trajectory. The required number of active days and evaluated
   elements are explicit pipeline parameters.
2. **Future adjusted trajectory:** a continuous, partially pooled slope of
   performance during the future window, adjusted for content difficulty and
   context. This outcome will be implemented after the data layer and baseline.

Positive/stable/negative trajectory labels are operational translations of the
continuous slope, not the primary statistical target.

## Data layers

1. `events_clean.parquet`: validated event-grain data derived from the raw JSON.
2. `student_coverage.parquet`: one row for every raw student, including explicit
   observability status and exclusion reason.
3. `student_prediction.parquet`: first-week student features joined to outcomes;
   implemented after outcome definitions are validated.

The legacy CSV is retained only as an exploratory baseline and is not the source
of truth for the redesigned pipeline.
