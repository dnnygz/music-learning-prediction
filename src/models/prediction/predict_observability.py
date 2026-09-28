#!/usr/bin/env python3
"""Predict future non-observability from leakage-safe first-week features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.build_prediction_dataset import build_features
from src.evaluation.classification import (
    quantile_calibration,
    stratified_bootstrap_interval,
)
from src.models.prediction.two_stage_prediction import FEATURES


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()
    return {
        "roc_auc_not_observable": float(roc_auc_score(y, probability)),
        "average_precision_not_observable": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]))),
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(y, prediction)),
        "precision_not_observable_at_0p5": float(precision_score(y, prediction, zero_division=0)),
        "recall_not_observable_at_0p5": float(recall_score(y, prediction, zero_division=0)),
        "true_negative_at_0p5": int(true_negative),
        "false_positive_at_0p5": int(false_positive),
        "false_negative_at_0p5": int(false_negative),
        "true_positive_at_0p5": int(true_positive),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--coverage", type=Path, default=Path("data/interim/student_coverage.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/observability_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/prediction/observability_v1.md"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def render_report(metrics: pd.DataFrame, metadata: dict[str, object]) -> str:
    indexed = metrics.set_index("model")
    baseline = indexed.loc["prevalence_baseline"]
    logistic = indexed.loc["logistic_regression"]
    forest = indexed.loc["random_forest"]
    lift = logistic.average_precision_not_observable / baseline.average_precision_not_observable
    return f"""# First-week behavior predicts future observability, but probabilities require calibration

## Conclusion

Behavioral signals recorded during `[0, 7)` contain useful out-of-sample information
about whether a student will have enough evaluated activity in `[7, 31)` to estimate
a subsequent performance trajectory. The strongest ranking model reached ROC-AUC
{logistic.roc_auc_not_observable:.3f} and average precision
{logistic.average_precision_not_observable:.3f}, compared with a non-observability
prevalence of {baseline.average_precision_not_observable:.3f} ({lift:.1f}x lift).

This supports predictability of **future observability**, not a causal claim about
retention and not a claim that the model is ready for operational deployment.

## Operational outcome

`FutureObservable = 1` when a student has at least three future days containing
evaluated performance and at least 100 evaluated elements during `[7, 31)`.
It means that the platform collected enough longitudinal evidence to estimate
performance dynamics. It is an operational data-coverage definition, not an
intrinsic student attribute.

- Initially eligible students: {metadata['cohort']}.
- Future observable: {metadata['observable']}.
- Future not observable: {metadata['not_observable']}.
- Non-observability prevalence: {metadata['not_observable'] / metadata['cohort']:.1%}.

## Out-of-fold evaluation

Predictions use five-fold stratified cross-validation. All features come from
`[0, 7)` and the outcome comes from `[7, 31)`. No future activity is included in
the predictors. The table treats future non-observability as the positive class
because it is the rare operational case.

| Model | ROC-AUC | 95% bootstrap CI | Average precision | 95% bootstrap CI | Brier | Log-loss | Precision @ 0.5 | Recall @ 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prevalence baseline | {baseline.roc_auc_not_observable:.3f} | [{baseline.roc_auc_ci_low:.3f}, {baseline.roc_auc_ci_high:.3f}] | {baseline.average_precision_not_observable:.3f} | [{baseline.average_precision_ci_low:.3f}, {baseline.average_precision_ci_high:.3f}] | {baseline.brier_score:.3f} | {baseline.log_loss:.3f} | {baseline.precision_not_observable_at_0p5:.3f} | {baseline.recall_not_observable_at_0p5:.3f} |
| Balanced logistic regression | {logistic.roc_auc_not_observable:.3f} | [{logistic.roc_auc_ci_low:.3f}, {logistic.roc_auc_ci_high:.3f}] | {logistic.average_precision_not_observable:.3f} | [{logistic.average_precision_ci_low:.3f}, {logistic.average_precision_ci_high:.3f}] | {logistic.brier_score:.3f} | {logistic.log_loss:.3f} | {logistic.precision_not_observable_at_0p5:.3f} | {logistic.recall_not_observable_at_0p5:.3f} |
| Balanced random forest | {forest.roc_auc_not_observable:.3f} | [{forest.roc_auc_ci_low:.3f}, {forest.roc_auc_ci_high:.3f}] | {forest.average_precision_not_observable:.3f} | [{forest.average_precision_ci_low:.3f}, {forest.average_precision_ci_high:.3f}] | {forest.brier_score:.3f} | {forest.log_loss:.3f} | {forest.precision_not_observable_at_0p5:.3f} | {forest.recall_not_observable_at_0p5:.3f} |

The balanced logistic model provides the best discrimination and recovers most
non-observable students at the default threshold, but class weighting makes its
probabilities overconfident: its Brier score and log-loss are worse than the
prevalence baseline. The random forest is less aggressive and better calibrated,
but misses more non-observable students. Threshold selection and probability
calibration must therefore depend on an explicit operational cost before use.

At the provisional 0.5 threshold, logistic regression identifies
{int(logistic.true_positive_at_0p5)} of {metadata['not_observable']} non-observable
students, with {int(logistic.false_positive_at_0p5)} false positives. The random
forest identifies {int(forest.true_positive_at_0p5)}, with
{int(forest.false_positive_at_0p5)} false positives. These counts illustrate why
the threshold cannot be selected without defining the cost of missed and
unnecessary follow-up.

## Limitations

- Only {metadata['not_observable']} students belong to the minority class, so
  uncertainty remains material despite the observed ranking signal.
- The observability threshold is operational and should be included in later
  sensitivity analysis.
- Stratified cross-validation evaluates generalization within this 945-student
  cohort; it does not establish temporal or external generalization.
- Observability is not identical to learning, persistence, or platform retention.

## Answer to the observability sub-question

Yes, under the current operational definition and dataset, first-week behavioral
signals predict future observability better than a prevalence-only baseline.
The supported claim concerns ranking/discrimination. Calibrated risk estimation
and an operational decision threshold remain future work.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = build_features(pd.read_parquet(args.events))
    coverage = pd.read_parquet(args.coverage)
    cohort = coverage.loc[coverage.initial_eligible, [
        "user_id", "future_observable", "future_active", "reason_not_observable"
    ]].merge(features, on="user_id", how="inner", validate="one_to_one")
    y = (~cohort.future_observable).astype(int).to_numpy()
    X = cohort[FEATURES]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    models = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=600, min_samples_leaf=8, max_features=0.7,
                class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
            ),
        ),
    }
    rows, predictions, calibration = [], [], []
    for model_index, (name, model) in enumerate(models.items()):
        probability = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        model_metrics = evaluate(y, probability)
        auc_interval = stratified_bootstrap_interval(
            y, probability, roc_auc_score,
            draws=args.bootstrap_draws, seed=args.seed + model_index,
        )
        ap_interval = stratified_bootstrap_interval(
            y, probability, average_precision_score,
            draws=args.bootstrap_draws, seed=args.seed + 100 + model_index,
        )
        rows.append({
            "model": name, **model_metrics,
            "roc_auc_ci_low": auc_interval[0], "roc_auc_ci_high": auc_interval[1],
            "average_precision_ci_low": ap_interval[0],
            "average_precision_ci_high": ap_interval[1],
        })
        predictions.append(pd.DataFrame({
            "user_id": cohort.user_id, "future_not_observable": y,
            "model": name, "predicted_probability_not_observable": probability,
        }))
        model_calibration = quantile_calibration(y, probability)
        model_calibration.insert(0, "model", name)
        calibration.append(model_calibration)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(
        args.output_dir / "calibration.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "out_of_fold_predictions.parquet", index=False
    )
    metadata = {
        "cohort": len(cohort), "observable": int(cohort.future_observable.sum()),
        "not_observable": int((~cohort.future_observable).sum()),
        "initial_window": "[0, 7)", "outcome_window": "[7, 31)",
        "future_observable_definition": ">=3 evaluated days and >=100 evaluated elements",
        "features": FEATURES, "seed": args.seed,
        "cross_validation": "5-fold stratified out-of-fold",
        "bootstrap_draws": args.bootstrap_draws,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(metrics, metadata), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
