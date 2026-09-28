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
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.build_prediction_dataset import build_features
from src.models.prediction.two_stage_prediction import FEATURES


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "roc_auc_not_observable": float(roc_auc_score(y, probability)),
        "average_precision_not_observable": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]))),
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(y, prediction)),
        "precision_not_observable_at_0p5": float(precision_score(y, prediction, zero_division=0)),
        "recall_not_observable_at_0p5": float(recall_score(y, prediction, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--coverage", type=Path, default=Path("data/interim/student_coverage.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/observability_v1"))
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


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
    rows, predictions = [], []
    for name, model in models.items():
        probability = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        rows.append({"model": name, **evaluate(y, probability)})
        predictions.append(pd.DataFrame({
            "user_id": cohort.user_id, "future_not_observable": y,
            "model": name, "predicted_probability_not_observable": probability,
        }))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "out_of_fold_predictions.parquet", index=False
    )
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "cohort": len(cohort), "observable": int(cohort.future_observable.sum()),
        "not_observable": int((~cohort.future_observable).sum()),
        "initial_window": "[0, 7)", "outcome_window": "[7, 31)",
        "future_observable_definition": ">=3 evaluated days and >=100 evaluated elements",
        "features": FEATURES, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
