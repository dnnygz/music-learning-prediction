#!/usr/bin/env python3
"""Evaluate probabilistic and continuous two-stage trajectory prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "initial_active_days", "log1p_initial_events", "log1p_initial_sessions",
    "log1p_initial_time_playing", "log1p_initial_total_evaluated",
    "initial_note_accuracy", "initial_chord_accuracy", "initial_difficulty_mean",
    "initial_difficulty_sd", "initial_completion_rate", "initial_abandonment_rate",
    "initial_practice_mode_share", "initial_songs", "initial_exercises",
]


def ridge() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def random_forest(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.7,
            random_state=seed, n_jobs=-1,
        ),
    )


def soft_metrics(q: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    return {
        "expected_log_loss": float(np.mean(-q * np.log(prediction) - (1 - q) * np.log(1 - prediction))),
        "expected_brier": float(np.mean(q * (1 - prediction) ** 2 + (1 - q) * prediction**2)),
        "mae_probability": float(mean_absolute_error(q, prediction)),
        "correlation_probability": float(np.corrcoef(q, prediction)[0, 1]),
    }


def continuous_metrics(mean: np.ndarray, sd: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    squared = (prediction - mean) ** 2
    return {
        "rmse_posterior_mean": float(np.sqrt(np.mean(squared))),
        "mae_posterior_mean": float(mean_absolute_error(mean, prediction)),
        "r2_posterior_mean": float(r2_score(mean, prediction)),
        "correlation_posterior_mean": float(np.corrcoef(mean, prediction)[0, 1]),
        "expected_rmse_including_target_uncertainty": float(np.sqrt(np.mean(squared + sd**2))),
    }


def run(data: pd.DataFrame, folds: int, imputations: int, seed: int):
    all_predictions, metric_rows = [], []
    users = np.array(sorted(data.user_id.unique()))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_lookup = {}
    for fold, (_, test_idx) in enumerate(splitter.split(users)):
        fold_lookup.update({user: fold for user in users[test_idx]})
    data = data.copy()
    data["fold"] = data.user_id.map(fold_lookup)

    for response_type, response in data.groupby("response_type", observed=True):
        X = response[FEATURES]
        slope = response.adjusted_trajectory_slope.to_numpy()
        slope_sd = response.calibrated_slope_sd.to_numpy()
        probability = response.probability_slope_gt_0p02.to_numpy()
        for model_name in ("ridge", "random_forest"):
            pred_a = np.empty(len(response))
            model_imputations = imputations if model_name == "ridge" else min(10, imputations)
            pred_b_draws = np.empty((model_imputations, len(response)))
            rng = np.random.default_rng(seed + (0 if model_name == "ridge" else 1000))
            target_draws = rng.normal(
                slope[None, :], slope_sd[None, :], size=(model_imputations, len(response))
            )
            for fold in range(folds):
                train = response.fold.to_numpy() != fold
                test = ~train
                model_a = ridge() if model_name == "ridge" else random_forest(seed + fold)
                model_a.fit(X.loc[train], probability[train])
                pred_a[test] = np.clip(model_a.predict(X.loc[test]), 0, 1)
                for draw in range(model_imputations):
                    model_b = ridge() if model_name == "ridge" else random_forest(seed + fold + draw)
                    model_b.fit(X.loc[train], target_draws[draw, train])
                    pred_b_draws[draw, test] = model_b.predict(X.loc[test])
            pred_b = pred_b_draws.mean(axis=0)
            predictive_sd = pred_b_draws.std(axis=0, ddof=1)
            metric_rows.append({
                "formulation": "A_probability", "response_type": response_type,
                "model": model_name, **soft_metrics(probability, pred_a),
            })
            metric_rows.append({
                "formulation": "B_continuous_multiple_imputation",
                "response_type": response_type, "model": model_name,
                **continuous_metrics(slope, slope_sd, pred_b),
            })
            all_predictions.append(pd.DataFrame({
                "user_id": response.user_id.to_numpy(), "response_type": response_type,
                "fold": response.fold.to_numpy(), "model": model_name,
                "target_probability": probability, "predicted_probability": pred_a,
                "target_slope_mean": slope, "target_slope_sd": slope_sd,
                "predicted_slope": pred_b, "prediction_between_imputation_sd": predictive_sd,
            }))
    return pd.DataFrame(metric_rows), pd.concat(all_predictions, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_prediction.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/two_stage_v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--imputations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = run(pd.read_parquet(args.input), args.folds, args.imputations, args.seed)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_parquet(args.output_dir / "out_of_fold_predictions.parquet", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "folds": args.folds, "imputations": args.imputations, "seed": args.seed,
        "features": FEATURES, "delta": 0.02,
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
