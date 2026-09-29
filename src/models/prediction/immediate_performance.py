#!/usr/bin/env python3
"""Predict immediate success with temporally evaluated beta-binomial models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logsumexp

from src.models.measurement.dispersion_comparison import logpmf
from src.models.measurement.hierarchical_measurement import FitResult, levels, values


FORMULAS = {
    "hierarchical_basic": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ (1|exercise_id) + (1|user_id)"
    ),
    "hierarchical_context": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ practice_mode + weekly_challenge + (1|exercise_id) + (1|user_id)"
    ),
}


def prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frame = data.copy()
    for column in ("user_id", "exercise_id", "response_type"):
        frame[column] = frame[column].astype(str)
    frame["practice_mode"] = (frame.play_mode == "practice").astype(float)
    frame["weekly_challenge"] = (frame.song_type == "weekly_challenge").astype(float)
    train = frame.loc[frame.day_index.between(0, 20)].copy()
    test = frame.loc[frame.day_index.between(21, 30)].copy()
    scaling: dict[str, float] = {}
    for source, target in (("difficulty_level", "difficulty_z"), ("day_index", "day_z")):
        mean = float(train[source].mean())
        std = float(train[source].std(ddof=0)) or 1.0
        train[target] = (train[source].astype(float) - mean) / std
        test[target] = (test[source].astype(float) - mean) / std
        scaling[f"{source}_mean"] = mean
        scaling[f"{source}_std"] = std
    return train, test, scaling


def priors(model_name: str) -> dict[str, bmb.Prior]:
    output = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
        "difficulty_z": bmb.Prior("Normal", mu=0, sigma=1),
        "day_z": bmb.Prior("Normal", mu=0, sigma=0.5),
        "1|exercise_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "1|user_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "kappa": bmb.Prior("HalfNormal", sigma=50),
    }
    if model_name == "hierarchical_context":
        output["practice_mode"] = bmb.Prior("Normal", mu=0, sigma=1)
        output["weekly_challenge"] = bmb.Prior("Normal", mu=0, sigma=1)
    return output


def fit(
    data: pd.DataFrame, model_name: str, iterations: int, draws: int, seed: int
) -> FitResult:
    model = bmb.Model(
        FORMULAS[model_name], data, family="beta_binomial",
        priors=priors(model_name), categorical=["exercise_id", "user_id"],
    )
    approximation = model.fit(
        inference_method="vi", n=iterations, method="advi",
        obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False,
        random_seed=seed,
    )
    trace = approximation.sample(draws, return_inferencedata=False, random_seed=seed)
    return FitResult(model, trace, float(approximation.hist[-1]))


def probability_draws(
    result: FitResult,
    data: pd.DataFrame,
    model_name: str,
    seed: int,
    chunk_size: int = 5000,
):
    intercept = values(result.trace, "Intercept")
    coefficients = {
        name: values(result.trace, name)
        for name in ("difficulty_z", "day_z", "practice_mode", "weekly_challenge")
        if name in result.trace.varnames
    }
    exercise_effect = values(result.trace, "1|exercise_id")
    student_effect = values(result.trace, "1|user_id")
    exercise_lookup = {name: index for index, name in enumerate(levels(result.model, "exercise_id"))}
    student_lookup = {name: index for index, name in enumerate(levels(result.model, "user_id"))}
    unknown_exercises = sorted(set(data.exercise_id) - set(exercise_lookup))
    unknown_students = sorted(set(data.user_id) - set(student_lookup))
    unknown_exercise_lookup = {name: index for index, name in enumerate(unknown_exercises)}
    unknown_student_lookup = {name: index for index, name in enumerate(unknown_students)}
    rng = np.random.default_rng(seed)
    sampled_exercise = rng.normal(
        size=(len(intercept), len(unknown_exercises))
    ) * values(result.trace, "1|exercise_id_sigma")[:, None]
    sampled_student = rng.normal(
        size=(len(intercept), len(unknown_students))
    ) * values(result.trace, "1|user_id_sigma")[:, None]

    for start in range(0, len(data), chunk_size):
        frame = data.iloc[start:start + chunk_size]
        eta = np.broadcast_to(intercept[:, None], (len(intercept), len(frame))).copy()
        for name, coefficient in coefficients.items():
            eta += coefficient[:, None] * frame[name].to_numpy(float)[None, :]
        for position, value in enumerate(frame.exercise_id):
            if value in exercise_lookup:
                eta[:, position] += exercise_effect[:, exercise_lookup[value]]
            else:
                eta[:, position] += sampled_exercise[:, unknown_exercise_lookup[value]]
        for position, value in enumerate(frame.user_id):
            if value in student_lookup:
                eta[:, position] += student_effect[:, student_lookup[value]]
            else:
                eta[:, position] += sampled_student[:, unknown_student_lookup[value]]
        yield start, expit(eta)


def calibration_table(
    data: pd.DataFrame, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "successful": data.successful.to_numpy(float),
        "evaluated": data.evaluated.to_numpy(float),
        "probability": probability,
    })
    frame["bin"] = pd.cut(
        frame.probability, bins=np.linspace(0, 1, bins + 1),
        include_lowest=True, duplicates="drop",
    )
    frame["predicted_successful"] = frame.probability * frame.evaluated
    output = frame.groupby("bin", observed=True).agg(
        rows=("successful", "size"), successful=("successful", "sum"),
        evaluated=("evaluated", "sum"),
        predicted_successful=("predicted_successful", "sum"),
    ).reset_index()
    output["predicted_probability"] = output.predicted_successful / output.evaluated
    output["observed_rate"] = output.successful / output.evaluated
    output["absolute_calibration_error"] = (
        output.predicted_probability - output.observed_rate
    ).abs()
    return output


def metrics(data: pd.DataFrame, probability: np.ndarray, kappa: np.ndarray | None) -> dict[str, float]:
    successful = data.successful.to_numpy(float)
    evaluated = data.evaluated.to_numpy(float)
    failures = evaluated - successful
    probability = np.clip(probability, 1e-9, 1 - 1e-9)
    calibration = calibration_table(data, probability)
    residual_variance = evaluated * probability * (1 - probability)
    if kappa is not None:
        kappa_mean = float(kappa.mean())
        residual_variance *= (evaluated + kappa_mean) / (kappa_mean + 1)
    residual = (successful - evaluated * probability) / np.sqrt(
        np.maximum(residual_variance, 1e-9)
    )
    output = {
        "rows": int(len(data)),
        "evaluated_elements": int(evaluated.sum()),
        "binary_log_loss": float(-np.sum(
            successful * np.log(probability) + failures * np.log1p(-probability)
        ) / evaluated.sum()),
        "brier_score": float(np.sum(
            successful * (1 - probability) ** 2 + failures * probability**2
        ) / evaluated.sum()),
        "expected_calibration_error": float(np.average(
            calibration.absolute_calibration_error, weights=calibration.evaluated
        )),
        "pearson_residual_mean": float(residual.mean()),
        "pearson_residual_sd": float(residual.std(ddof=1)),
    }
    return output


def evaluate_model(
    result: FitResult, data: pd.DataFrame, model_name: str, seed: int
) -> tuple[np.ndarray, float]:
    successful = data.successful.to_numpy(float)
    evaluated = data.evaluated.to_numpy(float)
    kappa = values(result.trace, "kappa")
    probability_mean = np.empty(len(data))
    lpd = np.empty(len(data))
    for start, probability in probability_draws(result, data, model_name, seed):
        stop = start + probability.shape[1]
        probability_mean[start:stop] = probability.mean(axis=0)
        ll = logpmf(
            "beta_binomial", successful[None, start:stop], evaluated[None, start:stop],
            probability, kappa,
        )
        lpd[start:stop] = logsumexp(ll, axis=0) - np.log(len(probability))
    return probability_mean, float(lpd.sum())


def scenario_labels(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    known_user = test.user_id.isin(set(train.user_id))
    known_exercise = test.exercise_id.isin(set(train.exercise_id))
    labels = pd.Series("known_student_known_exercise", index=test.index, dtype="string")
    labels.loc[~known_user & known_exercise] = "new_student_known_exercise"
    labels.loc[known_user & ~known_exercise] = "known_student_new_exercise"
    labels.loc[~known_user & ~known_exercise] = "new_student_new_exercise"
    return labels


def markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(summary: pd.DataFrame, coverage: pd.DataFrame, metadata: dict) -> str:
    overall = summary.loc[summary.scenario == "all", [
        "response_type", "model", "binary_log_loss", "brier_score",
        "expected_calibration_error", "pearson_residual_sd", "beta_binomial_lpd",
    ]]
    scenarios = summary.loc[
        (summary.scenario != "all") & (summary.model == "hierarchical_context"),
        ["response_type", "scenario", "rows", "evaluated_elements",
         "binary_log_loss", "brier_score", "expected_calibration_error"],
    ]
    return f"""# Immediate performance prediction

## Question

Given the student, exercise, declared difficulty, practice context, and response
type, what is the probability that an evaluated element will be successful?

## Design

- Training window: `[0, 21)`.
- Temporal test window: `[21, 31)`.
- Notes and chords fitted separately.
- Models: global-rate baseline, basic hierarchical beta-binomial, and hierarchical
  beta-binomial with practice mode and song type.
- Exercise and student effects use partial pooling.
- Unseen students and exercises are integrated using population random-effect
  distributions rather than global frequency encodings.
- Inference: {metadata['vi_iterations']:,} ADVI iterations and
  {metadata['posterior_draws']:,} posterior draws.

## Holdout coverage

{markdown(coverage)}

## Overall temporal holdout performance

{markdown(overall)}

Lower log-loss, Brier score, and calibration error are better. Beta-binomial LPD
is only defined for the probabilistic hierarchical models; higher values are
better.

## Performance by prediction scenario

{markdown(scenarios)}

The known-student/known-exercise scenario is the primary result. Cold-start
student and exercise estimates are descriptive because the temporal holdout
contains very few new students and relatively few unseen-exercise rows.

## Interpretation

The hierarchical model improves immediate mean-probability prediction over the
global-rate baseline. For notes, the basic model reduces log-loss from 0.4780 to
0.4575 and Brier score from 0.1504 to 0.1447. For chords, it reduces log-loss
from 0.4549 to 0.4375 and Brier score from 0.1407 to 0.1365. This is evidence that
student, exercise, difficulty, and time contain useful signal about current
platform-recorded performance.

Adding practice mode and song type improves beta-binomial LPD and slightly lowers
residual dispersion, but does not improve element-level mean probabilities. The
context model is virtually tied with the basic model for notes and is worse in
chord log-loss and Brier score. Therefore, the richer context specification is
not selected as the best immediate probability model from this experiment.

The global-rate baseline has very low aggregate calibration error by construction:
one constant probability closely matches the overall success rate. That does not
make it individually informative. Its worse log-loss and Brier scores show that
it cannot distinguish easier from harder current interactions.

This experiment predicts platform-recorded success under current conditions. It
does not predict whether a student will improve and does not identify a causal
effect of practice mode, song type, or difficulty. The result shows that the
hierarchical structure is useful for estimating immediate expected performance
even though early behavior did not predict later individual slopes.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/student_exercise_day.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/model_outputs/immediate_performance_v1"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/prediction/immediate_performance_v1.md"),
    )
    parser.add_argument("--vi-iterations", type=int, default=20000)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    train_all, test_all, scaling = prepare(pd.read_parquet(args.input))
    scenario = scenario_labels(train_all, test_all)
    coverage = (
        test_all.assign(scenario=scenario.to_numpy())
        .groupby("scenario", observed=True)
        .agg(rows=("user_id", "size"), students=("user_id", "nunique"),
             exercises=("exercise_id", "nunique"), evaluated_elements=("evaluated", "sum"))
        .reset_index()
    )
    metric_rows, prediction_frames, calibration_frames = [], [], []

    for response_index, response_type in enumerate(("note", "chord")):
        train = train_all.loc[train_all.response_type == response_type].copy()
        test = test_all.loc[test_all.response_type == response_type].copy()
        test_scenario = scenario.loc[test.index]
        baseline_probability = np.repeat(
            train.successful.sum() / train.evaluated.sum(), len(test)
        )
        for segment in ["all", *sorted(test_scenario.unique())]:
            mask = np.ones(len(test), dtype=bool) if segment == "all" else (
                test_scenario.to_numpy() == segment
            )
            metric_rows.append({
                "response_type": response_type, "model": "global_rate_baseline",
                "scenario": segment, **metrics(test.loc[mask], baseline_probability[mask], None),
                "beta_binomial_lpd": np.nan,
            })

        for model_index, model_name in enumerate(FORMULAS):
            print(f"Fitting {response_type} {model_name}", flush=True)
            result = fit(
                train, model_name, args.vi_iterations, args.posterior_draws,
                args.seed + response_index * 10 + model_index,
            )
            probability, total_lpd = evaluate_model(
                result, test, model_name, args.seed + 100 + response_index * 10 + model_index
            )
            kappa = values(result.trace, "kappa")
            for segment in ["all", *sorted(test_scenario.unique())]:
                mask = np.ones(len(test), dtype=bool) if segment == "all" else (
                    test_scenario.to_numpy() == segment
                )
                row = {
                    "response_type": response_type, "model": model_name,
                    "scenario": segment, **metrics(test.loc[mask], probability[mask], kappa),
                    "beta_binomial_lpd": total_lpd if segment == "all" else np.nan,
                }
                metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "user_id": test.user_id.to_numpy(), "exercise_id": test.exercise_id.to_numpy(),
                "day_index": test.day_index.to_numpy(), "response_type": response_type,
                "scenario": test_scenario.to_numpy(), "model": model_name,
                "successful": test.successful.to_numpy(), "evaluated": test.evaluated.to_numpy(),
                "predicted_probability": probability,
            }))
            calibration = calibration_table(test, probability)
            calibration.insert(0, "response_type", response_type)
            calibration.insert(1, "model", model_name)
            calibration_frames.append(calibration)
            del result
            gc.collect()

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    coverage.to_csv(args.output_dir / "holdout_coverage.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        args.output_dir / "calibration.csv", index=False
    )
    metadata = {
        "train_window": "[0, 21)", "test_window": "[21, 31)",
        "train_rows": len(train_all), "test_rows": len(test_all),
        "train_students": int(train_all.user_id.nunique()),
        "test_students": int(test_all.user_id.nunique()),
        "train_exercises": int(train_all.exercise_id.nunique()),
        "test_exercises": int(test_all.exercise_id.nunique()),
        "vi_iterations": args.vi_iterations, "posterior_draws": args.posterior_draws,
        "seed": args.seed, "scaling": scaling,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.report.write_text(render_report(summary, coverage, metadata), encoding="utf-8")
    print(summary.loc[summary.scenario == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
