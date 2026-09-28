#!/usr/bin/env python3
"""Build leakage-safe first-week features joined to uncertain future slopes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


UNCERTAINTY_INFLATION = {"note": 1 / 0.573788, "chord": 1 / 0.705704}


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    initial = events.loc[events.days_since_signup.between(0, 7, inclusive="left")].copy()
    initial["practice_mode"] = (initial.play_mode == "practice").astype(float)
    grouped = initial.groupby("user_id", observed=True)
    features = grouped.agg(
        initial_events=("source_row_number", "size"),
        initial_active_days=("day_index", "nunique"),
        initial_sessions=("session_index", "nunique"),
        initial_time_playing=("time_playing", "sum"),
        initial_songs=("song_id", "nunique"),
        initial_exercises=("exercise_id", "nunique"),
        initial_notes_successful=("notes_successful", "sum"),
        initial_notes_evaluated=("notes_evaluated", "sum"),
        initial_chords_successful=("chords_successful", "sum"),
        initial_chords_evaluated=("chords_evaluated", "sum"),
        initial_difficulty_mean=("difficulty_level", "mean"),
        initial_difficulty_sd=("difficulty_level", "std"),
        initial_completion_rate=("completed", "mean"),
        initial_abandonment_rate=("abandoned", "mean"),
        initial_practice_mode_share=("practice_mode", "mean"),
    ).reset_index()
    features["initial_note_accuracy"] = (
        features.initial_notes_successful / features.initial_notes_evaluated.replace(0, np.nan)
    )
    features["initial_chord_accuracy"] = (
        features.initial_chords_successful / features.initial_chords_evaluated.replace(0, np.nan)
    )
    features["initial_total_evaluated"] = (
        features.initial_notes_evaluated + features.initial_chords_evaluated
    )
    for column in [
        "initial_events", "initial_sessions", "initial_time_playing",
        "initial_total_evaluated", "initial_notes_evaluated", "initial_chords_evaluated",
    ]:
        features[f"log1p_{column}"] = np.log1p(features[column])
    return features


def join_targets(features: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    targets = slopes.loc[slopes.window == "7_31"].copy()
    targets["uncertainty_inflation_factor"] = targets.response_type.map(UNCERTAINTY_INFLATION)
    targets["calibrated_slope_sd"] = (
        targets.slope_sd * targets.uncertainty_inflation_factor
    )
    for delta in (0.0, 0.01, 0.02):
        suffix = str(delta).replace(".", "p")
        targets[f"probability_slope_gt_{suffix}"] = ndtr(
            (targets.adjusted_trajectory_slope - delta)
            / targets.calibrated_slope_sd.clip(lower=1e-9)
        )
    targets["target_uncertainty_source"] = "advi_sd_inflated_by_subsample_nuts_ratio"
    return features.merge(targets, on="user_id", how="inner", validate="one_to_many")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument("--slopes", type=Path, default=Path("data/model_outputs/random_slope_v1/student_adjusted_trajectory_slopes.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/student_prediction.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = join_targets(build_features(pd.read_parquet(args.events)), pd.read_parquet(args.slopes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    print(f"Wrote {len(dataset):,} student-response rows to {args.output}")


if __name__ == "__main__":
    main()
