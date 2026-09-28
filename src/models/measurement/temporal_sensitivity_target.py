#!/usr/bin/env python3
"""Estimate a common [21, 31) adjusted trajectory target for sensitivity tests."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from src.models.measurement.hierarchical_measurement import RANDOM_SEED, prepare_data, standardize
from src.models.measurement.random_slope_measurement import fit_model, extract_student_slopes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/student_exercise_day.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/model_outputs/temporal_sensitivity_target_v1"),
    )
    parser.add_argument("--vi-iterations", type=int, default=20000)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_parquet(args.input)
    future = prepare_data(all_data, 21, 31)
    slope_frames, population_frames = [], []
    for response_index, response_type in enumerate(("note", "chord")):
        response = future.loc[future.response_type == response_type].copy()
        response, _, scaling = standardize(response, response.iloc[:0].copy())
        print(f"Fitting common target: {response_type} [21, 31)", flush=True)
        result = fit_model(
            response, "random_slope", args.vi_iterations, args.posterior_draws,
            args.seed + response_index,
        )
        slopes, population = extract_student_slopes(
            result, response_type, "21_31", scaling["day_index_std"]
        )
        slope_frames.append(slopes)
        population_frames.append(population)
        del result
        gc.collect()

    slopes = pd.concat(slope_frames, ignore_index=True)
    population = pd.concat(population_frames, ignore_index=True)
    slopes.to_parquet(args.output_dir / "student_adjusted_trajectory_slopes.parquet", index=False)
    population.to_csv(args.output_dir / "population_slope_variation.csv", index=False)
    metadata = {
        "target_window": "[21, 31)",
        "vi_iterations": args.vi_iterations,
        "posterior_draws": args.posterior_draws,
        "seed": args.seed,
        "slope_unit": "log_odds_per_day",
        "response_types": ["note", "chord"],
        "inference": "mean-field ADVI",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote common trajectory target to {args.output_dir}")


if __name__ == "__main__":
    main()
