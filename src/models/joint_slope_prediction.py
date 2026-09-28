#!/usr/bin/env python3
"""Evaluate early features directly inside the beta-binomial slope model."""

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

from src.models.dispersion_comparison import logpmf
from src.models.hierarchical_measurement import RANDOM_SEED, FitResult, prepare_data, split_users, values
from src.models.random_slope_measurement import model_priors


EARLY_FEATURES = [
    "initial_active_days", "log1p_initial_time_playing",
    "log1p_initial_total_evaluated", "initial_note_accuracy",
    "initial_chord_accuracy", "initial_difficulty_mean",
    "initial_completion_rate", "initial_practice_mode_share",
]


def prepare_joined(events: pd.DataFrame, students: pd.DataFrame, train_users: set[str]):
    feature_table = students.drop_duplicates("user_id")[["user_id", *EARLY_FEATURES]].copy()
    scaling = {}
    train_mask = feature_table.user_id.isin(train_users)
    for feature in EARLY_FEATURES:
        median = float(feature_table.loc[train_mask, feature].median())
        raw = feature_table[feature].fillna(median).astype(float)
        mean = float(raw[train_mask].mean())
        std = float(raw[train_mask].std(ddof=0)) or 1.0
        feature_table[f"z_{feature}"] = (raw - mean) / std
        scaling[feature] = {"median": median, "mean": mean, "std": std}
    joined = events.merge(feature_table, on="user_id", how="inner", validate="many_to_one")
    for source, target in (("difficulty_level", "difficulty_z"), ("day_index", "day_z")):
        train_values = joined.loc[joined.user_id.isin(train_users), source].astype(float)
        mean, std = float(train_values.mean()), float(train_values.std(ddof=0))
        joined[target] = (joined[source].astype(float) - mean) / (std or 1.0)
        scaling[source] = {"mean": mean, "std": std or 1.0}
    return joined, scaling


def formula(joint: bool) -> str:
    rhs = "1 + difficulty_z + day_z + (1|exercise_id) + (1 + day_z|user_id)"
    if joint:
        rhs += " + " + " + ".join(f"day_z:z_{feature}" for feature in EARLY_FEATURES)
    return f"proportion(successful, evaluated) ~ {rhs}"


def fit(data: pd.DataFrame, joint: bool, iterations: int, draws: int, seed: int) -> FitResult:
    priors = model_priors("random_slope")
    if joint:
        for feature in EARLY_FEATURES:
            priors[f"day_z:z_{feature}"] = bmb.Prior("Normal", mu=0, sigma=0.25)
    model = bmb.Model(
        formula(joint), data, family="beta_binomial", priors=priors,
        categorical=["exercise_id", "user_id"],
    )
    approximation = model.fit(
        inference_method="vi", n=iterations, method="advi",
        obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False, random_seed=seed,
    )
    trace = approximation.sample(draws, return_inferencedata=False, random_seed=seed)
    return FitResult(model, trace, float(approximation.hist[-1]))


def probability_draws(result: FitResult, data: pd.DataFrame, joint: bool, seed: int, chunk=5000):
    trace = result.trace
    intercept, difficulty, day = [values(trace, name) for name in ("Intercept", "difficulty_z", "day_z")]
    interactions = {
        feature: values(trace, f"day_z:z_{feature}") for feature in EARLY_FEATURES
    } if joint else {}
    exercise = values(trace, "1|exercise_id")
    levels = [str(x) for x in result.model.backend.model.coords["exercise_id__factor_dim"]]
    exercise_lookup = {name: index for index, name in enumerate(levels)}
    users = sorted(data.user_id.unique())
    user_lookup = {name: index for index, name in enumerate(users)}
    rng = np.random.default_rng(seed)
    user_intercept = rng.normal(size=(len(intercept), len(users))) * values(trace, "1|user_id_sigma")[:, None]
    user_slope = rng.normal(size=(len(intercept), len(users))) * values(trace, "day_z|user_id_sigma")[:, None]
    for start in range(0, len(data), chunk):
        frame = data.iloc[start:start + chunk]
        day_z = frame.day_z.to_numpy()
        eta = intercept[:, None] + difficulty[:, None] * frame.difficulty_z.to_numpy()[None, :] + day[:, None] * day_z
        for feature, coefficient in interactions.items():
            eta += coefficient[:, None] * day_z[None, :] * frame[f"z_{feature}"].to_numpy()[None, :]
        ex_idx = np.array([exercise_lookup.get(x, -1) for x in frame.exercise_id])
        known = ex_idx >= 0
        if known.any(): eta[:, known] += exercise[:, ex_idx[known]]
        user_idx = np.array([user_lookup[x] for x in frame.user_id])
        eta += user_intercept[:, user_idx] + user_slope[:, user_idx] * day_z[None, :]
        yield start, expit(eta)


def evaluate(result: FitResult, data: pd.DataFrame, joint: bool, seed: int):
    y, n = data.successful.to_numpy(float), data.evaluated.to_numpy(float)
    kappa = values(result.trace, "kappa")
    users = sorted(data.user_id.unique()); lookup = {u:i for i,u in enumerate(users)}
    codes = data.user_id.map(lookup).to_numpy(); user_ll=None; p_mean=np.empty(len(data)); var=np.empty(len(data))
    for start,p in probability_draws(result,data,joint,seed):
        stop=start+p.shape[1]; nn=n[None,start:stop]; yy=y[None,start:stop]
        ll=logpmf("beta_binomial",yy,nn,p,kappa)
        if user_ll is None: user_ll=np.zeros((len(p),len(users)))
        for d in range(len(p)): np.add.at(user_ll[d],codes[start:stop],ll[d])
        cm=nn*p; cv=nn*p*(1-p)*(nn+kappa[:,None])/(kappa[:,None]+1)
        p_mean[start:stop]=p.mean(0); var[start:stop]=cv.mean(0)+cm.var(0,ddof=1)
    failures=n-y; residual=(y-n*p_mean)/np.sqrt(np.maximum(var,1e-9))
    return {
        "joint_user_lpd":float(np.sum(logsumexp(user_ll,axis=0)-np.log(len(user_ll)))),
        "binary_log_loss":float(-np.sum(y*np.log(np.clip(p_mean,1e-9,1))+failures*np.log(np.clip(1-p_mean,1e-9,1)))/n.sum()),
        "brier_score":float(np.sum(y*(1-p_mean)**2+failures*p_mean**2)/n.sum()),
        "pearson_residual_sd":float(residual.std(ddof=1)),
    }


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events",type=Path,default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--students",type=Path,default=Path("data/processed/student_prediction.parquet"))
    parser.add_argument("--output-dir",type=Path,default=Path("data/model_outputs/joint_model_v1"))
    parser.add_argument("--vi-iterations",type=int,default=20000)
    parser.add_argument("--posterior-draws",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    events=prepare_data(pd.read_parquet(args.events),7,31)
    train_users,test_users=split_users(events,0.2,args.seed)
    joined,scaling=prepare_joined(events,pd.read_parquet(args.students),train_users)
    rows=[]; coefficients=[]
    for ri,response_type in enumerate(("note","chord")):
        response=joined[joined.response_type==response_type]
        train=response[response.user_id.isin(train_users)]; test=response[response.user_id.isin(test_users)]
        baseline=None
        for mi,(name,joint) in enumerate((('random_slope_baseline',False),('joint_early_features',True))):
            print(f"Fitting {response_type} {name}",flush=True)
            result=fit(train,joint,args.vi_iterations,args.posterior_draws,args.seed+ri*10+mi)
            metrics=evaluate(result,test,joint,args.seed+100+ri*10+mi)
            if baseline is None: baseline=metrics['joint_user_lpd']
            rows.append({'response_type':response_type,'model':name,**metrics,'delta_lpd_vs_baseline':metrics['joint_user_lpd']-baseline,'elbo_final_loss':result.elbo_final})
            if joint:
                for feature in EARLY_FEATURES:
                    draws=values(result.trace,f"day_z:z_{feature}")
                    coefficients.append({'response_type':response_type,'feature':feature,'estimate':draws.mean(),'posterior_sd':draws.std(ddof=1),'hdi_3pct':np.quantile(draws,.03),'hdi_97pct':np.quantile(draws,.97)})
            del result; gc.collect()
    pd.DataFrame(rows).to_csv(args.output_dir/'model_comparison.csv',index=False)
    pd.DataFrame(coefficients).to_csv(args.output_dir/'early_feature_slope_coefficients.csv',index=False)
    (args.output_dir/'run_metadata.json').write_text(json.dumps({'train_users':len(train_users),'test_users':len(test_users),'features':EARLY_FEATURES,'scaling':scaling,'vi_iterations':args.vi_iterations,'posterior_draws':args.posterior_draws},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__': main()
