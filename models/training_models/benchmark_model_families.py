"""
Benchmark model families for F1 training tasks.

Usage:
  python benchmark_model_families.py --task winner
  python benchmark_model_families.py --task tire
  python benchmark_model_families.py --task all
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

import xgboost as xgb

from _shared_utils import (
    MODEL_DIR,
    add_track_type_features,
    apply_clean_lap_filter,
    apply_fewshot_filter,
    compute_fewshot_weights,
    compute_gap_and_pace_features,
    compute_pit_features,
    compute_speed_features,
    compute_teammate_and_compound_features,
    load_lap_telemetry,
    load_reference_data,
)
from train_winner_model import (
    _compute_actual_position,
    _compute_gap_to_leader,
    _compute_position_jump,
    _detect_sc_laps,
)


def _try_optional_classifiers() -> Dict[str, Callable[[], object]]:
    out: Dict[str, Callable[[], object]] = {}

    try:
        from lightgbm import LGBMClassifier

        out["lgbm_clf"] = lambda: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        out["catboost_clf"] = lambda: CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
        )
    except Exception:
        pass

    return out


def _try_optional_regressors() -> Dict[str, Callable[[], object]]:
    out: Dict[str, Callable[[], object]] = {}

    try:
        from lightgbm import LGBMRegressor

        out["lgbm_reg"] = lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor

        out["catboost_reg"] = lambda: CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
    except Exception:
        pass

    return out


def _winner_frame() -> Tuple[pd.DataFrame, List[str]]:
    race_results, features_df, qualifying = load_reference_data()

    lap_df, _, _ = load_lap_telemetry(race_results)
    lap_df = apply_clean_lap_filter(lap_df)
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    lap_df = compute_pit_features(lap_df)

    lap_df = _detect_sc_laps(lap_df)
    lap_df = _compute_actual_position(lap_df)
    lap_df = _compute_gap_to_leader(lap_df)
    lap_df = _compute_position_jump(lap_df, n=2)
    lap_df = _compute_position_jump(lap_df, n=3)

    all_laps = []
    for _, group in lap_df.groupby(["driver", "event", "year"]):
        group = group.sort_values("LapNumber")
        early = group[group["LapNumber"] <= 15]
        early_samples = early[early["LapNumber"] % 5 == 0]
        mid = group[(group["LapNumber"] > 15) & (group["LapNumber"] <= 35)]
        mid_samples = mid[mid["LapNumber"] % 3 == 0]
        late = group[group["LapNumber"] > 35]
        all_laps.append(pd.concat([early_samples, mid_samples, late]))

    mid_race = pd.concat(all_laps, ignore_index=True).copy()

    winner_targets = race_results[race_results["status"] == "Finished"][[
        "driver_code", "event", "year", "final_position", "grid_position"
    ]].rename(columns={"driver_code": "driver"})

    feats_merge = features_df[[
        "driver_code", "event", "year", "avg_finish_last5", "points_last5", "dnf_rate_last5"
    ]].rename(columns={"driver_code": "driver"})

    quali_merge = qualifying[[
        "driver_code", "event", "year", "best_quali_lap"
    ]].rename(columns={"driver_code": "driver"})

    winner_df = (
        mid_race[[
            "driver", "event", "year", "LapNumber", "laps_remaining",
            "avg_speed", "speed_rank_pct", "delta_vs_field",
            "tire_age", "tire_age_pct", "compound_enc", "team",
            "sc_active", "current_position", "position_pct", "n_active",
            "gap_to_leader_s", "position_jump_2", "position_jump_3",
        ]]
        .merge(winner_targets, on=["driver", "event", "year"], how="inner")
        .merge(feats_merge, on=["driver", "event", "year"], how="left")
        .merge(quali_merge, on=["driver", "event", "year"], how="left")
    )

    winner_df = add_track_type_features(winner_df, MODEL_DIR)
    winner_df["target_win"] = (winner_df["final_position"] == 1).astype(int)
    winner_df["grid_position_group"] = winner_df["grid_position"].apply(
        lambda gp: 1 if gp <= 5 else (2 if gp <= 15 else 3)
    )
    winner_df["position_gain_pct"] = (
        winner_df["speed_rank_pct"] - (winner_df["grid_position"] / 20.0).clip(0, 1)
    ).fillna(0)

    total_laps_map = (
        winner_df.groupby(["event", "year"])["LapNumber"].max().rename("total_laps").reset_index()
    )
    winner_df = winner_df.merge(total_laps_map, on=["event", "year"], how="left")
    winner_df["total_laps"] = winner_df["total_laps"].fillna(60)
    winner_df["lap_progress"] = winner_df["LapNumber"] / winner_df["total_laps"].clip(lower=1)
    winner_df["is_late_race"] = (winner_df["lap_progress"] > 0.75).astype(int)
    winner_df["tire_freshness"] = 1.0 / (1.0 + winner_df["tire_age_pct"])
    winner_df["gap_laps_remaining"] = (
        winner_df["gap_to_leader_s"] / winner_df["laps_remaining"].clip(lower=1)
    )
    winner_df["is_leading"] = (winner_df["current_position"] == 1.0).astype(int)
    winner_df["leading_and_late"] = winner_df["position_pct"] * winner_df["lap_progress"]
    winner_df["gap_urgency"] = winner_df["delta_vs_field"] / winner_df["laps_remaining"].clip(lower=1)
    winner_df["sc_beneficiary"] = (
        (winner_df["sc_active"] == 1) & (winner_df["position_jump_2"] > 2)
    ).astype(int)

    # Fast team encoding for benchmark only.
    winner_df["team_enc"] = winner_df["team"].fillna("Unknown").astype("category").cat.codes

    winner_df["best_quali_lap"] = winner_df["best_quali_lap"].fillna(winner_df["best_quali_lap"].median())
    winner_df["grid_position"] = winner_df["grid_position"].fillna(20)
    winner_df["avg_finish_last5"] = winner_df["avg_finish_last5"].fillna(10)
    winner_df["points_last5"] = winner_df["points_last5"].fillna(0)
    winner_df["dnf_rate_last5"] = winner_df["dnf_rate_last5"].fillna(0.2)

    winner_feats = [
        "grid_position",
        "grid_position_group",
        "avg_finish_last5",
        "points_last5",
        "dnf_rate_last5",
        "team_enc",
        "best_quali_lap",
        "track_type_enc",
        "laps_remaining",
        "speed_rank_pct",
        "delta_vs_field",
        "tire_age",
        "tire_age_pct",
        "compound_enc",
        "current_position",
        "position_pct",
        "is_leading",
        "sc_active",
        "position_jump_2",
        "position_jump_3",
        "sc_beneficiary",
        "gap_to_leader_s",
        "gap_laps_remaining",
        "lap_progress",
        "is_late_race",
        "gap_urgency",
        "leading_and_late",
        "position_gain_pct",
        "tire_freshness",
    ]

    winner_df[winner_feats] = winner_df[winner_feats].fillna(winner_df[winner_feats].median(numeric_only=True))
    winner_df = apply_fewshot_filter(winner_df)
    return winner_df, winner_feats


def _tire_frame() -> Tuple[pd.DataFrame, List[str]]:
    race_results, _, _ = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)
    lap_df = apply_clean_lap_filter(lap_df)
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    lap_df = add_track_type_features(lap_df, MODEL_DIR)

    tire_feats = [
        "compound_enc",
        "tire_age",
        "tire_age_pct",
        "stint_len_med",
        "stint_laps_left",
        "stint_progress",
        "laps_remaining",
        "LapNumber",
        "track_type_enc",
        "avg_throttle",
        "avg_brake",
        "hard_brake_rate",
        "avg_drs",
        "avg_rpm",
        "track_temp",
        "air_temp",
        "rainfall",
        "delta_vs_field",
        "speed_rank_pct",
        "gap_to_below_proxy",
        "pace_drop_5",
        "delta_vs_teammate",
    ]

    tire_df = lap_df[
        (lap_df["is_clean_lap"] == 1) & (lap_df["tire_age"] > 0)
    ][tire_feats + ["rel_speed_delta", "event", "year"]].copy()
    tire_df = tire_df.fillna(tire_df.median(numeric_only=True))
    tire_df = apply_fewshot_filter(tire_df)
    return tire_df, tire_feats


def run_winner_benchmark() -> pd.DataFrame:
    print("\n[WINNER BENCHMARK]")
    df, feats = _winner_frame()
    X = df[feats]
    y = df["target_win"]
    groups = df["event"] + "_" + df["year"].astype(str)

    gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    tr_idx, va_idx = next(gss.split(X, y, groups))

    w = compute_fewshot_weights(df)
    lap_weight = np.exp(3.0 * df["lap_progress"].values)
    lap_weight = lap_weight / lap_weight.mean()
    sample_w = w * lap_weight

    pos = int((y.iloc[tr_idx] == 1).sum())
    neg = int((y.iloc[tr_idx] == 0).sum())
    ratio = (neg / max(pos, 1))

    candidates: Dict[str, Callable[[], object]] = {
        "xgb_clf": lambda: xgb.XGBClassifier(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.9,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gb_clf": lambda: HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=8,
            max_iter=600,
            random_state=42,
        ),
        "rf_clf": lambda: RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "logreg": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    }
    candidates.update(_try_optional_classifiers())

    rows = []
    for name, builder in candidates.items():
        try:
            model = builder()
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sample_w.iloc[tr_idx])
            prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            rows.append({
                "model": name,
                "roc_auc": roc_auc_score(y.iloc[va_idx], prob),
                "pr_auc": average_precision_score(y.iloc[va_idx], prob),
                "logloss": log_loss(y.iloc[va_idx], prob),
                "brier": brier_score_loss(y.iloc[va_idx], prob),
                "precision@0.5": precision_score(y.iloc[va_idx], pred, zero_division=0),
                "recall@0.5": recall_score(y.iloc[va_idx], pred, zero_division=0),
                "f1@0.5": f1_score(y.iloc[va_idx], pred, zero_division=0),
            })
        except Exception as exc:
            print(f"  skip {name}: {exc}")

    out = pd.DataFrame(rows).sort_values(["logloss", "brier"], ascending=[True, True])
    print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return out


def run_tire_benchmark() -> pd.DataFrame:
    print("\n[TIRE BENCHMARK]")
    df, feats = _tire_frame()
    X = df[feats]
    y = df["rel_speed_delta"]
    groups = df["event"] + "_" + df["year"].astype(str)

    gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    tr_idx, va_idx = next(gss.split(X, y, groups))

    w = compute_fewshot_weights(df)

    candidates: Dict[str, Callable[[], object]] = {
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=20,
        ),
        "xgb_reg": lambda: xgb.XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=15,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gb_reg": lambda: HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=8,
            max_iter=600,
            random_state=42,
        ),
        "rf_reg": lambda: RandomForestRegressor(
            n_estimators=800,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }
    candidates.update(_try_optional_regressors())

    rows = []
    for name, builder in candidates.items():
        try:
            model = builder()
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=w.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            rows.append({
                "model": name,
                "mae": mean_absolute_error(y.iloc[va_idx], pred),
                "rmse": np.sqrt(mean_squared_error(y.iloc[va_idx], pred)),
                "r2": r2_score(y.iloc[va_idx], pred),
            })
        except Exception as exc:
            print(f"  skip {name}: {exc}")

    out = pd.DataFrame(rows).sort_values(["mae", "rmse"], ascending=[True, True])
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model families on grouped F1 splits")
    parser.add_argument("--task", choices=["winner", "tire", "all"], default="all")
    parser.add_argument("--save-csv", action="store_true", help="Save benchmark tables to CSV files")
    args = parser.parse_args()

    if args.task in ("winner", "all"):
        winner_out = run_winner_benchmark()
        if args.save_csv:
            winner_out.to_csv("winner_benchmark_results.csv", index=False)

    if args.task in ("tire", "all"):
        tire_out = run_tire_benchmark()
        if args.save_csv:
            tire_out.to_csv("tire_benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
