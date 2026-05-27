"""
Train a ranking probability distribution model instead of binary win prediction.

This model outputs full probability distributions:
- P(driver finishes P1), P(driver finishes P2), ..., P(driver finishes P20)
- Aggregate to P(Top 5), P(Top 10), P(Points), P(Podium)

Key difference from winner_model.py:
- winner_model predicts: binary win/no-win per lap-snapshot
- ranking_model predicts: multinomial finish position across entire field

This enables:
- Full probability distributions for strategy analysis
- Monte Carlo simulation input (ranking likelihoods)
- Better uncertainty quantification
"""

import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_clean_lap_filter,
    compute_speed_features,
    compute_gap_and_pace_features,
    compute_teammate_and_compound_features,
    compute_pit_features,
    apply_fewshot_filter,
    compute_fewshot_weights,
    add_track_type_features,
    save_artifact,
    MODEL_DIR,
)


def train_ranking_distribution():
    """
    Train a multinomial classifier that predicts finish positions 1-22.
    
    Strategy:
    1. Load race data and lap telemetry
    2. For each driver-race, take the FINAL lap as the feature snapshot
    3. Use driver's ALL historical finishing positions as examples
    4. Train XGBClassifier with 22-class multinomial
    5. Apply isotonic calibration on held-out set
    6. Output calibrated probabilities for each position class
    """
    
    print("\n[Ranking] Training ranking distribution model...")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    race_results, features_df, qualifying = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)
    
    print("Cleaning laps...")
    lap_df = apply_clean_lap_filter(lap_df)
    
    print("Computing telemetry features...")
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    lap_df = compute_pit_features(lap_df)
    
    print("Loading skill features...")
    feats_merge = features_df[[
        "driver_code", "event", "year",
        "avg_finish_last5", "points_last5", "dnf_rate_last5"
    ]].rename(columns={"driver_code": "driver"})
    
    quali_merge = qualifying[[
        "driver_code", "event", "year", "best_quali_lap"
    ]].rename(columns={"driver_code": "driver"})
    
    # =========================================================================
    # FEATURE EXTRACTION: Use FINAL LAP as snapshot
    # =========================================================================
    # Group by driver-race, take last lap
    final_lap_df = (
        lap_df.sort_values(["driver", "event", "year", "LapNumber"])
        .groupby(["driver", "event", "year"])
        .tail(1)
        .reset_index(drop=True)
    )
    
    print(f"Extracted {len(final_lap_df):,} final-lap snapshots")
    
    # =========================================================================
    # MERGE WITH RESULTS (target)
    # =========================================================================
    ranking_targets = race_results[[
        "driver_code", "event", "year", "final_position", "grid_position", "status"
    ]].rename(columns={"driver_code": "driver"})
    
    # Only keep finished races
    ranking_targets = ranking_targets[ranking_targets["status"] == "Finished"]
    
    ranking_df = (
        final_lap_df[[
            "driver", "event", "year", "LapNumber",
            "avg_speed", "speed_rank_pct", "delta_vs_field",
            "tire_age", "tire_age_pct", "compound_enc", "team",
            "laps_remaining",
        ]]
        .merge(ranking_targets, on=["driver", "event", "year"], how="inner")
        .merge(feats_merge, on=["driver", "event", "year"], how="left")
        .merge(quali_merge, on=["driver", "event", "year"], how="left")
    )
    
    ranking_df = add_track_type_features(ranking_df, MODEL_DIR)
    assert len(ranking_df) > 0, "ranking_df empty after merges"
    
    print(f"Training set size: {len(ranking_df):,}")
    
    # =========================================================================
    # TARGET: multinomial position (0-21, zero-indexed for XGBoost; 22 drivers in 2026)
    # =========================================================================
    ranking_df["target_position"] = (ranking_df["final_position"].astype(int).clip(1, 22) - 1)
    
    # =========================================================================
    # FEATURES (same as winner model, but applied to final lap)
    # =========================================================================
    ranking_df["best_quali_lap"] = ranking_df["best_quali_lap"].fillna(
        ranking_df["best_quali_lap"].median()
    )
    ranking_df["grid_position"] = ranking_df["grid_position"].fillna(22)
    ranking_df["avg_finish_last5"] = ranking_df["avg_finish_last5"].fillna(10)
    ranking_df["points_last5"] = ranking_df["points_last5"].fillna(0)
    ranking_df["dnf_rate_last5"] = ranking_df["dnf_rate_last5"].fillna(0.2)
    
    ranking_df["lap_progress"] = ranking_df["LapNumber"] / max(ranking_df["LapNumber"].max(), 1)
    # Use live position proxy features only (no final-position leakage).
    n_grid = 22.0
    ranking_df["current_position"] = (ranking_df["speed_rank_pct"] * n_grid).clip(1.0, n_grid)
    ranking_df["position_pct"] = (
        1.0 - ((ranking_df["current_position"] - 1.0) / (n_grid - 1.0))
    ).clip(0.0, 1.0)
    
    ranking_df["gap_urgency"] = ranking_df["delta_vs_field"] * (
        1.0 / (ranking_df["laps_remaining"] + 1.0)
    )
    
    ranking_df["tire_freshness"] = 1.0 / (1.0 + ranking_df["tire_age_pct"])
    ranking_df["position_gain_pct"] = (
        ranking_df["position_pct"] - (1.0 - ranking_df["grid_position"] / n_grid).clip(0, 1)
    ).fillna(0)
    
    # =========================================================================
    # TEAM ENCODING
    # =========================================================================
    le_team = LabelEncoder()
    ranking_df["team_enc"] = le_team.fit_transform(
        ranking_df["team"].fillna("Unknown")
    )
    
    RANKING_FEATS = [
        "grid_position", "avg_finish_last5", "points_last5", "dnf_rate_last5",
        "team_enc", "best_quali_lap", "track_type_enc",
        "speed_rank_pct", "delta_vs_field", "tire_age", "tire_age_pct", "compound_enc",
        "lap_progress", "gap_urgency", "tire_freshness", "position_gain_pct",
    ]
    
    ranking_df[RANKING_FEATS] = ranking_df[RANKING_FEATS].fillna(
        ranking_df[RANKING_FEATS].median(numeric_only=True)
    )
    
    # =========================================================================
    # TRAIN/CAL/VAL SPLIT
    # =========================================================================
    ranking_df = apply_fewshot_filter(ranking_df)
    X = ranking_df[RANKING_FEATS]
    y = ranking_df["target_position"]
    groups = ranking_df["event"] + "_" + ranking_df["year"].astype(str)
    w = compute_fewshot_weights(ranking_df)
    
    gss_train = GroupShuffleSplit(test_size=0.4, n_splits=1, random_state=42)
    train_idx, holdout_idx = next(gss_train.split(X, y, groups))
    
    gss_cal = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    holdout_groups = groups.iloc[holdout_idx]
    cal_idx_local, val_idx_local = next(
        gss_cal.split(X.iloc[holdout_idx], y.iloc[holdout_idx], holdout_groups)
    )
    cal_idx = holdout_idx[cal_idx_local]
    val_idx = holdout_idx[val_idx_local]
    
    print(f"Split: train={len(train_idx):,}, cal={len(cal_idx):,}, val={len(val_idx):,}")
    
    # =========================================================================
    # BASE MULTINOMIAL MODEL
    # =========================================================================
    # num_class must match exactly what the data contains — derive dynamically
    # to avoid "different number of classes" crashes when splits don't contain
    # every position class (e.g. P22 only exists in 2023 but not 2024 val set).
    n_classes = int(y.max()) + 1  # positions are 0-indexed: 0..N-1
    base_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    
    base_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        sample_weight=w.iloc[train_idx],
    )
    
    # =========================================================================
    # CALIBRATION (isotonic on multinomial)
    # =========================================================================
    model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv="prefit"
    )
    model.fit(X.iloc[cal_idx], y.iloc[cal_idx])
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    val_probs = model.predict_proba(X.iloc[val_idx])  # Shape: (n_samples, 22)
    val_preds = model.predict(X.iloc[val_idx])
    
    val_y = y.iloc[val_idx].values
    all_labels = list(range(n_classes))   # all possible position classes 0..n_classes-1

    acc = accuracy_score(val_y, val_preds)
    ll  = log_loss(val_y, val_probs, labels=all_labels)
    
    # Top-5 accuracy: did model put true position in top 5?
    top5_acc = np.mean([
        np.argsort(-val_probs[i])[:5] for i in range(len(val_probs))
    ])  # This needs fixing — let me do it properly
    
    # Better: for each sample, check if true position is in top-5 predicted
    top_5_mask = np.array([
        val_y[i] in np.argsort(-val_probs[i])[:5]
        for i in range(len(val_y))
    ])
    top5_acc = top_5_mask.mean()
    
    print(f"\n  Validation accuracy (exact): {acc:.4f}")
    print(f"  Validation log-loss: {ll:.4f}")
    print(f"  Top-5 accuracy: {top5_acc:.4f}")
    
    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================
    save_artifact(MODEL_DIR, "ranking_model.pkl", model)
    save_artifact(MODEL_DIR, "ranking_feats.pkl", RANKING_FEATS)
    save_artifact(MODEL_DIR, "ranking_team_encoder.pkl", le_team)
    
    print("\n✓ Ranking distribution model trained and saved")
    
    return model, RANKING_FEATS


if __name__ == "__main__":
    train_ranking_distribution()
