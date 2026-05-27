"""
Train pit stop prediction model.

This model predicts whether a driver will pit within the next PIT_HORIZON laps.
It learns from tire degradation signals, strategic factors, and on-track pressure
(undercut threats, competitor pit timing, compound swap requirements).
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

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
    PIT_HORIZON,
)


def train_pit_model():
    """Train and return the pit stop prediction model."""
    print(f"\n[5/7] Training pit model (horizon={PIT_HORIZON} laps ahead)...")

    # Load and process lap data
    race_results, _, _ = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)
    lap_df = apply_clean_lap_filter(lap_df)
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    lap_df = compute_pit_features(lap_df)

    lap_df = add_track_type_features(lap_df, MODEL_DIR)

    PIT_FEATS = [
        "compound_enc",
        "tire_age",
        "tire_age_pct",
        "stint_len_med",
        "stint_laps_left",
        "stint_progress",
        "Stint",
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
        "teammate_pitted",
        "delta_vs_teammate",
        "must_change_compound",
        "delta_vs_field",
        "speed_rank_pct",
        "gap_to_below_proxy",
        "rel_speed_delta",
        "pace_drop_5",
    ]

    pit_df = lap_df[
        ~lap_df["is_pit_lap"].astype(bool)
    ][PIT_FEATS + ["pit_within_horizon", "event", "year"]].copy()
    pit_df = pit_df.fillna(pit_df.median(numeric_only=True))
    pit_df = apply_fewshot_filter(pit_df)

    X_pit = pit_df[PIT_FEATS]
    y_pit = pit_df["pit_within_horizon"]
    g_pit = pit_df["event"] + "_" + pit_df["year"].astype(str)
    w_pit = compute_fewshot_weights(pit_df)

    pos_pit = int(y_pit.sum())
    neg_pit = len(y_pit) - pos_pit
    ratio   = neg_pit / pos_pit if pos_pit > 0 else 1
    print(f"  horizon={PIT_HORIZON}: positives={pos_pit:,}  negatives={neg_pit:,}  "
          f"scale_pos_weight={ratio:.1f}")

    # Use same split as tire model for consistency
    gss_pit = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    tr_pit, va_pit = next(gss_pit.split(X_pit, y_pit, g_pit))

    pit_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.04,
        scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    pit_model.fit(X_pit.iloc[tr_pit], y_pit.iloc[tr_pit], sample_weight=w_pit.iloc[tr_pit])

    pit_probs_val = pit_model.predict_proba(X_pit.iloc[va_pit])[:, 1]
    acc_pit = accuracy_score(y_pit.iloc[va_pit], pit_model.predict(X_pit.iloc[va_pit]))
    ll_pit = log_loss(y_pit.iloc[va_pit], pit_probs_val)
    print(f"  val accuracy : {acc_pit:.4f}")
    print(f"  val log-loss : {ll_pit:.4f}")
    print(f"  val prob range: [{pit_probs_val.min():.3f}, {pit_probs_val.max():.3f}]  "
          f"mean {pit_probs_val.mean():.3f}")

    # Save artifacts
    save_artifact(MODEL_DIR, "pit_model.pkl", pit_model)
    save_artifact(MODEL_DIR, "pit_feats.pkl", PIT_FEATS)
    save_artifact(MODEL_DIR, "pit_horizon.pkl", PIT_HORIZON)

    return pit_model, PIT_FEATS


if __name__ == "__main__":
    train_pit_model()
