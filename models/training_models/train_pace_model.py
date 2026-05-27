"""
Train lap pace baseline model.

This model predicts absolute lap speed (pace) as a function of tire state,
track conditions, driving inputs, and race context. Provides a reference
for anomaly detection and expected pace under different scenarios.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_clean_lap_filter,
    compute_speed_features,
    apply_fewshot_filter,
    compute_fewshot_weights,
    add_track_type_features,
    save_artifact,
    MODEL_DIR,
)


def train_pace_model(train_groups=None, val_groups=None):
    """Train and return the pace baseline model."""
    print("\n[6/7] Training lap pace baseline...")

    # Load and process lap data
    race_results, _, _ = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)
    lap_df = apply_clean_lap_filter(lap_df)
    lap_df = compute_speed_features(lap_df)

    lap_df = add_track_type_features(lap_df, MODEL_DIR)

    PACE_FEATS = [
        "compound_enc",
        "tire_age",
        "tire_age_pct",
        "laps_remaining",
        "LapNumber",
        "track_type_enc",
        "avg_throttle",
        "avg_brake",
        "hard_brake_rate",
        "avg_drs",
        "track_temp",
        "air_temp",
        "rainfall",
        "delta_vs_field",
    ]

    pace_src = lap_df[
        lap_df["is_clean_lap"] == 1
    ][PACE_FEATS + ["avg_speed", "event", "year"]].copy()
    pace_src = pace_src.fillna(pace_src.median(numeric_only=True))
    pace_src = apply_fewshot_filter(pace_src)

    X_pace = pace_src[PACE_FEATS]
    y_pace = pace_src["avg_speed"]
    w_pace = compute_fewshot_weights(pace_src)

    g_pace = pace_src["event"] + "_" + pace_src["year"].astype(str)

    # If group splits aren't provided, generate them
    if train_groups is None or val_groups is None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
        tr_idx, va_idx = next(gss.split(X_pace, y_pace, g_pace))
    else:
        train_mask = g_pace.isin(train_groups)
        val_mask = g_pace.isin(val_groups)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
            tr_idx, va_idx = next(gss.split(X_pace, y_pace, g_pace))
        else:
            tr_idx = np.where(train_mask.to_numpy())[0]
            va_idx = np.where(val_mask.to_numpy())[0]

    pace_model = Ridge(alpha=1.0)
    pace_model.fit(X_pace.iloc[tr_idx], y_pace.iloc[tr_idx], sample_weight=w_pace.iloc[tr_idx])

    mae_pace = mean_absolute_error(y_pace.iloc[va_idx], pace_model.predict(X_pace.iloc[va_idx]))
    print(f"  val MAE: {mae_pace:.4f} km/h")

    # Save artifacts
    save_artifact(MODEL_DIR, "pace_model.pkl", pace_model)
    save_artifact(MODEL_DIR, "pace_feats.pkl", PACE_FEATS)

    return pace_model, PACE_FEATS


if __name__ == "__main__":
    train_pace_model()
