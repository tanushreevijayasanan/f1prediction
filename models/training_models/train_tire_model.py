import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_clean_lap_filter,
    compute_speed_features,
    compute_gap_and_pace_features,
    compute_teammate_and_compound_features,
    apply_fewshot_filter,
    compute_fewshot_weights,
    add_track_type_features,
    save_artifact,
    MODEL_DIR,
)


def train_tire_model():
    print("\n[4/7] Training tire degradation model (clean laps only)...")

    race_results, _, _ = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)
    lap_df = apply_clean_lap_filter(lap_df)
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)

    lap_df = add_track_type_features(lap_df, MODEL_DIR)

    TIRE_FEATS = [
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
    ][TIRE_FEATS + ["rel_speed_delta", "event", "year"]].copy()
    tire_df = tire_df.fillna(tire_df.median(numeric_only=True))
    tire_df = apply_fewshot_filter(tire_df)

    print(f"  clean laps for degradation training: {len(tire_df):,}")

    assert tire_df["laps_remaining"].std() > 0, "laps_remaining zero variance"
    assert tire_df["tire_age_pct"].std() > 0, "tire_age_pct zero variance"

    X_tire = tire_df[TIRE_FEATS]
    y_tire = tire_df["rel_speed_delta"]
    g_tire = tire_df["event"] + "_" + tire_df["year"].astype(str)
    w_tire = compute_fewshot_weights(tire_df)

    gss_tire = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    tr_tire, va_tire = next(gss_tire.split(X_tire, y_tire, g_tire))

    tire_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.06, max_depth=5,
        subsample=0.8, min_samples_leaf=20,
    )
    tire_model.fit(X_tire.iloc[tr_tire], y_tire.iloc[tr_tire], sample_weight=w_tire.iloc[tr_tire])

    mae_tire = mean_absolute_error(y_tire.iloc[va_tire], tire_model.predict(X_tire.iloc[va_tire]))
    print(f"  val MAE (rel_speed_delta): {mae_tire:.6f}  ({mae_tire*100:.3f}%)")

    save_artifact(MODEL_DIR, "tire_model.pkl", tire_model)
    save_artifact(MODEL_DIR, "tire_feats.pkl", TIRE_FEATS)

    train_groups = set(g_tire.iloc[tr_tire])
    val_groups = set(g_tire.iloc[va_tire])

    return tire_model, TIRE_FEATS, tr_tire, va_tire, X_tire, y_tire, train_groups, val_groups


if __name__ == "__main__":
    train_tire_model()
