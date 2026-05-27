"""
Train safety car / neutralisation detector.

This module detects laps where the field was neutralised (safety car, virtual
safety car) by comparing field-wide speed to race baseline. Creates a simple
threshold-based detector rather than a complex model.
"""

import pandas as pd

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_fewshot_filter,
    add_track_type_features,
    save_artifact,
    MODEL_DIR,
    SC_SPEED_TOL,
)


def train_sc_detector():
    """Train and return the safety car detector threshold."""
    print("\n[7/7] Training safety car / neutralisation detector...")

    race_results, _, _ = load_reference_data()
    lap_df, _, _ = load_lap_telemetry(race_results)

    lap_df = apply_fewshot_filter(lap_df)
    lap_df = add_track_type_features(lap_df, MODEL_DIR)

    field_speed_by_lap = (
        lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"].mean().reset_index()
    )
    race_avg_speed = (
        lap_df.groupby(["event", "year"])["avg_speed"].quantile(0.75).reset_index()
        .rename(columns={"avg_speed": "race_p75_speed"})
    )
    field_speed_by_lap = field_speed_by_lap.merge(race_avg_speed, on=["event", "year"])
    field_speed_by_lap["sc_flag"] = (
        field_speed_by_lap["avg_speed"] < field_speed_by_lap["race_p75_speed"] * (1 - SC_SPEED_TOL)
    ).astype(int)

    sc_threshold = float(
        (field_speed_by_lap["race_p75_speed"] * (1 - SC_SPEED_TOL)).median()
    )
    print(f"  SC speed threshold (median across races): {sc_threshold:.1f} km/h")

    save_artifact(MODEL_DIR, "sc_speed_threshold.pkl", sc_threshold)

    return sc_threshold


if __name__ == "__main__":
    train_sc_detector()
