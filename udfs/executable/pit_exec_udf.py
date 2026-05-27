#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd


MODEL_DIR = os.getenv("MODEL_DIR", "/var/lib/clickhouse/user_scripts/models")
MODEL_PATH = os.path.join(MODEL_DIR, "pit_model.pkl")
FEATS_PATH = os.path.join(MODEL_DIR, "pit_feats.pkl")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(FEATS_PATH, "rb") as f:
    FEATS = pickle.load(f)

# Keep this order in sync with executable_function config argument list.
INPUT_COLUMNS = [
    "compound_enc",
    "tire_age",
    "tire_age_pct",
    "stint_len_med",
    "stint_laps_left",
    "stint_progress",
    "current_lap",
    "track_type_enc",
    "avg_throttle",
    "rainfall",
    "delta_vs_field",
    "speed_rank_pct",
    "grid_position",
    "avg_finish_last5",
    "points_last5",
    "dnf_rate_last5",
    "team_enc",
    "best_quali_lap",
    "lap_progress",
    "gap_urgency",
    "tire_freshness",
    "position_gain_pct",
    "Stint",
    "teammate_pitted",
    "must_change_compound",
    "gap_to_below_proxy",
    "rel_speed_delta",
    "pace_drop_5",
]


def parse_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(INPUT_COLUMNS):
        return None
    try:
        vals = [float(x) for x in parts]
        return dict(zip(INPUT_COLUMNS, vals))
    except ValueError:
        return None


def main():
    out = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        row = parse_line(line)
        if row is None:
            out.append("0.0")
            continue
        df = pd.DataFrame([row])[FEATS].fillna(0)
        p = float(MODEL.predict_proba(df)[0][1])
        if not np.isfinite(p):
            p = 0.0
        out.append(f"{min(max(p, 0.0), 1.0):.9f}")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
