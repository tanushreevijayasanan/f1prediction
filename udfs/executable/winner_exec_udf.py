#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd


MODEL_DIR = os.getenv("MODEL_DIR", "/var/lib/clickhouse/user_scripts/models")
MODEL_PATH = os.path.join(MODEL_DIR, "winner_model.pkl")
FEATS_PATH = os.path.join(MODEL_DIR, "winner_feats.pkl")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(FEATS_PATH, "rb") as f:
    FEATS = pickle.load(f)

# Keep this order in sync with executable_function config argument list.
INPUT_COLUMNS = [
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
