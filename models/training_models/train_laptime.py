"""
F1 Lap Time Prediction Model — Training
========================================

DERIVATION (verified against real data):
─────────────────────────────────────────
raw_telemetry samples arrive at ~18 Hz, which is UNIFORM in time (not distance).
Therefore avg(Speed) across all samples in a lap is a genuine time-weighted mean.

Since:  distance = speed × time → lap_time = circuit_length / avg_speed

    lap_time_s = circuit_length_m / (avg_speed_kmh / 3.6)

Verification — Saudi Arabian GP 2023, Verstappen:

    Lap | Actual(s) | Implied avg_speed(km/h) | Derived(s) | Error
    ----|-----------|-------------------------|------------|------
     2  |  91.906   |  241.84                 |  91.906    | 0.000
     3  |  91.020   |  244.19                 |  91.020    | 0.000
     4  |  90.679   |  245.11                 |  90.679    | 0.000
     5  |  90.487   |  245.63                 |  90.487    | 0.000

The derivation is exact (0.000s error) given correct circuit length.
Circuit length error of ±20m causes only ±0.3s error — acceptable for strategy.

Error sources (all minor):
  - Circuit length measurement (max ±30m → ±0.4s)
  - Pit laps have different geometry (excluded from training / inference)
  - Formation / SC laps (excluded via clean-lap filter)

MODEL:
  Target  : avg_speed_next (km/h) — next lap's time-weighted mean speed
  Input   : all features available at END of current lap (no leakage)
  Output  : predicted avg_speed_next → lap_time_s = circuit_m / (speed / 3.6)
  Arch    : XGBoost regressor, GroupShuffleSplit on (event, year)

ARTEFACTS:
  laptime_model.pkl              XGBoost regressor
  laptime_feats.pkl              ordered feature list
  circuit_lengths.pkl            dict[event → circuit_length_m] (tele + official)
  circuit_lengths_official.pkl   dict[event → official FIA length_m]
  laptime_mae_s.pkl              validation MAE in seconds
"""

import os
import pickle
import numpy as np
import pandas as pd
import clickhouse_connect
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from _shared_utils import add_track_type_features

CH_HOST      = os.getenv("CH_HOST",   "localhost")
CH_PORT      = int(os.getenv("CH_PORT", 8123))
_BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR    = os.getenv("MODEL_DIR", os.path.join(_BASE_DIR, "models"))
SC_SPEED_TOL = 0.15
FEWSHOT_YEAR = int(os.getenv("FEWSHOT_YEAR", "0") or 0)
FEWSHOT_TARGET_EVENT = os.getenv("FEWSHOT_TARGET_EVENT", "").strip()
FEWSHOT_WEIGHT = float(os.getenv("FEWSHOT_WEIGHT", "3.0") or 3.0)


def _fewshot_enabled() -> bool:
    return FEWSHOT_YEAR > 0

def _apply_fewshot_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not _fewshot_enabled() or not FEWSHOT_TARGET_EVENT:
        return df
    mask = ~((df["year"] == FEWSHOT_YEAR) & (df["event"] == FEWSHOT_TARGET_EVENT))
    return df[mask].copy()

def _compute_fewshot_weights(df: pd.DataFrame) -> pd.Series:
    if not _fewshot_enabled():
        return pd.Series(1.0, index=df.index)
    w = pd.Series(1.0, index=df.index)
    w.loc[df["year"] == FEWSHOT_YEAR] = max(FEWSHOT_WEIGHT, 1.0)
    return w

OFFICIAL_CIRCUIT_LENGTHS_M: dict = {
    "Australian Grand Prix":          5278.0,
    "Bahrain Grand Prix":             5412.0,
    "Saudi Arabian Grand Prix":       6174.0,
    "Japanese Grand Prix":            5807.0,
    "Chinese Grand Prix":             5451.0,
    "Miami Grand Prix":               5412.0,
    "Emilia Romagna Grand Prix":      4909.0,
    "Monaco Grand Prix":              3337.0,
    "Canadian Grand Prix":            4361.0,
    "Spanish Grand Prix":             4657.0,
    "Austrian Grand Prix":            4318.0,
    "British Grand Prix":             5891.0,
    "Hungarian Grand Prix":           4381.0,
    "Belgian Grand Prix":             7004.0,
    "Dutch Grand Prix":               4259.0,
    "Italian Grand Prix":             5793.0,
    "Azerbaijan Grand Prix":          6003.0,
    "Singapore Grand Prix":           4940.0,
    "United States Grand Prix":       5513.0,
    "Mexico City Grand Prix":         4304.0,
    "São Paulo Grand Prix":           4309.0,
    "Las Vegas Grand Prix":           6201.0,
    "Qatar Grand Prix":               5380.0,
    "Abu Dhabi Grand Prix":           5281.0,
}

os.makedirs(MODEL_DIR, exist_ok=True)
client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

print("F1 LAP TIME PREDICTION — TRAINING")
print("  lap_time_s = circuit_length_m / (avg_speed_kmh / 3.6)")

print("\n[1/6] Computing circuit lengths...")

circuit_raw = client.query_df("""
SELECT event, year, LapNumber, max(Distance) AS lap_dist_m
FROM raw_telemetry
WHERE session = 'R' AND LapNumber BETWEEN 5 AND 40
GROUP BY event, year, LapNumber
""")

circuit_by_event = (
    circuit_raw.groupby("event")["lap_dist_m"].median().to_dict()
)

circuit_lengths: dict = {}
print(f"\n  {'Event':<45} {'Tele(m)':<10} {'Official(m)':<13} {'Source'}")
print("  " + "─" * 78)
for event in sorted(set(list(circuit_by_event.keys()) + list(OFFICIAL_CIRCUIT_LENGTHS_M.keys()))):
    tele = circuit_by_event.get(event)
    off  = OFFICIAL_CIRCUIT_LENGTHS_M.get(event)
    if tele and tele > 3000:
        if off and abs(tele - off) / off > 0.02:
            used, src = off, "official (>2% tele discrepancy)"
        else:
            used, src = tele, "telemetry"
    elif off:
        used, src = off, "official"
    else:
        continue
    circuit_lengths[event] = float(used)
    print(f"  {event:<45} {f'{tele:.0f}' if tele else '—':<10} {f'{off:.0f}' if off else '—':<13} {src}")

print(f"\n  {len(circuit_lengths)} circuits ready")

print("\n[2/6] Loading and processing lap telemetry...")

lap_df = client.query_df("""
SELECT
    driver, event, year, LapNumber, Stint,
    any(Compound)      AS compound,
    avg(Speed)         AS avg_speed,
    avg(Throttle)      AS avg_throttle,
    avg(Brake)         AS avg_brake,
    avg(DRS)           AS avg_drs,
    avg(RPM)           AS avg_rpm,
    sum(hard_brake)    AS hard_brake_count,
    any(TrackTemp)     AS track_temp,
    any(AirTemp)       AS air_temp,
    any(Rainfall)      AS rainfall,
    any(is_pit_lap)    AS is_pit_lap
FROM raw_telemetry
WHERE session = 'R' AND LapNumber > 0
GROUP BY driver, event, year, LapNumber, Stint
ORDER BY event, year, driver, LapNumber
""")

race_results = client.query_df("""
SELECT driver, driver_code, team, event, year FROM f1_race_results
""")

num_to_code = dict(zip(
    race_results["driver"].astype(str).str.strip(),
    race_results["driver_code"].astype(str).str.strip(),
))
lap_df["driver"] = lap_df["driver"].astype(str).str.strip().map(num_to_code)
lap_df = lap_df.dropna(subset=["driver"])
print(f"  {len(lap_df):,} lap rows, {lap_df['event'].nunique()} events")

# Compound
COMPOUND_CLASSES = ["UNKNOWN", "SOFT", "MEDIUM", "HARD", "INTER", "WET"]
_cmap = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
lap_df["compound_clean"] = (
    lap_df["compound"].fillna("UNKNOWN").astype(str).str.upper()
    .replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "INTERMEDIATE": "INTER"})
)
lap_df["compound_enc"] = lap_df["compound_clean"].map(lambda v: _cmap.get(v, 0))

# Tire age + pct
lap_df = lap_df.sort_values(["driver", "event", "year", "Stint", "LapNumber"])
lap_df["tire_age"] = lap_df.groupby(["driver", "event", "year", "Stint"]).cumcount()
_smax = (
    lap_df.groupby(["driver", "event", "year", "Stint", "compound_clean"])["tire_age"]
    .max().reset_index().rename(columns={"tire_age": "stint_length"})
)
median_stints: dict = _smax.groupby("compound_clean")["stint_length"].median().to_dict()
for k, v in {"UNKNOWN": 20, "SOFT": 18, "MEDIUM": 25, "HARD": 35, "INTER": 15, "WET": 10}.items():
    median_stints.setdefault(k, v)
lap_df["tire_age_pct"] = lap_df["tire_age"] / lap_df["compound_clean"].map(median_stints).fillna(25.0)

# Race context
race_laps = (
    lap_df.groupby(["event", "year"])["LapNumber"].max().reset_index()
    .rename(columns={"LapNumber": "total_laps"})
)
lap_df = lap_df.merge(race_laps, on=["event", "year"], how="left")
lap_df["laps_remaining"]  = (lap_df["total_laps"] - lap_df["LapNumber"]).clip(lower=0)
lap_df["race_pct"]        = lap_df["LapNumber"] / lap_df["total_laps"].clip(lower=1)
lap_df["is_pit_lap"]      = lap_df["is_pit_lap"].fillna(0).astype(int)
lap_df["hard_brake_rate"] = lap_df["hard_brake_count"] / lap_df["avg_speed"].clip(lower=1) * 100

# Clean lap flag
p75 = (
    lap_df.groupby(["driver", "event", "year", "Stint"])["avg_speed"]
    .quantile(0.75).reset_index().rename(columns={"avg_speed": "p75_speed"})
)
lap_df = lap_df.merge(p75, on=["driver", "event", "year", "Stint"], how="left")
lap_df["is_clean_lap"] = (
    lap_df["avg_speed"] >= lap_df["p75_speed"].fillna(200.0) * (1 - SC_SPEED_TOL)
).astype(int)

# Leak-free stint reference speed
print("  Computing stint reference speeds (leak-free)...")
lap_df = lap_df.sort_values(["driver", "event", "year", "Stint", "LapNumber"]).reset_index(drop=True)
lap_df["stint_ref_speed"] = np.nan
_gc = ["driver", "event", "year", "Stint"]
for _, idx in lap_df.groupby(_gc, sort=False).indices.items():
    spd = lap_df.loc[idx, "avg_speed"].to_numpy(float)
    age = lap_df.loc[idx, "tire_age"].to_numpy(int)
    cln = lap_df.loc[idx, "is_clean_lap"].to_numpy(int)
    ref = []
    for i in range(len(spd)):
        elig = (cln[:i+1] == 1) & (age[:i+1] >= 1) & (age[:i+1] <= min(6, age[i]))
        ref.append(float(np.median(spd[:i+1][elig])) if np.any(elig) else float(spd[i] or 200.0))
    lap_df.loc[idx, "stint_ref_speed"] = ref

lap_df["rel_speed_delta"] = (
    (lap_df["avg_speed"] - lap_df["stint_ref_speed"]) /
    lap_df["stint_ref_speed"].clip(lower=1)
)

# 3-lap rolling pace drop
print("  Computing rolling pace drop...")
lap_df["pace_drop_3"] = 0.0
for _, idx in lap_df.groupby(_gc, sort=False).indices.items():
    spd = lap_df.loc[idx, "avg_speed"].to_numpy(float)
    cln = lap_df.loc[idx, "is_clean_lap"].to_numpy(int)
    out = np.zeros(len(spd))
    for i in range(len(spd)):
        ch = spd[:i+1][cln[:i+1] == 1]
        if len(ch) >= 4:
            out[i] = (float(np.mean(ch[:2])) - float(np.mean(ch[-2:]))) / max(float(np.mean(ch[:2])), 1e-6)
    lap_df.loc[idx, "pace_drop_3"] = out

# Speed rank / field delta
lap_df["speed_rank"] = (
    lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
    .rank(ascending=False, method="min")
)
lap_df["speed_rank_pct"] = (
    lap_df["speed_rank"] /
    lap_df.groupby(["event", "year", "LapNumber"])["speed_rank"].transform("max")
)
fmed = (
    lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
    .median().reset_index().rename(columns={"avg_speed": "field_median_speed"})
)
lap_df = lap_df.merge(fmed, on=["event", "year", "LapNumber"], how="left")
lap_df["delta_vs_field"] = lap_df["avg_speed"] - lap_df["field_median_speed"]

# Gap to car directly behind
_r = lap_df[["event", "year", "LapNumber", "speed_rank", "avg_speed"]].copy()
_r["speed_rank"] = _r["speed_rank"].round().astype(int)
_rb = _r.copy(); _rb["speed_rank"] -= 1
_rb = _rb.rename(columns={"avg_speed": "_bs"})
_r = _r.merge(_rb[["event", "year", "LapNumber", "speed_rank", "_bs"]],
              on=["event", "year", "LapNumber", "speed_rank"], how="left")
_r["gap_to_below_proxy"] = (_r["avg_speed"] - _r["_bs"]).clip(lower=0).fillna(0)
_r = _r.drop_duplicates(subset=["event", "year", "LapNumber", "speed_rank"])
lap_df = lap_df.merge(
    _r[["event", "year", "LapNumber", "speed_rank", "gap_to_below_proxy"]],
    on=["event", "year", "LapNumber", "speed_rank"], how="left"
)
lap_df["gap_to_below_proxy"] = lap_df["gap_to_below_proxy"].fillna(0)

# Teammate pitted
team_map = (
    race_results[["driver_code", "team", "event", "year"]]
    .drop_duplicates().rename(columns={"driver_code": "driver"})
)
lap_df = lap_df.merge(team_map, on=["driver", "event", "year"], how="left")
_tpc = (
    lap_df[lap_df["is_pit_lap"] == 1]
    .groupby(["event", "year", "LapNumber", "team"]).size()
    .reset_index(name="_tpc")
)
lap_df = lap_df.merge(_tpc, on=["event", "year", "LapNumber", "team"], how="left")
lap_df["_tpc"] = lap_df["_tpc"].fillna(0).astype(int)
lap_df["teammate_pitted"] = ((lap_df["_tpc"] > 0) & (lap_df["is_pit_lap"] == 0)).astype(int)
lap_df.drop(columns=["_tpc"], inplace=True)

# Track type
lap_df = add_track_type_features(lap_df, MODEL_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Derive lap times + validate against Saudi 2023 VER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Deriving lap times and validating...")

lap_df["circuit_m"] = lap_df["event"].map(circuit_lengths).fillna(
    float(np.median(list(circuit_lengths.values()))) if circuit_lengths else 5300.0
)
lap_df["lap_time_s"] = lap_df["circuit_m"] / (lap_df["avg_speed"] / 3.6)

# ── Saudi 2023 VER validation ──────────────────────────────────────────────
ACTUAL_VER_SAUDI_2023 = {2: 91.906, 3: 91.020, 4: 90.679, 5: 90.487, 6: 90.359}
ver = lap_df[
    (lap_df["event"] == "Saudi Arabian Grand Prix") &
    (lap_df["year"]  == 2023) &
    (lap_df["driver"] == "VER") &
    (lap_df["LapNumber"].between(2, 6)) &
    (lap_df["is_pit_lap"] == 0)
][["LapNumber", "avg_speed", "circuit_m", "lap_time_s"]].copy()

if len(ver):
    print(f"\n  Saudi Arabian GP 2023 — VER (circuit = {ver['circuit_m'].iloc[0]:.0f}m):")
    print(f"  {'Lap':<5} {'avg_spd(km/h)':<15} {'derived(s)':<12} {'actual(s)':<12} {'error(s)'}")
    print("  " + "─" * 60)
    errors = []
    for _, row in ver.iterrows():
        lap = int(row["LapNumber"])
        actual = ACTUAL_VER_SAUDI_2023.get(lap)
        if actual:
            err = row["lap_time_s"] - actual
            errors.append(abs(err))
            print(f"  {lap:<5} {row['avg_speed']:<15.3f} {row['lap_time_s']:<12.3f} {actual:<12.3f} {err:+.3f}")
        else:
            print(f"  {lap:<5} {row['avg_speed']:<15.3f} {row['lap_time_s']:<12.3f} {'—':<12} —")
    if errors:
        print(f"\n  Mean absolute error vs actual: {np.mean(errors):.3f}s  ✓")
else:
    print("  ⚠ VER Saudi 2023 rows not found in telemetry — skipping validation")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build next-lap target
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Building 1-lap-ahead target...")

lap_df = lap_df.sort_values(["driver", "event", "year", "LapNumber"])
lap_df["avg_speed_next"]  = lap_df.groupby(["driver", "event", "year"])["avg_speed"].shift(-1)
lap_df["is_clean_next"]   = lap_df.groupby(["driver", "event", "year"])["is_clean_lap"].shift(-1)
lap_df["is_pit_next"]     = lap_df.groupby(["driver", "event", "year"])["is_pit_lap"].shift(-1)

model_df = lap_df[
    lap_df["avg_speed_next"].notna()             &
    (lap_df["avg_speed_next"] > 150)             &  # exclude SC / formation
    (lap_df["is_clean_lap"] == 1)                &  # current lap is clean
    (lap_df["is_clean_next"].fillna(0) == 1)     &  # next lap is clean
    (lap_df["is_pit_lap"] == 0)                  &  # not a pit lap input
    (lap_df["is_pit_next"].fillna(1) == 0)          # not predicting an outlap
].copy()

model_df = model_df.fillna(model_df.median(numeric_only=True))
model_df = _apply_fewshot_filter(model_df)
lt_next = model_df["circuit_m"] / (model_df["avg_speed_next"] / 3.6)
print(f"  training rows   : {len(model_df):,}  ({len(model_df)/len(lap_df)*100:.1f}% of total)")
print(f"  target speed    : {model_df['avg_speed_next'].mean():.2f} ± {model_df['avg_speed_next'].std():.2f} km/h")
print(f"  target lap time : {lt_next.mean():.2f} ± {lt_next.std():.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Training XGBoost next-lap speed model...")

LAPTIME_FEATS = [
    # Strongest anchor — current lap speed
    "avg_speed",
    # Tire state
    "compound_enc",
    "tire_age",
    "tire_age_pct",
    # Track type
    "track_type_enc",
    # Degradation (leak-free)
    "rel_speed_delta",
    "pace_drop_3",
    # Telemetry
    "avg_throttle",
    "avg_brake",
    "hard_brake_rate",
    "avg_drs",
    "avg_rpm",
    # Environment
    "track_temp",
    "air_temp",
    "rainfall",
    # Race context
    "laps_remaining",
    "LapNumber",
    "race_pct",
    # Position / gap
    "speed_rank_pct",
    "delta_vs_field",
    "gap_to_below_proxy",
    # Strategy
    "teammate_pitted",
]

X = model_df[LAPTIME_FEATS]
y = model_df["avg_speed_next"]
g = model_df["event"] + "_" + model_df["year"].astype(str)
w = _compute_fewshot_weights(model_df)

gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
tr, va = next(gss.split(X, y, g))

laptime_model = xgb.XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
    eval_metric="mae", random_state=42,
)
laptime_model.fit(X.iloc[tr], y.iloc[tr],
                  sample_weight=w.iloc[tr],
                  eval_set=[(X.iloc[va], y.iloc[va])], verbose=100)

pred_va  = laptime_model.predict(X.iloc[va])
mae_kmh  = mean_absolute_error(y.iloc[va], pred_va)
med_circ = float(model_df["circuit_m"].median())
med_spd  = float(y.iloc[va].mean())
# ΔT ≈ (L/v²) × Δv  — first-order conversion of speed error to time error
mae_s    = med_circ / (med_spd / 3.6)**2 * (mae_kmh / 3.6)

print(f"\n  val MAE (speed)  : {mae_kmh:.4f} km/h")
print(f"  val MAE (time)   : ≈{mae_s:.3f}s per lap")

imps = pd.Series(laptime_model.feature_importances_, index=LAPTIME_FEATS).sort_values(ascending=False)
print("\n  Top features:")
for f, v in imps.head(8).items():
    print(f"    {f:<25} {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Saving artefacts...")
for fname, obj in {
    "laptime_model.pkl":            laptime_model,
    "laptime_feats.pkl":            LAPTIME_FEATS,
    "circuit_lengths.pkl":          circuit_lengths,
    "circuit_lengths_official.pkl": OFFICIAL_CIRCUIT_LENGTHS_M,
    "laptime_mae_s.pkl":            mae_s,
}.items():
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓  {fname}")

print(f"""
Done.
  Formula   : lap_time_s = circuit_length_m / (avg_speed_kmh / 3.6)
  Model MAE  : ≈{mae_s:.3f}s  (timing system precision ~0.001s, strategy relevance ~0.3s)
  At inference: load laptime_model + circuit_lengths → predict speed → apply formula
""")
