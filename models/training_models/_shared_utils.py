import os
import pickle
import numpy as np
import pandas as pd
import clickhouse_connect
from sklearn.preprocessing import LabelEncoder

CH_HOST      = os.getenv("CH_HOST",   "localhost")
CH_PORT      = int(os.getenv("CH_PORT", 8123))
_BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR    = os.getenv("MODEL_DIR", os.path.join(_BASE_DIR, "models"))
PIT_HORIZON  = 3   # laps: "will this driver pit within the next N laps?"
SC_SPEED_TOL = 0.15  # clean-lap filter: drop laps >15% below P75 stint speed
FEWSHOT_YEAR = int(os.getenv("FEWSHOT_YEAR", "0") or 0)
FEWSHOT_TARGET_EVENT = os.getenv("FEWSHOT_TARGET_EVENT", "").strip()
FEWSHOT_WEIGHT = float(os.getenv("FEWSHOT_WEIGHT", "3.0") or 3.0)

TRACK_TYPE_MAP = {
    "Australian Grand Prix": "BALANCED",
    "Bahrain Grand Prix": "POWER",
    "Saudi Arabian Grand Prix": "STREET_FAST",
    "Japanese Grand Prix": "HIGH_SPEED",
    "Chinese Grand Prix": "BALANCED",
    "Miami Grand Prix": "STREET",
    "Emilia Romagna Grand Prix": "DOWNFORCE",
    "Monaco Grand Prix": "STREET",
    "Canadian Grand Prix": "POWER",
    "Spanish Grand Prix": "DOWNFORCE",
    "Austrian Grand Prix": "SHORT",
    "British Grand Prix": "HIGH_SPEED",
    "Hungarian Grand Prix": "DOWNFORCE",
    "Belgian Grand Prix": "HIGH_SPEED",
    "Dutch Grand Prix": "DOWNFORCE",
    "Italian Grand Prix": "HIGH_SPEED",
    "Azerbaijan Grand Prix": "STREET_FAST",
    "Singapore Grand Prix": "STREET",
    "United States Grand Prix": "BALANCED",
    "Mexico City Grand Prix": "ALTITUDE",
    "Sao Paulo Grand Prix": "BALANCED",
    "Las Vegas Grand Prix": "STREET_FAST",
    "Qatar Grand Prix": "HIGH_SPEED",
    "Abu Dhabi Grand Prix": "BALANCED",
}
TRACK_TYPE_DEFAULT = "UNKNOWN"


def _normalize_event_name(event: str) -> str:
    if not isinstance(event, str):
        return str(event)
    return event.replace("S\u00e3o", "Sao")


def load_track_type_map() -> dict:
    path = os.getenv("TRACK_TYPE_MAP_PATH", "").strip()
    if path and os.path.isfile(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}
    return dict(TRACK_TYPE_MAP)


def get_track_type_encoder(model_dir: str) -> LabelEncoder:
    """Load or fit track type encoder, and persist artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    enc_path = os.path.join(model_dir, "track_type_encoder.pkl")
    map_path = os.path.join(model_dir, "track_type_map.pkl")

    if os.path.isfile(enc_path) and os.path.isfile(map_path):
        with open(enc_path, "rb") as f:
            return pickle.load(f)

    track_map = load_track_type_map()
    classes = sorted(set(track_map.values()) | {TRACK_TYPE_DEFAULT})
    le = LabelEncoder()
    le.fit(classes)

    try:
        with open(enc_path, "wb") as f:
            pickle.dump(le, f)
        with open(map_path, "wb") as f:
            pickle.dump(track_map, f)
    except PermissionError:
        # Non-fatal in constrained/locked environments: training can continue with in-memory encoder.
        pass

    return le


def add_track_type_features(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """Add track_type and track_type_enc columns based on event."""
    track_map = load_track_type_map()
    le = get_track_type_encoder(model_dir)
    df = df.copy()
    df["track_type"] = df["event"].map(_normalize_event_name).map(track_map).fillna(TRACK_TYPE_DEFAULT)
    df["track_type_enc"] = le.transform(df["track_type"])
    return df

COMPOUND_CLASSES = ["UNKNOWN", "SOFT", "MEDIUM", "HARD", "INTER", "WET"]
_COMP_MAP = {c: i for i, c in enumerate(COMPOUND_CLASSES)}


def get_client():
    """Get ClickHouse client."""
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)


def _fewshot_enabled() -> bool:
    return FEWSHOT_YEAR > 0


def apply_fewshot_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude target race from training when few-shot config is active.
    Keeps historical data, but removes the target race for the specified year.
    """
    if not _fewshot_enabled() or not FEWSHOT_TARGET_EVENT:
        return df
    mask = ~((df["year"] == FEWSHOT_YEAR) & (df["event"] == FEWSHOT_TARGET_EVENT))
    return df[mask].copy()


def compute_fewshot_weights(df: pd.DataFrame) -> pd.Series:
    """
    Upweight all races in FEWSHOT_YEAR (excluding target event via filter),
    while keeping historical data with weight 1.0.
    """
    if not _fewshot_enabled():
        return pd.Series(1.0, index=df.index)
    w = pd.Series(1.0, index=df.index)
    w.loc[df["year"] == FEWSHOT_YEAR] = max(FEWSHOT_WEIGHT, 1.0)
    return w


def load_reference_data():
    """Load reference datasets from ClickHouse."""
    client = get_client()
    
    race_results = client.query_df("""
    SELECT driver, driver_code, team, grid_position, final_position,
           points, status, event, year
    FROM f1_race_results
    """)

    features_df = client.query_df("""
    SELECT driver_code, team, event, year,
           avg_finish_last5, points_last5, dnf_rate_last5
    FROM f1_features
    """)

    qualifying = client.query_df("""
    SELECT driver_code, event, year,
           best_quali_lap, grid_position AS quali_grid
    FROM qualifying_results
    WHERE best_quali_lap > 0
    """)

    assert len(race_results) > 0, "race_results is empty"
    assert len(features_df)  > 0, "features_df is empty"

    return race_results, features_df, qualifying


def load_lap_telemetry(race_results):
    """Build lap-level dataset from raw telemetry."""
    client = get_client()
    
    lap_df = client.query_df("""
    SELECT
        driver, event, year, LapNumber, Stint,
        any(Compound)          AS compound,
        avg(Speed)             AS avg_speed,
        avg(Throttle)          AS avg_throttle,
        avg(Brake)             AS avg_brake,
        avg(DRS)               AS avg_drs,
        sum(hard_brake)        AS hard_brake_count,
        sum(full_throttle)     AS full_throttle_count,
        avg(RPM)               AS avg_rpm,
        any(TrackTemp)         AS track_temp,
        any(AirTemp)           AS air_temp,
        any(Rainfall)          AS rainfall,
        any(is_pit_lap)        AS is_pit_lap,
        count()                AS tele_samples,
        max(Time_ms)           AS lap_finish_ms
    FROM raw_telemetry
    WHERE session = 'R' AND LapNumber > 0
    GROUP BY driver, event, year, LapNumber, Stint
    ORDER BY event, year, driver, LapNumber
    """)

    assert len(lap_df) > 0, "lap_df is empty — no telemetry found"
    
    # Total laps per race
    race_laps = (
        lap_df.groupby(["event", "year"])["LapNumber"]
        .max().reset_index()
        .rename(columns={"LapNumber": "total_laps"})
    )

    # Driver number → code translation
    num_to_code_map = dict(zip(
        race_results["driver"].astype(str).str.strip(),
        race_results["driver_code"].astype(str).str.strip(),
    ))
    lap_df["driver"] = lap_df["driver"].astype(str).str.strip().map(num_to_code_map)
    lap_df = lap_df.dropna(subset=["driver"])

    # Tire age & compound
    lap_df = lap_df.sort_values(["driver", "event", "year", "Stint", "LapNumber"])

    def _clean_compound(series: pd.Series) -> pd.Series:
        return (
            series.fillna("UNKNOWN").astype(str).str.upper()
            .replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "INTERMEDIATE": "INTER"})
        )

    lap_df["compound_clean"] = _clean_compound(lap_df["compound"])
    lap_df["compound_enc"]   = lap_df["compound_clean"].map(lambda v: _COMP_MAP.get(v, 0))

    lap_df["tire_age"] = (
        lap_df.groupby(["driver", "event", "year", "Stint"]).cumcount()
    )

    # Median stint lengths
    _stint_max = (
        lap_df.groupby(["driver", "event", "year", "Stint", "compound_clean"])["tire_age"]
        .max().reset_index().rename(columns={"tire_age": "stint_length"})
    )
    median_stint_by_compound: dict = (
        _stint_max.groupby("compound_clean")["stint_length"].median().to_dict()
    )
    _defaults = {"UNKNOWN": 20.0, "SOFT": 18.0, "MEDIUM": 25.0,
                 "HARD": 35.0, "INTER": 15.0, "WET": 10.0}
    for k, v in _defaults.items():
        median_stint_by_compound.setdefault(k, v)

    lap_df["stint_len_med"] = lap_df["compound_clean"].map(median_stint_by_compound).fillna(25.0)
    lap_df["tire_age_pct"] = lap_df["tire_age"] / lap_df["stint_len_med"].clip(lower=1.0)
    lap_df["stint_laps_left"] = (lap_df["stint_len_med"] - lap_df["tire_age"]).clip(lower=0)
    lap_df["stint_progress"] = lap_df["tire_age"] / lap_df["stint_len_med"].clip(lower=1.0)

    # Laps remaining
    lap_df = lap_df.merge(race_laps, on=["event", "year"], how="left")
    lap_df["laps_remaining"] = (lap_df["total_laps"] - lap_df["LapNumber"]).clip(lower=0)

    # ── Cumulative race time → true gap_to_leader_s and cumul_position ──────────
    # Time_ms is lap-relative (resets to 0 each lap), so max(Time_ms) = lap duration.
    # Pit-in laps (is_pit_lap=1): add PIT_LOSS_S because stationary service is
    # absent from telemetry, causing max(Time_ms) to undercount by ~23 s.
    PIT_LOSS_S = 23.0
    lap_df = lap_df.sort_values(["driver", "event", "year", "LapNumber"]).reset_index(drop=True)
    if "lap_finish_ms" in lap_df.columns:
        lap_df["lap_finish_ms"] = pd.to_numeric(lap_df["lap_finish_ms"], errors="coerce").fillna(0)
        _pit_mask = lap_df["is_pit_lap"].fillna(0).astype(int) == 1
        _lap_s = lap_df["lap_finish_ms"] / 1000.0 + (_pit_mask.astype(float) * PIT_LOSS_S)
        _field_med = lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"].transform("median")
        _fallback = 5500.0 / (_field_med / 3.6).clip(lower=10)
        lap_df["lap_time_s"] = _lap_s.where(_lap_s > 10.0, _fallback)
    else:
        _field_med = lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"].transform("median")
        lap_df["lap_time_s"] = 5500.0 / (_field_med / 3.6).clip(lower=10)
    lap_df["cumul_race_time_s"] = lap_df.groupby(["driver", "event", "year"])["lap_time_s"].cumsum()
    _leader = (
        lap_df.groupby(["event", "year", "LapNumber"])["cumul_race_time_s"]
        .min().reset_index().rename(columns={"cumul_race_time_s": "leader_cumul_s"})
    )
    lap_df = lap_df.merge(_leader, on=["event", "year", "LapNumber"], how="left")
    lap_df["gap_to_leader_s"] = (lap_df["cumul_race_time_s"] - lap_df["leader_cumul_s"]).clip(lower=0.0)
    lap_df["cumul_position"] = (
        lap_df.groupby(["event", "year", "LapNumber"])["cumul_race_time_s"]
        .rank(method="min", ascending=True)
    )
    _n = lap_df.groupby(["event", "year", "LapNumber"])["driver"].transform("count").clip(lower=2)
    lap_df["cumul_position_pct"] = (1.0 - (lap_df["cumul_position"] - 1) / (_n - 1)).clip(0.0, 1.0)
    print(f"  Race-time features: gap_to_leader_s median={lap_df['gap_to_leader_s'].median():.1f}s")

    return lap_df, race_laps, median_stint_by_compound


def apply_clean_lap_filter(lap_df):
    """Apply clean lap filter (SC/VSC exclusion)."""
    p75_speed = (
        lap_df.groupby(["driver", "event", "year", "Stint"])["avg_speed"]
        .quantile(0.75).reset_index().rename(columns={"avg_speed": "p75_speed"})
    )
    lap_df = lap_df.merge(p75_speed, on=["driver", "event", "year", "Stint"], how="left")
    lap_df["is_clean_lap"] = (
        lap_df["avg_speed"] >= lap_df["p75_speed"] * (1 - SC_SPEED_TOL)
    ).astype(int)
    lap_df["p75_speed"] = lap_df["p75_speed"].fillna(200.0)

    n_dirty = (~lap_df["is_clean_lap"].astype(bool)).sum()
    print(f"[FIX-T2] SC/VSC laps flagged: {n_dirty:,} / {len(lap_df):,} "
          f"({n_dirty/len(lap_df)*100:.1f}%)")

    return lap_df


def compute_speed_features(lap_df):
    """Compute speed-based features: speed delta, degradation, position rank, etc."""
    # Speed delta (degradation target) — leak-free per lap
    lap_df = lap_df.sort_values(["driver", "event", "year", "Stint", "LapNumber"]).reset_index(drop=True)
    lap_df["stint_ref_speed"] = np.nan
    group_cols = ["driver", "event", "year", "Stint"]
    for _, idx in lap_df.groupby(group_cols, sort=False).indices.items():
        speeds = lap_df.loc[idx, "avg_speed"].to_numpy(dtype=float)
        ages = lap_df.loc[idx, "tire_age"].to_numpy(dtype=int)
        clean = lap_df.loc[idx, "is_clean_lap"].to_numpy(dtype=int)
        ref = np.empty(len(speeds), dtype=float)
        for i in range(len(speeds)):
            age_i = ages[i]
            elig = (
                (clean[: i + 1] == 1)
                & (ages[: i + 1] >= 1)
                & (ages[: i + 1] <= min(6, age_i))
            )
            if np.any(elig):
                ref[i] = float(np.median(speeds[: i + 1][elig]))
            else:
                ref[i] = float(speeds[i] if speeds[i] > 0 else 200.0)
        lap_df.loc[idx, "stint_ref_speed"] = ref
    lap_df["speed_delta"]     = lap_df["avg_speed"] - lap_df["stint_ref_speed"]
    lap_df["rel_speed_delta"] = lap_df["speed_delta"] / lap_df["stint_ref_speed"].clip(lower=1)
    lap_df["is_pit_lap"]      = lap_df["is_pit_lap"].fillna(0).astype(int)

    # Hard brake rate
    lap_df["hard_brake_rate"] = lap_df["hard_brake_count"] / lap_df["avg_speed"].clip(lower=1) * 100

    # Position proxies
    lap_df["speed_rank"] = (
        lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
        .rank(ascending=False, method="min")
    )
    lap_df["speed_rank_pct"] = (
        lap_df["speed_rank"] /
        lap_df.groupby(["event", "year", "LapNumber"])["speed_rank"].transform("max")
    )

    # Δ speed vs field median
    field_median_speed = (
        lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
        .median().reset_index().rename(columns={"avg_speed": "field_median_speed"})
    )
    lap_df = lap_df.merge(field_median_speed, on=["event", "year", "LapNumber"], how="left")
    lap_df["delta_vs_field"] = lap_df["avg_speed"] - lap_df["field_median_speed"]

    return lap_df


def compute_gap_and_pace_features(lap_df):
    """Compute gap to car behind and pace drop features."""
    # Gap to car directly behind in speed rank
    _r = lap_df[["event", "year", "LapNumber", "speed_rank", "avg_speed"]].copy()
    _r["speed_rank"] = _r["speed_rank"].round().astype(int)
    _r_below = _r.copy()
    _r_below["speed_rank"] = _r_below["speed_rank"] - 1
    _r_below = _r_below.rename(columns={"avg_speed": "_below_speed"})
    _r = _r.merge(
        _r_below[["event", "year", "LapNumber", "speed_rank", "_below_speed"]],
        on=["event", "year", "LapNumber", "speed_rank"], how="left"
    )
    _r["gap_to_below_proxy"] = (_r["avg_speed"] - _r["_below_speed"]).clip(lower=0).fillna(0)
    _r = _r.drop_duplicates(subset=["event", "year", "LapNumber", "speed_rank"])
    lap_df = lap_df.merge(
        _r[["event", "year", "LapNumber", "speed_rank", "gap_to_below_proxy"]],
        on=["event", "year", "LapNumber", "speed_rank"], how="left"
    )
    lap_df["gap_to_below_proxy"] = lap_df["gap_to_below_proxy"].fillna(0)

    # Pace drop (stint-local clean pace fade)
    import numpy as np
    print("  Computing pace_drop_5 (stint-local clean pace fade)...")
    _gc = ["driver", "event", "year", "Stint"]
    lap_df = lap_df.sort_values(_gc + ["LapNumber"]).reset_index(drop=True)
    lap_df["pace_drop_5"] = 0.0
    for _, idx in lap_df.groupby(_gc, sort=False).indices.items():
        spd = lap_df.loc[idx, "avg_speed"].to_numpy(dtype=float)
        cln = lap_df.loc[idx, "is_clean_lap"].to_numpy(dtype=int)
        out = np.zeros(len(spd))
        for i in range(len(spd)):
            hist_clean = spd[:i+1][cln[:i+1] == 1]
            if len(hist_clean) >= 6:
                early = float(np.mean(hist_clean[:3]))
                late  = float(np.mean(hist_clean[-3:]))
                out[i] = (early - late) / max(early, 1e-6)
        lap_df.loc[idx, "pace_drop_5"] = out

    return lap_df


def compute_teammate_and_compound_features(lap_df, race_results):
    """Compute teammate and compound-related features."""
    # Team merge
    team_map = (
        race_results[["driver_code", "team", "event", "year"]]
        .drop_duplicates()
        .rename(columns={"driver_code": "driver"})
    )
    lap_df = lap_df.merge(team_map, on=["driver", "event", "year"], how="left")

    # Δ speed vs teammate
    team_speed = (
        lap_df.groupby(["event", "year", "LapNumber", "team"])["avg_speed"]
        .mean().reset_index().rename(columns={"avg_speed": "team_avg_speed"})
    )
    lap_df = lap_df.merge(team_speed, on=["event", "year", "LapNumber", "team"], how="left")
    lap_df["delta_vs_teammate"] = lap_df["avg_speed"] - lap_df["team_avg_speed"]

    # Mandatory compound flag
    lap_df["must_change_compound"] = (
        (lap_df["Stint"] == 1) &
        (lap_df["LapNumber"] / lap_df["total_laps"] > 0.60)
    ).astype(int)

    return lap_df


def compute_pit_features(lap_df):
    """Compute pit-related features."""
    pit_flags = lap_df[["driver", "event", "year", "LapNumber", "is_pit_lap"]].copy()
    pit_flags = pit_flags.rename(columns={"LapNumber": "pit_lap_num"})
    pit_flags = pit_flags[pit_flags["is_pit_lap"] == 1]

    # Mark preceding PIT_HORIZON laps as "will pit soon"
    upcoming_rows = []
    for offset in range(1, PIT_HORIZON + 1):
        tmp = pit_flags.copy()
        tmp["LapNumber"] = tmp["pit_lap_num"] - offset
        tmp["laps_until_pit"] = offset
        upcoming_rows.append(tmp[["driver", "event", "year", "LapNumber", "laps_until_pit"]])

    upcoming_pits = pd.concat(upcoming_rows, ignore_index=True)
    upcoming_pits = upcoming_pits.drop_duplicates(subset=["driver", "event", "year", "LapNumber"])
    upcoming_pits["pit_within_horizon"] = 1

    lap_df = lap_df.merge(
        upcoming_pits[["driver", "event", "year", "LapNumber", "pit_within_horizon"]],
        on=["driver", "event", "year", "LapNumber"], how="left",
    )
    lap_df["pit_within_horizon"] = lap_df["pit_within_horizon"].fillna(0).astype(int)

    # Teammate pitted same lap
    pit_laps = lap_df[lap_df["is_pit_lap"] == 1][
        ["event", "year", "LapNumber", "team", "driver"]
    ].rename(columns={"driver": "pit_driver"})

    lap_df = lap_df.merge(
        pit_laps.groupby(["event", "year", "LapNumber", "team"])
                 .agg(pit_drivers=("pit_driver", set)).reset_index(),
        on=["event", "year", "LapNumber", "team"], how="left",
    )
    lap_df["teammate_pitted"] = lap_df.apply(
        lambda r: int(
            isinstance(r["pit_drivers"], set)
            and any(d != r["driver"] for d in r["pit_drivers"])
            and r["is_pit_lap"] == 0
        ), axis=1,
    )
    lap_df.drop(columns=["pit_drivers"], inplace=True)

    return lap_df


def save_artifact(model_dir, name, obj):
    """Save a pickle artifact."""
    import time
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, name)
    tmp_path = filepath + ".tmp"
    last_err = None
    for _ in range(5):
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
            os.replace(tmp_path, filepath)
            last_err = None
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.25)
    if last_err is not None:
        raise last_err
    print(f"  ✓  {name}")
    return filepath
