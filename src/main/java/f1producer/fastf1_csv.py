import os
import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("f1_cache")


def extract_raw_telemetry(year, event, session_type="R"):
    session = fastf1.get_session(year, event, session_type)
    session.load()

    drivers = session.drivers
    all_data = []

    for drv in drivers:
        laps = session.laps.pick_drivers(drv)

        for _, lap in laps.iterrows():
            tel = lap.get_car_data().add_distance()

            # Guard: skip entirely empty car data laps (e.g. DNF on formation lap)
            if tel.empty:
                continue

            # FIX: reindex pos to tel's index instead of concat with mismatched
            # row counts. When get_pos_data() returns fewer rows than get_car_data()
            # (common during SC laps and for drivers in incidents), a naive concat
            # produces NaT in the Time column for non-overlapping rows, which the
            # downstream Time.notna() filter then drops — silently losing entire laps.
            try:
                pos = lap.get_pos_data()
                if pos.empty:
                    raise ValueError("empty pos data")
                pos_xyz = pos[["X", "Y", "Z"]].reindex(tel.index)
            except Exception:
                pos_xyz = pd.DataFrame(
                    {"X": np.nan, "Y": np.nan, "Z": np.nan},
                    index=tel.index
                )

            df = pd.concat([tel, pos_xyz], axis=1)

            # basic metadata
            df["driver"] = str(drv)
            df["event"] = event
            df["year"] = year
            df["session"] = session_type

            df["LapNumber"] = lap["LapNumber"]
            df["Stint"] = lap["Stint"]
            df["Compound"] = lap["Compound"]

            # time — filter on car data Time only (pos NaT no longer bleeds in)
            df = df[df["Time"].notna()].copy()
            if df.empty:
                continue

            df["Time"] = df["Time"].dt.total_seconds()
            df["Time_ms"] = pd.to_numeric(df["Time"] * 1000, errors="coerce")
            df = df[df["Time_ms"].notna()].copy()
            if df.empty:
                continue

            df["Time_ms"] = df["Time_ms"].astype("int64")
            df["Time"] = pd.to_datetime(df["Time"], unit="s")

            # derived distance
            try:
                track_len = session.event["Circuit"]["length"]
            except Exception:
                track_len = df["Distance"].max()
            if not track_len or track_len == 0:
                track_len = df["Distance"].max() or 1.0
            df["RelativeDistance"] = df["Distance"] / track_len

            # pit detection
            df["is_pit_lap"] = 1 if lap["PitInTime"] is not pd.NaT else 0

            # braking detection
            df["hard_brake"] = ((df["Brake"] == 1) & (df["Speed"].diff() < -15)).astype(int)

            # throttle detection
            df["full_throttle"] = (df["Throttle"] >= 98).astype(int)

            # weather
            weather = session.weather_data.iloc[-1]
            df["TrackTemp"] = weather.get("TrackTemp")
            df["AirTemp"] = weather.get("AirTemp")
            df["Rainfall"] = weather.get("Rainfall")
            df["weather"] = "WET" if weather.get("Rainfall", 0) else "DRY"

            # placeholder track info
            df["corner_id"] = 0
            df["track_segment"] = "STRAIGHT"

            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    final = pd.concat(all_data, ignore_index=True)

    final = final[
        [
            "Time", "Time_ms", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS",
            "Distance", "RelativeDistance",
            "X", "Y", "Z",
            "LapNumber", "Stint", "Compound",
            "is_pit_lap",
            "TrackTemp", "AirTemp", "Rainfall", "weather",
            "corner_id", "track_segment",
            "hard_brake", "full_throttle",
            "driver", "event", "year", "session",
        ]
    ]

    return final


def write_csv_per_driver(df: pd.DataFrame, out_dir: str, year: int, event: str, session_type: str):
    os.makedirs(out_dir, exist_ok=True)
    event_key = event.replace(" ", "_")
    for drv, g in df.groupby("driver"):
        out_name = f"{year}_{event_key}_{session_type}_{drv}.csv"
        out_path = os.path.join(out_dir, out_name)
        g.to_csv(out_path, index=False)


def extract_qualifying_results(year, event):
    q_session = fastf1.get_session(year, event, "Q")
    q_session.load()

    results = q_session.results.copy()
    if results is None or results.empty:
        return pd.DataFrame()

    def _lap_to_seconds(val):
        if pd.isna(val):
            return None
        try:
            return float(val.total_seconds())
        except Exception:
            return None

    results["driver"] = results["DriverNumber"].astype(str)
    if "FullName" in results.columns:
        results["driver_name"] = results["FullName"].astype(str)
    elif "BroadcastName" in results.columns:
        results["driver_name"] = results["BroadcastName"].astype(str)
    else:
        results["driver_name"] = results["Abbreviation"].astype(str)
    results["driver_code"] = results["Abbreviation"].astype(str)
    results["team"] = results["TeamName"].astype(str)
    results["grid_position"] = pd.to_numeric(results["Position"], errors="coerce")

    results["Q1"] = results["Q1"].apply(_lap_to_seconds)
    results["Q2"] = results["Q2"].apply(_lap_to_seconds)
    results["Q3"] = results["Q3"].apply(_lap_to_seconds)

    results["best_quali_lap"] = results[["Q1", "Q2", "Q3"]].min(axis=1, skipna=True)

    results["event"] = event
    results["year"] = year

    return results[
        [
            "driver",
            "driver_name",
            "driver_code",
            "team",
            "grid_position",
            "Q1",
            "Q2",
            "Q3",
            "best_quali_lap",
            "event",
            "year",
        ]
    ]


def write_qualifying_csv(df: pd.DataFrame, out_dir: str, year: int, event: str):
    os.makedirs(out_dir, exist_ok=True)
    event_key = event.replace(" ", "_")
    out_name = f"{year}_{event_key}_qualifying_results.csv"
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    year = 2026
    event = "Miami Grand Prix"
    session_type = "R"

    qdf = extract_qualifying_results(year, event)
    df = extract_raw_telemetry(year, event, session_type)

    if df.empty:
        print("ERROR: No telemetry data extracted. Check FastF1 cache and session availability.")
        exit(1)

    # Enrich race telemetry with driver display info from qualifying results.
    if not qdf.empty:
        meta = qdf[["driver", "driver_name", "driver_code", "team"]].drop_duplicates(subset=["driver"])
        df["driver"] = df["driver"].astype(str)
        df = df.merge(meta, on="driver", how="left")
    else:
        df["driver_name"] = df["driver"].astype(str)
        df["driver_code"] = df["driver"].astype(str)
        df["team"] = None

    # Quick sanity check — print lap counts per driver so missing data is obvious
    print("\n=== Lap counts per driver ===")
    lap_counts = df.groupby("driver")["LapNumber"].nunique().sort_values(ascending=False)
    print(lap_counts.to_string())
    print()

    write_csv_per_driver(df, out_dir="raw_telemetry_per_driver", year=year, event=event, session_type=session_type)
    print(df.head())

    if not qdf.empty:
        write_qualifying_csv(qdf, out_dir="raw_telemetry_per_driver", year=year, event=event)