import glob
import os
import pandas as pd
def load_prerace_from_clickhouse(ch, event, year, session, race_laps, canonical_event_name, normalize_driver_code):
    event = canonical_event_name(event)
    try:
        driver_map_df = ch.query_df(
            """
            select toString(r.driver) as driver_num, r.driver_code, r.final_position, r.status, q.team, q.grid_position, q.best_quali_lap
            from f1_race_results r
            left join qualifying_results q
                on  r.driver_code = q.driver_code
                and r.event       = q.event
                and r.year        = q.year
            where r.event = {event:String}
              and r.year  = {year:Int32}
            """,
            parameters={"event": event, "year": int(year)}
        )
    except Exception:
        driver_map_df = None

    num_to_code = {}
    if driver_map_df is not None and not driver_map_df.empty:
        num_to_code = dict(zip(driver_map_df["driver_num"], driver_map_df["driver_code"]))

    pre_race_ctx = {}
    try:
        feat_df = ch.query_df(
            """
            select f.driver_code, f.team, f.grid_position, f.avg_finish_last5, f.points_last5, f.dnf_rate_last5, q.best_quali_lap
            from f1_features f
            left join qualifying_results q
                on  f.driver_code = q.driver_code
                and f.event       = q.event
                and f.year        = q.year
            where f.event = {event:String}
              and f.year  = {year:Int32}
            """,
            parameters={"event": event, "year": int(year)}
        )

        for _, row in feat_df.iterrows():
            code = str(row.driver_code)
            pre_race_ctx[code] = dict(
                driver_code=code,
                team=str(row.team),
                grid_position=int(row.grid_position),
                avg_finish_last5=float(row.avg_finish_last5),
                points_last5=float(row.points_last5),
                dnf_rate_last5=float(row.dnf_rate_last5),
                best_quali_lap=float(row.best_quali_lap) if pd.notna(row.best_quali_lap) else 90.0,
                final_position=None,
                final_status="",
            )
    except Exception:
        pass

    try:
        quali_df = ch.query_df(
            """
            select toString(driver) as driver_num, driver_code, team, grid_position, best_quali_lap
            from qualifying_results
            where event = {event:String}
              and year  = {year:Int32}
            """,
            parameters={"event": event, "year": int(year)}
        )
        for _, row in quali_df.iterrows():
            code = str(row.driver_code)
            num = str(row.driver_num) if pd.notna(row.driver_num) else ""
            if num and code:
                num_to_code[num] = code
            if code not in pre_race_ctx:
                pre_race_ctx[code] = dict(
                    driver_code=code,
                    team=str(row.team) if pd.notna(row.team) else "Unknown",
                    grid_position=int(row.grid_position) if pd.notna(row.grid_position) else 10,
                    avg_finish_last5=10.0,
                    points_last5=0.0,
                    dnf_rate_last5=0.2,
                    best_quali_lap=float(row.best_quali_lap) if pd.notna(row.best_quali_lap) else 90.0,
                    final_position=None,
                    final_status="",
                )
            else:
                if pd.notna(row.grid_position):
                    pre_race_ctx[code]["grid_position"] = int(row.grid_position)
                if pd.notna(row.best_quali_lap):
                    pre_race_ctx[code]["best_quali_lap"] = float(row.best_quali_lap)
                if pd.notna(row.team) and row.team:
                    pre_race_ctx[code]["team"] = str(row.team)
    except Exception:
        pass

    if driver_map_df is not None and not driver_map_df.empty:
        for _, row in driver_map_df.iterrows():
            code = str(row.driver_code)
            if code not in pre_race_ctx:
                pre_race_ctx[code] = dict(
                    driver_code=code, team=str(row.team) if pd.notna(row.team) else "Unknown",
                    grid_position=10, avg_finish_last5=10.0, points_last5=0.0,
                    dnf_rate_last5=0.2, best_quali_lap=90.0,
                )
            pre_race_ctx[code]["final_position"] = int(row.final_position) if pd.notna(row.final_position) else None
            pre_race_ctx[code]["final_status"] = str(row.status) if pd.notna(row.status) else ""

    if not num_to_code:
        csv_event = event.replace(" ", "_")
        search_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "main", "java", "f1producer", "raw_telemetry_per_driver"),
            os.environ.get("DATASET_ROOT", ""),
        ]
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            matches = glob.glob(os.path.join(d, f"{year}_{csv_event}_qualifying_results.csv"))
            if not matches:
                continue
            try:
                quali_csv = pd.read_csv(matches[0])
                for _, row in quali_csv.iterrows():
                    num = str(int(row["driver"])) if pd.notna(row.get("driver")) else ""
                    code = str(row.get("driver_code", "")).strip()
                    if num and code and len(code) == 3:
                        num_to_code[num] = code
                        if code not in pre_race_ctx:
                            pre_race_ctx[code] = dict(
                                driver_code=code, team=str(row.get("team", "Unknown")),
                                grid_position=int(row["grid_position"]) if pd.notna(row.get("grid_position")) else 10,
                                avg_finish_last5=10.0, points_last5=0.0, dnf_rate_last5=0.2,
                                best_quali_lap=float(row["best_quali_lap"]) if pd.notna(row.get("best_quali_lap")) else 90.0,
                            )
                        else:
                            if pd.notna(row.get("team")) and row["team"]:
                                pre_race_ctx[code]["team"] = str(row["team"])
                            if pd.notna(row.get("grid_position")):
                                pre_race_ctx[code]["grid_position"] = int(row["grid_position"])
                            if pd.notna(row.get("best_quali_lap")):
                                pre_race_ctx[code]["best_quali_lap"] = float(row["best_quali_lap"])
                print(f"  âœ“ Loaded {len(num_to_code)} driver mappings from qualifying CSV")
                break
            except Exception as e:
                print(f"  âš  Failed to load qualifying CSV: {e}")

    try:
        drv_df = ch.query_df(
            """
            select distinct toString(driver) as driver_num
            from raw_telemetry
            where event   = {event:String}
              and year    = {year:Int32}
              and session = {session:String}
            """,
            parameters={"event": event, "year": int(year), "session": str(session)}
        )
        for _, row in drv_df.iterrows():
            raw_num = str(row.driver_num).strip()
            code = num_to_code.get(raw_num, raw_num)
            if code not in pre_race_ctx:
                pre_race_ctx[code] = dict(
                    driver_code=code, team="Unknown", grid_position=10, avg_finish_last5=10.0,
                    points_last5=0.0, dnf_rate_last5=0.2, best_quali_lap=90.0,
                )
    except Exception:
        pass

    for raw_num, code in list(num_to_code.items()):
        if raw_num in pre_race_ctx and code != raw_num:
            pre_race_ctx[code] = pre_race_ctx.pop(raw_num)
            pre_race_ctx[code]["driver_code"] = code

    total_laps = race_laps.get(event, 0)
    if total_laps == 0:
        try:
            total_df = ch.query_df(
                """
                select max(LapNumber) as max_lap
                from raw_telemetry
                where event   = {event:String}
                  and year    = {year:Int32}
                  and session = {session:String}
                """,
                parameters={"event": event, "year": int(year), "session": str(session)}
            )
            if not total_df.empty and pd.notna(total_df.iloc[0]["max_lap"]):
                total_laps = int(total_df.iloc[0]["max_lap"])
        except Exception:
            pass

    normalized_map = {}
    for raw_num, code in num_to_code.items():
        n = str(raw_num).strip()
        c = normalize_driver_code(code)
        if n and c:
            normalized_map[n] = c
    num_to_code = normalized_map

    normalized_prerace = {}
    for key, ctx in pre_race_ctx.items():
        norm_key = normalize_driver_code(key)
        if not norm_key:
            norm_key = normalize_driver_code(ctx.get("driver_code"))
        if not norm_key:
            mapped = num_to_code.get(str(key).strip())
            norm_key = normalize_driver_code(mapped)
        if not norm_key:
            continue
        ctx["driver_code"] = norm_key
        normalized_prerace[norm_key] = ctx
    pre_race_ctx = normalized_prerace

    return pre_race_ctx, num_to_code, total_laps


def clickhouse_lap_to_rows(ch, event, year, session, lap_no, num_to_code, canonical_event_name, normalize_driver_code):
    event = canonical_event_name(event)
    lap_tele = ch.query_df(
        """
        select toString(driver) as driver_num, LapNumber, Stint, any(Compound) as Compound, avg(Speed) as avg_speed, max(Speed) as max_speed,
               avg(Throttle) as avg_throttle, avg(Brake) as avg_brake, sum(hard_brake) as hard_brake_count, sum(full_throttle) as full_throttle_count,
               avg(DRS) as avg_drs, avg(RPM) as avg_rpm, any(weather) as weather, any(TrackTemp) as track_temp, any(AirTemp) as air_temp,
               any(Rainfall) as rainfall, any(is_pit_lap) as is_pit_lap, max(Time_ms) as lap_finish_ms, max(RelativeDistance) as max_rel_dist
        from raw_telemetry
        where event      = {event:String}
          and year       = {year:Int32}
          and session    = {session:String}
          and LapNumber  = {lap:Int32}
        group by driver_num, LapNumber, Stint
        order by LapNumber, driver_num
        """,
        parameters={"event": event, "year": int(year), "session": str(session), "lap": int(lap_no)}
    )
    if lap_tele.empty:
        return []

    def _resolve_driver_code(row):
        mapped = normalize_driver_code(num_to_code.get(str(row.get("driver_num")).strip()))
        return mapped if mapped else ""

    lap_tele["driver_code"] = lap_tele.apply(_resolve_driver_code, axis=1)
    return lap_tele.where(pd.notnull(lap_tele), None).to_dict(orient="records")


def load_cumulative_times(ch, event, year, session, up_to_lap, num_to_code, pit_stop_loss_s):
    if up_to_lap < 1:
        return {}
    df = ch.query_df(
        f"""
        select driver_num,
               sum(case when is_pit = 1 then lap_time_s + {pit_stop_loss_s} else lap_time_s end) as total_time_s,
               max(LapNumber) as last_lap,
               count() as laps_done
        from (
            select toString(driver) as driver_num, LapNumber, max(Time_ms) / 1000.0 as lap_time_s, any(is_pit_lap) as is_pit
            from raw_telemetry
            where event   = {{event:String}}
              and year    = {{year:Int32}}
              and session = {{session:String}}
              and LapNumber <= {{up_to_lap:Int32}}
            group by driver_num, LapNumber
        )
        group by driver_num
        """,
        parameters={"event": event, "year": int(year), "session": str(session), "up_to_lap": int(up_to_lap)}
    )
    out = {}
    for _, row in df.iterrows():
        code = num_to_code.get(str(row["driver_num"]), str(row["driver_num"]))
        out[code] = (float(row["total_time_s"]), int(row["last_lap"]))
    return out


def clickhouse_max_lap(ch, event, year, session):
    df = ch.query_df(
        """
        select max(LapNumber) as max_lap
        from raw_telemetry
        where event   = {event:String}
          and year    = {year:Int32}
          and session = {session:String}
        """,
        parameters={"event": event, "year": int(year), "session": str(session)}
    )
    if df.empty or pd.isna(df.iloc[0]["max_lap"]):
        return 0
    return int(df.iloc[0]["max_lap"])

