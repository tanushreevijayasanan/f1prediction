import asyncio
import json
import math
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import clickhouse_connect
    HAS_CLICKHOUSE = True
except ImportError:
    HAS_CLICKHOUSE = False
    print("Warning: clickhouse-connect not installed. pip install clickhouse-connect")

app = FastAPI()

CH_HOST     = os.getenv("CH_HOST", "localhost")
CH_PORT     = int(os.getenv("CH_PORT", "8123"))
CH_USER     = os.getenv("CH_USER", "default")
CH_PASSWORD = os.getenv("CH_PASSWORD", "")
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()
REQUIRE_INTERNAL_AUTH = os.getenv("REQUIRE_INTERNAL_AUTH", "0").strip().lower() in {"1", "true", "yes"}
DEBUG_ENDPOINT_ENABLED = os.getenv("DEBUG_ENDPOINT_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def _require_internal_auth(x_internal_token: Optional[str]) -> None:
    if not REQUIRE_INTERNAL_AUTH:
        return
    if not INTERNAL_API_TOKEN or x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
def _canonical_event_name(event: str) -> str:
    return re.sub(r"\s+", " ", str(event or "").replace("_", " ").strip())


F1_EVENT    = _canonical_event_name(os.getenv("F1_EVENT", ""))
F1_YEAR     = int(os.getenv("F1_YEAR", "0") or 0)
F1_SESSION  = os.getenv("F1_SESSION", "R")

ch_client = None
if HAS_CLICKHOUSE:
    try:
        ch_client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT,
            username=CH_USER, password=CH_PASSWORD
        )
        ch_client.ping()
        print(f"✓ Connected to ClickHouse at {CH_HOST}:{CH_PORT}")
        print(f"✓ Event: '{F1_EVENT}'  Year: {F1_YEAR}")
    except Exception as e:
        print(f"⚠ ClickHouse connection failed: {e}")
        ch_client = None

clients: List[WebSocket] = []
clients_lock    = asyncio.Lock()
latest_commentary: List[str] = []
commentary_lock = asyncio.Lock()
latest_predictions: dict = {}

UI_DIR = os.path.dirname(os.path.abspath(__file__))

EMPTY_STATE = {
    "lap": 0,
    "event": F1_EVENT,
    "year": F1_YEAR,
    "positions": [],
    "leaderboard_rows": [],
    "predictions": {},
    "pit_strategies": {},
    "sc_active": False,
    "commentary": [],
    "top_speed": None,
    "avg_speed": None,
    "avg_lap_ms": None,
    "lap_times": {},
    "lap_times_history": {},
    "tire_history": {},
    "corner_data": [],
    "aggression_data": [],
    "speed_trace": {"buckets": [], "straight_zones": []},
    "timestamp": "",
}


def sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def _circuit_map_event(event: str) -> str:
    """circuit_map uses underscored event names e.g. Japanese_Grand_Prix"""
    return event.replace(" ", "_")


class PredictionAggregator:
    def __init__(self):
        self.current_lap = 0
        self.event = F1_EVENT
        self.year  = F1_YEAR
        self.session = F1_SESSION

        self._lap_time_history: Dict[str, list] = {}
        self._tire_history: Dict[str, list]      = {}
        self._pred_cols: Optional[set] = None

        # Loaded once from circuit_map
        self._corner_layout: Optional[List[dict]] = None
        self._straight_zones: Optional[List[dict]] = None

    def _q(self, sql: str, params=None):
        if not ch_client:
            return []
        try:
            result = ch_client.query(sql, parameters=(params or {}))
            cols = result.column_names
            return [dict(zip(cols, row)) for row in result.result_rows]
        except Exception as e:
            print(f"ClickHouse query error: {e}")
            traceback.print_exc()
            return []

    # ── Static layout (loaded once per session) ───────────────────────────────

    def _load_circuit_layout(self):
        cm_event = _circuit_map_event(self.event)

        # Reference speed/throttle per corner from circuit_map
        self._corner_layout = self._q(
            """
            SELECT
                corner_id,
                avg(Speed)    AS ref_speed,
                avg(Throttle) AS ref_throttle
            FROM circuit_map
            WHERE event     = {event:String}
              AND corner_id > 0
            GROUP BY corner_id
            ORDER BY corner_id
            """,
            {"event": cm_event}
        )

        # Straight zone distance ranges — group consecutive STRAIGHT rows
        self._straight_zones = self._q(
            """
            SELECT
                min(Distance) AS zone_start,
                max(Distance) AS zone_end
            FROM (
                SELECT
                    Distance,
                    segment_type,
                    row_number() OVER (ORDER BY Distance) -
                    row_number() OVER (PARTITION BY segment_type ORDER BY Distance) AS grp
                FROM circuit_map
                WHERE event = {event:String}
            )
            WHERE segment_type = 'STRAIGHT'
            GROUP BY grp
            ORDER BY zone_start
            """,
            {"event": cm_event}
        )

        print(f"✓ Circuit layout: {len(self._corner_layout or [])} corners, "
              f"{len(self._straight_zones or [])} straight zones")

    def _prediction_columns(self) -> set:
        if self._pred_cols is not None:
            return self._pred_cols
        rows = self._q("DESCRIBE TABLE prediction_results")
        self._pred_cols = {str(r.get("name", "")).strip() for r in rows if r.get("name")}
        return self._pred_cols

    # ── Prediction table queries ──────────────────────────────────────────────

    def get_latest_lap(self) -> int:
        """Latest lap with ML predictions (from prediction_results)."""
        rows = self._q(
            "SELECT max(lap_no) AS m FROM prediction_results "
            "WHERE event={event:String} AND year={year:Int32} AND session={session:String}",
            {"event": self.event, "year": self.year, "session": self.session}
        )
        return int(rows[0]["m"]) if rows and rows[0]["m"] else 0

    def get_live_lap(self) -> int:
        """Latest lap currently streaming in raw_telemetry (may be ahead of predictions)."""
        rows = self._q(
            "SELECT max(LapNumber) AS m FROM raw_telemetry "
            "WHERE event={event:String} AND year={year:Int32} AND session={session:String}",
            {"event": self.event, "year": self.year, "session": self.session}
        )
        return int(rows[0]["m"]) if rows and rows[0]["m"] else 0

    def get_avg_lap_ms(self) -> Optional[float]:
        lap = self.current_lap
        if lap <= 0:
            return None
        rows = self._q(
            """
            SELECT avg(curr_laptime) AS m
            FROM (
                SELECT driver_code, curr_laptime
                FROM (
                    SELECT
                        driver_code,
                        curr_laptime,
                        row_number() OVER (PARTITION BY driver_code ORDER BY ts DESC) AS rn
                    FROM prediction_results
                    WHERE event={event:String}
                      AND year={year:Int32}
                      AND session={session:String}
                      AND lap_no={lap:Int32}
                )
                WHERE rn = 1
            )
            """,
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )
        val = rows[0]["m"] if rows and rows[0]["m"] else None
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f * 1000

    def get_speed_kpis(self, lap: int) -> Dict[str, Optional[float]]:
        if lap <= 0:
            return {"top_speed": None, "avg_speed": None}
        rows = self._q(
            """
            SELECT
                max(Speed) AS top_speed,
                avg(Speed) AS avg_speed
            FROM raw_telemetry
            WHERE event      = {event:String}
              AND year       = {year:Int32}
              AND session    = {session:String}
              AND LapNumber  = {lap:Int32}
              AND is_pit_lap = 0
            """,
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )
        if not rows:
            return {"top_speed": None, "avg_speed": None}
        top_speed = rows[0].get("top_speed")
        avg_speed = rows[0].get("avg_speed")
        top_speed_f = float(top_speed) if top_speed is not None else None
        avg_speed_f = float(avg_speed) if avg_speed is not None else None
        if top_speed_f is not None and (math.isnan(top_speed_f) or math.isinf(top_speed_f)):
            top_speed_f = None
        if avg_speed_f is not None and (math.isnan(avg_speed_f) or math.isinf(avg_speed_f)):
            avg_speed_f = None
        return {"top_speed": top_speed_f, "avg_speed": avg_speed_f}

    def get_lap_times_current(self, lap: int) -> Dict[str, float]:
        rows = self._q(
            "SELECT driver_code, curr_laptime FROM prediction_results "
            "WHERE event={event:String} AND year={year:Int32} AND session={session:String} AND lap_no={lap:Int32}",
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )
        result = {}
        for r in rows:
            v = r.get("curr_laptime")
            if v is not None:
                f = float(v) * 1000
                if not math.isnan(f) and not math.isinf(f):
                    result[str(r["driver_code"])] = f
        return result

    def update_lap_time_history(self, lap: int) -> Dict[str, list]:
        rows = self._q(
            "SELECT driver_code, lap_no, curr_laptime FROM prediction_results "
            "WHERE event={event:String} AND year={year:Int32} AND session={session:String} AND lap_no={lap:Int32}",
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )
        for r in rows:
            code = str(r["driver_code"])
            v = r.get("curr_laptime")
            if v is not None:
                f = float(v) * 1000
                if math.isnan(f) or math.isinf(f):
                    continue
                if code not in self._lap_time_history:
                    self._lap_time_history[code] = []
                self._lap_time_history[code].append({"lap": lap, "ms": f})
                if len(self._lap_time_history[code]) > 50:
                    self._lap_time_history[code].pop(0)
        return self._lap_time_history

    def update_tire_history(self, lap: int) -> Dict[str, list]:
        rows = self._q(
            "SELECT driver_code, lap_no, tire, curr_laptime FROM prediction_results "
            "WHERE event={event:String} AND year={year:Int32} AND session={session:String} AND lap_no={lap:Int32}",
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )
        for r in rows:
            key = f"{r['driver_code']}_{r.get('tire', 'UNK')}"
            v = r.get("curr_laptime")
            if v is not None:
                f = float(v) * 1000
                if math.isnan(f) or math.isinf(f):
                    continue
                if key not in self._tire_history:
                    self._tire_history[key] = []
                self._tire_history[key].append({"lap": lap, "speed": f})
                if len(self._tire_history[key]) > 50:
                    self._tire_history[key].pop(0)
        return self._tire_history

    # ── raw_telemetry queries ─────────────────────────────────────────────────

    def get_corner_data(self, lap: int) -> List[dict]:
        """Live avg speed & throttle per corner_id for this lap.

        corner_id is sourced from circuit_map by joining on distance buckets,
        so this still works even when raw_telemetry.corner_id is 0/null.
        """
        if self._corner_layout is None:
            self._load_circuit_layout()

        cm_event = _circuit_map_event(self.event)
        live_rows = self._q(
            """
            SELECT
                cm.corner_id AS corner_id,
                avg(rt.Speed)    AS live_speed,
                avg(rt.Throttle) AS live_throttle
            FROM raw_telemetry rt
            INNER JOIN
            (
                SELECT
                    toInt32(Distance / 5) * 5 AS dist_bucket,
                    corner_id
                FROM circuit_map
                WHERE event = {cm_event:String}
                  AND corner_id > 0
                GROUP BY dist_bucket, corner_id
            ) cm
            ON toInt32(rt.Distance / 5) * 5 = cm.dist_bucket
            WHERE rt.event      = {event:String}
              AND rt.year       = {year:Int32}
              AND rt.session    = {session:String}
              AND rt.LapNumber  = {lap:Int32}
              AND rt.is_pit_lap = 0
            GROUP BY cm.corner_id
            ORDER BY cm.corner_id
            """,
            {"event": self.event, "cm_event": cm_event, "year": self.year, "session": self.session, "lap": lap}
        )

        live_map = {int(r["corner_id"]): r for r in live_rows}

        result = []
        for ref in (self._corner_layout or []):
            cid      = int(ref["corner_id"])
            live     = live_map.get(cid, {})
            speed    = float(live.get("live_speed",    ref.get("ref_speed",    0)) or 0)
            throttle = float(live.get("live_throttle", ref.get("ref_throttle", 0)) or 0)
            if math.isnan(speed)    or math.isinf(speed):    speed    = 0.0
            if math.isnan(throttle) or math.isinf(throttle): throttle = 0.0
            result.append({"corner_id": cid, "speed": speed, "throttle": throttle})
        return result

    def get_speed_trace(self, lap: int) -> dict:
        """Speed in 100m distance buckets for current lap + straight zone ranges."""
        if self._straight_zones is None:
            self._load_circuit_layout()

        rows = self._q(
            """
            SELECT
                toInt32(Distance / 100) * 100 AS dist_bucket,
                avg(Speed)                    AS avg_speed
            FROM raw_telemetry
            WHERE event     = {event:String}
              AND year       = {year:Int32}
              AND session    = {session:String}
              AND LapNumber  = {lap:Int32}
              AND is_pit_lap = 0
            GROUP BY dist_bucket
            ORDER BY dist_bucket
            """,
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )

        buckets = []
        for r in rows:
            s = float(r.get("avg_speed") or 0)
            if not math.isnan(s) and not math.isinf(s):
                buckets.append({"dist": int(r["dist_bucket"]), "speed": s})

        zones = [
            {"start": float(z["zone_start"]), "end": float(z["zone_end"])}
            for z in (self._straight_zones or [])
        ]

        return {"buckets": buckets, "straight_zones": zones}

    def get_aggression_data(self, lap: int) -> List[dict]:
        """Per-driver aggression metrics for current lap from streamed telemetry."""
        rows = self._q(
            """
            SELECT
                driver,
                avg(Throttle)          AS throttle_avg,
                sum(toInt32(hard_brake)) AS hard_brakes
            FROM raw_telemetry
            WHERE event      = {event:String}
              AND year       = {year:Int32}
              AND session    = {session:String}
              AND LapNumber  = {lap:Int32}
              AND is_pit_lap = 0
            GROUP BY driver
            ORDER BY throttle_avg DESC
            LIMIT 20
            """,
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )

        result = []
        for r in rows:
            thr = float(r.get("throttle_avg") or 0)
            hb = float(r.get("hard_brakes") or 0)
            if math.isnan(thr) or math.isinf(thr):
                thr = 0.0
            if math.isnan(hb) or math.isinf(hb):
                hb = 0.0
            result.append({
                "driver": str(r.get("driver") or ""),
                "throttle_avg": thr,
                "hard_brakes": hb,
            })
        return result

    # ── Main state builder ────────────────────────────────────────────────────

    def get_dashboard_state(self) -> Dict[str, Any]:
        self.current_lap = self.get_latest_lap()   # ML-predicted lap
        lap = self.current_lap
        live_lap = self.get_live_lap()              # freshest telemetry lap
        speed_kpis = self.get_speed_kpis(live_lap)

        pred_cols = self._prediction_columns()
        has_speed_rank = "speed_rank" in pred_cols
        has_gap_to_leader = "gap_to_leader" in pred_cols
        order_clause = "ORDER BY speed_rank ASC, win_proba DESC" if has_speed_rank else "ORDER BY pace_score DESC, win_proba DESC"
        rank_select = "speed_rank," if has_speed_rank else ""
        gap_select = "gap_to_leader," if has_gap_to_leader else ""
        rows = self._q(
            f"""
            SELECT
                driver_code, team, win_proba, pace_score,
                pit_prob, pit_alert, tire, tire_age,
                obs_deg, delta_vs_field, curr_laptime, pred_laptime,
                {rank_select}
                {gap_select}
                1 AS _keep_select_valid
            FROM (
                SELECT
                    driver_code, team, win_proba, pace_score,
                    pit_prob, pit_alert, tire, tire_age,
                    obs_deg, delta_vs_field, curr_laptime, pred_laptime, ts,
                    {rank_select}
                    {gap_select}
                    row_number() OVER (PARTITION BY driver_code ORDER BY ts DESC) AS rn
                FROM prediction_results
                WHERE event  = {{event:String}}
                  AND year    = {{year:Int32}}
                  AND session = {{session:String}}
                  AND lap_no  = {{lap:Int32}}
            )
            WHERE rn = 1
            {order_clause}
            """,
            {"event": self.event, "year": self.year, "session": self.session, "lap": lap}
        )

        prev_pos_map: Dict[str, int] = {}
        if lap > 0:
            if has_speed_rank:
                prev_rows = self._q(
                    """
                    SELECT driver_code, speed_rank
                    FROM prediction_results
                    WHERE event={event:String}
                      AND year={year:Int32}
                      AND session={session:String}
                      AND lap_no={lap:Int32}
                    ORDER BY speed_rank ASC
                    """,
                    {"event": self.event, "year": self.year, "session": self.session, "lap": lap - 1}
                )
                prev_pos_map = {
                    str(r.get("driver_code")): int(r.get("speed_rank") or 0)
                    for r in prev_rows if r.get("driver_code")
                }
            else:
                prev_rows = self._q(
                    """
                    SELECT driver_code
                    FROM prediction_results
                    WHERE event={event:String}
                      AND year={year:Int32}
                      AND session={session:String}
                      AND lap_no={lap:Int32}
                    ORDER BY pace_score DESC, win_proba DESC
                    """,
                    {"event": self.event, "year": self.year, "session": self.session, "lap": lap - 1}
                )
                for i, r in enumerate(prev_rows, start=1):
                    code = str(r.get("driver_code") or "")
                    if code:
                        prev_pos_map[code] = i

        positions, predictions, pit_strategies = [], {}, {}
        delta_values = []
        leaderboard_rows = []

        # Build a mapping from numeric driver ids -> 3-letter driver_code (UI expects).
        # inference_engine->ClickHouse sometimes stores numeric driver_code, but UI
        # renders only 3-letter alphabetic codes (e.g., VER).
        numeric_driver_codes: set[str] = set()
        for r in rows:
            raw = str(r.get("driver_code") or "").strip().upper()
            if not (len(raw) == 3 and raw.isalpha()):
                if raw:
                    numeric_driver_codes.add(raw)

        driver_code_map: Dict[str, str] = {}
        if numeric_driver_codes and ch_client:
            # Prefer qualifying_results mapping (has driver_code + driver/driver_num).
            # If that fails, fall back to race results mapping.
            in_list = ",".join(numeric_driver_codes)  # both sides are UInt/Int-like strings
            try:
                prev = self._q(
                    f"""
                    SELECT
                        toString(driver) AS driver_num,
                        driver_code
                    FROM qualifying_results
                    WHERE event={ { 'event:String': 1 } }
                    """,
                    None
                )
            except Exception:
                prev = []

            # ClickHouse placeholders: reuse the aggregator._q formatter and keep it simple.
            # We'll just run two separate queries using the configured event/year/session.
            # Note: qualifying_results typically doesn't have session; race/qualifying are static.
            try:
                qrows = self._q(
                    """
                    SELECT
                        toString(driver) AS driver_num,
                        driver_code
                    FROM qualifying_results
                    WHERE event={event:String}
                      AND year={year:Int32}
                    """,
                    {"event": self.event, "year": self.year}
                )
                for rr in qrows:
                    dn = str(rr.get("driver_num") or "").strip()
                    dc = str(rr.get("driver_code") or "").strip().upper()
                    if dn and (len(dc) == 3 and dc.isalpha()):
                        driver_code_map[dn] = dc
            except Exception:
                pass

            if not driver_code_map:
                try:
                    qrows = self._q(
                        """
                        SELECT
                            toString(driver) AS driver_num,
                            driver_code
                        FROM f1_race_results
                        WHERE event={event:String}
                          AND year={year:Int32}
                        """,
                        {"event": self.event, "year": self.year}
                    )
                    for rr in qrows:
                        dn = str(rr.get("driver_num") or "").strip()
                        dc = str(rr.get("driver_code") or "").strip().upper()
                        if dn and (len(dc) == 3 and dc.isalpha()):
                            driver_code_map[dn] = dc
                except Exception:
                    pass

        def _ui_driver_code(raw_code: str) -> Optional[str]:
            raw_code = str(raw_code or "").strip().upper()
            if len(raw_code) == 3 and raw_code.isalpha():
                return raw_code
            # Try numeric->mapped 3-letter code
            mapped = driver_code_map.get(raw_code)
            if mapped and (len(mapped) == 3 and mapped.isalpha()):
                return mapped
            return None

        for r in rows:
            raw_code = str(r.get("driver_code") or "").strip().upper()
            code = _ui_driver_code(raw_code)
            if not code:
                # If we can't map the driver id to a 3-letter UI code, drop it.
                continue

            delta    = float(r.get("delta_vs_field") or 0)
            win_prob = float(r.get("win_proba")      or 0)
            pit_prob = float(r.get("pit_prob")       or 0)
            obs_deg  = float(r.get("obs_deg")        or 0)
            for val in [delta, win_prob, pit_prob, obs_deg]:
                pass  # sanitize() handles these at the end

            gap_val = r.get("gap_to_leader") if has_gap_to_leader else None
            if gap_val is None:
                gap_val = delta
            gap_val = float(gap_val)
            positions.append((code, float(gap_val)))
            delta_values.append(delta)
            predictions[code] = {"win_prob": win_prob, "pit_prob": pit_prob}
            pit_strategies[code] = {
                "pit_probability":  pit_prob,
                "pit_urgency":      "HIGH" if pit_prob > 0.7 else "MEDIUM" if pit_prob > 0.4 else "LOW",
                "tire_age":         int(r.get("tire_age") or 0),
                "compound":         str(r.get("tire")     or "UNK"),
                "tire_degradation": obs_deg,
                "team":             str(r.get("team")     or ""),
                "stint":            0,
            }

            pos_now = int(r.get("speed_rank") or 0) if has_speed_rank else (len(leaderboard_rows) + 1)
            pos_prev = int(prev_pos_map.get(code, pos_now))
            curr_lt_s = float(r.get("curr_laptime") or 0.0)
            next_lt_s = float(r.get("pred_laptime") or 0.0)
            leaderboard_rows.append({
                "driver": code,
                "position": pos_now,
                "prev_position": pos_prev,
                "win_prob": win_prob,
                "pred_position": 0,  # placeholder — assigned below from win_prob ranking
                "pace_score": float(r.get("pace_score") or 0.0),
                "tire_symbol": (str(r.get("tire") or "UNK")[:1] or "?").upper(),
                "tire_name": str(r.get("tire") or "UNKNOWN"),
                "tire_age": int(r.get("tire_age") or 0),
                "deg_delta": obs_deg,
                "curr_laptime_ms": curr_lt_s * 1000.0,
                "next_laptime_ms": next_lt_s * 1000.0,
                "gap_s": gap_val,
                "pit_prob": pit_prob,
                "pit_alert": bool(r.get("pit_alert") or 0),
            })

        # Compute pred_position by ranking drivers on descending win_prob.
        # The driver with the highest win probability gets pred_position=1, etc.
        if leaderboard_rows:
            win_ranked = sorted(
                range(len(leaderboard_rows)),
                key=lambda i: leaderboard_rows[i]["win_prob"],
                reverse=True,
            )
            for pred_pos, idx in enumerate(win_ranked, start=1):
                leaderboard_rows[idx]["pred_position"] = pred_pos

        # Keep DB order as-is. SC heuristic should be based on relative field deltas,
        # not on displayed leaderboard gap values.
        sc_active = bool(
            delta_values and
            float(np.std(delta_values)) < 5.0 and
            float(np.mean(delta_values)) < -30
        )

        state = {
            "lap":               lap,
            "event":             self.event,
            "year":              self.year,
            "positions":         positions,
            "leaderboard_rows":  leaderboard_rows,
            "predictions":       predictions,
            "pit_strategies":    pit_strategies,
            "sc_active":         sc_active,
            "commentary":        latest_commentary,
            "top_speed":         speed_kpis["top_speed"],
            "avg_speed":         speed_kpis["avg_speed"],
            "avg_lap_ms":        self.get_avg_lap_ms(),
            "lap_times":         self.get_lap_times_current(lap),
            "lap_times_history": self.update_lap_time_history(lap),
            "tire_history":      self.update_tire_history(lap),
            "corner_data":       self.get_corner_data(live_lap),
            "aggression_data":   self.get_aggression_data(live_lap),
            "speed_trace":       self.get_speed_trace(live_lap),
            "live_lap":          live_lap,
            "timestamp":         datetime.now().isoformat(),
        }
        return sanitize(state)


aggregator = PredictionAggregator()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with clients_lock:
        clients.append(websocket)
    try:
        while True:
            try:
                state = aggregator.get_dashboard_state()
                async with commentary_lock:
                    state["commentary"] = list(latest_commentary)  # Copy list to avoid mutation
                # Include latest predictions from inference engine
                if latest_predictions:
                    state.update(latest_predictions)
                await websocket.send_text(json.dumps(state))
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"State build error: {e}")
                traceback.print_exc()
                fallback = sanitize(dict(EMPTY_STATE))
                fallback["timestamp"]  = datetime.now().isoformat()
                async with commentary_lock:
                    fallback["commentary"] = list(latest_commentary)  # Copy list to avoid mutation
                # Include latest predictions from inference engine
                if latest_predictions:
                    fallback.update(latest_predictions)
                fallback["error"]      = str(e)
                try:
                    await websocket.send_text(json.dumps(fallback))
                except WebSocketDisconnect:
                    raise
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket fatal error: {e}")
        traceback.print_exc()
    finally:
        async with clients_lock:
            if websocket in clients:
                clients.remove(websocket)


@app.get("/state")
async def get_state():
    return aggregator.get_dashboard_state()


@app.get("/debug")
async def debug(x_internal_token: Optional[str] = Header(default=None)):
    if not DEBUG_ENDPOINT_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    _require_internal_auth(x_internal_token)
    try:
        all_events = aggregator._q(
            "SELECT event, year, count() AS n, max(lap_no) AS max_lap "
            "FROM prediction_results GROUP BY event, year ORDER BY year DESC"
        )
        latest_rows = aggregator._q(
            "SELECT * FROM prediction_results "
            "WHERE event={event:String} AND year={year:Int32} "
            "ORDER BY lap_no DESC LIMIT 5",
            {"event": aggregator.event, "year": aggregator.year}
        )
        return {
            "configured_event":     aggregator.event,
            "configured_year":      aggregator.year,
            "clickhouse_connected": ch_client is not None,
            "all_events_in_db":     all_events,
            "latest_5_rows":        latest_rows,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/predictions")
async def update_predictions(data: dict, x_internal_token: Optional[str] = Header(default=None)):
    _require_internal_auth(x_internal_token)
    """Receive prediction data from inference engine"""
    global latest_predictions
    latest_predictions = data
    return {"status": "ok", "received": True}


@app.post("/commentary")
async def update_commentary(items: List[str], x_internal_token: Optional[str] = Header(default=None)):
    _require_internal_auth(x_internal_token)
    global latest_commentary
    async with commentary_lock:
        latest_commentary = items[:7]
    return {"status": "ok", "count": len(latest_commentary)}


@app.get("/health")
async def health_check():
    return {
        "status":               "ok",
        "clickhouse_connected": ch_client is not None,
        "current_lap":          aggregator.current_lap,
        "event":                aggregator.event,
        "year":                 aggregator.year,
        "timestamp":            datetime.now().isoformat(),
    }


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
