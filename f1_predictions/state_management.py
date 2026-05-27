"""
State management functions for safety car, weather, and lap tracking.
"""
import numpy as np
from typing import List

# These will be injected by inference_engine.py
_state = None
_is_sc_lap_fn = None
_SC_THRESHOLD = 0
_SC_FIELD_DROP = 0.85
_RAINFALL_ONSET_DELTA = 0.05
_log = None


def set_state_management_context(state, is_sc_lap_fn, SC_THRESHOLD, SC_FIELD_DROP, RAINFALL_ONSET_DELTA, log=None):
    """Initialize state management with state reference and functions."""
    global _state, _is_sc_lap_fn, _SC_THRESHOLD, _SC_FIELD_DROP, _RAINFALL_ONSET_DELTA, _log
    
    _state = state
    _is_sc_lap_fn = is_sc_lap_fn
    _SC_THRESHOLD = SC_THRESHOLD
    _SC_FIELD_DROP = SC_FIELD_DROP
    _RAINFALL_ONSET_DELTA = RAINFALL_ONSET_DELTA
    _log = log


def update_sc_state(tele_rows: List, lap_no: int) -> None:
    """Update safety car state based on telemetry."""
    speeds = [float(r.get("avg_speed") or 0) for r in tele_rows if (r.get("avg_speed") or 0) > 50]
    if not speeds:
        return
    
    _state["all_speeds_ever"].extend(speeds)
    if lap_no >= 5 and len(_state["all_speeds_ever"]) > 100:
        _state["race_p75_speed"] = np.percentile(list(_state["all_speeds_ever"]), 75)
    
    field_med = float(np.median(speeds))
    was_sc = _state["sc_active"]
    _state["sc_active"] = _is_sc_lap_fn(field_med, _SC_THRESHOLD, _SC_FIELD_DROP)
    
    if _state["sc_active"]:
        _state["sc_laps"] += 1
        if not was_sc:
            msg = f"🟡 SAFETY CAR DEPLOYED  lap {lap_no}  field speed {field_med:.0f} km/h"
            _state["suggestions"].insert(0, msg)
            _state["predicted_laptime"].clear()
            if _log:
                _log.info(f"LAP {lap_no:3d} | {msg}")
    elif was_sc:
        msg = f"🟢 SAFETY CAR ENDING    lap {lap_no} — GREEN FLAG"
        _state["suggestions"].insert(0, msg)
        _state["sc_end_lap"] = lap_no
        if _log:
            _log.info(f"LAP {lap_no:3d} | {msg}")


def check_weather(tele_rows: List, lap_no: int) -> None:
    """Check weather conditions and update state."""
    rainfalls = [float(r.get("rainfall") or 0) for r in tele_rows]
    if not rainfalls:
        return
    
    current_rain = float(np.mean(rainfalls))
    last_rain = _state["last_rainfall"]
    
    if current_rain > 0.5:
        _state["weather_state"] = "WET"
    elif current_rain > 0.1 and last_rain and last_rain > 0.3:
        _state["weather_state"] = "WET_CLEARING"
    elif current_rain < 0.05:
        _state["weather_state"] = "CLEAR"
    
    if last_rain is not None:
        delta = current_rain - last_rain
        if delta > _RAINFALL_ONSET_DELTA:
            msg = (f"🌧  RAIN ONSET  lap {lap_no}  "
                   f"rainfall {last_rain:.2f}→{current_rain:.2f}  CONSIDER INTER/WET")
            _state["suggestions"].insert(0, msg)
            _state["weather_state"] = "RAIN_ONSET"
            if _log:
                _log.info(f"LAP {lap_no:3d} | {msg}")
        elif delta < -_RAINFALL_ONSET_DELTA and last_rain > 0.1:
            msg = (f"☀️  TRACK DRYING  lap {lap_no}  "
                   f"rainfall {last_rain:.2f}→{current_rain:.2f}  SLICK WINDOW OPENING")
            _state["suggestions"].insert(0, msg)
            _state["weather_state"] = "WET_CLEARING"
            if _log:
                _log.info(f"LAP {lap_no:3d} | {msg}")
    
    _state["last_rainfall"] = current_rain


def estimate_fuel_load(code: str) -> float:
    """
    [FIX-FEAT] Estimate fuel load as % of full tank.
    Assumes ~1.8 kg fuel burned per lap at race pace.
    ~110 kg full tank → 61 laps at race pace.
    """
    hist = _state["lap_history"].get(code, [])
    if not hist:
        return 0.95
    
    laps_used = len(hist)
    pit_count = _state["current_stint"].get(code, 1) - 1
    
    laps_per_tank = 61.0
    fuel_remaining = max(0.0, 1.0 - (laps_used / laps_per_tank)) + (pit_count * 0.95)
    return min(1.0, fuel_remaining)
