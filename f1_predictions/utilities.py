"""
Utility functions for inference engine.
Includes grid ranking, tire age, lap time conversions, and other helpers.
"""
import numpy as np
from typing import Dict, List

# These will be injected by inference_engine.py
_state = None
_CIRCUIT_LENGTHS = None
_MEDIAN_STINT_LENGTHS = None
_GRID_PRIOR_LAPS = None
_GRID_BLEND_LAPS = None
_GRID_POSITION_REG = None
_MIN_ACTIVE_FOR_RANK = None
_SPARK_BARS = None
_COMPOUND_COLOR = None
_COMPOUND_SYMBOL = None


def set_utilities_context(state, CIRCUIT_LENGTHS, MEDIAN_STINT_LENGTHS, 
                          GRID_PRIOR_LAPS, GRID_BLEND_LAPS, GRID_POSITION_REG, 
                          MIN_ACTIVE_FOR_RANK, SPARK_BARS, COMPOUND_COLOR, COMPOUND_SYMBOL):
    """Initialize utilities with references to global state and constants."""
    global _state, _CIRCUIT_LENGTHS, _MEDIAN_STINT_LENGTHS
    global _GRID_PRIOR_LAPS, _GRID_BLEND_LAPS, _GRID_POSITION_REG, _MIN_ACTIVE_FOR_RANK
    global _SPARK_BARS, _COMPOUND_COLOR, _COMPOUND_SYMBOL
    
    _state = state
    _CIRCUIT_LENGTHS = CIRCUIT_LENGTHS
    _MEDIAN_STINT_LENGTHS = MEDIAN_STINT_LENGTHS
    _GRID_PRIOR_LAPS = GRID_PRIOR_LAPS
    _GRID_BLEND_LAPS = GRID_BLEND_LAPS
    _GRID_POSITION_REG = GRID_POSITION_REG
    _MIN_ACTIVE_FOR_RANK = MIN_ACTIVE_FOR_RANK
    _SPARK_BARS = SPARK_BARS
    _COMPOUND_COLOR = COMPOUND_COLOR
    _COMPOUND_SYMBOL = COMPOUND_SYMBOL


def grid_rank_map(codes: List[str]) -> Dict[str, float]:
    """
    Qualifying grid order as a rank pct in [0,1].
    Lower grid_position => better rank (smaller pct).
    """
    if not codes:
        return {}
    keyed = []
    for c in codes:
        pr = _state["pre_race"].get(c, {})
        gp = pr.get("grid_position", 999)
        try:
            gp = int(gp)
        except Exception:
            gp = 999
        keyed.append((gp, c))
    keyed.sort(key=lambda x: (x[0], x[1]))
    n = len(keyed)
    return {c: (i + 1) / max(n, 1) for i, (_, c) in enumerate(keyed)}


def grid_blend_weight(lap_no: int, n_active: int) -> float:
    """
    Weight for qualifying grid prior in speed-rank.
    1.0 at/before GRID_PRIOR_LAPS, then linearly decays to GRID_POSITION_REG by GRID_BLEND_LAPS.
    After GRID_BLEND_LAPS, retains residual GRID_POSITION_REG influence to prevent impossible
    overtakes (e.g., P20 can't suddenly be faster than top 3 on a single lap).
    """
    if n_active < _MIN_ACTIVE_FOR_RANK:
        return 1.0
    if _GRID_BLEND_LAPS <= 0:
        return _GRID_POSITION_REG
    if lap_no <= _GRID_PRIOR_LAPS:
        return 1.0
    if lap_no >= _GRID_BLEND_LAPS:
        return _GRID_POSITION_REG
    
    span = max(_GRID_BLEND_LAPS - _GRID_PRIOR_LAPS, 1)
    fraction = (lap_no - _GRID_PRIOR_LAPS) / span
    return 1.0 - (fraction * (1.0 - _GRID_POSITION_REG))


def laps_remaining(lap_no: int) -> int:
    """Calculate remaining laps in the race."""
    return max(0, _state["total_laps"] - lap_no)


def tire_age_pct(compound: str, age: int) -> float:
    """Calculate tire age as percentage of median stint length."""
    median = _MEDIAN_STINT_LENGTHS.get(compound.upper().replace("INTERMEDIATE", "INTER"), 25.0)
    return age / median if median > 0 else 0.0


def stint_len_med(compound: str) -> float:
    """Get median stint length for a compound."""
    return float(_MEDIAN_STINT_LENGTHS.get(compound.upper().replace("INTERMEDIATE", "INTER"), 25.0))


def hard_brake_rate(hard_brake_count: float, avg_speed: float) -> float:
    """Calculate hard braking rate normalized by speed."""
    return hard_brake_count / max(avg_speed, 1.0) * 100.0


def speed_to_laptime(speed_kmh: float, event: str) -> float:
    """Convert average speed to lap time (seconds)."""
    circuit_m = _CIRCUIT_LENGTHS.get(event, 5300.0)
    if speed_kmh <= 0:
        return 0.0
    return circuit_m / (speed_kmh / 3.6)


def laptime_str(lt_s: float) -> str:
    """Format lap time as M:SS.sss string."""
    if lt_s <= 0:
        return "—"
    m = int(lt_s // 60)
    s = lt_s - m * 60
    return f"{m}:{s:06.3f}"


def get_current_laptime(code: str) -> float:
    """Get current lap time for a driver."""
    return _state["current_laptime"].get(code, 0.0)


def get_formatted_team(code: str) -> str:
    """Get formatted team name for a driver."""
    team = _state["pre_race"].get(code, {}).get("team", "")
    return team if team else code


def sparkline(values: List[float], width: int = 10) -> str:
    """Generate a sparkline representation of values."""
    if not values:
        return ""
    chunk = values[-width:]
    lo, hi = min(chunk), max(chunk)
    span = hi - lo or 1
    return "".join(_SPARK_BARS[min(7, int((v - lo) / span * 7))] for v in chunk)


def field_median_speed(tele_rows: List) -> float:
    """Calculate median speed across the field."""
    speeds = [float(r.get("avg_speed") or 0) for r in tele_rows if (r.get("avg_speed") or 0) > 50]
    return float(np.median(speeds)) if speeds else 0.0


def is_sc_lap(field_median: float, SC_THRESHOLD: float, SC_FIELD_DROP: float) -> bool:
    """Determine if lap is under safety car based on field median speed."""
    if SC_THRESHOLD and SC_THRESHOLD > 0:
        return field_median < SC_THRESHOLD
    ref = _state["race_p75_speed"]
    if ref <= 0:
        return False
    return field_median < ref * SC_FIELD_DROP


def current_stint_clean_speeds(code: str) -> List[float]:
    """
    Return avg_speeds for clean laps in the CURRENT stint only.
    
    The original code used all of lap_history[code], which spans multiple stints.
    After a pit stop, laps from the previous stint contaminate pace_drop and
    rel_speed_delta with cross-stint degradation signals that make the model
    think the driver is aggressively degrading when they're actually on fresh
    rubber.
    """
    current_stint_id = _state["current_stint"][code]
    hist = _state["lap_history"].get(code, [])
    return [
        h["avg_speed"]
        for h in hist
        if h.get("stint") == current_stint_id
        and not h.get("is_sc")
        and h["avg_speed"] > 50
    ]


def pace_drop_from_clean(clean_speeds: List[float], n: int = 3) -> float:
    """
    Compute pace drop as (mean of first n clean laps) - (mean of last n clean laps)
    divided by the early mean. Positive = driver is getting slower.
    Returns 0.0 if not enough data.
    """
    if len(clean_speeds) < n * 2:
        return 0.0
    early = float(np.mean(clean_speeds[:n]))
    late = float(np.mean(clean_speeds[-n:]))
    return (early - late) / max(early, 1e-6)
