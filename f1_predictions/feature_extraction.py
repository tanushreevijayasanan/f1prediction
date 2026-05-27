"""
Feature extraction functions for all models (winner, tire, pit, laptime, ranking).
Extracts features from telemetry and state for model inference.
"""
import numpy as np
import pandas as pd
from typing import Dict

# These will be injected by inference_engine.py
_state = None
_enc_compound_fn = None
_enc_track_type_fn = None
_enc_team_fn = None
_tire_age_pct_fn = None
_stint_len_med_fn = None
_hard_brake_rate_fn = None
_laps_remaining_fn = None
_current_stint_clean_speeds_fn = None
_pace_drop_from_clean_fn = None


def set_feature_extraction_context(enc_compound, enc_track_type, enc_team, 
                                     tire_age_pct, stint_len_med, hard_brake_rate, 
                                     laps_remaining, current_stint_clean_speeds, pace_drop_from_clean,
                                     state):
    """Initialize feature extraction with encoder and utility functions."""
    global _enc_compound_fn, _enc_track_type_fn, _enc_team_fn
    global _tire_age_pct_fn, _stint_len_med_fn, _hard_brake_rate_fn
    global _laps_remaining_fn, _current_stint_clean_speeds_fn, _pace_drop_from_clean_fn
    global _state
    
    _enc_compound_fn = enc_compound
    _enc_track_type_fn = enc_track_type
    _enc_team_fn = enc_team
    _tire_age_pct_fn = tire_age_pct
    _stint_len_med_fn = stint_len_med
    _hard_brake_rate_fn = hard_brake_rate
    _laps_remaining_fn = laps_remaining
    _current_stint_clean_speeds_fn = current_stint_clean_speeds
    _pace_drop_from_clean_fn = pace_drop_from_clean
    _state = state


def laptime_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    """
    [FIX-F2] Uses current-stint-only clean speeds for pace_drop_3.
    [FIX-F4] Uses populated gap_to_below.
    """
    age = _state["tire_age"][code]
    avg_spd = float(row.get("avg_speed") or 0.0)
    ref_s = float(_state["stint_ref_speed"].get(code, 0.0) or 0.0)
    rel_d = ((avg_spd - ref_s) / max(ref_s, 1.0)) if ref_s > 0 and avg_spd > 0 else 0.0

    clean = _current_stint_clean_speeds_fn(code)
    pace_d3 = _pace_drop_from_clean_fn(clean, n=2)

    return {
        "avg_speed": avg_spd,
        "compound_enc": _enc_compound_fn(comp),
        "tire_age": age,
        "tire_age_pct": _tire_age_pct_fn(comp, age),
        "track_type_enc": _enc_track_type_fn(_state["event"]),
        "rel_speed_delta": rel_d,
        "pace_drop_3": pace_d3,
        "avg_throttle": float(row.get("avg_throttle") or 85.0),
        "avg_brake": float(row.get("avg_brake") or 5.0),
        "hard_brake_rate": _hard_brake_rate_fn(float(row.get("hard_brake_count") or 0), avg_spd),
        "avg_drs": float(row.get("avg_drs") or 0.5),
        "avg_rpm": float(row.get("avg_rpm") or 10000.0),
        "track_temp": float(row.get("track_temp") or 35.0),
        "air_temp": float(row.get("air_temp") or 28.0),
        "rainfall": float(row.get("rainfall") or 0.0),
        "laps_remaining": _laps_remaining_fn(lap_no),
        "LapNumber": lap_no,
        "race_pct": lap_no / max(_state["total_laps"], 1),
        "speed_rank_pct": _state["speed_rank"].get(code, 0.5),
        "delta_vs_field": _state["delta_vs_field"].get(code, 0.0),
        "gap_to_below_proxy": float(_state["gap_to_below"].get(code, 0.0)),
        "teammate_pitted": int(_state["teammate_pitted"].get(code, False)),
    }


def winner_feats(code: str, lap_no: int) -> dict:
    """
    Build feature vector for the win probability model.

    Key fixes vs original:
      - sc_active: binary SC lap flag (was missing entirely)
      - position_jump_2/3: positions gained in 2/3 laps (SC beneficiary signal)
      - gap_to_leader_s: actual seconds behind leader from cumulative-time block
      - gap_laps_remaining: gap_s / laps_remaining (recovery difficulty)
      - is_leading: direct binary flag instead of implied by speed_rank_pct
      - position_history stores integer positions so jumps compute correctly
    """
    pr = _state["pre_race"].get(code, {})
    grid_pos = pr.get("grid_position", 10)

    grid_group = 1 if grid_pos <= 5 else (2 if grid_pos <= 15 else 3)

    speed_rank_pct = _state["speed_rank"].get(code, 0.5)
    n_drivers = max(_state.get("n_drivers", 20), 2)
    current_position = float(np.clip(speed_rank_pct * n_drivers, 1.0, float(n_drivers)))
    position_pct = float(
        np.clip(1.0 - ((current_position - 1.0) / max(n_drivers - 1, 1)), 0.0, 1.0)
    )

    is_leading = int(current_position <= 1.5)

    sc_active = int(_state.get("sc_active", False))

    pos_hist = list(_state.get("position_history", {}).get(code, []))

    if len(pos_hist) >= 2:
        position_jump_2 = float(pos_hist[-2] - current_position)
    else:
        position_jump_2 = 0.0

    if len(pos_hist) >= 3:
        position_jump_3 = float(pos_hist[-3] - current_position)
    else:
        position_jump_3 = 0.0

    sc_beneficiary = int(sc_active == 1 and position_jump_2 > 2.0)

    gap_to_leader_s = float(_state.get("gap_to_leader", {}).get(code, 0.0))

    laps_remaining = _laps_remaining_fn(lap_no)
    gap_laps_remaining = gap_to_leader_s / max(laps_remaining, 1)

    total_laps = max(_state.get("total_laps", 53), 1)
    lap_progress = lap_no / total_laps
    is_late_race = 1 if lap_no > total_laps * 0.75 else 0

    delta_vs_field = float(_state["delta_vs_field"].get(code, 0.0))
    gap_urgency = delta_vs_field / max(laps_remaining, 1)

    leading_and_late = position_pct * lap_progress

    comp = _state["current_compound"].get(code, "UNKNOWN")
    tire_age_pct = _tire_age_pct_fn(comp, _state["tire_age"][code])
    tire_freshness = 1.0 / (1.0 + tire_age_pct)

    position_gain_pct = position_pct - (1.0 - min(grid_pos / 20.0, 1.0))

    return {
        "grid_position": grid_pos,
        "grid_position_group": grid_group,
        "avg_finish_last5": pr.get("avg_finish_last5", 10.0),
        "points_last5": pr.get("points_last5", 0.0),
        "dnf_rate_last5": pr.get("dnf_rate_last5", 0.2),
        "team_enc": _enc_team_fn(pr.get("team", "Unknown")),
        "best_quali_lap": pr.get("best_quali_lap", 90.0),
        "track_type_enc": _enc_track_type_fn(_state["event"]),
        "laps_remaining": laps_remaining,
        "speed_rank_pct": speed_rank_pct,
        "delta_vs_field": delta_vs_field,
        "tire_age": _state["tire_age"][code],
        "tire_age_pct": tire_age_pct,
        "compound_enc": _enc_compound_fn(comp),
        "current_position": current_position,
        "position_pct": position_pct,
        "is_leading": is_leading,
        "sc_active": sc_active,
        "position_jump_2": position_jump_2,
        "position_jump_3": position_jump_3,
        "sc_beneficiary": sc_beneficiary,
        "gap_to_leader_s": gap_to_leader_s,
        "gap_laps_remaining": gap_laps_remaining,
        "lap_progress": lap_progress,
        "is_late_race": is_late_race,
        "gap_urgency": gap_urgency,
        "leading_and_late": leading_and_late,
        "position_gain_pct": position_gain_pct,
        "tire_freshness": tire_freshness,
    }


def tire_feats(code: str, comp: str, row: dict) -> dict:
    """Extract tire degradation model features."""
    age = _state["tire_age"][code]
    avg_spd = float(row.get("avg_speed") or 200.0)
    stint_len = _stint_len_med_fn(comp)
    clean = _current_stint_clean_speeds_fn(code)
    pace_d5 = _pace_drop_from_clean_fn(clean, n=3)
    return {
        "compound_enc": _enc_compound_fn(comp),
        "tire_age": age,
        "tire_age_pct": _tire_age_pct_fn(comp, age),
        "stint_len_med": stint_len,
        "stint_laps_left": max(0.0, stint_len - age),
        "stint_progress": age / max(stint_len, 1.0),
        "laps_remaining": _laps_remaining_fn(_state["current_lap"]),
        "LapNumber": _state["current_lap"],
        "track_type_enc": _enc_track_type_fn(_state["event"]),
        "avg_throttle": float(row.get("avg_throttle") or 85.0),
        "avg_brake": float(row.get("avg_brake") or 5.0),
        "hard_brake_rate": _hard_brake_rate_fn(float(row.get("hard_brake_count") or 0), avg_spd),
        "avg_drs": float(row.get("avg_drs") or 0.5),
        "avg_rpm": float(row.get("avg_rpm") or 10000.0),
        "track_temp": float(row.get("track_temp") or 35.0),
        "air_temp": float(row.get("air_temp") or 28.0),
        "rainfall": float(row.get("rainfall") or 0),
        "delta_vs_field": _state["delta_vs_field"].get(code, 0.0),
        "speed_rank_pct": _state["speed_rank"].get(code, 0.5),
        "gap_to_below_proxy": float(_state["gap_to_below"].get(code, 0.0)),
        "pace_drop_5": pace_d5,
        "delta_vs_teammate": _state["delta_vs_teammate"].get(code, 0.0),
    }


def ranking_feats(code: str, lap_no: int) -> dict:
    """
    Build features for ranking distribution model (multinomial finishing position).
    Uses same core features as winner model but applied to final-lap context.
    """
    pr = _state["pre_race"].get(code, {})
    grid_pos = pr.get("grid_position", 10)

    speed_rank_pct = _state["speed_rank"].get(code, 0.5)
    n_drivers = max(_state.get("n_drivers", 20), 2)
    current_position = float(np.clip(speed_rank_pct * n_drivers, 1.0, float(n_drivers)))
    position_pct = float(np.clip(1.0 - ((current_position - 1.0) / max(n_drivers - 1, 1)), 0.0, 1.0))

    total_laps = _state.get("total_laps", 53)
    lap_progress = lap_no / max(total_laps, 1)

    delta_vs_field = float(_state["delta_vs_field"].get(code, 0.0))
    laps_remaining = _laps_remaining_fn(lap_no)
    gap_urgency = delta_vs_field * (1.0 / max(laps_remaining, 1))

    n_grid = 22.0
    grid_pct_front = (1.0 - min(grid_pos / n_grid, 1.0))
    position_gain = position_pct - grid_pct_front

    tire_age_pct = _tire_age_pct_fn(
        _state["current_compound"].get(code, "UNKNOWN"),
        _state["tire_age"][code]
    )
    tire_freshness = 1.0 / (1.0 + tire_age_pct)

    return {
        "grid_position": grid_pos,
        "avg_finish_last5": pr.get("avg_finish_last5", 10.0),
        "points_last5": pr.get("points_last5", 0.0),
        "dnf_rate_last5": pr.get("dnf_rate_last5", 0.2),
        "team_enc": _enc_team_fn(pr.get("team", "Unknown")),
        "best_quali_lap": pr.get("best_quali_lap", 90.0),
        "track_type_enc": _enc_track_type_fn(_state["event"]),
        "speed_rank_pct": speed_rank_pct,
        "delta_vs_field": delta_vs_field,
        "tire_age": _state["tire_age"][code],
        "tire_age_pct": tire_age_pct,
        "compound_enc": _enc_compound_fn(_state["current_compound"].get(code, "UNKNOWN")),
        "lap_progress": lap_progress,
        "gap_urgency": gap_urgency,
        "tire_freshness": tire_freshness,
        "position_gain_pct": position_gain,
    }


def pit_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    """
    Extract pit stop decision model features.
    
    [FIX-F2] pace_drop_5 uses current-stint-only clean speeds.
    [FIX-F4] gap_to_below_proxy now populated from state.
    [FIX-F7] must_change_compound aligned with training.
    """
    feats = tire_feats(code, comp, row)
    feats["Stint"] = int(_state["current_stint"].get(code, 1))
    feats["teammate_pitted"] = int(_state["teammate_pitted"].get(code, False))
    feats["must_change_compound"] = must_change_compound(code, lap_no)
    feats["track_type_enc"] = _enc_track_type_fn(_state["event"])
    feats["gap_to_below_proxy"] = float(_state["gap_to_below"].get(code, 0.0))

    avg_s = float(row.get("avg_speed") or 0.0)
    ref_s = float(_state["stint_ref_speed"].get(code, 0.0) or 0.0)
    feats["rel_speed_delta"] = ((avg_s - ref_s) / max(ref_s, 1.0)) if ref_s > 0 and avg_s > 0 else 0.0

    clean = _current_stint_clean_speeds_fn(code)
    feats["pace_drop_5"] = _pace_drop_from_clean_fn(clean, n=3)

    return feats


def must_change_compound(code: str, lap_no: int) -> int:
    """
    [FIX-F7] Align with training definition.
    
    Training used: (Stint == 1) & (race_pct > 0.60)
    i.e. the driver is still on their FIRST stint past 60% of the race.
    """
    stint = int(_state["current_stint"].get(code, 1))
    if stint != 1:
        return 0
    
    total_laps = max(_state.get("total_laps", 53), 1)
    race_pct = lap_no / total_laps
    return 1 if race_pct > 0.60 else 0
