from collections import defaultdict, deque


def _bounded_defaultdict(maxsize):
    class BoundedDefaultDict(defaultdict):
        def __missing__(self, key):
            val = deque(maxlen=maxsize)
            self[key] = val
            return val
    return BoundedDefaultDict(list)


def create_initial_state(mem_limit_speeds, mem_limit_lap_hist, mem_limit_stint_spd, mem_limit_pos_hist):
    return {
        "event": "",
        "year": 0,
        "total_laps": 0,
        "pre_race": {},
        "_num_to_code": {},
        "current_lap": 0,
        "n_drivers": 20,
        "tire_age": defaultdict(int),
        "current_compound": {},
        "current_stint": defaultdict(int),
        "stint_speed_samples": _bounded_defaultdict(mem_limit_stint_spd),
        "stint_ref_speed": defaultdict(float),
        "lap_history": _bounded_defaultdict(mem_limit_lap_hist),
        "pace_trend": _bounded_defaultdict(mem_limit_lap_hist),
        "obs_deg": {},
        "model_deg": {},
        "pit_prob": defaultdict(float),
        "pit_alert": defaultdict(bool),
        "teammate_pitted": defaultdict(bool),
        "win_proba": {},
        "pace_score": {},
        "speed_rank": {},
        "speed_rank_ema": {},
        "delta_vs_field": {},
        "delta_vs_teammate": {},
        "gap_to_below": {},
        "position_history": _bounded_defaultdict(mem_limit_pos_hist),
        "constructor_state": defaultdict(dict),
        "suggestions": [],
        "corner_alerts": [],
        "pattern_insights": [],
        "sc_active": False,
        "sc_end_lap": 0,
        "sc_laps": 0,
        "race_p75_speed": 0.0,
        "all_speeds_ever": deque(maxlen=mem_limit_speeds),
        "last_rainfall": None,
        "compounds_used": defaultdict(set),
        "predicted_laptime": {},
        "current_laptime": {},
        "fan_commentary": [],
        "speed_collapsed": set(),
        "win_proba_ema": {},
        "position_probabilities": {},
        "monte_carlo_results": {},
        "last_model_eval_lap": 0,
        "driver_status": {},
        "active_drivers": set(),
        "eligible_drivers": set(),
        "dnf_lap": {},
        "last_seen_lap": defaultdict(int),
        "inactive_lap_streak": defaultdict(int),
        "drs_proximity": {},
        "drs_available": {},
        "weather_state": "CLEAR",
        "fuel_state": {},
        "tire_delta_to_leader": {},
        "constructor_delta": {},
        "damage_estimate": {},
        "cumulative_time": defaultdict(float),
        "laps_completed": defaultdict(int),
        "_cumul_laps_counted": 0,
    }

