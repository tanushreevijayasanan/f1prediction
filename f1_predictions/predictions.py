"""
Prediction functions for ranking and win probability models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

# These will be injected by inference_engine.py
_state = None
_HAS_RANKING_MODEL = False
_HAS_MONTE_CARLO = False
_ranking_model = None
_laptime_model = None
_pit_model = None
_RANKING_FEATS = []
_LAPTIME_FEATS = []
_log = None


def set_predictions_context(HAS_RANKING_MODEL, HAS_MONTE_CARLO, ranking_model, laptime_model, pit_model,
                            RANKING_FEATS, LAPTIME_FEATS, state, log=None):
    """Initialize predictions with models and state."""
    global _HAS_RANKING_MODEL, _HAS_MONTE_CARLO, _ranking_model, _laptime_model, _pit_model
    global _RANKING_FEATS, _LAPTIME_FEATS, _state, _log
    
    _HAS_RANKING_MODEL = HAS_RANKING_MODEL
    _HAS_MONTE_CARLO = HAS_MONTE_CARLO
    _ranking_model = ranking_model
    _laptime_model = laptime_model
    _pit_model = pit_model
    _RANKING_FEATS = RANKING_FEATS
    _LAPTIME_FEATS = LAPTIME_FEATS
    _state = state
    _log = log


def compute_ranking_probabilities(lap_no: int, ranking_feats_fn, predict_fn) -> Dict[str, Dict[str, float]]:
    """
    Compute finishing position probability distributions using ranking model.
    
    For each driver, outputs:
    - p1, p2, ..., p20: finish position probabilities
    - podium: P(finish ≤ 3)
    - top5: P(finish ≤ 5)
    - top10: P(finish ≤ 10)
    - points: P(finish ≤ 10) [F1 points awarded to top 10]
    - expected_position: E[finishing position]
    - win_probability: same as p1
    
    [FIX-DNF] Skip drivers with status != "ACTIVE"
    Returns: {driver_code -> {metric_name -> probability}}
    """
    if not _HAS_RANKING_MODEL:
        return {}
    
    result = {}
    active_only = {c: _state["driver_status"].get(c, "ACTIVE") == "ACTIVE" 
                   for c in _state["pre_race"].keys()}
    
    try:
        for code in _state["active_drivers"]:
            if not active_only.get(code, True):
                continue

            feats = ranking_feats_fn(code, lap_no)
            feat_df = pd.DataFrame([feats])[_RANKING_FEATS].fillna(0)

            prob_dist = _ranking_model.predict_proba(feat_df)[0]
            n_classes = len(prob_dist)

            position_probs = {f"p{i+1}": float(prob_dist[i]) for i in range(n_classes)}
            for i in range(n_classes, 20):
                position_probs[f"p{i+1}"] = 0.0

            podium_p = float(sum(prob_dist[0:min(3, n_classes)]))
            top5_p = float(sum(prob_dist[0:min(5, n_classes)]))
            top10_p = float(sum(prob_dist[0:min(10, n_classes)]))
            expected_pos = float(sum((i + 1) * p for i, p in enumerate(prob_dist)))

            result[code] = {
                **position_probs,
                "podium": podium_p,
                "top5": top5_p,
                "top10": top10_p,
                "points": top10_p,
                "expected_position": expected_pos,
                "win_probability": float(prob_dist[0]) if len(prob_dist) > 0 else 0.0,
            }
    except Exception as e:
        if _log:
            _log.warning(f"Ranking probabilities failed at lap {lap_no}: {e}")
        return {}
    
    return result


def calibrate_late_race_probabilities(result: Dict[str, Any], lap_no: int, total_laps: int,
                                      log=None) -> Dict[str, Any]:
    """
    Calibrate win probabilities in late race to favor the leader.
    
    Applies late-race dynamics: leader's probability rises based on gap and race completion.
    """
    race_completion_pct = lap_no / max(total_laps, 1)
    is_late_race = race_completion_pct > 0.70
    
    if log:
        log.info(f"LAP {lap_no}: Calibration check - completion {race_completion_pct:.2%}, late_race={is_late_race}")
    
    if not is_late_race or not result:
        return result
    
    sample_val = next(iter(result.values()))
    nested_shape = isinstance(sample_val, dict)
    if nested_shape:
        return result

    win_probs = {
        code: float(prob)
        for code, prob in result.items()
        if isinstance(prob, (int, float, np.floating))
    }

    if not win_probs:
        return result

    leader = min(win_probs.keys(), key=lambda k: _state.get("speed_rank", {}).get(k, 999))
    leader_rank = _state.get("speed_rank", {}).get(leader, 999)
    
    if log:
        log.info(f"LAP {lap_no}: Leader identified as {leader} (rank {leader_rank})")
    
    progress = np.clip((race_completion_pct - 0.70) / 0.30, 0.0, 1.0)
    gap_s = float(_state.get("gap_to_leader", {}).get(leader, 0.0) or 0.0)
    gap_factor = np.clip(gap_s / 6.0, 0.0, 1.0)
    leader_boost = (0.03 + (0.22 * progress)) * (0.6 + 0.4 * gap_factor)
    
    if log:
        log.info(f"LAP {lap_no}: Leader boost = {leader_boost:.3f} (progress={progress:.3f})")
    
    calibrated_probs: Dict[str, float] = {}
    reduction_factor = max(0.70, 1.0 - (leader_boost * 0.55))

    for code, old_win_prob in win_probs.items():
        if code == leader:
            new_win_prob = min(0.92, old_win_prob + leader_boost)
            calibrated_probs[code] = new_win_prob
            
            if log:
                log.info(f"LAP {lap_no}: {code} win prob: {old_win_prob:.3f} -> {new_win_prob:.3f}")
        else:
            new_win_prob = old_win_prob * reduction_factor
            calibrated_probs[code] = new_win_prob
            
            if log and abs(old_win_prob - new_win_prob) > 0.01:
                log.info(f"LAP {lap_no}: {code} win prob: {old_win_prob:.3f} -> {new_win_prob:.3f}")
    
    total_win_prob = sum(calibrated_probs.values())
    if total_win_prob > 0:
        for code in list(calibrated_probs.keys()):
            calibrated_probs[code] /= total_win_prob

    for code, p in calibrated_probs.items():
        result[code] = p
    
    if log:
        leader_prob = float(result.get(leader, 0.0))
        log.info(f"LAP {lap_no}: Calibration complete - leader {leader} now has {leader_prob:.1%} win prob")
    
    return result


def run_monte_carlo_simulation(lap_no: int, MonteCarloRaceSimulator, TireDegradationModel,
                               n_simulations: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Run Monte Carlo simulation of remaining race to predict outcome probability distributions.
    
    Only runs periodically (every 5 laps after lap 20) to avoid computational overhead.
    Returns same format as compute_ranking_probabilities.
    """
    if not _HAS_MONTE_CARLO:
        return {}
    
    if lap_no < 20 or lap_no - _state.get("last_model_eval_lap", 0) < 5:
        return {}
    
    try:
        tire_deg_model = TireDegradationModel()
        sim = MonteCarloRaceSimulator(
            laptime_model=_laptime_model,
            tire_model=tire_deg_model,
            pit_model=_pit_model,
            current_state=_state,
            n_simulations=n_simulations,
        )
        
        results = sim.simulate()
        _state["last_model_eval_lap"] = lap_no
        _state["monte_carlo_results"] = results
        
        return results
    
    except Exception as e:
        if _log:
            _log.warning(f"Monte Carlo simulation failed: {e}")
        return {}
