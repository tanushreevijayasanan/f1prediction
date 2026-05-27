import sys, os, logging, time, re
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests
try:
    from nltk.tokenize import wordpunct_tokenize
    from nltk.stem import PorterStemmer
    _nltk_stemmer = PorterStemmer()
    HAS_NLTK = True
except Exception:
    wordpunct_tokenize = None
    _nltk_stemmer = None
    HAS_NLTK = False
                                               
try:
    from config_loader import get_config, get
    CONFIG = get_config()
except Exception as e:
    logging.warning(f"Could not load config.yaml: {e}. Using hardcoded defaults.")
    CONFIG = {}
    get = lambda path, default=None: default

try:
    from config import parse_cli_args
except ImportError:
    from f1_predictions.config import parse_cli_args

try:
    from clickhouse_io import (
        load_prerace_from_clickhouse as io_load_prerace_from_clickhouse,
        clickhouse_lap_to_rows as io_clickhouse_lap_to_rows,
        load_cumulative_times as io_load_cumulative_times,
        clickhouse_max_lap as io_clickhouse_max_lap,
    )
except ImportError:
    from f1_predictions.clickhouse_io import (
        load_prerace_from_clickhouse as io_load_prerace_from_clickhouse,
        clickhouse_lap_to_rows as io_clickhouse_lap_to_rows,
        load_cumulative_times as io_load_cumulative_times,
        clickhouse_max_lap as io_clickhouse_max_lap,
    )

try:
    from models_runtime import load_runtime_artifacts
except ImportError:
    from f1_predictions.models_runtime import load_runtime_artifacts
try:
    from state import create_initial_state
except ImportError:
    from f1_predictions.state import create_initial_state
try:
    from constants import RACE_LAPS, COMPOUND_COLOR, COMPOUND_SYMBOL, COMPOUND_DEG_RATE, MEDALS, SPARK_BARS
except ImportError:
    from f1_predictions.constants import RACE_LAPS, COMPOUND_COLOR, COMPOUND_SYMBOL, COMPOUND_DEG_RATE, MEDALS, SPARK_BARS
try:
    from inference_utils import canonical_event_name, normalize_driver_code, safe_float
except ImportError:
    from f1_predictions.inference_utils import canonical_event_name, normalize_driver_code, safe_float

args = parse_cli_args()
HISTORY_WINDOW = args.history
if args.log:
    logging.basicConfig(filename=args.log, level=logging.INFO,
                        format="%(asctime)s  %(message)s")
    log = logging.getLogger("f1")
else:
    log = None

logger = logging.getLogger("f1_probabilities")                                           
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
console = Console(highlight=False) if HAS_RICH else None
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(_BASE_DIR, "models"))

                                 
_pred_client = None
_pred_table = None
UI_API_TOKEN = os.getenv("UI_API_TOKEN", "").strip()


def _is_safe_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", str(name or "")))


def _status_msg(msg: str, color: str = "green") -> None:
    (console.print(f"[{color}]{msg}[/{color}]") if HAS_RICH else print(msg))


def _warn_msg(msg: str) -> None:
    _status_msg(msg, "yellow")


def _fail_msg(msg: str) -> None:
    _status_msg(msg, "red")


try:
    _artifacts = load_runtime_artifacts(
        model_dir=MODEL_DIR,
        manifest_path=os.getenv("MODEL_HASH_MANIFEST", "").strip(),
        strict=os.getenv("MODEL_HASH_STRICT", "0").strip().lower() in {"1", "true", "yes"},
        status=lambda m: _status_msg(m, "green"),
        fail=_fail_msg,
        warn=_warn_msg,
    )
except SystemExit:
    raise
except Exception as e:
    _fail_msg(f"✗ Model loading failed: {e}")
    sys.exit(1)

winner_model = _artifacts.winner_model
tire_model = _artifacts.tire_model
pit_model = _artifacts.pit_model
pace_model = _artifacts.pace_model
le_team = _artifacts.le_team
COMPOUND_CLASSES = _artifacts.compound_classes
MEDIAN_STINT_LENGTHS = _artifacts.median_stint_lengths
WINNER_FEATS = _artifacts.winner_feats
TIRE_FEATS = _artifacts.tire_feats
PIT_FEATS = _artifacts.pit_feats
SC_THRESHOLD = _artifacts.sc_threshold
TRACK_TYPE_MAP = _artifacts.track_type_map
TRACK_TYPE_ENCODER = _artifacts.track_type_encoder
PIT_HORIZON = _artifacts.pit_horizon
PIT_ALERT_THRESHOLD = _artifacts.pit_alert_threshold
laptime_model = _artifacts.laptime_model
LAPTIME_FEATS = _artifacts.laptime_feats
CIRCUIT_LENGTHS = _artifacts.circuit_lengths
LAPTIME_MAE_S = _artifacts.laptime_mae_s
ranking_model = _artifacts.ranking_model
RANKING_FEATS = _artifacts.ranking_feats
HAS_RANKING_MODEL = _artifacts.has_ranking_model
try:
    from monte_carlo_simulator import MonteCarloRaceSimulator, TireDegradationModel, format_prob_distribution
    HAS_MONTE_CARLO = True
    msg = f"✓ Monte Carlo simulator enabled"
    (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
except ImportError as e:
    HAS_MONTE_CARLO = False
    msg = f"⚠ Monte Carlo simulator not available: {e}"
    (console.print(f"[yellow]{msg}[/yellow]") if HAS_RICH else print(msg))

COMPOUND_MAP = {c: i for i, c in enumerate(COMPOUND_CLASSES)}

                                               
BLEND_MAX            = get("inference.blend_max", 0.70)
BLEND_RAMP_LAPS      = get("inference.blend_ramp_laps", 12)
COLLAPSE_THRESHOLD   = get("inference.collapse_threshold", 20.0)
SC_FIELD_DROP        = get("inference.sc_field_drop", 0.85)
RAINFALL_ONSET_DELTA = get("inference.rainfall_onset_delta", 0.05)
RANKING_PRIOR_WEIGHT = min(max(float(get("inference.ranking_prior_weight", 0.35)), 0.0), 1.0)
LIVE_ORDER_BLEND_MAX = min(max(float(get("inference.live_order_blend_max", 0.10)), 0.0), 0.30)

                                                                           
                                               
GRID_PRIOR_LAPS      = max(0, int(args.grid_prior_laps or get("inference.grid_prior_laps", 3)))
GRID_BLEND_LAPS      = max(GRID_PRIOR_LAPS, int(args.grid_blend_laps or get("inference.grid_blend_laps", 20)))
GRID_PRIOR_WEIGHT    = min(max(float(args.grid_prior_weight or get("inference.grid_prior_weight", 0.95)), 0.0), 1.0)
GRID_POSITION_REG    = min(max(float(args.grid_position_regularization or get("inference.grid_position_regularization", 0.15)), 0.0), 0.5)
MIN_ACTIVE_FOR_RANK  = max(1, int(args.min_active_for_speed_rank or get("inference.min_active_for_speed_rank", 10)))
SPEED_RANK_EMA_ALPHA = min(max(float(args.speed_rank_ema or get("inference.speed_rank_ema", 0.35)), 0.0), 1.0)
WIN_PROBA_EMA_ALPHA  = min(max(float(args.win_proba_ema or get("inference.win_proba_ema", 0.15)), 0.0), 1.0)
WIN_PROBA_MAX_DELTA  = max(0.0, float(args.win_proba_max_delta or get("inference.win_proba_max_delta", 0.03)))

                                                        
                                                                               
                                                                                      
PIT_STOP_LOSS_S     = get("inference.pit_stop_loss_s",       23.0)
CUMUL_TIME_MIN_LAPS = int(get("inference.cumulative_time_min_laps", 5))
DNS_DETECT_LAP      = int(get("inference.dns_detect_lap", 5))
DNF_MISSING_LAPS    = int(get("inference.dnf_missing_laps", 5))

                                                                                     
MEM_LIMIT_SPEEDS = get("inference.state_memory.all_speeds_keep_last", 100)
MEM_LIMIT_LAP_HIST = get("inference.state_memory.lap_history_keep_last", 30)
MEM_LIMIT_STINT_SPD = get("inference.state_memory.stint_speeds_keep_last", 50)
MEM_LIMIT_PIT_HIST = get("inference.state_memory.pit_history_keep_last", 20)
MEM_LIMIT_POS_HIST = get("inference.state_memory.position_history_keep_last", 50)

state: dict = create_initial_state(
    mem_limit_speeds=MEM_LIMIT_SPEEDS,
    mem_limit_lap_hist=MEM_LIMIT_LAP_HIST,
    mem_limit_stint_spd=MEM_LIMIT_STINT_SPD,
    mem_limit_pos_hist=MEM_LIMIT_POS_HIST,
)

# Import new modules for separation of concerns
try:
    import utilities
    import feature_extraction
    import predictions
    import state_management
    import commentary as commentary_module
except ImportError:
    import f1_predictions.utilities as utilities
    import f1_predictions.feature_extraction as feature_extraction
    import f1_predictions.predictions as predictions
    import f1_predictions.state_management as state_management
    import f1_predictions.commentary as commentary_module

# Initialize module contexts
utilities.set_utilities_context(
    state=state,
    CIRCUIT_LENGTHS=CIRCUIT_LENGTHS,
    MEDIAN_STINT_LENGTHS=MEDIAN_STINT_LENGTHS,
    GRID_PRIOR_LAPS=GRID_PRIOR_LAPS,
    GRID_BLEND_LAPS=GRID_BLEND_LAPS,
    GRID_POSITION_REG=GRID_POSITION_REG,
    MIN_ACTIVE_FOR_RANK=MIN_ACTIVE_FOR_RANK,
    SPARK_BARS=SPARK_BARS,
    COMPOUND_COLOR=COMPOUND_COLOR,
    COMPOUND_SYMBOL=COMPOUND_SYMBOL,
)

# Encoding helpers for feature extraction
def _enc_compound(val: str) -> int:
    return COMPOUND_MAP.get(str(val).upper().replace("INTERMEDIATE", "INTER"), 0)

def _enc_track_type(event: str) -> int:
    ev = canonical_event_name(event)
    t = TRACK_TYPE_MAP.get(ev, "UNKNOWN")
    try:
        return int(TRACK_TYPE_ENCODER.transform([t])[0])
    except Exception:
        return 0

def _enc_team(val: str) -> int:
    try:
        return int(le_team.transform([str(val)])[0])
    except Exception:
        return 0

feature_extraction.set_feature_extraction_context(
    enc_compound=_enc_compound,
    enc_track_type=_enc_track_type,
    enc_team=_enc_team,
    tire_age_pct=utilities.tire_age_pct,
    stint_len_med=utilities.stint_len_med,
    hard_brake_rate=utilities.hard_brake_rate,
    laps_remaining=utilities.laps_remaining,
    current_stint_clean_speeds=utilities.current_stint_clean_speeds,
    pace_drop_from_clean=utilities.pace_drop_from_clean,
    state=state,
)

state_management.set_state_management_context(
    state=state,
    is_sc_lap_fn=utilities.is_sc_lap,
    SC_THRESHOLD=SC_THRESHOLD,
    SC_FIELD_DROP=SC_FIELD_DROP,
    RAINFALL_ONSET_DELTA=RAINFALL_ONSET_DELTA,
    log=log,
)

predictions.set_predictions_context(
    HAS_RANKING_MODEL=HAS_RANKING_MODEL,
    HAS_MONTE_CARLO=HAS_MONTE_CARLO,
    ranking_model=ranking_model,
    laptime_model=laptime_model,
    pit_model=pit_model,
    RANKING_FEATS=RANKING_FEATS,
    LAPTIME_FEATS=LAPTIME_FEATS,
    state=state,
    log=log,
)

commentary_module.set_commentary_context(
    state=state,
    HAS_NLTK=HAS_NLTK,
    wordpunct_tokenize=wordpunct_tokenize,
    nltk_stemmer=_nltk_stemmer,
    get_current_laptime_fn=utilities.get_current_laptime,
    get_formatted_team_fn=utilities.get_formatted_team,
    laptime_str_fn=utilities.laptime_str,
)

                                                                                

def _enc_compound(val: str) -> int:
    return COMPOUND_MAP.get(str(val).upper().replace("INTERMEDIATE", "INTER"), 0)

def _canonical_event_name(event: str) -> str:
    return canonical_event_name(event)

def _enc_track_type(event: str) -> int:
    ev = _canonical_event_name(event)
    t = TRACK_TYPE_MAP.get(ev, "UNKNOWN")
    try:
        return int(TRACK_TYPE_ENCODER.transform([t])[0])
    except Exception:
        return 0

def _enc_team(val: str) -> int:
    try:
        return int(le_team.transform([str(val)])[0])
    except Exception:
        return 0

def _predict(feats: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame([feats])[cols].fillna(0)


def _normalize_driver_code(val) -> str:
    """Wrapper for inference_utils.normalize_driver_code"""
    return normalize_driver_code(val)

def _safe_float(v, default=0.0) -> float:
    """Wrapper for inference_utils.safe_float"""
    return safe_float(v, default=default)

def _driver_code_from_row(row: dict) -> str:
    """Extract driver code from telemetry row"""
    for field in ("driver_code", "driver_name"):
        code = _normalize_driver_code(row.get(field))
        if code:
            return code
    driver_num = str(row.get("driver_num") or row.get("driver") or "").strip()
    if driver_num:
        mapped = _normalize_driver_code(state.get("_num_to_code", {}).get(driver_num))
        if mapped:
            return mapped
    return ""

def _row_is_valid_activity(row: dict) -> bool:
    """Activity guard for live inference"""
    code = _driver_code_from_row(row)
    if not code:
        return False
    spd = _safe_float(row.get("avg_speed"), 0.0)
    if spd <= 50.0:
        return False
    has_signal = any(
        row.get(k) is not None
        for k in ("lap_finish_ms", "avg_throttle", "avg_rpm", "avg_brake", "avg_drs")
    )
    return bool(has_signal)

# Utility wrappers that delegate to the utilities module
def _sparkline(values: list, width: int = 10) -> str:
    return utilities.sparkline(values, width)

def _grid_rank_map(codes: list[str]) -> dict:
    return utilities.grid_rank_map(codes)

def _grid_blend_weight(lap_no: int, n_active: int) -> float:
    return utilities.grid_blend_weight(lap_no, n_active)

def _laps_remaining(lap_no: int) -> int:
    return utilities.laps_remaining(lap_no)

def _tire_age_pct(compound: str, age: int) -> float:
    return utilities.tire_age_pct(compound, age)

def _stint_len_med(compound: str) -> float:
    return utilities.stint_len_med(compound)

def _hard_brake_rate(hard_brake_count: float, avg_speed: float) -> float:
    return utilities.hard_brake_rate(hard_brake_count, avg_speed)

def _speed_to_laptime(speed_kmh: float, event: str) -> float:
    return utilities.speed_to_laptime(speed_kmh, event)

def _laptime_str(lt_s: float) -> str:
    return utilities.laptime_str(lt_s)

def _lt(code: str) -> float:
    return utilities.get_current_laptime(code)

def _lt_str(lt_s: float) -> str:
    return utilities.laptime_str(lt_s)

def _fteam(code: str) -> str:
    return utilities.get_formatted_team(code)

def _current_stint_clean_speeds(code: str) -> list[float]:
    return utilities.current_stint_clean_speeds(code)

def _pace_drop_from_clean(clean_speeds: list[float], n: int = 3) -> float:
    return utilities.pace_drop_from_clean(clean_speeds, n)

def _field_median_speed(tele_rows: list) -> float:
    return utilities.field_median_speed(tele_rows)

def _is_sc_lap(field_median: float) -> bool:
    return utilities.is_sc_lap(field_median, SC_THRESHOLD, SC_FIELD_DROP)

def _predict(feats: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame([feats])[cols].fillna(0)

# Feature extraction wrappers that delegate to the feature_extraction module
def _laptime_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    return feature_extraction.laptime_feats(code, comp, row, lap_no)

def _must_change_compound(code: str, lap_no: int) -> int:
    return feature_extraction.must_change_compound(code, lap_no)

def _winner_feats(code: str, lap_no: int) -> dict:
    return feature_extraction.winner_feats(code, lap_no)

def _tire_feats(code: str, comp: str, row: dict) -> dict:
    return feature_extraction.tire_feats(code, comp, row)

def _ranking_feats(code: str, lap_no: int) -> dict:
    return feature_extraction.ranking_feats(code, lap_no)

def _pit_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    return feature_extraction.pit_feats(code, comp, row, lap_no)




# State management wrappers that delegate to the state_management module
def _update_sc_state(tele_rows: list, lap_no: int) -> None:
    state_management.update_sc_state(tele_rows, lap_no)

def _check_weather(tele_rows: list, lap_no: int) -> None:
    state_management.check_weather(tele_rows, lap_no)

# Prediction wrappers that delegate to the predictions module
def _compute_ranking_probabilities(lap_no: int) -> Dict[str, Dict[str, float]]:
    result = predictions.compute_ranking_probabilities(lap_no, _ranking_feats, _predict)
    
    # Additional calibration logic specific to inference_engine
    if not result:
        return {}
    
    for code in state["driver_status"].keys():
        if code not in result and state["driver_status"].get(code) in ("DNF", "DNS"):
            result[code] = {f"p{i+1}": 0.0 for i in range(20)}
            result[code]["status"] = state["driver_status"][code]
    
    return predictions.calibrate_late_race_probabilities(result, lap_no, state["total_laps"], log)

def _calibrate_late_race_probabilities(result: Dict[str, Any], lap_no: int, total_laps: int) -> Dict[str, Any]:
    return predictions.calibrate_late_race_probabilities(result, lap_no, total_laps, log)

def _run_monte_carlo_simulation(lap_no: int, n_simulations: int = 500) -> Dict[str, Dict[str, float]]:
    try:
        from monte_carlo_simulator import MonteCarloRaceSimulator, TireDegradationModel
        return predictions.run_monte_carlo_simulation(lap_no, MonteCarloRaceSimulator, TireDegradationModel, n_simulations)
    except ImportError:
        try:
            from f1_predictions.monte_carlo_simulator import MonteCarloRaceSimulator, TireDegradationModel
            return predictions.run_monte_carlo_simulation(lap_no, MonteCarloRaceSimulator, TireDegradationModel, n_simulations)
        except ImportError:
            return {}


            result[code]["dnf_lap"] = state["dnf_lap"].get(code, 0)
    
    return result


def _calibrate_late_race_probabilities(result: Dict[str, Any], lap_no: int, total_laps: int) -> Dict[str, Any]:
    """
    Apply realistic late-race probability calibration.
    
    At lap 52/53 (98% completion), leader should have 85-95% win probability.
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

                                                           
    leader = min(win_probs.keys(), key=lambda k: state.get("speed_rank", {}).get(k, 999))
    leader_rank = state.get("speed_rank", {}).get(leader, 999)
    
    if log:
        log.info(f"LAP {lap_no}: Leader identified as {leader} (rank {leader_rank})")
    
                                                     
    progress = np.clip((race_completion_pct - 0.70) / 0.30, 0.0, 1.0)
    gap_s = float(state.get("gap_to_leader", {}).get(leader, 0.0) or 0.0)
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


def _run_monte_carlo_simulation(lap_no: int, n_simulations: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Run Monte Carlo simulation of remaining race to predict outcome probability distributions.
    
    Only runs periodically (every 5 laps after lap 20) to avoid computational overhead.
    Returns same format as _compute_ranking_probabilities.
    """
    if not HAS_MONTE_CARLO:
        return {}
    
                                                      
    if lap_no < 20 or lap_no - state["last_model_eval_lap"] < 5:
        return {}
    
    try:
                              
        tire_deg_model = TireDegradationModel()
        sim = MonteCarloRaceSimulator(
            laptime_model=laptime_model,
            tire_model=tire_deg_model,
            pit_model=pit_model,
            current_state=state,
            n_simulations=n_simulations,
        )
        
                         
        results = sim.simulate()
        
                                    
        state["last_model_eval_lap"] = lap_no
        state["monte_carlo_results"] = results
        
        return results
    
    except Exception as e:
        logger.warning(f"Monte Carlo simulation failed: {e}")
        return {}


                                                                                

def _update_sc_state(tele_rows: list, lap_no: int) -> None:
    speeds = [float(r.get("avg_speed") or 0) for r in tele_rows if (r.get("avg_speed") or 0) > 50]
    if not speeds:
        return
    state["all_speeds_ever"].extend(speeds)
    if lap_no >= 5 and len(state["all_speeds_ever"]) > 100:
        state["race_p75_speed"] = np.percentile(list(state["all_speeds_ever"]), 75)
    field_med = float(np.median(speeds))
    was_sc    = state["sc_active"]
    state["sc_active"] = _is_sc_lap(field_med)
    if state["sc_active"]:
        state["sc_laps"] += 1
        if not was_sc:
            msg = f"🟡 SAFETY CAR DEPLOYED  lap {lap_no}  field speed {field_med:.0f} km/h"
            state["suggestions"].insert(0, msg)
                                                                                
                                                         
            state["predicted_laptime"].clear()
            if log:
                log.info(f"LAP {lap_no:3d} | {msg}")
    elif was_sc:
        msg = f"🟢 SAFETY CAR ENDING    lap {lap_no} — GREEN FLAG"
        state["suggestions"].insert(0, msg)
        state["sc_end_lap"] = lap_no
        if log:
            log.info(f"LAP {lap_no:3d} | {msg}")


def _check_weather(tele_rows: list, lap_no: int) -> None:
    rainfalls = [float(r.get("rainfall") or 0) for r in tele_rows]
    if not rainfalls:
        return
    current_rain = float(np.mean(rainfalls))
    last_rain    = state["last_rainfall"]
    
                                     
    if current_rain > 0.5:
        state["weather_state"] = "WET"
    elif current_rain > 0.1 and last_rain and last_rain > 0.3:
        state["weather_state"] = "WET_CLEARING"
    elif current_rain < 0.05:
        state["weather_state"] = "CLEAR"
    
    if last_rain is not None:
        delta = current_rain - last_rain
        if delta > RAINFALL_ONSET_DELTA:
            msg = (f"🌧  RAIN ONSET  lap {lap_no}  "
                   f"rainfall {last_rain:.2f}→{current_rain:.2f}  CONSIDER INTER/WET")
            state["suggestions"].insert(0, msg)
            state["weather_state"] = "RAIN_ONSET"
            if log:
                log.info(f"LAP {lap_no:3d} | {msg}")
        elif delta < -RAINFALL_ONSET_DELTA and last_rain > 0.1:
            msg = (f"☀️  TRACK DRYING  lap {lap_no}  "
                   f"rainfall {last_rain:.2f}→{current_rain:.2f}  SLICK WINDOW OPENING")
            state["suggestions"].insert(0, msg)
            state["weather_state"] = "WET_CLEARING"
            if log:
                log.info(f"LAP {lap_no:3d} | {msg}")
    state["last_rainfall"] = current_rain


                                                                                

_narr_cooldown: dict = {}

def _estimate_fuel_load(code: str, lap_no: int) -> float:
    """
    [FIX-FEAT] Estimate fuel load as % of full tank.
    Assumes ~1.8 kg fuel burned per lap at race pace.
    ~110 kg full tank → 61 laps at race pace.
    """
    hist = state["lap_history"].get(code, [])
    if not hist:
        return 0.95                             
    
    laps_used = len(hist)
                                       
    pit_count = state["current_stint"].get(code, 1) - 1
    
                                       
    laps_per_tank = 61.0
    fuel_remaining = max(0.0, 1.0 - (laps_used / laps_per_tank)) + (pit_count * 0.95)
    return min(1.0, fuel_remaining)


def _ncool(key: str, lap_no: int, min_laps: int = 4) -> bool:
    return lap_no - _narr_cooldown.get(key, -99) >= min_laps

def _nfire(key: str, lap_no: int) -> None:
    _narr_cooldown[key] = lap_no

def _ranked() -> list[str]:
    return [c for c, _ in sorted(state["speed_rank"].items(), key=lambda x: x[1])]

def _active(min_spd: float = 100.0) -> list[str]:
    return [c for c, t in state["pace_trend"].items() if t and t[-1] > min_spd]

_COMMENTARY_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "are", "was", "were",
    "into", "after", "before", "over", "under", "next", "lap", "field", "watch", "right",
    "more", "less", "than", "then", "will", "would", "could", "should", "now", "one", "two",
    "three", "four", "five", "driver", "drivers", "team", "teams"
}

_COMMENTARY_VARIANTS = [
    ("leads restart", ["controls restart", "heads restart"]),
    ("pits after", ["dives in after", "boxes after"]),
    ("pulling away", ["stretching the gap", "edging clear"]),
    ("CLOSING", ["HUNTING", "CLOSING IN"]),
    ("Next-lap predictions", ["Projected next-lap pace", "Next-lap pace forecast"]),
    ("undercut threat building", ["undercut window opening", "undercut pressure rising"]),
    ("Pit window", ["Stop window", "Pit phase"]),
    ("in trouble", ["under pressure", "in difficulty"])
]

def _stem_signature(text: str) -> set[str]:
    if not HAS_NLTK or not wordpunct_tokenize or _nltk_stemmer is None:
        return {w for w in text.lower().split() if len(w) > 2}
    toks = wordpunct_tokenize(text.lower())
    return {
        _nltk_stemmer.stem(t)
        for t in toks
        if t.isalpha() and len(t) > 2 and t not in _COMMENTARY_STOPWORDS
    }

def _apply_commentary_variation(text: str, lap_no: int, idx: int) -> str:
    out = " ".join(str(text).split())
    for base, choices in _COMMENTARY_VARIANTS:
        if base in out and choices:
            pick = choices[(lap_no + idx + len(out)) % len(choices)]
            out = out.replace(base, pick, 1)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out

def _nltk_refine_commentary(lines: list[str], lap_no: int) -> list[str]:
    if not lines:
        return []
    polished: list[str] = []
    signatures: list[set[str]] = []
    for idx, raw in enumerate(lines):
        line = _apply_commentary_variation(raw, lap_no, idx)
        sig = _stem_signature(line)
        is_dup = False
        for prev in signatures:
            union = len(sig | prev) or 1
            overlap = len(sig & prev) / union
            if overlap >= 0.78:
                is_dup = True
                break
        if is_dup:
            continue
        polished.append(line)
        signatures.append(sig)
    return polished[:7]

def _build_fan_commentary(lap_no: int) -> list[str]:
    stories: list[tuple[int, str]] = []
    ranked = _ranked()
    active = _active()

    if state["sc_active"] and _ncool("sc_active", lap_no, 1):
        old_tyres = [c for c in active if state["tire_age"].get(c, 0) >= 8]
        names = ", ".join(old_tyres[:4])
        tail  = f" {names} have a free stop." if names else ""
        stories.append((100, f"🟡 SAFETY CAR — field bunching up.{tail}"))
        _nfire("sc_active", lap_no)

    sc_end = state.get("sc_end_lap", 0)
    if not state["sc_active"] and sc_end == lap_no and _ncool("sc_end", lap_no, 1):
        p1 = ranked[0] if ranked else "?"
        p2 = ranked[1] if len(ranked) > 1 else "?"
        stories.append((99, f"🟢 GREEN FLAG! {p1} leads restart, {p2} right behind — watch for passes!"))
        _nfire("sc_end", lap_no)

    rain_alerts = [x for x in state["suggestions"] if "RAIN ONSET" in x or "TRACK DRYING" in x]
    if rain_alerts and _ncool("weather", lap_no, 2):
        slick_n = sum(1 for c in active if state["current_compound"].get(c,"") not in ("INTER","WET"))
        if "RAIN ONSET" in rain_alerts[0]:
            stories.append((98, f"🌧️ Rain starting! {slick_n} drivers on slicks — the intermediate call is open NOW."))
        else:
            inters = [c for c in active if state["current_compound"].get(c,"") in ("INTER","WET")]
            names  = ", ".join(inters[:3]) or "Several drivers"
            stories.append((97, f"☀️ Track drying — {names} must decide: box for slicks or gamble one more lap?"))
        _nfire("weather", lap_no)

    for code in list(state.get("speed_collapsed", set())):
        key = f"col_{code}"
        if not _ncool(key, lap_no, 3): continue
        lt_self  = _lt(code)
        all_lts  = [_lt(c) for c in active if _lt(c) > 0]
        lt_field = float(np.median(all_lts)) if all_lts else 0
        gap_s    = lt_self - lt_field
        stories.append((95, f"🚨 {code} ({_fteam(code)}) in trouble — losing {gap_s:.0f}s/lap to the field."))
        _nfire(key, lap_no)

    for alert in state["suggestions"]:
        if "🔵 PITTED" not in alert: continue
        try: code = alert.split("]")[1].strip().split()[0]
        except: continue
        key = f"pit_{code}_{lap_no}"
        if not _ncool(key, lap_no, 1): continue
        comp = state["current_compound"].get(code, "?")
        age  = state["tire_age"].get(code, 0)
        sc_c = " — under Safety Car!" if state.get("sc_active") else "."
        stories.append((90, f"🔵 {code} ({_fteam(code)}) pits after {age+1} laps, back on {comp.lower()}{sc_c}"))
        _nfire(key, lap_no)

    if len(ranked) >= 2 and _ncool("podium", lap_no, 3):
        p1, p2 = ranked[0], ranked[1]
        lt1, lt2 = _lt(p1), _lt(p2)
        if lt1 > 0 and lt2 > 0:
            gap = lt2 - lt1
            w   = state["win_proba"].get(p1, 0)
            if gap > 0.4:
                stories.append((88, f"🏁 {p1} ({_fteam(p1)}) pulling away — {gap:.2f}s/lap faster than {p2}. Win: {w:.0%}."))
            elif gap < -0.4:
                stories.append((88, f"🏁 {p2} ({_fteam(p2)}) CLOSING — {abs(gap):.2f}s/lap quicker than leader {p1}. Lead at risk."))
            else:
                stories.append((85, f"🏁 {p1} ({_fteam(p1)}) leads, {p2} within {abs(gap):.2f}s/lap. {w:.0%} win probability."))
        _nfire("podium", lap_no)

    if not state["sc_active"] and _ncool("lt_table", lap_no, 4):
        rows = []
        for code in ranked[:5]:
            curr = state["current_laptime"].get(code, 0)
            pred = state["predicted_laptime"].get(code, 0)
            if curr > 0 and pred > 0:
                diff  = pred - curr
                arrow = f"↑{abs(diff):.2f}s faster" if diff < -0.05 else (f"↓{abs(diff):.2f}s slower" if diff > 0.05 else "~same")
                rows.append(f"{code} {_lt_str(pred)} ({arrow})")
        if rows:
            mae_tag = f" ±{LAPTIME_MAE_S:.2f}s" if LAPTIME_MAE_S else ""
            stories.append((84, f"⏱️ Next-lap predictions{mae_tag}: " + "  |  ".join(rows)))
            _nfire("lt_table", lap_no)

    for alert in state["pattern_insights"]:
        if "UNDERCUT RISK" not in alert: continue
        try:
            older = alert.split("]")[1].strip().split("(")[0].strip()
            fresh = alert.split("vs")[1].strip().split("(")[0].strip()
        except: continue
        key = f"uc_{older}_{fresh}"
        if not _ncool(key, lap_no, 4): continue
        age_o, age_f = state["tire_age"].get(older, 0), state["tire_age"].get(fresh, 0)
        lt_o,  lt_f  = _lt(older), _lt(fresh)
        pace_adv = lt_o - lt_f
        stories.append((78, f"⚡ {_fteam(older)}: {older} (lap {age_o}) vs {fresh} (lap {age_f}). {fresh} {pace_adv:+.2f}s/lap quicker — undercut threat building."))
        _nfire(key, lap_no)

    for alert in state["pattern_insights"]:
        if "TYRE CLIFF" not in alert: continue
        try: code = alert.split("]")[1].strip().split()[0]
        except: continue
        key = f"cliff_{code}"
        if not _ncool(key, lap_no, 5): continue
        age  = state["tire_age"].get(code, 0)
        comp = state["current_compound"].get(code, "?")
        pred = state["predicted_laptime"].get(code, 0)
        curr = state["current_laptime"].get(code, 0)
        loss = f" — {pred-curr:+.2f}s forecast next lap." if pred > 0 and curr > 0 and abs(pred-curr) > 0.1 else "."
        stories.append((75, f"📉 {code} ({_fteam(code)}) hitting the cliff — {age} laps on {comp.lower()}{loss} Pit window urgent."))
        _nfire(key, lap_no)

    pit_flagged = [c for c in active if state["pit_alert"].get(c)]
    if pit_flagged and _ncool("pit_grp", lap_no, 3):
        names = ", ".join(pit_flagged[:4])
        tail  = " 🟡 FREE stop under SC!" if state["sc_active"] else "."
        stories.append((72, f"🔧 Pit window: {names} flagged by model{tail}"))
        _nfire("pit_grp", lap_no)

    fade = [a for a in state["suggestions"] if "PACE FADE" in a]
    if fade and _ncool("fade_grp", lap_no, 5):
        try:
            code = fade[0].split("]")[1].strip().split()[0]
            curr, pred = state["current_laptime"].get(code, 0), state["predicted_laptime"].get(code, 0)
            trend_str  = f" — next lap forecast {_lt_str(pred)}" if pred > 0 else ""
            stories.append((65, f"🐢 {code} ({_fteam(code)}) fading, lap times getting slower each stint lap{trend_str}."))
        except: pass
        _nfire("fade_grp", lap_no)

    stories.sort(key=lambda x: -x[0])
    seen, result = set(), []
    for _, text in stories:
        k = text[:40]
        if k not in seen:
            seen.add(k); result.append(text)
        if len(result) >= 7: break
    return result


                                                                                 

def run_inference(lap_no: int, tele_rows: list, corner_rows: list, is_tick: bool = False) -> None:
    state["current_lap"]      = lap_no
    state["suggestions"]      = []
    state["corner_alerts"]    = []
    state["pattern_insights"] = []
    state["speed_collapsed"]  = set()

    _update_sc_state(tele_rows, lap_no)
    _check_weather(tele_rows, lap_no)

    observed_speeds = {}
    for r in tele_rows:
        if not _row_is_valid_activity(r):
            continue
        code = _driver_code_from_row(r)
        observed_speeds[code] = _safe_float(r.get("avg_speed"), 0.0)

    driver_universe = set()
    driver_universe.update(state["driver_status"].keys())
    driver_universe.update(state["last_seen_lap"].keys())
    driver_universe.update(observed_speeds.keys())
    for _pr_code in state["pre_race"].keys():
        driver_universe.add(_pr_code)

    for code in driver_universe:
        status = state["driver_status"].get(code, "ACTIVE")
        if status in ("DNF", "DNS"):
            continue
        if code in observed_speeds:
            if state["driver_status"].get(code) not in ("DNF", "DNS"):
                state["driver_status"][code] = "ACTIVE"
                state["last_seen_lap"][code] = lap_no
                state["inactive_lap_streak"][code] = 0
        else:
            seen_lap = int(state["last_seen_lap"].get(code, 0))
            if seen_lap == 0:
                if lap_no >= DNS_DETECT_LAP:
                    state["driver_status"][code] = "DNS"
                    state["dnf_lap"].setdefault(code, 0)
                continue
            state["inactive_lap_streak"][code] = state["inactive_lap_streak"].get(code, 0) + 1
            _dnf_threshold = max(DNF_MISSING_LAPS, 5)
            if state["inactive_lap_streak"][code] >= _dnf_threshold:
                state["driver_status"][code] = "DNF"
                state["dnf_lap"].setdefault(code, seen_lap + 1)

    for code in observed_speeds.keys():
        state["driver_status"].setdefault(code, "ACTIVE")
        if state["driver_status"][code] == "ACTIVE":
            state["last_seen_lap"][code] = lap_no
            state["inactive_lap_streak"][code] = 0

    active_speeds = {
        c: s for c, s in observed_speeds.items()
        if state["driver_status"].get(c, "ACTIVE") == "ACTIVE"
    }
    state["active_drivers"] = set(active_speeds.keys())
    state["eligible_drivers"] = set(active_speeds.keys())

    all_lap_speeds = list(active_speeds.values())
    field_med_spd  = float(np.median(all_lap_speeds)) if all_lap_speeds else 0.0
    field_max_spd  = max(all_lap_speeds) if all_lap_speeds else 1.0
    n_active       = len(active_speeds)

                                               
                                                                 
    team_multipliers = {
        "Mercedes": 1.00, "McLaren": 1.02, "Ferrari": 1.01, "Red Bull Racing": 1.00,
        "Aston Martin": 0.99, "Alpine": 0.99, "Haas F1 Team": 0.98, "Williams": 0.98,
        "Racing Bulls": 0.98, "Audi": 0.97, "Cadillac": 0.97
    }
    
    adjusted_speeds = {}
    for code, speed in active_speeds.items():
        team = state.get("pre_race", {}).get(code, {}).get("team", "Unknown")
        compound = state["current_compound"].get(code, "MEDIUM")
        tire_age = state["tire_age"].get(code, 0)
        
        team_mult = team_multipliers.get(team, 1.0)
        
        compound_adj = {"HARD": 0.98, "MEDIUM": 1.0, "SOFT": 1.02}.get(compound, 1.0)
        age_penalty = max(0.90, 1.0 - (tire_age * 0.004))                                         
        
        adjusted_speeds[code] = speed * team_mult * compound_adj * age_penalty
    
    if lap_no <= 1:
        sorted_codes = sorted(
            adjusted_speeds.keys(),
            key=lambda c: (int(state.get("pre_race", {}).get(c, {}).get("grid_position", 999)), c)
        )
    else:
        sorted_codes = sorted(adjusted_speeds, key=lambda c: -adjusted_speeds[c])

    speed_rank_map = {code: (i + 1) / max(n_active, 1) for i, code in enumerate(sorted_codes)}
    grid_rank_map  = _grid_rank_map(list(active_speeds.keys()))

    state["n_drivers"] = n_active

    grid_w = _grid_blend_weight(lap_no, n_active) * GRID_PRIOR_WEIGHT
    for code in sorted_codes:
        sr = speed_rank_map.get(code, 0.5)
        gr = grid_rank_map.get(code, sr)
        blended = (grid_w * gr) + ((1.0 - grid_w) * sr)

        if lap_no <= 1:
            state["speed_rank_ema"][code] = sr
            state["speed_rank"][code] = sr
        else:
            prev = state["speed_rank_ema"].get(code, blended)
            ema  = (1.0 - SPEED_RANK_EMA_ALPHA) * prev + SPEED_RANK_EMA_ALPHA * blended
            state["speed_rank_ema"][code] = ema
            state["speed_rank"][code]     = ema

        state["delta_vs_field"][code] = active_speeds[code] - field_med_spd
        
    
                                            
                                                        
                                         
                                                                 
                                                                                 
    if lap_no <= 15:
                                                                              
        _early_alpha = min(SPEED_RANK_EMA_ALPHA, 0.15)                     
        for code in sorted_codes:
            sr = speed_rank_map.get(code, 0.5)
            gr = grid_rank_map.get(code, sr)
            blended = (grid_w * gr) + ((1.0 - grid_w) * sr)
                                                       
            prev = state["speed_rank"].get(code, blended)
            state["speed_rank"][code] = (1.0 - _early_alpha) * prev + _early_alpha * blended

                                                                  
                                                                   
                                                                                 
    if len(sorted_codes) >= 2:
        for i, code in enumerate(sorted_codes):
            if i + 1 < len(sorted_codes):
                next_code = sorted_codes[i + 1]
                state["gap_to_below"][code] = (
                    active_speeds[code] - active_speeds[next_code]
                )
                                                                        
                state["drs_proximity"][code] = state["gap_to_below"][code]
                state["drs_available"][code] = (state["gap_to_below"][code] < 1.0 and state["gap_to_below"][code] > -0.5)
            else:
                state["gap_to_below"][code] = 0.0
                state["drs_proximity"][code] = float('inf')
                state["drs_available"][code] = False
    else:
        for code in sorted_codes:
            state["gap_to_below"][code] = 0.0
            state["drs_proximity"][code] = float('inf')
            state["drs_available"][code] = False

                                                          
    for code in state["active_drivers"]:
        state["fuel_state"][code] = _estimate_fuel_load(code, lap_no)
    
                                             
                                                                   
    try:
        leader = sorted_codes[0] if sorted_codes else None
        if leader:
            leader_age = state["tire_age"].get(leader, 0)
            leader_compound = state["current_compound"].get(leader, "MEDIUM")
            for code in state["active_drivers"]:
                if code == leader:
                    state["tire_delta_to_leader"][code] = 0.0
                else:
                    code_age = state["tire_age"].get(code, 0)
                    code_compound = state["current_compound"].get(code, "MEDIUM")
                                                                            
                    age_rate = COMPOUND_DEG_RATE.get(code_compound, 0.20)
                    age_delta = code_age - leader_age
                    state["tire_delta_to_leader"][code] = age_delta * age_rate * 0.25
    except Exception:
        pass                                 
    
                                                        
    field_mean_speed = field_med_spd
    team_speeds = defaultdict(list)
    for code in state["active_drivers"]:
        team = state["pre_race"].get(code, {}).get("team", "")
        spd = active_speeds.get(code, 0.0)
        if team and spd > 0:
            team_speeds[team].append(spd)
    
    for team, speeds in team_speeds.items():
        avg_team_speed = float(np.mean(speeds))
        state["constructor_delta"][team] = avg_team_speed - field_mean_speed
    
                                                      
    for code in state["active_drivers"]:
        spd = active_speeds.get(code, 0.0)
        if lap_no > 3 and field_med_spd > 0 and spd > 50:
            speed_gap = (field_med_spd - spd) / field_med_spd
            state["damage_estimate"][code] = min(1.0, max(0.0, speed_gap * 1.5))
        else:
            state["damage_estimate"][code] = 0.0
    team_speeds = defaultdict(list)
    for code in sorted_codes:
        team = state["pre_race"].get(code, {}).get("team", "")
        if team:
            team_speeds[team].append(active_speeds.get(code, 0.0))
    team_avg = {t: float(np.mean(v)) for t, v in team_speeds.items() if v}
    for code in sorted_codes:
        team = state["pre_race"].get(code, {}).get("team", "")
        if team and team in team_avg:
            state["delta_vs_teammate"][code] = active_speeds.get(code, 0.0) - team_avg[team]
        else:
            state["delta_vs_teammate"][code] = 0.0

                                        
    pitted_this_lap: dict[str, str] = {}
    for row in tele_rows:
        code  = _driver_code_from_row(row)
        stint = int(row.get("Stint") or 1)
        if not code or code not in state["active_drivers"]:
            continue
        prev  = state["current_stint"][code]
        if code and prev > 0 and stint != prev:
            pr   = state["pre_race"].get(code, {})
            team = pr.get("team", "")
            if team:
                pitted_this_lap[code] = team

    for row in tele_rows:
        code = _driver_code_from_row(row)
        if not code or code not in state["active_drivers"]:
            continue
        pr   = state["pre_race"].get(code, {})
        team = pr.get("team", "")
        state["teammate_pitted"][code] = any(
            t == team and d != code for d, t in pitted_this_lap.items()
        )

                                                                                  
                                                                 
                                                                 
                                                                                
                                                                          
     
                                                        
                                                                                   
                                                                            
                                                                                
                                                                                  
                                                                  
                                                                   
    if not is_tick:
                                                                                   
        _finish_ms:   dict[str, int]  = {}
        _is_pit_in:   dict[str, bool] = {}
        for _row in tele_rows:
            _c   = _driver_code_from_row(_row)
            _fms = _row.get("lap_finish_ms")
            _pit = int(_row.get("is_pit_lap") or 0)
            if _c and _c in state["active_drivers"]:
                if _fms is not None:
                    try:
                        _finish_ms[_c] = int(_fms)
                    except (TypeError, ValueError):
                        pass
                _is_pit_in[_c] = bool(_pit)

        _circuit_m   = CIRCUIT_LENGTHS.get(state["event"], 5800.0)
        _field_lap_s = _circuit_m / (field_med_spd / 3.6) if field_med_spd > 50 else 95.0

        for _code, _spd in active_speeds.items():
            _fms = _finish_ms.get(_code)
            if _code in pitted_this_lap:
                                                                            
                                                        
                _lap_s = (_fms / 1000.0) if (_fms and _fms > 5000) else _field_lap_s
            elif _is_pit_in.get(_code):
                                                                             
                                                                                      
                                                                                 
                _base_lap_s = (_fms / 1000.0) if (_fms and _fms > 5000) else _field_lap_s
                if _base_lap_s < (_field_lap_s * 0.80):
                    _lap_s = _base_lap_s + PIT_STOP_LOSS_S
                else:
                    _lap_s = _base_lap_s
            elif _fms and _fms > 5000:
                                                               
                _lap_s = _fms / 1000.0
            elif _spd > 50:
                                                                         
                _lap_s = _circuit_m / (_spd / 3.6)
            else:
                continue
            state["cumulative_time"][_code] += _lap_s
            state["laps_completed"][_code]   = lap_no
        state["_cumul_laps_counted"] += 1

                                                                                  
                                                                                     
                                                                                
                                                                                
                                                                   
    if sorted_codes and field_med_spd > 50:
        state["gap_to_leader"] = {}
        _use_cumul = (
            lap_no >= CUMUL_TIME_MIN_LAPS
            and len(state["cumulative_time"]) >= 2
        )
        if _use_cumul:
            _cumul_active = {
                c: state["cumulative_time"][c]
                for c in active_speeds
                if state["cumulative_time"].get(c, 0) > 0
            }
            if len(_cumul_active) >= 2:
                _leader_code = min(_cumul_active, key=_cumul_active.get)
                _leader_cumul = _cumul_active[_leader_code]
                for c in active_speeds:
                    _c_cumul = _cumul_active.get(c)
                    if _c_cumul is not None:
                        state["gap_to_leader"][c] = max(0.0, _c_cumul - _leader_cumul)
                    else:
                        state["gap_to_leader"][c] = 0.0
            else:
                _use_cumul = False  # fall through to speed-based
        if not _use_cumul:
            _circuit_m_gap = CIRCUIT_LENGTHS.get(state["event"], 5800.0)
            _leader_code = sorted_codes[0]
            _leader_spd  = active_speeds[_leader_code]
            _avg_lap_s   = _circuit_m_gap / (_leader_spd / 3.6)
            for c in active_speeds:
                _spd_diff = _leader_spd - active_speeds[c]
                _gap_s = max(0.0, (_spd_diff / max(_leader_spd, 1.0)) * _avg_lap_s)
                state["gap_to_leader"][c] = _gap_s

                                                                              
                                                                              
    for _code in active_speeds:
        _rank_pct = float(state["speed_rank"].get(_code, 0.5))
        _pos = int(round(np.clip(_rank_pct * max(n_active, 1), 1.0, float(max(n_active, 1)))))
        state["position_history"][_code].append(_pos)

                                  
    for row in tele_rows:
        code  = _driver_code_from_row(row)
        if not code or code not in state["active_drivers"]:
            continue

        comp  = str(row.get("Compound") or "UNKNOWN").upper().replace("INTERMEDIATE", "INTER")
        stint = int(row.get("Stint") or 1)

        prev_stint = state["current_stint"][code]
        pitted     = (prev_stint > 0 and stint != prev_stint)
        state["current_stint"][code] = stint

        if pitted or code not in state["current_compound"]:
            state["tire_age"][code]              = 0
            state["stint_ref_speed"][code]       = 0.0
                                                                    
            state["stint_speed_samples"][code]   = deque(maxlen=MEM_LIMIT_STINT_SPD)
                                                                             
            state["predicted_laptime"].pop(code, None)

        else:
            if not is_tick:
                state["tire_age"][code] += 1

        state["current_compound"][code] = comp
        state["compounds_used"][code].add(comp)

        avg_spd = float(row.get("avg_speed") or 0)
        age     = state["tire_age"][code]

        if not state["sc_active"] and avg_spd > 50 and age <= 6:
            state["stint_speed_samples"][code].append(avg_spd)
            samples = state["stint_speed_samples"][code]
            if len(samples) >= 2:
                state["stint_ref_speed"][code] = float(np.median(samples))

        state["pace_trend"][code].append(avg_spd)
                                                                                   
                                                                         
                                                                            
                                                           
        if len(state["pace_trend"][code]) > HISTORY_WINDOW:
            state["pace_trend"][code].popleft()

        state["lap_history"][code].append({
            "lap":              lap_no,
            "avg_speed":        avg_spd,
            "hard_brake_count": float(row.get("hard_brake_count") or 0),
            "avg_throttle":     float(row.get("avg_throttle") or 0),
            "avg_drs":          float(row.get("avg_drs") or 0),
            "compound":         comp,
            "stint":            stint,
            "tire_age":         age,
            "is_sc":            state["sc_active"],
        })

        ref_s = state["stint_ref_speed"][code]
        if ref_s > 0 and not state["sc_active"] and age >= 2:
            state["obs_deg"][code] = avg_spd - ref_s
        else:
            state["obs_deg"][code] = 0.0

                    
        if tire_model:
            tf = _tire_feats(code, comp, row)
            state["model_deg"][code] = float(
                tire_model.predict(_predict(tf, TIRE_FEATS))[0]
            )

                   
        if pit_model and not pitted:
            pf       = _pit_feats(code, comp, row, lap_no)
            pit_prob = float(pit_model.predict_proba(_predict(pf, PIT_FEATS))[0][1])
            state["pit_prob"][code]  = pit_prob
            state["pit_alert"][code] = pit_prob > PIT_ALERT_THRESHOLD
            if state["pit_alert"][code]:
                pr     = state["pre_race"].get(code, {})
                sc_tag = " 🟡 FREE STOP" if state["sc_active"] else ""
                msg = (
                    f"🔧 PIT WINDOW [{pr.get('team', code):<17}] {code}  "
                    f"{comp}  age {age}  "
                    f"({_tire_age_pct(comp, age):.0%} of median)  "
                    f"prob {pit_prob:.0%}  within {PIT_HORIZON} laps{sc_tag}"
                )
                state["suggestions"].append(msg)
                if log:
                    log.info(f"LAP {lap_no:3d} | {msg}")

        if state["sc_active"] and not pitted and age >= 8:
            pr = state["pre_race"].get(code, {})
            sc_msg = (
                f"🟡 FREE PIT WINDOW  [{pr.get('team', code):<17}] {code}  "
                f"{comp}  age {age}  ({_tire_age_pct(comp, age):.0%} of median)"
            )
            if not any(code in s and "FREE PIT" in s for s in state["suggestions"]):
                state["suggestions"].append(sc_msg)
                if log:
                    log.info(f"LAP {lap_no:3d} | {sc_msg}")

        if _must_change_compound(code, lap_no) and not state["sc_active"]:
            pr  = state["pre_race"].get(code, {})
            state["suggestions"].append(
                f"⚠️  MANDATORY STOP  [{pr.get('team', code)}] {code}  "
                f"only used {', '.join(state['compounds_used'].get(code, {code}))}  "
                f"rule penalty risk"
            )

        if not state["sc_active"]:
            hist = state["lap_history"][code]
            clean_hist = [h for h in hist if not h.get("is_sc")]

            if len(clean_hist) >= 6:
                hb_now  = hist[-1]["hard_brake_count"]
                hb_prev = np.mean([h["hard_brake_count"] for h in list(clean_hist)[-5:-1]])
                if hb_prev > 0 and hb_now > hb_prev * 1.6 and hb_now > 5:
                    pr = state["pre_race"].get(code, {})
                    state["suggestions"].append(
                        f"⚠️  BRAKE SPIKE  [{pr.get('team', code)}] {code}  "
                        f"{hb_now:.0f} vs avg {hb_prev:.1f}  (+{(hb_now/hb_prev-1)*100:.0f}%)"
                    )

            if len(clean_hist) >= 6:
                th_now  = hist[-1]["avg_throttle"]
                th_prev = np.mean([h["avg_throttle"] for h in list(clean_hist)[-5:-1]])
                if th_prev > 0 and th_now < th_prev * 0.90:
                    pr = state["pre_race"].get(code, {})
                    state["suggestions"].append(
                        f"📉 THROTTLE DROP  [{pr.get('team', code)}] {code}  "
                        f"{th_now:.1f}% vs {th_prev:.1f}%  (↓{(1-th_now/th_prev)*100:.0f}%)"
                    )

            clean_trend = [h["avg_speed"] for h in clean_hist if h["avg_speed"] > 50]
            if len(clean_trend) >= 5:
                drop = (
                    (np.mean(clean_trend[:3]) - np.mean(clean_trend[-3:]))
                    / (np.mean(clean_trend[:3]) + 1e-6) * 100
                )
                if drop > 1.5:
                    pr = state["pre_race"].get(code, {})
                    state["suggestions"].append(
                        f"🐢 PACE FADE  [{pr.get('team', code)}] {code}  "
                        f"↓{drop:.1f}% over {len(clean_trend)} clean laps"
                    )

            if pitted:
                pr     = state["pre_race"].get(code, {})
                sc_tag = " (SC window)" if state["sc_active"] else ""
                state["suggestions"].append(
                    f"🔵 PITTED  [{pr.get('team', code):<17}] {code}  "
                    f"→ {comp}  age 0{sc_tag}"
                )
                if len(clean_trend) >= 3:
                    pre_pit  = np.mean(list(clean_trend)[-3:])
                    open_spd = state["stint_ref_speed"][code]
                    if open_spd > 0 and open_spd > pre_pit * 1.008:
                        state["suggestions"].append(
                            f"🟢 PIT GAIN  [{pr.get('team', code):<17}] {code}  "
                            f"+{((open_spd / pre_pit) - 1) * 100:.1f}% on fresh set"
                        )

                          
    if lap_no > 3 and not state["sc_active"]:
        for code, spd in active_speeds.items():
            if (field_med_spd - spd) > COLLAPSE_THRESHOLD:
                pr  = state["pre_race"].get(code, {})
                gap = field_med_spd - spd
                state["speed_collapsed"].add(code)
                msg = (
                    f"🚨 SPEED COLLAPSE  [{pr.get('team', code)}] {code}  "
                    f"{spd:.1f} km/h  ({gap:.1f} below median)"
                )
                if not any("SPEED COLLAPSE" in s and code in s for s in state["suggestions"]):
                    state["suggestions"].insert(0, msg)
                    if log:
                        log.info(f"LAP {lap_no:3d} | {msg}")

                   
    for row in corner_rows:
        code = str(row.get("driver_code") or row.get("driver_num") or "")
        hb   = float(row.get("hard_brakes") or 0)
        if hb >= 3:
            pr     = state["pre_race"].get(code, {})
            turn   = row.get("turn_number") or row.get("corner_id", "?")
            letter = str(row.get("corner_letter") or "")
            label  = f"T{turn}{letter}" if letter else f"T{turn}"
            spd    = float(row.get("avg_speed_corner") or 0)
            state["corner_alerts"].append(
                f"{label:<6}  {code:<4} [{pr.get('team', code):<17}]  "
                f"hard brakes: {hb:.0f}   spd: {spd:.1f} km/h"
            )

                      
    teams_seen: dict[str, list] = defaultdict(list)
    for code, pr in state["pre_race"].items():
        if state["lap_history"].get(code):
            teams_seen[pr.get("team", "")].append(code)

    for team, drivers in teams_seen.items():
        if len(drivers) >= 2:
            d0, d1 = drivers[0], drivers[1]
            age_diff = abs(state["tire_age"][d0] - state["tire_age"][d1])
            if age_diff >= 5:
                older   = d0 if state["tire_age"][d0] > state["tire_age"][d1] else d1
                fresher = d1 if older == d0 else d0
                comp_o  = state["current_compound"].get(older, "MEDIUM")
                deg_rate    = COMPOUND_DEG_RATE.get(comp_o, 0.20)
                time_adv_s  = age_diff * deg_rate * 0.25
                if (state["current_laptime"].get(older, 0) > 0 and
                        state["current_laptime"].get(fresher, 0) > 0):
                    state["pattern_insights"].append(
                        f"⚡ UNDERCUT RISK  [{team}]  "
                        f"{older}(age {state['tire_age'][older]}) vs "
                        f"{fresher}(age {state['tire_age'][fresher]})  "
                        f"~{time_adv_s:.1f}s/lap pace disadvantage"
                    )

    for code, deg in state["model_deg"].items():
        if deg < -0.03 and state["tire_age"][code] > 10:
            pr = state["pre_race"].get(code, {})
            state["pattern_insights"].append(
                f"📉 TYRE CLIFF  [{pr.get('team', code)}] {code}  "
                f"model deg {deg*100:+.1f}%  age {state['tire_age'][code]}"
            )

    if state["sc_active"]:
        state["pattern_insights"].insert(
            0,
            f"🟡 SAFETY CAR ACTIVE  (lap {lap_no})  — "
            f"pace/deg metrics suppressed  [{state['sc_laps']} SC laps total]"
        )

                        
    team_agg: dict = defaultdict(lambda: {"codes": [], "speeds": [], "degs": [], "pits": 0})
    for code in state["lap_history"]:
        hist = state["lap_history"][code]
        if not hist or hist[-1]["avg_speed"] < 50:
            continue
        pr   = state["pre_race"].get(code, {})
        team = pr.get("team", "Unknown")
        team_agg[team]["codes"].append(code)
        team_agg[team]["speeds"].append(hist[-1]["avg_speed"])
        team_agg[team]["degs"].append(state["obs_deg"].get(code, 0.0))
        if state["pit_alert"][code]:
            team_agg[team]["pits"] += 1

    for team, agg in team_agg.items():
        state["constructor_state"][team] = {
"drivers":   agg["codes"],
            "avg_speed": float(np.mean(list(agg["speeds"]))),
            "avg_deg": float(np.mean(list(agg["degs"]))),
            "pit_flags": agg["pits"],
        }

                          
                                                                        
                                                                            
                                                                              
                                                                            
                                                      
    _event = state["event"]
    for _row in tele_rows:
        _code = _driver_code_from_row(_row)
        if not _code or _code not in state["active_drivers"]:
            continue
        _spd = float(_row.get("avg_speed") or 0)
                                                     
        if _spd > 50:
            state["current_laptime"][_code] = _speed_to_laptime(_spd, _event)
            _comp = state["current_compound"].get(_code, "UNKNOWN")
            _lf   = _laptime_feats(_code, _comp, _row, lap_no)
            try:
                _pred = float(laptime_model.predict(_predict(_lf, LAPTIME_FEATS))[0])
                state["predicted_laptime"][_code] = _speed_to_laptime(_pred, _event)
            except Exception:
                                                               
                state["predicted_laptime"][_code] = state["current_laptime"][_code]

    state["fan_commentary"] = _nltk_refine_commentary(_build_fan_commentary(lap_no), lap_no)

                                   
    if args.fan_insights and state["fan_commentary"]:
        try:
            _headers = {"X-Internal-Token": UI_API_TOKEN} if UI_API_TOKEN else None
            requests.post(
                f"{args.ui_backend}/commentary",
                json=state["fan_commentary"],
                headers=_headers,
                timeout=1
            )
        except Exception:
            pass                                           

                                                                          
    state["position_probabilities"] = _compute_ranking_probabilities(lap_no)

                                                                                        
                                                                                
    if state["sc_active"]:
        blend_w = 0.0
    else:
        blend_w = min(LIVE_ORDER_BLEND_MAX, lap_no / max(BLEND_RAMP_LAPS, 1) * LIVE_ORDER_BLEND_MAX)

                                                          
    _race_progress = lap_no / max(state.get("total_laps", 53), 1)
    effective_ranking_prior = RANKING_PRIOR_WEIGHT * max(0.0, 1.0 - _race_progress)
    if log:
        log.info(
            f"LAP {lap_no}: effective_ranking_prior={effective_ranking_prior:.3f} "
            f"(base={RANKING_PRIOR_WEIGHT}, progress={_race_progress:.2%})"
        )

    raw: dict = {}
    for code in state["active_drivers"]:
        wf = _winner_feats(code, lap_no)
        model_p = float(winner_model.predict_proba(
            _predict(wf, WINNER_FEATS)
        )[0][1])

        last_spd = state["pace_trend"].get(code, [0])[-1]
        if last_spd < 100:  # pitted / very slow — treat as last
            rank_score = 0.0
        else:
            rank_score = 1.0 - state["speed_rank"].get(code, 0.5)

        rank_probs = state["position_probabilities"].get(code, {})
        ranking_p1 = float(rank_probs.get("p1", model_p))

        model_prior = (1.0 - effective_ranking_prior) * model_p + (effective_ranking_prior * ranking_p1)
        score = (1.0 - blend_w) * model_prior + blend_w * rank_score

        glr = float(wf.get("gap_laps_remaining", 0.0))
        lp  = float(wf.get("lap_progress", 0.0))
        penalty = float(np.exp(-1.6 * glr * max(lp, 0.35)))
        if wf.get("laps_remaining", 99) <= 5 and wf.get("gap_to_leader_s", 0.0) > 15.0:
            penalty *= 0.05
        raw[code] = max(0.0, score * penalty)

        trend = state["pace_trend"].get(code, [])
        state["pace_score"][code] = np.mean(list(trend)[-3:] if trend else [])

    total = sum(raw.values()) or 1.0
    base_probs = {c: v / total for c, v in raw.items()}

                                                                    
    smoothed = {}
    for code, p in base_probs.items():
        prev = state["win_proba_ema"].get(code, p)
        ema  = (1.0 - WIN_PROBA_EMA_ALPHA) * prev + WIN_PROBA_EMA_ALPHA * p
        if WIN_PROBA_MAX_DELTA > 0:
            lo = prev - WIN_PROBA_MAX_DELTA
            hi = prev + WIN_PROBA_MAX_DELTA
            ema = max(lo, min(hi, ema))
        smoothed[code] = ema

    s_total = sum(smoothed.values()) or 1.0
    state["win_proba"] = {c: v / s_total for c, v in smoothed.items()}
    state["win_proba_ema"] = dict(state["win_proba"])

    _dnf_zeroed = False
    for _code in list(state["win_proba"].keys()):
        if state["driver_status"].get(_code) in ("DNF", "DNS"):
            if state["win_proba"][_code] > 0.0:
                if log:
                    log.warning(
                        f"LAP {lap_no}: FIX-1 zeroing DNF/DNS driver {_code} "
                        f"(had win_proba={state['win_proba'][_code]:.4f})"
                    )
            state["win_proba"][_code] = 0.0
            _dnf_zeroed = True
    if _dnf_zeroed:
        _renorm_total = sum(state["win_proba"].values()) or 1.0
        state["win_proba"] = {c: v / _renorm_total for c, v in state["win_proba"].items()}

    state["win_proba"] = _calibrate_late_race_probabilities(state["win_proba"], lap_no, state.get("total_laps", 53))

                                                                               
                                                          
    if HAS_MONTE_CARLO and lap_no >= 20 and lap_no % 5 == 0:
        mc_results = _run_monte_carlo_simulation(lap_no, n_simulations=500)
        if mc_results:
            state["monte_carlo_results"] = mc_results

    if _pred_client and _pred_table:
        _emit_predictions(lap_no + 1)


                                                                                

def render(lap_no: int) -> None:
    if not HAS_RICH:
        _plain(lap_no)
        return
    if not args.no_clear:
        console.clear()

    pct   = lap_no / max(state["total_laps"], 1) * 100
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    sc_tag = "  [bold yellow blink]🟡 SC[/bold yellow blink]" if state["sc_active"] else ""
    console.print(Rule(
        f"[bold cyan]🏎  {state['event']}  {state['year']}  ·  "
        f"Lap [white]{lap_no}[/white]/{state['total_laps']}  "
        f"[dim]{bar} {pct:.0f}%[/dim]{sc_tag}[/bold cyan]",
        style="cyan",
    ))

    if not args.alerts_only:
        active = set(state.get("active_drivers", set()))
        sorted_d = sorted(
            [(c, r) for c, r in state["speed_rank"].items() if c in active],
            key=lambda x: x[1]
        )
        if not sorted_d:
            sorted_d = [(c, i/len(state["win_proba"]))
                        for i, (c, _) in enumerate(
                            sorted(
                                [(c, p) for c, p in state["win_proba"].items() if c in active],
                                key=lambda x: -x[1]
                            )
                        )]

        lb = Table(
            title="[bold white]Live Race Order (speed-rank proxy)[/bold white]",
            box=box.SIMPLE_HEAD, expand=False, min_width=128, padding=(0, 1),
        )
        lb.add_column("",        width=3)
        lb.add_column("Driver",  style="bold cyan", width=6)
        lb.add_column("Win %",                      width=7)
        lb.add_column("Pred P",  justify="right",   width=6)                                
        lb.add_column("Pace",    justify="right",   width=6)
        lb.add_column("Tire",                       width=9)
        lb.add_column("Age",     justify="right",   width=4)
        lb.add_column("Deg δ",   justify="right",   width=8)
        lb.add_column("Curr LT", justify="right",   width=9)
        lb.add_column("Next LT", justify="right",   width=9)
        lb.add_column("Gap",     justify="right",   width=8)                            
        lb.add_column("Pit %",   justify="right",   width=6)
        lb.add_column("!",                          width=2)

        for pos_i, (code, _) in enumerate(sorted_d, 1):
            pr    = state["pre_race"].get(code, {})
            comp  = state["current_compound"].get(code, "—")
            age   = state["tire_age"][code]
            obs_d = state["obs_deg"].get(code, 0.0)
            pace  = state["pace_score"].get(code, 0.0)
            trend = state["pace_trend"].get(code, [])
            prob  = state["win_proba"].get(code, 0.0)
            pit_p = state["pit_prob"].get(code, 0.0)
            dfield= state["delta_vs_field"].get(code, 0.0)
            flag  = "🔴" if state["pit_alert"][code] else "  "

            cc      = COMPOUND_COLOR.get(comp, "white")
            sym     = COMPOUND_SYMBOL.get(comp, comp[:1])
            deg_col = "red" if obs_d < -2.0 else ("yellow" if obs_d < -0.5 else "green")

                                                                                
                                                                 
            _gap_s = state.get("gap_to_leader", {}).get(code)
            if _gap_s is not None:
                if _gap_s < 0.05:
                    gap_str = "[bold green]+Leader[/bold green]"
                else:
                    gap_str = f"[cyan]+{_gap_s:.3f}s[/cyan]"
            else:
                dfield  = state["delta_vs_field"].get(code, 0.0)
                df_col  = "green" if dfield > 0 else "red"
                gap_str = f"[{df_col}]{dfield:+.1f}[/{df_col}]"

            curr_lt = state["current_laptime"].get(code, 0)
            pred_lt = state["predicted_laptime"].get(code, 0)
            curr_lt_str = _laptime_str(curr_lt) if curr_lt > 0 else "—"
            if pred_lt > 0 and curr_lt > 0:
                diff    = pred_lt - curr_lt
                lt_col  = "red" if diff > 0.2 else ("yellow" if diff > 0.05 else "green")
                pred_str = f"[{lt_col}]{_laptime_str(pred_lt)}[/{lt_col}]"
            else:
                pred_str = "[dim]—[/dim]"

                                                                            
                                                                     
            pos_probs = state["position_probabilities"].get(code, {})
            if pos_probs and "expected_position" in pos_probs:
                pred_pos = round(pos_probs["expected_position"])
            else:
                pred_pos = pos_i                                           
            if pred_pos < pos_i:
                pp_col, pp_arrow = "green",  f"▲{pred_pos}"
            elif pred_pos > pos_i:
                pp_col, pp_arrow = "red",    f"▼{pred_pos}"
            else:
                pp_col, pp_arrow = "white",  f" {pred_pos}"
            pred_pos_str = f"[{pp_col}]{pp_arrow}[/{pp_col}]"

            lb.add_row(
                MEDALS.get(pos_i, f"[dim]{pos_i:2d}[/dim]"),
                code,
                f"{prob:.1%}",
                pred_pos_str,
                f"{pace:.2f}",
                f"[{cc}]{sym} {comp:<6}[/{cc}]",
                str(age),
                f"[{deg_col}]{obs_d:+.1f}[/{deg_col}]",
                curr_lt_str,
                pred_str,
                gap_str,
                f"{pit_p:.0%}",
                flag,
            )
        console.print(lb)
        terminal = []
        for code, status in sorted(state.get("driver_status", {}).items()):
            if status not in ("DNF", "DNS"):
                continue
            when = state.get("dnf_lap", {}).get(code, 0)
            lap_tag = f" @L{when}" if when else ""
            terminal.append(f"{code}: {status}{lap_tag}")
        if terminal:
            console.print(Panel(
                "\n".join(terminal),
                title="[bold red]Retired / Not Started[/bold red]",
                border_style="red", expand=False, padding=(0, 1),
            ))

                                                                                  
                                                                                  
        if state.get("position_probabilities"):
            active_codes = [c for c, _ in sorted_d]
            pred_top10 = set()
            for c in active_codes:
                pos_probs = state["position_probabilities"].get(c, {})
                exp_pos = pos_probs.get("expected_position")
                if exp_pos is not None and exp_pos <= 10.5:
                    pred_top10.add(c)

            actual_top10 = set()
            for c in active_codes:
                pr = state["pre_race"].get(c, {})
                fp = pr.get("final_position")
                fs = str(pr.get("final_status", "")).strip().lower()
                if fp is not None and fs == "finished" and int(fp) <= 10:
                    actual_top10.add(c)

            if actual_top10:
                hits = len(pred_top10 & actual_top10)
                metric = f"Top-10 hit: {hits}/10  (target 8/10)"
                console.print(f"[bold cyan]{metric}[/bold cyan]")

    for items, title, style in [
        (state["corner_alerts"][:10],   "⚠  Corner Events This Lap",  "yellow"),
        (state["suggestions"][:14],     "🔔  Strategist Alerts",       "magenta"),
        (state["pattern_insights"][:8], "🔍  Pattern Insights",        "blue"),
    ]:
        if items:
            console.print(Panel(
                "\n".join(items),
                title=f"[bold {style}]{title}[/bold {style}]",
                border_style=style, expand=False, padding=(0, 1),
            ))

    if args.fan_insights:
        commentary = state.get("fan_commentary", [])
        if commentary:
            console.print(Panel(
                "\n".join(commentary),
                title="[bold white]📡  Live Commentary[/bold white]",
                border_style="white", expand=False, padding=(0, 1),
            ))

    lines = []
    for code, trend in sorted(state["pace_trend"].items()):
        if not trend or trend[-1] <= 50:
            continue
        pr   = state["pre_race"].get(code, {})
        comp = state["current_compound"].get(code, "—")
        cc   = COMPOUND_COLOR.get(comp, "white")
        sym  = COMPOUND_SYMBOL.get(comp, "?")
        clean_trend = [s for s in trend if s > 50]
        deg_str = (
            f"  {state['obs_deg'].get(code, 0.0):+.1f}"
            if state["stint_ref_speed"].get(code, 0) > 0 else ""
        )
        lines.append(
            f"  {code:<5} [{cc}]{sym}[/{cc}]  "
            f"{_sparkline(clean_trend)}  {trend[-1]:.1f} km/h{deg_str}"
        )
    if lines:
        console.print(Panel(
            "\n".join(lines),
            title=f"[bold white]📈  Speed Trend (last {HISTORY_WINDOW} laps)[/bold white]",
            border_style="white", expand=False, padding=(0, 1),
        ))
    console.print()


def _plain(lap_no: int) -> None:
    W = 88
    sc_tag = "  [SC ACTIVE]" if state["sc_active"] else ""
    print(f"\n{'═'*W}\n  {state['event']} {state['year']}  |  "
          f"Lap {lap_no}/{state['total_laps']}{sc_tag}\n{'═'*W}")
    active = set(state.get("active_drivers", set()))
    sorted_d = sorted(
        [(c, r) for c, r in state["speed_rank"].items() if c in active],
        key=lambda x: x[1]
    ) or \
               [(c, i) for i, (c, _) in enumerate(
                   sorted(
                       [(c, p) for c, p in state["win_proba"].items() if c in active],
                       key=lambda x: -x[1]
                   ))]
    print(f"\n  {'P':<4}{'DRV':<6}{'Win%':>6}  "
          f"{'Pace':>5}  {'Tire':<7}{'Age':>4}  {'DegΔ':>7}  {'ΔField':>7}  {'Pit%':>5}  {'!'}")
    print("  " + "─" * 78)
    for pos_i, (code, _) in enumerate(sorted_d, 1):
        pr    = state["pre_race"].get(code, {})
        comp  = state["current_compound"].get(code, "—")[:6]
        obs_d = state["obs_deg"].get(code, 0.0)
        pace  = state["pace_score"].get(code, 0.0)
        pit_p = state["pit_prob"].get(code, 0.0)
        dfield= state["delta_vs_field"].get(code, 0.0)
        prob  = state["win_proba"].get(code, 0.0)
        flag  = "PIT!" if state["pit_alert"][code] else ""
        print(
            f"  {pos_i:<4}{code:<6}{prob:>5.1%}  "
            f"{pace:>5.2f}  {comp:<7}{state['tire_age'][code]:>4}  "
            f"{obs_d:>+7.1f}  {dfield:>+7.1f}  {pit_p:>5.0%}  {flag}"
        )
    terminal = []
    for code, status in sorted(state.get("driver_status", {}).items()):
        if status in ("DNF", "DNS"):
            when = state.get("dnf_lap", {}).get(code, 0)
            lap_tag = f" @L{when}" if when else ""
            terminal.append(f"{code}: {status}{lap_tag}")
    if terminal:
        print("\n  -- RETIRED/NOT STARTED")
        for row in terminal:
            print(f"  {row}")
    for label, items in [
        ("ALERTS",   state["suggestions"]),
        ("PATTERNS", state["pattern_insights"]),
        ("CORNERS",  state["corner_alerts"][:8]),
    ]:
        if items:
            print(f"\n  ── {label}")
            for s in items:
                print(f"  {s}")


                                                                                

def _load_prerace_from_clickhouse(ch, event: str, year: int, session: str):
    return io_load_prerace_from_clickhouse(
        ch=ch,
        event=event,
        year=year,
        session=session,
        race_laps=RACE_LAPS,
        canonical_event_name=_canonical_event_name,
        normalize_driver_code=_normalize_driver_code,
    )


def _clickhouse_lap_to_rows(ch, event: str, year: int, session: str, lap_no: int, num_to_code: dict) -> list:
    return io_clickhouse_lap_to_rows(
        ch=ch,
        event=event,
        year=year,
        session=session,
        lap_no=lap_no,
        num_to_code=num_to_code,
        canonical_event_name=_canonical_event_name,
        normalize_driver_code=_normalize_driver_code,
    )


def _load_cumulative_times(ch, event: str, year: int, session: str,
                           up_to_lap: int, num_to_code: dict) -> None:
    try:
        loaded = io_load_cumulative_times(
            ch=ch,
            event=event,
            year=year,
            session=session,
            up_to_lap=up_to_lap,
            num_to_code=num_to_code,
            pit_stop_loss_s=PIT_STOP_LOSS_S,
        )
        for code, (t, lap) in loaded.items():
            if t > 0:
                state["cumulative_time"][code] = t
                state["laps_completed"][code] = lap
        state["_cumul_laps_counted"] = int(up_to_lap)
        n = len(loaded)
        msg = (f"  ⏱  Pre-loaded cumulative race times for {n} drivers "
               f"(laps 1–{up_to_lap}, using actual lap durations from telemetry)")
        (console.print(f"[cyan]{msg}[/cyan]") if HAS_RICH else print(msg))
    except Exception as exc:
        logger.warning(f"Could not pre-load cumulative times: {exc}")


def _clickhouse_max_lap(ch, event: str, year: int, session: str) -> int:
    return io_clickhouse_max_lap(ch=ch, event=event, year=year, session=session)
def _clickhouse_live_loop() -> None:
    if not args.event or not args.year:
        raise SystemExit("--clickhouse-live requires --event and --year (or env F1_EVENT/F1_YEAR)")

    try:
        import clickhouse_connect
    except Exception:
        raise SystemExit("ClickHouse client missing. Run: pip install -r requirements.txt")

    ch = clickhouse_connect.get_client(host=args.ch_host, port=args.ch_port)
    ch.ping()

    event_name = _canonical_event_name(args.event)
    pre_race_ctx, num_to_code, total_laps = _load_prerace_from_clickhouse(
        ch, event_name, args.year, args.session
    )

    state["event"]      = event_name
    state["year"]       = args.year
    state["total_laps"] = max(total_laps, 0)
    state["pre_race"]   = pre_race_ctx
    state["_num_to_code"] = num_to_code

    msg = f"  ✓ ClickHouse live mode @ {args.ch_host}:{args.ch_port}  ({len(pre_race_ctx)} drivers)"
    (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
    
                                        
    debug_msg = f"  📊 Query params: event='{event_name}', year={args.year}, session='{args.session}', total_laps={state['total_laps']}"
    (console.print(f"[cyan]{debug_msg}[/cyan]") if HAS_RICH else print(debug_msg))

                                                                             
                                                                             
                                                                              
                                                            
                                                                   
    if args.start_lap == 0:
        try:
            _boot_max = _clickhouse_max_lap(ch, event_name, args.year, args.session)
        except Exception:
            _boot_max = 0
        emitted_lap = max(_boot_max - args.lap_buffer, 0)
        msg = f"  ⏭  Live mode: skipping to lap {emitted_lap} (use --start-lap 1 to replay from the beginning)"
        (console.print(f"[cyan]{msg}[/cyan]") if HAS_RICH else print(msg))
    else:
        emitted_lap = max(args.start_lap - 1, 0)

                                                                        
                                                                                  
                                                                                 
                                                                                    
    if emitted_lap > 0:
        _load_cumulative_times(
            ch, event_name, args.year, args.session, emitted_lap, num_to_code
        )

    last_tick_emit = 0.0

    while True:
        try:
            max_lap = _clickhouse_max_lap(ch, event_name, args.year, args.session)
        except Exception as e:
            msg = f"ClickHouse poll failed: {e}"
            (console.print(f"[red]{msg}[/red]") if HAS_RICH else print(msg))
            time.sleep(args.poll_interval_ms / 1000.0)
            continue

        if max_lap > state["total_laps"]:
            state["total_laps"] = max_lap
            debug_msg = f"  📈 Updated total_laps to {state['total_laps']}"
            (console.print(f"[dim]{debug_msg}[/dim]") if HAS_RICH else print(debug_msg))

        emit_upto = max_lap - args.lap_buffer

                                                                               
        if args.tick_seconds and args.tick_seconds > 0:
            now = time.time()
            if now - last_tick_emit < args.tick_seconds:
                time.sleep(args.poll_interval_ms / 1000.0)
                continue

            if emit_upto <= 0:
                time.sleep(args.poll_interval_ms / 1000.0)
                continue

            tele_rows = _clickhouse_lap_to_rows(
                ch, event_name, args.year, args.session, emit_upto, num_to_code
            )
            if tele_rows:
                run_inference(emit_upto, tele_rows, [], is_tick=True)
                render(emit_upto)
                last_tick_emit = now
            else:
                time.sleep(args.poll_interval_ms / 1000.0)
            continue

                                  
        if emit_upto <= emitted_lap:
            if max_lap > emitted_lap:
                debug_msg = f"  ⏳ Waiting: emit_upto={emit_upto}, emitted_lap={emitted_lap}, max_lap={max_lap}, buf={args.lap_buffer}"
                (console.print(f"[dim]{debug_msg}[/dim]") if HAS_RICH else print(debug_msg))
            time.sleep(args.poll_interval_ms / 1000.0)
            continue

        for lap in range(emitted_lap + 1, emit_upto + 1):
            tele_rows = _clickhouse_lap_to_rows(
                ch, event_name, args.year, args.session, lap, num_to_code
            )
            if not tele_rows:
                msg = f"  ⚠  Lap {lap}: No telemetry rows returned from ClickHouse (event={event_name}, year={args.year}, session={args.session})"
                (console.print(f"[yellow]{msg}[/yellow]") if HAS_RICH else print(msg))
                continue

                                                                          
                                                                          
                                                                            
                                                                   
            _n_drivers_this_lap = len(set(
                str(r.get("driver_code") or r.get("driver_num") or "").strip()
                for r in tele_rows
                if (r.get("avg_speed") or 0) > 50
            ))
            _min_drivers = 15 if lap <= 30 else 10                            
            if _n_drivers_this_lap < _min_drivers:
                msg = f"  ⏳ Lap {lap}: Only {_n_drivers_this_lap}/{_min_drivers} drivers — waiting for more data"
                (console.print(f"[dim]{msg}[/dim]") if HAS_RICH else print(msg))
                break                                                       

            msg = f"  ✓ Lap {lap}: Processing {len(tele_rows)} telemetry records (forecasting lap {lap + 1})"
            (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
            run_inference(lap, tele_rows, [])
            render(lap)
            emitted_lap = lap

        if state["total_laps"] > 0 and emitted_lap >= state["total_laps"]:
            break


                                                     

def _emit_predictions(lap_no: int) -> None:
    """
    Emit per-driver predictions for the current lap to ClickHouse.

    Uses list-of-lists with explicit column_names — the only insert form
    that works reliably across all clickhouse_connect versions. Passing a
    list-of-dicts together with column_names causes a silent type mismatch
    in some versions and drops the write entirely.
    """
    try:
        cols = [
            "ts", "event", "year", "session", "lap_no",
            "driver_code", "team", "win_proba", "pace_score",
            "pit_prob", "pit_alert", "tire", "tire_age",
            "obs_deg", "delta_vs_field", "curr_laptime", "pred_laptime",
            "speed_rank", "gap_to_leader",
        ]
        ts = datetime.now(timezone.utc)
        session = args.session if args.session else "R"
        ranked_codes = [c for c, _ in sorted(state.get("speed_rank", {}).items(), key=lambda x: x[1])]
        rank_pos = {code: i + 1 for i, code in enumerate(ranked_codes)}
        rows    = []
        for code in sorted(state["win_proba"].keys()):
            pr = state["pre_race"].get(code, {})
            rows.append([
                ts,
                str(state["event"]),
                int(state["year"] or 0),
                session,
                int(lap_no),
                code,
                pr.get("team", "Unknown"),
                float(state["win_proba"].get(code, 0.0)),
                float(state["pace_score"].get(code, 0.0)),
                float(state["pit_prob"].get(code, 0.0)),
                int(bool(state["pit_alert"].get(code, False))),
                str(state["current_compound"].get(code, "UNKNOWN")),
                int(state["tire_age"].get(code, 0)),
                float(state["obs_deg"].get(code, 0.0)),
                float(state["delta_vs_field"].get(code, 0.0)),
                float(state["current_laptime"].get(code, 0.0)),
                float(state["predicted_laptime"].get(code, 0.0)),
                int(rank_pos.get(code, 0)),
                float(state.get("gap_to_leader", {}).get(code, 0.0)),
            ])

        if rows:
            _pred_client.insert(_pred_table, rows, column_names=cols)
            if log:
                log.info(f"LAP {lap_no:3d} | prediction_results: wrote {len(rows)} rows")
    except Exception as e:
        msg = f"Prediction write failed (lap {lap_no}): {e}"
        (console.print(f"[red]{msg}[/red]") if HAS_RICH else print(msg))
        if log:
            log.exception(f"LAP {lap_no:3d} | _emit_predictions failed")

def main() -> None:
    global _pred_client, _pred_table
    if args.write_preds:
        try:
            import clickhouse_connect
            _pred_client = clickhouse_connect.get_client(host=args.ch_host, port=args.ch_port)
            _pred_client.ping()
            _pred_table = args.preds_table
            if not _is_safe_identifier(_pred_table):
                raise ValueError(
                    f"Unsafe predictions table name: '{_pred_table}'. Use [A-Za-z_][A-Za-z0-9_]*"
                )
                                                      
            _pred_client.command(f"""
                create table IF NOT EXISTS {_pred_table}
                (
                    ts             DateTime,
                    event          String,
                    year           UInt16,
                    session        String,
                    lap_no         UInt16,
                    driver_code    String,
                    team           String,
                    win_proba      Float32,
                    pace_score     Float32,
                    pit_prob       Float32,
                    pit_alert      UInt8,
                    tire           String,
                    tire_age       UInt16,
                    obs_deg        Float32,
                    delta_vs_field Float32,
                    curr_laptime   Float32,
                    pred_laptime   Float32,
                    speed_rank     UInt8,
                    gap_to_leader  Float32
                )
                ENGINE = MergeTree
                partition by (year, event)
                order by (event, year, session, lap_no, driver_code)
            """)
            _pred_client.command(f"alter table {_pred_table} ADD COLUMN IF NOT EXISTS speed_rank UInt8")
            _pred_client.command(f"alter table {_pred_table} ADD COLUMN IF NOT EXISTS gap_to_leader Float32")
            msg = f"  ✓ Prediction output enabled: {args.preds_table}"
            (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
        except Exception as e:
            msg = f"Prediction output disabled (ClickHouse error): {e}"
            (console.print(f"[yellow]{msg}[/yellow]") if HAS_RICH else print(msg))
            _pred_client = None
            _pred_table = None

    if not args.clickhouse_live:
        raise SystemExit("Socket mode removed. Run with --clickhouse-live.")

    _clickhouse_live_loop()
    return


if __name__ == "__main__":
    main()


