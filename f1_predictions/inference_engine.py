import sys, os, pickle, argparse, logging, time, hashlib, json
import re
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, Optional, List, Any

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

parser = argparse.ArgumentParser()
parser.add_argument("--log",         default=None)
parser.add_argument("--alerts-only", action="store_true")
parser.add_argument("--no-clear",    action="store_true")
parser.add_argument("--history",     type=int, default=10)
parser.add_argument("--fan-insights", dest="fan_insights", action="store_true", default=True)
parser.add_argument("--no-commentary", dest="fan_insights", action="store_false")
parser.add_argument("--clickhouse-live", action="store_true",
                    help="Poll ClickHouse raw_telemetry table for new laps and render live")
parser.add_argument("--event",   default=os.getenv("F1_EVENT", ""))
parser.add_argument("--year",    type=int, default=int(os.getenv("F1_YEAR", "0") or 0))
parser.add_argument("--session", default=os.getenv("F1_SESSION", "R"))
parser.add_argument("--ch-host", default=os.getenv("CH_HOST", "localhost"))
parser.add_argument("--ch-port", type=int, default=int(os.getenv("CH_PORT", "8123")))
parser.add_argument("--poll-interval-ms", type=int, default=1000)
parser.add_argument("--lap-buffer", type=int, default=1)
parser.add_argument("--start-lap", type=int, default=0,
                    help="First lap to emit. 0 (default) = skip pre-existing ClickHouse data and only process new laps.")
parser.add_argument("--tick-seconds", type=int, default=0,
                    help="Emit predictions every N seconds of wall-clock time. ClickHouse live mode only.")
parser.add_argument("--write-preds", action="store_true",
                    help="Write predictions to ClickHouse after each emit.")
parser.add_argument("--preds-table", default=os.getenv("PRED_TABLE", "prediction_results"), help="ClickHouse predictions table name")
parser.add_argument("--grid-prior-laps", type=int, default=3,
                    help="Use qualifying grid as primary order for the first N laps.")
parser.add_argument("--grid-blend-laps", type=int, default=20,                                               
                    help="Blend out qualifying grid influence by this lap.")
parser.add_argument("--grid-prior-weight", type=float, default=0.95,
                    help="Max weight of qualifying grid in speed-rank blend (0-1).")
parser.add_argument("--grid-position-regularization", type=float, default=0.15,
                    help="Residual grid influence weight after blend laps (0-1). Prevents impossible overtakes.")
parser.add_argument("--min-active-for-speed-rank", type=int, default=10,
                    help="If fewer active cars than this, fall back to grid ordering.")
parser.add_argument("--speed-rank-ema", type=float, default=0.35,
                    help="EMA alpha for speed-rank stability (0-1). Lower = smoother.")
parser.add_argument("--win-proba-ema", type=float, default=0.20,
                    help="EMA alpha for win probability stability (0-1). Lower = smoother.")
parser.add_argument("--win-proba-max-delta", type=float, default=0.05,
                    help="Max per-emit change in win probability (fraction, e.g. 0.08 = 8%).")
parser.add_argument("--ui-backend", default="http://localhost:8000",
                    help="UI backend URL for pushing commentary (e.g., http://localhost:8000)")
args = parser.parse_args()
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
MODEL_HASH_MANIFEST = os.getenv("MODEL_HASH_MANIFEST", "").strip()
MODEL_HASH_STRICT = os.getenv("MODEL_HASH_STRICT", "0").strip().lower() in {"1", "true", "yes"}
UI_API_TOKEN = os.getenv("UI_API_TOKEN", "").strip()


def _is_safe_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", str(name or "")))


def _verify_artifact_hash(path: str) -> None:
    if not MODEL_HASH_MANIFEST:
        return
    try:
        with open(MODEL_HASH_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected = str(manifest.get(os.path.basename(path), "")).strip().lower()
        if not expected:
            msg = f"Artifact hash missing in manifest for {os.path.basename(path)}"
            if MODEL_HASH_STRICT:
                raise ValueError(msg)
            logging.warning(msg)
            return
        h = hashlib.sha256()
        with open(path, "rb") as rf:
            for chunk in iter(lambda: rf.read(8192), b""):
                h.update(chunk)
        actual = h.hexdigest().lower()
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch for {os.path.basename(path)}")
    except Exception as e:
        if MODEL_HASH_STRICT:
            raise
        logging.warning(f"Artifact hash verification warning: {e}")

def load_pickle(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.isfile(path):
        msg = f"✗ Missing model file: {path}"
        (console.print(f"[red]{msg}[/red]") if HAS_RICH else print(msg))
        sys.exit(1)
    _verify_artifact_hash(path)
    with open(path, "rb") as f:
        return pickle.load(f)
                                                                         
try:
    winner_model         = load_pickle("winner_model.pkl")
    tire_model           = load_pickle("tire_model.pkl")
    pit_model            = load_pickle("pit_model.pkl")
    pace_model           = load_pickle("pace_model.pkl")
    le_team              = load_pickle("team_encoder.pkl")
    COMPOUND_CLASSES     = load_pickle("compound_classes.pkl")
    MEDIAN_STINT_LENGTHS = load_pickle("median_stint_lengths.pkl")
    WINNER_FEATS         = load_pickle("winner_feats.pkl")
    TIRE_FEATS           = load_pickle("tire_feats.pkl")
    PIT_FEATS            = load_pickle("pit_feats.pkl")
    SC_THRESHOLD         = load_pickle("sc_speed_threshold.pkl")
    TRACK_TYPE_MAP       = load_pickle("track_type_map.pkl")
    TRACK_TYPE_ENCODER   = load_pickle("track_type_encoder.pkl")
    PIT_HORIZON          = load_pickle("pit_horizon.pkl")
    msg = f"✓ Models loaded from {MODEL_DIR}  (pit horizon={PIT_HORIZON} laps)"
    (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
    required = {"laps_remaining", "speed_rank_pct", "delta_vs_field"}
    missing = required - set(WINNER_FEATS)
    assert not missing, f"Retrain needed — missing WINNER_FEATS: {missing}"
    assert "must_change_compound" in PIT_FEATS, "Retrain needed — must_change_compound not in PIT_FEATS"
    assert "hard_brake_rate" in PIT_FEATS, "Retrain needed — hard_brake_rate not in PIT_FEATS"
except SystemExit:
    raise
except Exception as e:
    msg = f"✗ Model loading failed: {e}"
    (console.print(f"[red]{msg}[/red]") if HAS_RICH else print(msg))
    sys.exit(1)

                                               
PIT_ALERT_THRESHOLD = 0.40
_pit_thr_path = os.path.join(MODEL_DIR, "pit_threshold.pkl")
if os.path.isfile(_pit_thr_path):
    try:
        _verify_artifact_hash(_pit_thr_path)
        with open(_pit_thr_path, "rb") as f:
            PIT_ALERT_THRESHOLD = float(pickle.load(f))
        msg = f"✓ Pit threshold loaded: {PIT_ALERT_THRESHOLD:.2f}"
        (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
    except Exception as e:
        msg = f"⚠ Pit threshold load failed: {e}"
        (console.print(f"[yellow]{msg}[/yellow]") if HAS_RICH else print(msg))

                              
laptime_model    = load_pickle("laptime_model.pkl")
LAPTIME_FEATS    = load_pickle("laptime_feats.pkl")
CIRCUIT_LENGTHS  = load_pickle("circuit_lengths.pkl")
LAPTIME_MAE_S    = load_pickle("laptime_mae_s.pkl")

                                                                     
                                                                
RACE_LAPS = {
    "Abu Dhabi Grand Prix": 58, "Australian Grand Prix": 58,
    "Austrian Grand Prix": 71, "Azerbaijan Grand Prix": 51,
    "Bahrain Grand Prix": 57, "Belgian Grand Prix": 44,
    "British Grand Prix": 52, "Canadian Grand Prix": 70,
    "Chinese Grand Prix": 56, "Dutch Grand Prix": 72,
    "Emilia Romagna Grand Prix": 63, "Hungarian Grand Prix": 70,
    "Italian Grand Prix": 53, "Japanese Grand Prix": 53,
    "Las Vegas Grand Prix": 50, "Mexico City Grand Prix": 71,
    "Miami Grand Prix": 57, "Monaco Grand Prix": 78,
    "Qatar Grand Prix": 57, "Saudi Arabian Grand Prix": 50,
    "Singapore Grand Prix": 62, "Spanish Grand Prix": 66,
    "São Paulo Grand Prix": 71, "United States Grand Prix": 56,
}

_lt_status = "lap time model loaded (required)"
(console.print(f"[green]{_lt_status}[/]") if HAS_RICH else print(_lt_status))

                                                                                    
try:
    ranking_model = load_pickle("ranking_model.pkl")
    RANKING_FEATS = load_pickle("ranking_feats.pkl")
    msg = f"✓ Ranking distribution model loaded (enables Monte Carlo simulation)"
    (console.print(f"[green]{msg}[/green]") if HAS_RICH else print(msg))
    HAS_RANKING_MODEL = True
except Exception as e:
    msg = f"⚠ Ranking model not available: {e} (Monte Carlo disabled)"
    (console.print(f"[yellow]{msg}[/yellow]") if HAS_RICH else print(msg))
    HAS_RANKING_MODEL = False

                              
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

COMPOUND_COLOR  = {"SOFT": "bold red", "MEDIUM": "bold yellow", "HARD": "bold white",
                   "INTER": "bold green", "WET": "bold blue"}
COMPOUND_SYMBOL = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTER": "I", "WET": "W"}
COMPOUND_DEG_RATE = {"SOFT": 0.35, "MEDIUM": 0.20, "HARD": 0.12, "INTER": 0.25, "WET": 0.18}
MEDALS           = {1: "🥇", 2: "🥈", 3: "🥉"}
SPARK_BARS       = "▁▂▃▄▅▆▇█"

                                               
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

                                                                                     
def _bounded_defaultdict(maxsize, factory=list):
    """Helper to create defaultdict that stores bounded deques"""
    class BoundedDefaultDict(defaultdict):
        def __missing__(self, key):
            val = deque(maxlen=maxsize)
            self[key] = val
            return val
    return BoundedDefaultDict(factory)

                                
MEM_LIMIT_SPEEDS = get("inference.state_memory.all_speeds_keep_last", 100)
MEM_LIMIT_LAP_HIST = get("inference.state_memory.lap_history_keep_last", 30)
MEM_LIMIT_STINT_SPD = get("inference.state_memory.stint_speeds_keep_last", 50)
MEM_LIMIT_PIT_HIST = get("inference.state_memory.pit_history_keep_last", 20)
MEM_LIMIT_POS_HIST = get("inference.state_memory.position_history_keep_last", 50)

state: dict = {
    "event":             "",
    "year":              0,
    "total_laps":        0,
    "pre_race":          {},
    "_num_to_code":      {},
    "current_lap":       0,
    "n_drivers":         20,
    "tire_age":          defaultdict(int),
    "current_compound":  {},
    "current_stint":     defaultdict(int),
    "stint_speed_samples": _bounded_defaultdict(MEM_LIMIT_STINT_SPD),                          
    "stint_ref_speed":   defaultdict(float),
    "lap_history":       _bounded_defaultdict(MEM_LIMIT_LAP_HIST),                             
    "pace_trend":        _bounded_defaultdict(MEM_LIMIT_LAP_HIST),                             
    "obs_deg":           {},
    "model_deg":         {},
    "pit_prob":          defaultdict(float),
    "pit_alert":         defaultdict(bool),
    "teammate_pitted":   defaultdict(bool),
    "win_proba":         {},
    "pace_score":        {},
    "speed_rank":        {},
    "speed_rank_ema":    {},
    "delta_vs_field":    {},
    "delta_vs_teammate": {},
                                                  
    "gap_to_below":      {},
                                        
    "position_history":  _bounded_defaultdict(MEM_LIMIT_POS_HIST),                             
    "constructor_state": defaultdict(dict),
    "suggestions":       [],
    "corner_alerts":     [],
    "pattern_insights":  [],
    "sc_active":         False,
    "sc_end_lap":        0,
    "sc_laps":           0,
    "race_p75_speed":    0.0,
    "all_speeds_ever":   deque(maxlen=MEM_LIMIT_SPEEDS),                          
    "last_rainfall":     None,
    "compounds_used":    defaultdict(set),
    "predicted_laptime": {},
    "current_laptime":   {},
    "fan_commentary":    [],
    "speed_collapsed":   set(),
    "win_proba_ema":     {},
                                                                     
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
                                                                                           
    "cumulative_time":     defaultdict(float),                                                
    "laps_completed":      defaultdict(int),                                         
    "_cumul_laps_counted": 0,                                                                  
}

                                                                                

def _enc_compound(val: str) -> int:
    return COMPOUND_MAP.get(str(val).upper().replace("INTERMEDIATE", "INTER"), 0)

def _normalize_event_name(event: str) -> str:
    if not isinstance(event, str):
        return str(event)
    return event.replace("S\u00e3o", "Sao")

def _canonical_event_name(event: str) -> str:
    """Normalize env/CLI event names (e.g. Miami_Grand_Prix -> Miami Grand Prix)."""
    ev = _normalize_event_name(str(event or "").strip().replace("_", " "))
    return re.sub(r"\s+", " ", ev).strip()

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


def _normalize_driver_code(val: Any) -> str:
    code = str(val or "").strip().upper()
    if len(code) == 3 and code.isalpha():
        return code
    return ""


def _driver_code_from_row(row: dict) -> str:
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


def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float(default)


def _row_is_valid_activity(row: dict) -> bool:
    """
    Activity guard for live inference:
    - require a real driver code
    - require meaningful speed (>50 km/h)
    - require at least one non-null telemetry signal so sparse/null rows don't
      keep retired drivers alive
    """
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

def _sparkline(values: list, width: int = 10) -> str:
    if not values:
        return ""
    chunk = values[-width:]
    lo, hi = min(chunk), max(chunk)
    span = hi - lo or 1
    return "".join(SPARK_BARS[min(7, int((v - lo) / span * 7))] for v in chunk)

def _grid_rank_map(codes: list[str]) -> dict:
    """
    Qualifying grid order as a rank pct in [0,1].
    Lower grid_position => better rank (smaller pct).
    """
    if not codes:
        return {}
    keyed = []
    for c in codes:
        pr = state["pre_race"].get(c, {})
        gp = pr.get("grid_position", 999)
        try:
            gp = int(gp)
        except Exception:
            gp = 999
        keyed.append((gp, c))
    keyed.sort(key=lambda x: (x[0], x[1]))
    n = len(keyed)
    return {c: (i + 1) / max(n, 1) for i, (_, c) in enumerate(keyed)}

def _grid_blend_weight(lap_no: int, n_active: int) -> float:
    """
    Weight for qualifying grid prior in speed-rank.
    1.0 at/before GRID_PRIOR_LAPS, then linearly decays to GRID_POSITION_REG by GRID_BLEND_LAPS.
    After GRID_BLEND_LAPS, retains residual GRID_POSITION_REG influence to prevent impossible
    overtakes (e.g., P20 can't suddenly be faster than top 3 on a single lap).
    """
    if n_active < MIN_ACTIVE_FOR_RANK:
        return 1.0
    if GRID_BLEND_LAPS <= 0:
        return GRID_POSITION_REG                                          
    if lap_no <= GRID_PRIOR_LAPS:
        return 1.0
    if lap_no >= GRID_BLEND_LAPS:
        return GRID_POSITION_REG                                         
    
                                                     
    span = max(GRID_BLEND_LAPS - GRID_PRIOR_LAPS, 1)
    fraction = (lap_no - GRID_PRIOR_LAPS) / span
    return 1.0 - (fraction * (1.0 - GRID_POSITION_REG))

def _laps_remaining(lap_no: int) -> int:
    return max(0, state["total_laps"] - lap_no)

def _tire_age_pct(compound: str, age: int) -> float:
    median = MEDIAN_STINT_LENGTHS.get(compound.upper().replace("INTERMEDIATE", "INTER"), 25.0)
    return age / median if median > 0 else 0.0

def _stint_len_med(compound: str) -> float:
    return float(MEDIAN_STINT_LENGTHS.get(compound.upper().replace("INTERMEDIATE", "INTER"), 25.0))

def _hard_brake_rate(hard_brake_count: float, avg_speed: float) -> float:
    return hard_brake_count / max(avg_speed, 1.0) * 100.0

def _speed_to_laptime(speed_kmh: float, event: str) -> float:
    circuit_m = CIRCUIT_LENGTHS.get(event, 5300.0)
    if speed_kmh <= 0:
        return 0.0
    return circuit_m / (speed_kmh / 3.6)

def _laptime_str(lt_s: float) -> str:
    if lt_s <= 0:
        return "—"
    m  = int(lt_s // 60)
    s  = lt_s - m * 60
    return f"{m}:{s:06.3f}"

def _lt(code: str) -> float:
    return state["current_laptime"].get(code, 0.0)

def _lt_str(lt_s: float) -> str:
    return _laptime_str(lt_s)

def _fteam(code: str) -> str:
    team = state["pre_race"].get(code, {}).get("team", "")
    return team if team else code


def _current_stint_clean_speeds(code: str) -> list[float]:
    """
    [FIX-F2] Return avg_speeds for clean laps in the CURRENT stint only.

    The original code used all of lap_history[code], which spans multiple stints.
    After a pit stop, laps from the previous stint contaminate pace_drop and
    rel_speed_delta with cross-stint degradation signals that make the model
    think the driver is aggressively degrading when they're actually on fresh
    rubber.
    """
    current_stint_id = state["current_stint"][code]
    hist = state["lap_history"].get(code, [])
    return [
        h["avg_speed"]
        for h in hist
        if h.get("stint") == current_stint_id
        and not h.get("is_sc")
        and h["avg_speed"] > 50
    ]


def _pace_drop_from_clean(clean_speeds: list[float], n: int = 3) -> float:
    """
    Compute pace drop as (mean of first n clean laps) - (mean of last n clean laps)
    divided by the early mean. Positive = driver is getting slower.
    Returns 0.0 if not enough data.
    """
    if len(clean_speeds) < n * 2:
        return 0.0
    early = float(np.mean(clean_speeds[:n]))
    late  = float(np.mean(clean_speeds[-n:]))
    return (early - late) / max(early, 1e-6)


                                                                                

def _laptime_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    """
    [FIX-F2] Uses current-stint-only clean speeds for pace_drop_3.
    [FIX-F4] Uses populated gap_to_below.
    """
    age     = state["tire_age"][code]
    avg_spd = float(row.get("avg_speed") or 0.0)
    ref_s   = float(state["stint_ref_speed"].get(code, 0.0) or 0.0)
    rel_d   = ((avg_spd - ref_s) / max(ref_s, 1.0)) if ref_s > 0 and avg_spd > 0 else 0.0

                                            
    clean = _current_stint_clean_speeds(code)
    pace_d3 = _pace_drop_from_clean(clean, n=2)

    return {
        "avg_speed":          avg_spd,
        "compound_enc":       _enc_compound(comp),
        "tire_age":           age,
        "tire_age_pct":       _tire_age_pct(comp, age),
        "track_type_enc":     _enc_track_type(state["event"]),
        "rel_speed_delta":    rel_d,
        "pace_drop_3":        pace_d3,
        "avg_throttle":       float(row.get("avg_throttle") or 85.0),
        "avg_brake":          float(row.get("avg_brake") or 5.0),
        "hard_brake_rate":    _hard_brake_rate(float(row.get("hard_brake_count") or 0), avg_spd),
        "avg_drs":            float(row.get("avg_drs") or 0.5),
        "avg_rpm":            float(row.get("avg_rpm") or 10000.0),
        "track_temp":         float(row.get("track_temp") or 35.0),
        "air_temp":           float(row.get("air_temp") or 28.0),
        "rainfall":           float(row.get("rainfall") or 0.0),
        "laps_remaining":     _laps_remaining(lap_no),
        "LapNumber":          lap_no,
        "race_pct":           lap_no / max(state["total_laps"], 1),
        "speed_rank_pct":     state["speed_rank"].get(code, 0.5),
        "delta_vs_field":     state["delta_vs_field"].get(code, 0.0),
                                                              
        "gap_to_below_proxy": float(state["gap_to_below"].get(code, 0.0)),
        "teammate_pitted":    int(state["teammate_pitted"].get(code, False)),
    }


def _field_median_speed(tele_rows: list) -> float:
    speeds = [float(r.get("avg_speed") or 0) for r in tele_rows if (r.get("avg_speed") or 0) > 50]
    return float(np.median(speeds)) if speeds else 0.0

def _is_sc_lap(field_median: float) -> bool:
    if SC_THRESHOLD and SC_THRESHOLD > 0:
        return field_median < SC_THRESHOLD
    ref = state["race_p75_speed"]
    if ref <= 0:
        return False
    return field_median < ref * SC_FIELD_DROP

def _must_change_compound(code: str, lap_no: int) -> int:
    """
    [FIX-F7] Align with training definition.

    Training used: (Stint == 1) & (race_pct > 0.60)
    i.e. the driver is still on their FIRST stint past 60% of the race.

    The original inference version used compound-set tracking (len(used) < 2),
    which diverges from the training label whenever:
      - A driver pits twice on the same compound (was allowed in certain wet races)
      - A driver is on Stint 2 with only 1 compound used (same as training intent
        but the set count is an imperfect proxy)

    We match training exactly: current_stint == 1 AND race_pct > 0.60.
    Inters/wet bypass as before.
    """
    current = state["current_compound"].get(code, "UNKNOWN")
    if current in ("INTER", "WET"):
        return 0
    race_pct = lap_no / max(state["total_laps"], 1)
    return int(state["current_stint"].get(code, 1) == 1 and race_pct > 0.60)


                                                                                

                                                                           
                                                  
                                                              
     
                                                                            
                                                                           
                                                                 
                                                                               
                                                                       
                                                                  
                                         
     
                                               
                                                                                      
                                                                                      
                                                                                     
                                                                           
     
                                                                               
                                                                                    
                                                                               

def _winner_feats(code: str, lap_no: int) -> dict:
    """
    Build feature vector for the win probability model.
 
    WINNER_FEATS (must match train_winner_model.py exactly):
      grid_position, grid_position_group, avg_finish_last5, points_last5,
      dnf_rate_last5, team_enc, best_quali_lap, track_type_enc,
      laps_remaining, speed_rank_pct, delta_vs_field, tire_age, tire_age_pct,
      compound_enc, current_position, position_pct, is_leading,
      sc_active, position_jump_2, position_jump_3, sc_beneficiary,
      gap_to_leader_s, gap_laps_remaining,
      lap_progress, is_late_race, gap_urgency, leading_and_late,
      position_gain_pct, tire_freshness
 
    Key fixes vs original:
      - sc_active: binary SC lap flag (was missing entirely)
      - position_jump_2/3: positions gained in 2/3 laps (SC beneficiary signal)
      - gap_to_leader_s: actual seconds behind leader from cumulative-time block
      - gap_laps_remaining: gap_s / laps_remaining (recovery difficulty)
      - is_leading: direct binary flag instead of implied by speed_rank_pct
      - position_history stores integer positions so jumps compute correctly
    """
    pr       = state["pre_race"].get(code, {})
    grid_pos = pr.get("grid_position", 10)
 
                                                                          
    grid_group = 1 if grid_pos <= 5 else (2 if grid_pos <= 15 else 3)
 
                                                                                
                                                                 
                                                 
    speed_rank_pct   = state["speed_rank"].get(code, 0.5)
    n_drivers        = max(state.get("n_drivers", 20), 2)
    current_position = float(np.clip(speed_rank_pct * n_drivers, 1.0, float(n_drivers)))
    position_pct     = float(
        np.clip(1.0 - ((current_position - 1.0) / max(n_drivers - 1, 1)), 0.0, 1.0)
    )
 
                                                 
    is_leading = int(current_position <= 1.5)
 
                                                                                
    sc_active = int(state.get("sc_active", False))
 
                                                       
                                                                         
                                                              
                                             
    pos_hist = list(state.get("position_history", {}).get(code, []))
 
    if len(pos_hist) >= 2:
        position_jump_2 = float(pos_hist[-2] - current_position)
    else:
        position_jump_2 = 0.0
 
    if len(pos_hist) >= 3:
        position_jump_3 = float(pos_hist[-3] - current_position)
    else:
        position_jump_3 = 0.0
 
                                                                                 
                                                                             
                                                        
    sc_beneficiary = int(sc_active == 1 and position_jump_2 > 2.0)
 
                                                                                
                                                                         
                                                            
                                                                        
    gap_to_leader_s = float(state.get("gap_to_leader", {}).get(code, 0.0))
 
                                                                  
                                                               
                                                                         
    laps_remaining     = _laps_remaining(lap_no)
    gap_laps_remaining = gap_to_leader_s / max(laps_remaining, 1)
 
                                                                                
    total_laps   = max(state.get("total_laps", 53), 1)
    lap_progress = lap_no / total_laps
    is_late_race = 1 if lap_no > total_laps * 0.75 else 0
 
                                                                                
    delta_vs_field = float(state["delta_vs_field"].get(code, 0.0))
    gap_urgency    = delta_vs_field / max(laps_remaining, 1)
 
                                                                                
                                                              
    leading_and_late = position_pct * lap_progress
 
                                                                                
    comp           = state["current_compound"].get(code, "UNKNOWN")
    tire_age_pct   = _tire_age_pct(comp, state["tire_age"][code])
    tire_freshness = 1.0 / (1.0 + tire_age_pct)
 
                                                                                
    position_gain_pct = position_pct - (1.0 - min(grid_pos / 20.0, 1.0))
 
    return {
                                                                                
        "grid_position":        grid_pos,
        "grid_position_group":  grid_group,
        "avg_finish_last5":     pr.get("avg_finish_last5", 10.0),
        "points_last5":         pr.get("points_last5", 0.0),
        "dnf_rate_last5":       pr.get("dnf_rate_last5", 0.2),
        "team_enc":             _enc_team(pr.get("team", "Unknown")),
        "best_quali_lap":       pr.get("best_quali_lap", 90.0),
        "track_type_enc":       _enc_track_type(state["event"]),
 
                                                                                
        "laps_remaining":       laps_remaining,
        "speed_rank_pct":       speed_rank_pct,
        "delta_vs_field":       delta_vs_field,
        "tire_age":             state["tire_age"][code],
        "tire_age_pct":         tire_age_pct,
        "compound_enc":         _enc_compound(comp),
 
                                                                                
        "current_position":     current_position,                    
        "position_pct":         position_pct,                            
        "is_leading":           is_leading,                    
 
                                                                                
        "sc_active":            sc_active,
        "position_jump_2":      position_jump_2,
        "position_jump_3":      position_jump_3,
        "sc_beneficiary":       sc_beneficiary,
 
                                                                                
        "gap_to_leader_s":      gap_to_leader_s,
        "gap_laps_remaining":   gap_laps_remaining,
 
                                                                                
        "lap_progress":         lap_progress,
        "is_late_race":         is_late_race,
        "gap_urgency":          gap_urgency,
        "leading_and_late":     leading_and_late,
 
                                                                                
        "position_gain_pct":    position_gain_pct,
        "tire_freshness":       tire_freshness,
    }
 
 

def _tire_feats(code: str, comp: str, row: dict) -> dict:
    age     = state["tire_age"][code]
    avg_spd = float(row.get("avg_speed") or 200.0)
    stint_len = _stint_len_med(comp)
    clean = _current_stint_clean_speeds(code)
    pace_d5 = _pace_drop_from_clean(clean, n=3)
    return {
        "compound_enc":     _enc_compound(comp),
        "tire_age":         age,
        "tire_age_pct":     _tire_age_pct(comp, age),
        "stint_len_med":    stint_len,
        "stint_laps_left":  max(0.0, stint_len - age),
        "stint_progress":   age / max(stint_len, 1.0),
        "laps_remaining":   _laps_remaining(state["current_lap"]),
        "LapNumber":        state["current_lap"],
        "track_type_enc":   _enc_track_type(state["event"]),
        "avg_throttle":     float(row.get("avg_throttle") or 85.0),
        "avg_brake":        float(row.get("avg_brake") or 5.0),
        "hard_brake_rate":  _hard_brake_rate(float(row.get("hard_brake_count") or 0), avg_spd),
        "avg_drs":          float(row.get("avg_drs") or 0.5),
        "avg_rpm":          float(row.get("avg_rpm") or 10000.0),
        "track_temp":       float(row.get("track_temp") or 35.0),
        "air_temp":         float(row.get("air_temp") or 28.0),
        "rainfall":         float(row.get("rainfall") or 0),
        "delta_vs_field":   state["delta_vs_field"].get(code, 0.0),
        "speed_rank_pct":   state["speed_rank"].get(code, 0.5),
        "gap_to_below_proxy": float(state["gap_to_below"].get(code, 0.0)),
        "pace_drop_5":      pace_d5,
        "delta_vs_teammate": state["delta_vs_teammate"].get(code, 0.0),
    }


def _ranking_feats(code: str, lap_no: int) -> dict:
    """
    Build features for ranking distribution model (multinomial finishing position).
    Uses same core features as winner model but applied to final-lap context.
    """
    pr = state["pre_race"].get(code, {})
    grid_pos = pr.get("grid_position", 10)
    
                                                                 
    speed_rank_pct = state["speed_rank"].get(code, 0.5)
    n_drivers = max(state.get("n_drivers", 20), 2)
    current_position = float(np.clip(speed_rank_pct * n_drivers, 1.0, float(n_drivers)))
    position_pct = float(np.clip(1.0 - ((current_position - 1.0) / max(n_drivers - 1, 1)), 0.0, 1.0))
    
    total_laps = state.get("total_laps", 53)
    lap_progress = lap_no / max(total_laps, 1)
    
    delta_vs_field = float(state["delta_vs_field"].get(code, 0.0))
    laps_remaining = _laps_remaining(lap_no)
    gap_urgency = delta_vs_field * (1.0 / max(laps_remaining, 1))
    
    n_grid = 22.0
    grid_pct_front = (1.0 - min(grid_pos / n_grid, 1.0))
    position_gain = position_pct - grid_pct_front
    
    tire_age_pct = _tire_age_pct(
        state["current_compound"].get(code, "UNKNOWN"),
        state["tire_age"][code]
    )
    tire_freshness = 1.0 / (1.0 + tire_age_pct)
    
    return {
        "grid_position":    grid_pos,
        "avg_finish_last5": pr.get("avg_finish_last5", 10.0),
        "points_last5":     pr.get("points_last5", 0.0),
        "dnf_rate_last5":   pr.get("dnf_rate_last5", 0.2),
        "team_enc":         _enc_team(pr.get("team", "Unknown")),
        "best_quali_lap":   pr.get("best_quali_lap", 90.0),
        "track_type_enc":   _enc_track_type(state["event"]),
        "speed_rank_pct":   speed_rank_pct,
        "delta_vs_field":   delta_vs_field,
        "tire_age":         state["tire_age"][code],
        "tire_age_pct":     tire_age_pct,
        "compound_enc":     _enc_compound(state["current_compound"].get(code, "UNKNOWN")),
        "lap_progress":     lap_progress,
        "gap_urgency":      gap_urgency,
        "tire_freshness":   tire_freshness,
        "position_gain_pct": position_gain,
    }

def _pit_feats(code: str, comp: str, row: dict, lap_no: int) -> dict:
    """
    [FIX-F2] pace_drop_5 uses current-stint-only clean speeds.
    [FIX-F4] gap_to_below_proxy now populated from state.
    [FIX-F7] must_change_compound aligned with training.
    """
    feats = _tire_feats(code, comp, row)
    feats["Stint"]                = int(state["current_stint"].get(code, 1))
    feats["teammate_pitted"]      = int(state["teammate_pitted"].get(code, False))
    feats["must_change_compound"] = _must_change_compound(code, lap_no)
    feats["track_type_enc"]       = _enc_track_type(state["event"])
                                             
    feats["gap_to_below_proxy"]   = float(state["gap_to_below"].get(code, 0.0))

    avg_s = float(row.get("avg_speed") or 0.0)
    ref_s = float(state["stint_ref_speed"].get(code, 0.0) or 0.0)
    feats["rel_speed_delta"] = ((avg_s - ref_s) / max(ref_s, 1.0)) if ref_s > 0 and avg_s > 0 else 0.0

                                 
    clean = _current_stint_clean_speeds(code)
    feats["pace_drop_5"] = _pace_drop_from_clean(clean, n=3)

    return feats


                                                                                

def _compute_ranking_probabilities(lap_no: int) -> Dict[str, Dict[str, float]]:
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
    if not HAS_RANKING_MODEL:
        return {}
    
    result = {}
                                                             
    active_only = {c: state["driver_status"].get(c, "ACTIVE") == "ACTIVE" 
                   for c in state["pre_race"].keys()}
    
    try:
        for code in state["active_drivers"]:
            if not active_only.get(code, True):
                continue

            feats = _ranking_feats(code, lap_no)
            feat_df = pd.DataFrame([feats])[RANKING_FEATS].fillna(0)

                                                                                      
                                                                                      
                                                                                      
            prob_dist = ranking_model.predict_proba(feat_df)[0]
            n_classes = len(prob_dist)                                           

                                                                            
            position_probs = {f"p{i+1}": float(prob_dist[i]) for i in range(n_classes)}
                                                                                   
            for i in range(n_classes, 20):
                position_probs[f"p{i+1}"] = 0.0

                                                                      
            podium_p   = float(sum(prob_dist[0:min(3,  n_classes)]))
            top5_p     = float(sum(prob_dist[0:min(5,  n_classes)]))
            top10_p    = float(sum(prob_dist[0:min(10, n_classes)]))
            expected_pos = float(sum((i + 1) * p for i, p in enumerate(prob_dist)))

            result[code] = {
                **position_probs,
                "podium":            podium_p,
                "top5":              top5_p,
                "top10":             top10_p,
                "points":            top10_p,
                "expected_position": expected_pos,                                   
                "win_probability":   float(prob_dist[0]),
            }

                                                                             
                                                                         
                                                                            
                                                             
        total_laps = max(state.get("total_laps", 53), 1)
        race_pct   = lap_no / total_laps
                                                                                 
        grid_alpha = max(0.25, 1.0 - race_pct * 1.5)

                                                                 
        dnf_codes = {c for c, s in state["driver_status"].items() if s in ("DNF", "DNS")}
        n_active_drivers = max(len(result), 1)

        for code in list(result.keys()):
            pr = state["pre_race"].get(code, {})
            grid_pos = pr.get("grid_position", 10)
            raw_model_pos = result[code]["expected_position"]

                                                                   
            blended = grid_alpha * grid_pos + (1.0 - grid_alpha) * raw_model_pos

                                  
            blended = max(1.0, min(float(n_active_drivers), blended))
            result[code]["expected_position"] = blended

    except Exception as e:
        logger.warning(f"Ranking model inference failed: {e}")
        return {}
    
                                                                   

                                            
    for code in state["driver_status"].keys():
        if code not in result and state["driver_status"].get(code) in ("DNF", "DNS"):
            result[code] = {f"p{i+1}": 0.0 for i in range(20)}
            result[code]["status"]  = state["driver_status"][code]
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
    event = _canonical_event_name(event)
                                                                    
    try:
        driver_map_df = ch.query_df(
            """
            SELECT
                toString(r.driver)  AS driver_num,
                r.driver_code,
                r.final_position,
                r.status,
                q.team,
                q.grid_position,
                q.best_quali_lap
            FROM f1_race_results r
            LEFT JOIN qualifying_results q
                ON  r.driver_code = q.driver_code
                AND r.event       = q.event
                AND r.year        = q.year
            WHERE r.event = {event:String}
              AND r.year  = {year:Int32}
            """,
            parameters={"event": event, "year": int(year)}
        )
    except Exception:
        driver_map_df = None

    num_to_code = {}
    if driver_map_df is not None and not driver_map_df.empty:
        num_to_code = dict(zip(driver_map_df["driver_num"], driver_map_df["driver_code"]))

                       
    pre_race_ctx: dict = {}
    try:
        feat_df = ch.query_df(
            """
            SELECT
                f.driver_code,
                f.team,
                f.grid_position,
                f.avg_finish_last5,
                f.points_last5,
                f.dnf_rate_last5,
                q.best_quali_lap
            FROM f1_features f
            LEFT JOIN qualifying_results q
                ON  f.driver_code = q.driver_code
                AND f.event       = q.event
                AND f.year        = q.year
            WHERE f.event = {event:String}
              AND f.year  = {year:Int32}
            """,
            parameters={"event": event, "year": int(year)}
        )

        for _, row in feat_df.iterrows():
            code = str(row.driver_code)
            pre_race_ctx[code] = dict(
                driver_code      = code,
                team             = str(row.team),
                grid_position    = int(row.grid_position),
                avg_finish_last5 = float(row.avg_finish_last5),
                points_last5     = float(row.points_last5),
                dnf_rate_last5   = float(row.dnf_rate_last5),
                best_quali_lap   = float(row.best_quali_lap) if pd.notna(row.best_quali_lap) else 90.0,
                final_position   = None,
                final_status     = "",
            )
    except Exception:
        pass

                                          
    try:
        quali_df = ch.query_df(
            """
            SELECT
                toString(driver) AS driver_num,
                driver_code,
                team,
                grid_position,
                best_quali_lap
            FROM qualifying_results
            WHERE event = {event:String}
              AND year  = {year:Int32}
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
                    driver_code      = code,
                    team             = str(row.team) if pd.notna(row.team) else "Unknown",
                    grid_position    = int(row.grid_position) if pd.notna(row.grid_position) else 10,
                    avg_finish_last5 = 10.0,
                    points_last5     = 0.0,
                    dnf_rate_last5   = 0.2,
                    best_quali_lap   = float(row.best_quali_lap) if pd.notna(row.best_quali_lap) else 90.0,
                    final_position   = None,
                    final_status     = "",
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
                    driver_code=code,
                    team=str(row.team) if pd.notna(row.team) else "Unknown",
                    grid_position=10,
                    avg_finish_last5=10.0,
                    points_last5=0.0,
                    dnf_rate_last5=0.2,
                    best_quali_lap=90.0,
                )
            pre_race_ctx[code]["final_position"] = (
                int(row.final_position) if pd.notna(row.final_position) else None
            )
            pre_race_ctx[code]["final_status"] = str(row.status) if pd.notna(row.status) else ""

    if not num_to_code:
        import glob
        csv_event = event.replace(" ", "_")
        search_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "main", "java", "f1producer",
                         "raw_telemetry_per_driver"),
            os.environ.get("DATASET_ROOT", ""),
        ]
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            pattern = os.path.join(d, f"{year}_{csv_event}_qualifying_results.csv")
            matches = glob.glob(pattern)
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
                                driver_code      = code,
                                team             = str(row.get("team", "Unknown")),
                                grid_position    = int(row["grid_position"]) if pd.notna(row.get("grid_position")) else 10,
                                avg_finish_last5 = 10.0,
                                points_last5     = 0.0,
                                dnf_rate_last5   = 0.2,
                                best_quali_lap   = float(row["best_quali_lap"]) if pd.notna(row.get("best_quali_lap")) else 90.0,
                            )
                        else:
                            if pd.notna(row.get("team")) and row["team"]:
                                pre_race_ctx[code]["team"] = str(row["team"])
                            if pd.notna(row.get("grid_position")):
                                pre_race_ctx[code]["grid_position"] = int(row["grid_position"])
                            if pd.notna(row.get("best_quali_lap")):
                                pre_race_ctx[code]["best_quali_lap"] = float(row["best_quali_lap"])
                print(f"  ✓ Loaded {len(num_to_code)} driver mappings from qualifying CSV")
                break
            except Exception as e:
                print(f"  ⚠ Failed to load qualifying CSV: {e}")

    try:
        drv_df = ch.query_df(
            """
            SELECT DISTINCT
                toString(driver) AS driver_num
            FROM raw_telemetry
            WHERE event   = {event:String}
              AND year    = {year:Int32}
              AND session = {session:String}
            """,
            parameters={"event": event, "year": int(year), "session": str(session)}
        )
        for _, row in drv_df.iterrows():
            raw_num = str(row.driver_num).strip()
            code = num_to_code.get(raw_num, raw_num)
            if code not in pre_race_ctx:
                pre_race_ctx[code] = dict(
                    driver_code      = code,
                    team             = "Unknown",
                    grid_position    = 10,
                    avg_finish_last5 = 10.0,
                    points_last5     = 0.0,
                    dnf_rate_last5   = 0.2,
                    best_quali_lap   = 90.0,
                )
    except Exception:
        pass

    for raw_num, code in list(num_to_code.items()):
        if raw_num in pre_race_ctx and code != raw_num:
            pre_race_ctx[code] = pre_race_ctx.pop(raw_num)
            pre_race_ctx[code]["driver_code"] = code

    total_laps = RACE_LAPS.get(event, 0)
    if total_laps == 0:
        try:
            total_df = ch.query_df(
                """
                SELECT max(LapNumber) AS max_lap
                FROM raw_telemetry
                WHERE event   = {event:String}
                  AND year    = {year:Int32}
                  AND session = {session:String}
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
        c = _normalize_driver_code(code)
        if n and c:
            normalized_map[n] = c
    num_to_code = normalized_map

    normalized_prerace = {}
    for key, ctx in pre_race_ctx.items():
        norm_key = _normalize_driver_code(key)
        if not norm_key:
            norm_key = _normalize_driver_code(ctx.get("driver_code"))
        if not norm_key:
            mapped = num_to_code.get(str(key).strip())
            norm_key = _normalize_driver_code(mapped)
        if not norm_key:
            continue
        ctx["driver_code"] = norm_key
        normalized_prerace[norm_key] = ctx
    pre_race_ctx = normalized_prerace

    return pre_race_ctx, num_to_code, total_laps


def _clickhouse_lap_to_rows(ch, event: str, year: int, session: str, lap_no: int, num_to_code: dict) -> list:
    event = _canonical_event_name(event)
    lap_tele = ch.query_df(
        """
        SELECT
            toString(driver)        AS driver_num,
            LapNumber,
            Stint,
            any(Compound)           AS Compound,
            avg(Speed)              AS avg_speed,
            max(Speed)              AS max_speed,
            avg(Throttle)           AS avg_throttle,
            avg(Brake)              AS avg_brake,
            sum(hard_brake)         AS hard_brake_count,
            sum(full_throttle)      AS full_throttle_count,
            avg(DRS)                AS avg_drs,
            avg(RPM)                AS avg_rpm,
            any(weather)            AS weather,
            any(TrackTemp)          AS track_temp,
            any(AirTemp)            AS air_temp,
            any(Rainfall)           AS rainfall,
            any(is_pit_lap)         AS is_pit_lap,
            max(Time_ms)            AS lap_finish_ms,
            max(RelativeDistance)   AS max_rel_dist
        FROM raw_telemetry
        WHERE event      = {event:String}
          AND year       = {year:Int32}
          AND session    = {session:String}
          AND LapNumber  = {lap:Int32}
        GROUP BY driver_num, LapNumber, Stint
        ORDER BY LapNumber, driver_num
        """,
        parameters={"event": event, "year": int(year), "session": str(session), "lap": int(lap_no)}
    )

    if lap_tele.empty:
        return []

    def _resolve_driver_code(row):
        mapped = _normalize_driver_code(num_to_code.get(str(row.get("driver_num")).strip()))
        if mapped:
            return mapped
        return ""

    lap_tele["driver_code"] = lap_tele.apply(_resolve_driver_code, axis=1)
    return lap_tele.where(pd.notnull(lap_tele), None).to_dict(orient="records")


def _load_cumulative_times(ch, event: str, year: int, session: str,
                           up_to_lap: int, num_to_code: dict) -> None:
    """
    Pre-populate state['cumulative_time'] and state['laps_completed'] from all
    historical laps in ClickHouse up to (and including) up_to_lap.

    max(Time_ms) per lap = actual lap duration in ms (Time_ms is lap-relative).
    Summing across laps gives total race time — the ground-truth position basis.

    Pit-in laps (is_pit_lap=1) get +PIT_STOP_LOSS_S because telemetry is absent
    during stationary pit service, so max(Time_ms) under-counts those laps.
    """
    if up_to_lap < 1:
        return
    try:
        df = ch.query_df(
            f"""
            SELECT
                driver_num,
                sum(case when is_pit = 1
                         then lap_time_s + {PIT_STOP_LOSS_S}
                         else lap_time_s
                    end)            AS total_time_s,
                max(LapNumber)      AS last_lap,
                count()             AS laps_done
            FROM (
                SELECT
                    toString(driver)        AS driver_num,
                    LapNumber,
                    max(Time_ms) / 1000.0   AS lap_time_s,
                    any(is_pit_lap)         AS is_pit
                FROM raw_telemetry
                WHERE event   = {{event:String}}
                  AND year    = {{year:Int32}}
                  AND session = {{session:String}}
                  AND LapNumber <= {{up_to_lap:Int32}}
                GROUP BY driver_num, LapNumber
            )
            GROUP BY driver_num
            """,
            parameters={"event": event, "year": int(year), "session": str(session), "up_to_lap": int(up_to_lap)}
        )
        for _, row in df.iterrows():
            code = num_to_code.get(str(row["driver_num"]), str(row["driver_num"]))
            t    = float(row["total_time_s"])
            lap  = int(row["last_lap"])
            if t > 0:
                state["cumulative_time"][code]   = t
                state["laps_completed"][code]    = lap
        state["_cumul_laps_counted"] = int(up_to_lap)
        n   = len(df)
        msg = (f"  ⏱  Pre-loaded cumulative race times for {n} drivers "
               f"(laps 1–{up_to_lap}, using actual lap durations from telemetry)")
        (console.print(f"[cyan]{msg}[/cyan]") if HAS_RICH else print(msg))
    except Exception as exc:
        logger.warning(f"Could not pre-load cumulative times: {exc}")



def _clickhouse_max_lap(ch, event: str, year: int, session: str) -> int:
    df = ch.query_df(
        """
        SELECT max(LapNumber) AS max_lap
        FROM raw_telemetry
        WHERE event   = {event:String}
          AND year    = {year:Int32}
          AND session = {session:String}
        """,
        parameters={"event": event, "year": int(year), "session": str(session)}
    )
    if df.empty or pd.isna(df.iloc[0]["max_lap"]):
        return 0
    return int(df.iloc[0]["max_lap"])


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
                CREATE TABLE IF NOT EXISTS {_pred_table}
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
                PARTITION BY (year, event)
                ORDER BY (event, year, session, lap_no, driver_code)
            """)
            _pred_client.command(f"ALTER TABLE {_pred_table} ADD COLUMN IF NOT EXISTS speed_rank UInt8")
            _pred_client.command(f"ALTER TABLE {_pred_table} ADD COLUMN IF NOT EXISTS gap_to_leader Float32")
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
