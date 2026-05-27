"""
Configuration loader for F1 Telemetry Producer
Provides centralized access to all constants and settings
"""

import os
import yaml
from pathlib import Path

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    Defaults to ./config.yaml
    """
    if config_path is None:
        # Try to find config.yaml in parent directory of this script
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_config():
    """
    Singleton-style access to configuration.
    Uses environment variable CONFIG_PATH if set, otherwise defaults to ./config.yaml
    """
    config_env = os.getenv("CONFIG_PATH")
    return load_config(config_env)

# Load config on module import (cached)
try:
    _config = get_config()
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}. Using hardcoded defaults.")
    _config = {
        "training": {
            "pit_horizon": 3,
            "fewshot_weight": 3.0,
            "train_test_split": 0.75,
        },
        "models": {
            "pit": {"scale_pos_weight": 0.3, "alert_threshold": 0.40},
            "winner": {"use_isotonic_calibration": True},
        },
        "inference": {
            "grid_prior_laps": 3,
            "grid_blend_laps": 20,
            "grid_prior_weight": 0.95,
            "grid_position_regularization": 0.15,
            "speed_rank_ema": 0.35,
            "win_proba_ema": 0.20,
            "state_memory": {
                "all_speeds_keep_last": 100,
                "lap_history_keep_last": 30,
                "stint_speeds_keep_last": 50,
                "pit_history_keep_last": 20,
            }
        }
    }

def get(key_path, default=None):
    """
    Get config value by dot-separated path.
    Example: get("inference.grid_prior_laps") -> 3
    """
    keys = key_path.split(".")
    value = _config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value or default
