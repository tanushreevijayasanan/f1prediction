"""
F1 Prediction System - Master Model Training Orchestrator

This script trains all 6 F1 prediction models:
  1. Winner (race victory prediction)
  2. Tire Degradation
  3. Pit Stop Prediction
  4. Lap Pace Baseline
  5. Safety Car Detector
  6. Lap Time (next-lap speed) Model

All models are trained on leak-free CV folds grouped by race to ensure
no cross-race leakage.
"""

import os
import sys
import pickle
import argparse
import re
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def detect_latest_race(cache_dir="c:/telemetry-producer/src/main/java/f1producer/f1_cache"):
    """Detect latest year/event from fastf1 cache dirs."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print("  [!] Cache dir not found, few-shot disabled")
        return None, None
    
    year_event_dirs = []
    for year_dir in cache_path.glob("20*/"):
        year = int(year_dir.name)
        for event_dir in year_dir.glob("*_Grand_Prix/"):
            # Parse 'YYYY-MM-DD_Event_Name' → event name
            match = re.match(r"^\d{4}-\d{2}-\d{2}_(.+)_Grand_Prix$", event_dir.name)
            if match:
                event_name = match.group(1).replace("_", " ") + " Grand Prix"
                year_event_dirs.append((year, event_name, event_dir.stat().st_mtime))
    
    if not year_event_dirs:
        return None, None
    
    # Most recent by dir mtime
    latest = max(year_event_dirs, key=lambda x: x[2])
    year, event = latest[0], latest[1]
    print(f"  [AUTO] Detected few-shot target: {year} '{event}'")
    return year, event

def setup_fewshot_auto():
    """Auto-configure few-shot env vars."""
    parser = argparse.ArgumentParser(description="F1 Model Training")
    parser.add_argument("--fewshot-year", type=int, default=None, help="Override few-shot year")
    parser.add_argument("--fewshot-event", type=str, default=None, help="Override few-shot event")
    args = parser.parse_args()
    
    year = args.fewshot_year
    event = args.fewshot_event
    
    if year is None:
        year, event = detect_latest_race()
    
    if year and event:
        os.environ["FEWSHOT_YEAR"] = str(year)
        os.environ["FEWSHOT_TARGET_EVENT"] = event
        os.environ["FEWSHOT_WEIGHT"] = "5.0"  # Aggressive upweight
        print(f"  [OK] Few-shot AUTO-ENABLED: year={year} event='{event}' weight=5.0")
        return True
    else:
        print("  [WARN] Few-shot disabled (no target race detected)")
        os.environ["FEWSHOT_YEAR"] = "0"
        return False

from _shared_utils import MODEL_DIR, FEWSHOT_YEAR, FEWSHOT_TARGET_EVENT, FEWSHOT_WEIGHT
from train_winner_model import train_winner_model
from train_tire_model import train_tire_model
from train_pit_model import train_pit_model
from train_pace_model import train_pace_model
from train_sc_detector import train_sc_detector
from train_ranking_distribution import train_ranking_distribution

# Auto-setup few-shot
fewshot_active = setup_fewshot_auto()

print("=" * 70)
print("F1 ML SYSTEM - TRAINING  (v2 Chief Engineer Edition)")
print("=" * 70)
if fewshot_active:
    print(f"Few-shot active: year={os.getenv('FEWSHOT_YEAR')}  target='{os.getenv('FEWSHOT_TARGET_EVENT')}'  weight={os.getenv('FEWSHOT_WEIGHT', '3.0')}")


def train_all_models():
    """Train all models sequentially."""
    
    # Train models
    print("\nTraining individual models...")
    print("-" * 70)
    
    try:
        # Train tire degradation first (returns train/val splits used by other models)
        tire_model, tire_feats, tr_tire, va_tire, X_tire, y_tire, tr_groups, va_groups = train_tire_model()
        
        # Train other models
        pit_model, pit_feats = train_pit_model()
        pace_model, pace_feats = train_pace_model(tr_groups, va_groups)
        winner_model, winner_feats, le_team = train_winner_model()
        sc_threshold = train_sc_detector()
        
        # Train ranking distribution model (NEW: for probability distributions)
        ranking_model, ranking_feats = train_ranking_distribution()

        # Train lap time model (required)
        import subprocess
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "train_laptime.py")])
        with open(os.path.join(MODEL_DIR, "laptime_feats.pkl"), "rb") as f:
            laptime_feats = pickle.load(f)
        
        # Common artifacts (not model-specific)
        from _shared_utils import (
            save_artifact,
            COMPOUND_CLASSES,
            load_reference_data,
            load_lap_telemetry,
        )
        
        race_results, _, _ = load_reference_data()
        lap_df, _, median_stint_by_compound = load_lap_telemetry(race_results)
        
        save_artifact(MODEL_DIR, "compound_classes.pkl", COMPOUND_CLASSES)
        save_artifact(MODEL_DIR, "median_stint_lengths.pkl", median_stint_by_compound)
        
        print("\n" + "=" * 70)
        print("All models trained successfully!")
        print("=" * 70)
        
        print("\nFeature summary:")
        print(f"  WINNER_FEATS  ({len(winner_feats)}): {winner_feats}")
        print(f"  RANKING_FEATS ({len(ranking_feats)}): {ranking_feats}")
        print(f"  TIRE_FEATS    ({len(tire_feats)}):   {tire_feats}")
        print(f"  PIT_FEATS     ({len(pit_feats)}):    {pit_feats}")
        print(f"  PACE_FEATS    ({len(pace_feats)}):   {pace_feats}")
        print(f"  LAPTIME_FEATS ({len(laptime_feats)}): {laptime_feats}")
        
        print(f"\nAll artifacts -> {MODEL_DIR}")
        print("\nTraining complete.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)
