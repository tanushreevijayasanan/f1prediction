# F1 Model Training - Modular Architecture

This directory contains the modularized F1 prediction model training pipeline. Each model has been split into its own file for better organization, maintainability, and reusability.

## Directory Structure

```
training_models/
├── __init__.py                      # Package initialization
├── _shared_utils.py                 # Shared utilities and common functions
├── train_all_models.py              # Master orchestrator script
├── train_winner_model.py            # Winner (race victory) prediction model
├── train_tire_model.py              # Tire degradation model
├── train_pit_model.py               # Pit stop prediction model
├── train_pace_model.py              # Lap pace baseline model
└── train_sc_detector.py             # Safety car / neutralisation detector
????????? train_laptime.py                 # Lap time / next-lap speed model
```

## Models Overview

### 1. Winner Model (`train_winner_model.py`)
- **Type**: XGBoost Classifier
- **Task**: Predicts probability a driver will win the race
- **Features**: Grid position, historical performance, on-track pace, tire state
- **Output**: `winner_model.pkl`, `winner_feats.pkl`, `team_encoder.pkl`

### 2. Tire Degradation Model (`train_tire_model.py`)
- **Type**: Gradient Boosting Regressor
- **Task**: Predicts tire pace degradation (relative speed delta)
- **Features**: Compound, tire age, driving inputs, track conditions
- **Training**: Clean laps only (excludes SC/VSC periods)
- **Output**: `tire_model.pkl`, `tire_feats.pkl`

### 3. Pit Stop Model (`train_pit_model.py`)
- **Type**: XGBoost Classifier
- **Task**: Predicts if driver will pit within next N laps (horizon)
- **Features**: Tire state, degradation signals, undercut pressure, strategic factors
- **Output**: `pit_model.pkl`, `pit_feats.pkl`, `pit_horizon.pkl`

### 4. Pace Baseline Model (`train_pace_model.py`)
- **Type**: Ridge Regression
- **Task**: Predicts absolute lap speed baseline
- **Features**: Tire state, driving inputs, track conditions, race context
- **Output**: `pace_model.pkl`, `pace_feats.pkl`

### 5. Safety Car Detector (`train_sc_detector.py`)
- **Type**: Threshold-based detector
- **Task**: Detects neutralized laps (safety car, VSC)
- **Method**: Compares field speed to race baseline
- **Output**: `sc_speed_threshold.pkl`

### 6. Lap Time Model (`train_laptime.py`)
- **Type**: XGBoost Regressor
- **Task**: Predicts next-lap average speed (converted to lap time)
- **Features**: Current lap pace, tire state, degradation, race context
- **Output**: `laptime_model.pkl`, `laptime_feats.pkl`, `circuit_lengths.pkl`, `laptime_mae_s.pkl`

### Track Type Features (All Models)
- **Type**: Event-level categorical feature
- **Task**: Encodes circuit style (e.g., street, high-speed, downforce)
- **Output**: `track_type_map.pkl`, `track_type_encoder.pkl`

### Shared Artifacts
- `compound_classes.pkl` - Tire compound encoding
- `median_stint_lengths.pkl` - Typical stint lengths by compound
- `team_encoder.pkl` - Team label encoder

## Usage

### Train All Models
```python
python train_all_models.py
```

This orchestrates training of all 6 models in sequence, with proper train/test splitting and cross-race validation.

### Train Individual Models
```python
from train_winner_model import train_winner_model
from train_tire_model import train_tire_model
from train_pit_model import train_pit_model
from train_pace_model import train_pace_model
from train_sc_detector import train_sc_detector

# Train individual models
winner_model, winner_feats, le_team = train_winner_model()
tire_model, tire_feats, tr, va, X, y = train_tire_model()
pit_model, pit_feats = train_pit_model()
pace_model, pace_feats = train_pace_model()
sc_threshold = train_sc_detector()
```

Lap time model:
```python
python train_laptime.py
```

### Evaluate Models (Single Race)
```python
python evaluate_winner_model.py --event "Japanese Grand Prix" --year 2026
python evaluate_tire_model.py --event "Japanese Grand Prix" --year 2026
python evaluate_pit_model.py --event "Japanese Grand Prix" --year 2026
python evaluate_pace_model.py --event "Japanese Grand Prix" --year 2026
python evaluate_laptime_model.py --event "Japanese Grand Prix" --year 2026
```

### Evaluate All Models (One Script)
```python
python evaluate_all_models.py
python evaluate_all_models.py --event "Japanese Grand Prix" --year 2026
```

### Benchmark Model Families
```python
python benchmark_model_families.py --task all
python benchmark_model_families.py --task winner
python benchmark_model_families.py --task tire --save-csv
```

### Pit Threshold Tuning
```python
python evaluate_all_models.py --tune-pit-threshold
python evaluate_all_models.py --tune-pit-threshold --pit-objective precision --pit-min-precision 0.5
```

## Key Improvements

### Clean Code Organization
- Each model has its own focused module
- Shared utilities extracted to `_shared_utils.py`
- No monolithic 1000+ line training script

### Easier Testing & Debugging
- Individual model training isolated
- Feature engineering pipeline is modular
- Easy to add new models or modify existing ones

### Better Maintainability
- Clear separation of concerns
- Feature sets defined in one place per model
- Shared preprocessing prevents duplication

### Production Ready
- Models saved as pickle artifacts
- Feature sets versioned with models
- Threshold-based detectors don't require heavy ML libraries

## Data Processing Pipeline

The training pipeline follows this sequence:

1. **Load Reference Data** (driver history, qualifying, race results)
2. **Load Lap Telemetry** (raw speed, throttle, brake, DRS per lap)
3. **Clean Lap Filter** (exclude SC/VSC periods)
4. **Compute Speed Features** (degradation, speed rank, gaps)
5. **Compute Tactical Features** (pit timing, team pressure, mandatory changes)
6. **Train Models** (group-shuffled CV by race to prevent leakage)

## Configuration

Environment variables:
- `CH_HOST` - ClickHouse server (default: localhost)
- `CH_PORT` - ClickHouse port (default: 8123)
- `MODEL_DIR` - Output directory for models (default: ./models)
- `FEWSHOT_YEAR` - Enable few-shot weighting for a season (e.g., 2026)
- `FEWSHOT_TARGET_EVENT` - Target race to exclude from training (exact event string)
- `FEWSHOT_WEIGHT` - Weight for all races in FEWSHOT_YEAR (default: 3.0)
- `TRACK_TYPE_MAP_PATH` - Optional JSON file to override the default event→track type mapping

Model hyperparameters are hardcoded in each training file based on extensive tuning.

## Feature Leakage Prevention

All models use `GroupShuffleSplit` on `(event, year)` to ensure:
- No laps from the same race appear in train and validation
- Models generalize to unseen races
- No temporal leakage from future race laps

## Integration

To use trained models in inference:

```python
import pickle

# Load models
with open('models/winner_model.pkl', 'rb') as f:
    winner_model = pickle.load(f)

with open('models/winner_feats.pkl', 'rb') as f:
    winner_feats = pickle.load(f)

# Prepare features and predict
predictions = winner_model.predict_proba(X[winner_feats])[:, 1]
```

See `inference_engine.py` for full production inference pipeline.
