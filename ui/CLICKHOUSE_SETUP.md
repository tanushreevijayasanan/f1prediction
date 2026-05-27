# UI ↔ ClickHouse Integration Guide

## Architecture

```
Inference Engine
    ↓ (writes to)
ClickHouse prediction_results table
    ↓ (reads from)
FastAPI Backend (main.py)
    ↓ (WebSocket broadcast)
Frontend (index.html + app.js)
    ↓ (displays)
Live Dashboard with Pit Strategy & Driver Insights
```

## Setup

### 1. Install Dependencies
```bash
cd ui
pip install -r requirements.txt
```

### 2. Ensure ClickHouse is Running
```bash
# Docker (if using docker-compose)
docker-compose up -d clickhouse

# Or native ClickHouse server
clickhouse server
```

### 3. Configure Environment (Optional)
```bash
export CH_HOST=localhost
export CH_PORT=8123
export CH_USER=default
export CH_PASSWORD=
```

Default values:
- Host: `localhost`
- Port: `8123`
- User: `default`
- Password: (empty)

### 4. Run Inference Engine
```bash
cd ../
python f1_predictions/inference_engine.py \
  --clickhouse-live \
  --event "2026_Japanese_Grand_Prix" \
  --year 2026 \
  --session R \
  --write-preds
```

This writes predictions to `prediction_results` table every lap.

### 5. Run UI Backend
```bash
cd ui
python main.py
# or
uvicorn main:app --reload
```

Server runs on `http://localhost:8000`

### 6. Open Dashboard
```
http://localhost:8000
```

## Data Flow

### Inference Engine → ClickHouse
```python
# Writes to prediction_results table each lap:
{
  "event": "2026_Japanese_Grand_Prix",
  "year": 2026,
  "LapNumber": 45,
  "driver": "VER",
  "speed_rank": 1,
  "avg_speed": 250.5,
  "win_proba": 0.85,
  "pit_within_horizon": 0,
  "pit_proba": 0.15,
  "tire_age": 8,
  "compound_enc": 2,
  "rel_speed_delta": 0.005,
  "stint": 1,
  "timestamp": "2026-03-31T12:34:56",
  ...
}
```

### ClickHouse → UI Backend
The backend queries:
1. **Latest lap** - `MAX(LapNumber)` to know current race status
2. **Leaderboard** - Current positions with predictions
3. **Pit Strategy** - Per-driver pit probability, tire age, compound
4. **Historical Stats** - Last 10 laps average performance

### UI Backend → Frontend
WebSocket broadcasts dashboard state every 500ms:
```json
{
  "lap": 45,
  "event": "2026_Japanese_Grand_Prix",
  "year": 2026,
  "positions": [["VER", 250.5], ["NOR", 248.2], ...],
  "predictions": {
    "VER": {"win_prob": 0.85, "pit_prob": 0.15},
    ...
  },
  "pit_strategies": {
    "VER": {
      "pit_probability": 0.15,
      "tire_age": 8,
      "compound": "HARD",
      "pit_urgency": "LOW",
      ...
    },
    ...
  },
  "sc_active": false
}
```

## Dashboard Features

### Leaderboard
- Position, Driver name, Speed, Win%, Pit Urgency
- Pit Urgency: HIGH (red) | MEDIUM (yellow) | LOW (green)

### Pit Strategy Panel
- **High Urgency Alerts** - Drivers needing pit stops soon
- **Pit Candidates** - Cards showing pit probability > 30%
  - Tire age, compound, degradation, stint number

### Driver Performance
- Last 10-lap stats for top drivers
- Tire degradation visualization
- Compound age tracking

### Race Aggregates
- Average field speed
- Active pit stop count
- Safety car status

## Queries Run by Backend

### Get Current Lap
```sql
SELECT MAX(LapNumber) as max_lap
FROM prediction_results
WHERE event = ? AND year = ?
```

### Get Leaderboard
```sql
SELECT 
  driver, LapNumber, speed_rank, avg_speed,
  win_proba, pit_within_horizon, gap_to_leader
FROM prediction_results
WHERE event = ? AND year = ? AND LapNumber = ?
ORDER BY speed_rank ASC
LIMIT 20
```

### Get Pit Strategy
```sql
SELECT 
  driver, pit_within_horizon, tire_age, 
  compound_enc, rel_speed_delta, stint
FROM prediction_results
WHERE event = ? AND year = ? AND driver = ? AND LapNumber = ?
ORDER BY LapNumber DESC
LIMIT 1
```

### Get Historical Stats (Last N Laps)
```sql
SELECT 
  driver, count(), avg(avg_speed), avg(win_proba),
  avg(pit_proba), max(avg_speed), min(avg_speed)
FROM prediction_results
WHERE event = ? AND year = ? AND driver = ?
  AND LapNumber >= ?
GROUP BY driver
```

## REST API Endpoints

- `WS` `/ws` - WebSocket: Real-time dashboard state (500ms updates)
- `GET` `/state` - Current dashboard state (JSON)
- `GET` `/predictions/{driver}` - Driver pit strategy & historical stats
- `GET` `/health` - Health check (ClickHouse connection status)

## Troubleshooting

### "ClickHouse connection failed"
- Ensure ClickHouse is running: `docker-compose up -d clickhouse`
- Check host/port: `CH_HOST=localhost CH_PORT=8123`
- Verify table exists: Query `SELECT * FROM prediction_results LIMIT 1` in ClickHouse CLI

### "No predictions in dashboard"
- Ensure inference engine is running and writing: `python f1_predictions/inference_engine.py --write-preds`
- Check prediction_results table for data: `SELECT count() FROM prediction_results`
- Verify event/year match: Default is `2026_Japanese_Grand_Prix` year `2026`

### "WebSocket not connecting"
- Check CORS: FastAPI should auto-allow localhost
- Verify port 8000 is accessible: `curl http://localhost:8000/health`
- Browser console for WebSocket errors: F12 → Console

### High latency / Slow queries
- Add indexes on (event, year, driver, LapNumber) on prediction_results
- Profile queries in ClickHouse: Add `FORMAT JSON` to queries
- Check ClickHouse memory/CPU usage

## Performance Tuning

### ClickHouse Table Optimization
```sql
-- Create indexes for faster lookups
ALTER TABLE prediction_results
ADD INDEX idx_event_year (event, year) TYPE hash,
ADD INDEX idx_driver (driver) TYPE hash,
ADD INDEX idx_lap (LapNumber) TYPE minmax;

-- Add TTL to auto-archive old races
ALTER TABLE prediction_results
MODIFY TTL toDateTime(timestamp) + INTERVAL 7 DAY;
```

### Materialized View (Optional)
```sql
-- Pre-aggregate pit strategies for faster queries
CREATE MATERIALIZED VIEW prediction_results_pit_mv AS
SELECT 
  event, year, LapNumber, driver,
  max(pit_within_horizon) as pit_prob,
  max(tire_age) as tire_age,
  max(compound_enc) as compound,
  max(stint) as stint
FROM prediction_results
GROUP BY event, year, LapNumber, driver;
```

## Next Steps

1. ✅ Run inference engine with `--write-preds` flag
2. ✅ Start UI backend: `python main.py`
3. ✅ Open dashboard: `http://localhost:8000`
4. ✅ Watch live pit strategy predictions!
