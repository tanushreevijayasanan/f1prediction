# Executable Model UDFs

This folder contains model-scoring executable UDF handlers:

- `winner_exec_udf.py` -> `f1_win_probability_exec`
- `pit_exec_udf.py` -> `f1_pit_probability_exec`
- `pace_exec_udf.py` -> `f1_pace_prediction_exec`

## Accuracy-first design

- Each script loads the same trained model artifact used by `inference_engine.py`.
- Feature names are read from the corresponding `*_feats.pkl` file.
- Input row values are mapped into those exact feature names before scoring.

## Deploy

1. Copy scripts into ClickHouse user scripts path:
   - `/var/lib/clickhouse/user_scripts/`
2. Copy model artifacts into:
   - `/var/lib/clickhouse/user_scripts/models/`
3. Copy [`clickhouse_executable_functions.xml`](./clickhouse_executable_functions.xml) into ClickHouse config dir (`config.d`).
4. Restart ClickHouse.
5. Validate:
   - `SELECT f1_win_probability_exec(...);`
   - `SELECT f1_pit_probability_exec(...);`
   - `SELECT f1_pace_prediction_exec(...);`

## Important

- `TabSeparated` argument order in XML must match each script's `INPUT_COLUMNS`.
- For maximum parity, feature engineering SQL/view logic must match the Python inference feature computation semantics.
