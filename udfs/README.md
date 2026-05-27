# F1 UDFs

## Current split

- Utility functions remain SQL UDFs.
- Model scoring functions are executable UDFs.

## Utility SQL UDFs

Use [`utilities_only.sql`](./utilities_only.sql):

- `f1_team_encoder()`
- `f1_track_encoder()`
- `f1_compound_encoder()`
- `f1_tire_encoder()`
- `f1_stint_length_median()`
- `f1_speed_threshold()`

## Model executable UDFs

See [`executable/README.md`](./executable/README.md):

- `f1_win_probability_exec()`
- `f1_pit_probability_exec()`
- `f1_pace_prediction_exec()`

## Migration steps

1. Remove legacy SQL model UDFs with [`drop_model_sql_udfs.sql`](./drop_model_sql_udfs.sql).
2. Install utility SQL UDFs with [`utilities_only.sql`](./utilities_only.sql).
3. Deploy executable scripts + XML config from [`executable/`](./executable/).

## Accuracy note

For highest parity with Python inference, executable UDF inputs must be fed with feature columns that match `inference_engine.py` feature engineering semantics.
