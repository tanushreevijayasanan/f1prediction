import argparse
import os
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clickhouse-live", action="store_true",
                        help="Poll ClickHouse raw_telemetry table for new laps and render live")
    parser.add_argument("--event", default=os.getenv("F1_EVENT", ""))
    parser.add_argument("--year", type=int, default=int(os.getenv("F1_YEAR", "0") or 0))
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
    parser.add_argument("--preds-table", default=os.getenv("PRED_TABLE", "prediction_results"),
                        help="ClickHouse predictions table name")
    parser.add_argument("--ui-backend", default="http://localhost:8000",
                        help="UI backend URL for pushing commentary (e.g., http://localhost:8000)")
    args = parser.parse_args()

    #backward-compatible defaults still used internally by inference_engine.
    args.log = None
    args.alerts_only = False
    args.no_clear = False
    args.history = 10
    args.fan_insights = True
    args.grid_prior_laps = 3
    args.grid_blend_laps = 20
    args.grid_prior_weight = 0.95
    args.grid_position_regularization = 0.15
    args.min_active_for_speed_rank = 10
    args.speed_rank_ema = 0.35
    args.win_proba_ema = 0.20
    args.win_proba_max_delta = 0.05
    return args

