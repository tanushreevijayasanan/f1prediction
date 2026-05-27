from __future__ import annotations

import os
import time
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_clean_lap_filter,
    compute_speed_features,
    compute_gap_and_pace_features,
    compute_teammate_and_compound_features,
    compute_pit_features,
    add_track_type_features,
    MODEL_DIR,
)
from train_winner_model import (
    _detect_sc_laps,
    _compute_actual_position,
    _compute_gap_to_leader,
    _compute_position_jump,
)


@dataclass
class StageTimer:
    name: str
    start: float


class Timings:
    def __init__(self) -> None:
        self._records: Dict[str, float] = {}
        self._stack: list[StageTimer] = []

    def start(self, name: str) -> None:
        self._stack.append(StageTimer(name=name, start=time.perf_counter()))

    def stop(self, name: str) -> None:
        if not self._stack:
            return
        timer = self._stack.pop()
        if timer.name != name:
            # best-effort: tolerate mis-nesting, still record timing
            pass
        dur = time.perf_counter() - timer.start
        self._records[name] = self._records.get(name, 0.0) + dur

    def report(self) -> None:
        total = sum(self._records.values()) or 1.0
        items = sorted(self._records.items(), key=lambda x: x[1], reverse=True)
        print("\n[BOTTLENECKS] Top stages by runtime:")
        for name, sec in items[:10]:
            pct = sec / total * 100
            print(f"  {name:<35} {sec:>7.2f}s  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<35} {total:>7.2f}s  (100.0%)")


def _safe_precision(y_true, y_pred) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return precision_score(y_true, y_pred, zero_division=0)


def _safe_recall(y_true, y_pred) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return recall_score(y_true, y_pred, zero_division=0)


def _safe_f1(y_true, y_pred) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return f1_score(y_true, y_pred, zero_division=0)


def _safe_roc_auc(y_true, y_prob) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)


def _safe_pr_auc(y_true, y_prob) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_prob)


def _fmt(maybe_val: Optional[float], digits: int = 4) -> str:
    if maybe_val is None or (isinstance(maybe_val, float) and np.isnan(maybe_val)):
        return "n/a"
    return f"{maybe_val:.{digits}f}"


def _best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
    min_precision: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Find threshold that maximizes objective in {"f1","precision","recall"}.
    Returns (threshold, precision, recall, f1).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan"), float("nan"), float("nan")

    best_t = 0.5
    best_score = -1.0
    best_p = best_r = best_f1 = 0.0

    # Evaluate over a dense grid; PR curve thresholds are fine too,
    # but this keeps it simple and stable.
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        if min_precision is not None and p < min_precision:
            continue

        if objective == "precision":
            score = p
        elif objective == "recall":
            score = r
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_t, best_p, best_r, best_f1 = t, p, r, f1

    return best_t, best_p, best_r, best_f1


def _normalize_event_name(event: str) -> str:
    if not isinstance(event, str):
        return str(event)
    return event.replace("S\u00e3o", "Sao").replace("SÃ£o", "Sao")


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_model_artifacts(model_dir: str) -> Dict[str, object]:
    artifacts = {}
    for name in [
        "winner_model.pkl",
        "winner_feats.pkl",
        "team_encoder.pkl",
        "pit_model.pkl",
        "pit_feats.pkl",
        "pit_horizon.pkl",
        "tire_model.pkl",
        "tire_feats.pkl",
        "pace_model.pkl",
        "pace_feats.pkl",
        "laptime_model.pkl",
        "laptime_feats.pkl",
        "circuit_lengths.pkl",
        "laptime_mae_s.pkl",
        "sc_speed_threshold.pkl",
    ]:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            artifacts[name] = _load_pickle(path)
        else:
            print(f"[WARN] Missing artifact: {name} (skipping related eval)")
    return artifacts


def _encode_team(le, series: pd.Series) -> pd.Series:
    vals = series.fillna("Unknown").astype(str)
    known = set(le.classes_)
    if "Unknown" not in known:
        vals = vals.apply(lambda v: v if v in known else le.classes_[0])
        return pd.Series(le.transform(vals), index=series.index)
    vals = vals.apply(lambda v: v if v in known else "Unknown")
    return pd.Series(le.transform(vals), index=series.index)


def _filter_event_year(df: pd.DataFrame, event: Optional[str], year: Optional[int]) -> pd.DataFrame:
    if event:
        df = df[df["event"].map(_normalize_event_name) == _normalize_event_name(event)]
    if year:
        df = df[df["year"] == int(year)]
    return df


def _compute_pace_drop_3(lap_df: pd.DataFrame) -> pd.DataFrame:
    # 3-lap rolling pace drop (stint-local clean pace fade)
    print("  Computing pace_drop_3 (stint-local clean pace fade)...")
    lap_df = lap_df.sort_values(["driver", "event", "year", "Stint", "LapNumber"]).reset_index(drop=True)
    lap_df["pace_drop_3"] = 0.0
    gc = ["driver", "event", "year", "Stint"]
    for _, idx in lap_df.groupby(gc, sort=False).indices.items():
        spd = lap_df.loc[idx, "avg_speed"].to_numpy(dtype=float)
        cln = lap_df.loc[idx, "is_clean_lap"].to_numpy(dtype=int)
        out = np.zeros(len(spd))
        for i in range(len(spd)):
            hist_clean = spd[:i + 1][cln[:i + 1] == 1]
            if len(hist_clean) >= 4:
                early = float(np.mean(hist_clean[:2]))
                late = float(np.mean(hist_clean[-2:]))
                out[i] = (early - late) / max(early, 1e-6)
        lap_df.loc[idx, "pace_drop_3"] = out
    return lap_df


def evaluate_models(
    event: Optional[str],
    year: Optional[int],
    tune_pit_threshold: bool,
    pit_objective: str,
    pit_min_precision: Optional[float],
) -> None:
    timings = Timings()

    timings.start("load_reference_data")
    race_results, features_df, qualifying = load_reference_data()
    timings.stop("load_reference_data")

    timings.start("load_lap_telemetry")
    lap_df, _, _ = load_lap_telemetry(race_results)
    timings.stop("load_lap_telemetry")

    timings.start("apply_clean_lap_filter")
    lap_df = apply_clean_lap_filter(lap_df)
    timings.stop("apply_clean_lap_filter")

    timings.start("compute_speed_features")
    lap_df = compute_speed_features(lap_df)
    timings.stop("compute_speed_features")

    timings.start("compute_gap_and_pace_features")
    lap_df = compute_gap_and_pace_features(lap_df)
    timings.stop("compute_gap_and_pace_features")

    timings.start("compute_teammate_and_compound_features")
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    timings.stop("compute_teammate_and_compound_features")

    timings.start("compute_pit_features")
    lap_df = compute_pit_features(lap_df)
    timings.stop("compute_pit_features")

    timings.start("add_track_type_features")
    lap_df = add_track_type_features(lap_df, MODEL_DIR)
    timings.stop("add_track_type_features")

    artifacts = _load_model_artifacts(MODEL_DIR)

    lap_df_f = _filter_event_year(lap_df, event, year)
    race_results_f = _filter_event_year(race_results, event, year)
    features_df_f = _filter_event_year(features_df, event, year)
    qualifying_f = _filter_event_year(qualifying, event, year)

    print("\n[SUMMARY] Evaluation scope:")
    if event or year:
        print(f"  event={event or 'ANY'}  year={year or 'ANY'}")
    else:
        print("  full dataset (all events/years)")
    print(f"  laps used (post-filter): {len(lap_df_f):,}")

    if "winner_model.pkl" in artifacts and "winner_feats.pkl" in artifacts and "team_encoder.pkl" in artifacts:
        print("\n[WINNER MODEL]")
        model = artifacts["winner_model.pkl"]
        feats = artifacts["winner_feats.pkl"]
        le_team = artifacts["team_encoder.pkl"]

        winner_src = lap_df_f.copy()
        winner_src = _detect_sc_laps(winner_src)
        winner_src = _compute_actual_position(winner_src)
        winner_src = _compute_gap_to_leader(winner_src)
        winner_src = _compute_position_jump(winner_src, n=2)
        winner_src = _compute_position_jump(winner_src, n=3)

        mid_race = winner_src[winner_src["LapNumber"] % 5 == 0].copy()
        winner_targets = race_results_f[race_results_f["status"] == "Finished"][
            ["driver_code", "event", "year", "final_position", "grid_position"]
        ].rename(columns={"driver_code": "driver"})

        _feats_merge = features_df_f[
            ["driver_code", "event", "year", "avg_finish_last5", "points_last5", "dnf_rate_last5"]
        ].rename(columns={"driver_code": "driver"})

        _quali_merge = qualifying_f[
            ["driver_code", "event", "year", "best_quali_lap"]
        ].rename(columns={"driver_code": "driver"})

        winner_df = (
            mid_race[
                [
                    "driver", "event", "year", "LapNumber", "laps_remaining",
                    "avg_speed", "speed_rank_pct", "delta_vs_field",
                    "tire_age", "tire_age_pct", "compound_enc", "team",
                    "sc_active",
                    "current_position", "position_pct", "n_active",
                    "gap_to_leader_s",
                    "position_jump_2", "position_jump_3",
                    "track_type_enc",
                ]
            ]
            .merge(winner_targets, on=["driver", "event", "year"], how="inner")
            .merge(_feats_merge, on=["driver", "event", "year"], how="left")
            .merge(_quali_merge, on=["driver", "event", "year"], how="left")
        )

        if len(winner_df) == 0:
            print("  No rows after merges; skipping.")
        else:
            winner_df["target_win"] = (winner_df["final_position"] == 1).astype(int)
            winner_df["grid_position_group"] = winner_df["grid_position"].apply(
                lambda gp: 1 if gp <= 5 else (2 if gp <= 15 else 3)
            )
            winner_df["position_gain_pct"] = (
                winner_df["speed_rank_pct"] - (winner_df["grid_position"] / 20.0).clip(0, 1)
            ).fillna(0)
            total_laps_map = (
                winner_df.groupby(["event", "year"])["LapNumber"].max()
                .rename("total_laps")
                .reset_index()
            )
            winner_df = winner_df.merge(total_laps_map, on=["event", "year"], how="left")
            winner_df["total_laps"] = winner_df["total_laps"].fillna(60)
            winner_df["lap_progress"] = winner_df["LapNumber"] / winner_df["total_laps"].clip(lower=1)
            winner_df["is_late_race"] = (winner_df["lap_progress"] > 0.75).astype(int)
            winner_df["tire_freshness"] = 1.0 / (1.0 + winner_df["tire_age_pct"])
            winner_df["gap_laps_remaining"] = (
                winner_df["gap_to_leader_s"] / winner_df["laps_remaining"].clip(lower=1)
            )
            winner_df["is_leading"] = (winner_df["current_position"] == 1.0).astype(int)
            winner_df["leading_and_late"] = winner_df["position_pct"] * winner_df["lap_progress"]
            winner_df["gap_urgency"] = (
                winner_df["delta_vs_field"] / winner_df["laps_remaining"].clip(lower=1)
            )
            winner_df["sc_beneficiary"] = (
                (winner_df["sc_active"] == 1) & (winner_df["position_jump_2"] > 2)
            ).astype(int)

            winner_df["team_enc"] = _encode_team(le_team, winner_df["team"])
            winner_df["best_quali_lap"] = winner_df["best_quali_lap"].fillna(
                winner_df["best_quali_lap"].median()
            )
            winner_df["grid_position"] = winner_df["grid_position"].fillna(20)
            winner_df["avg_finish_last5"] = winner_df["avg_finish_last5"].fillna(10.0)
            winner_df["points_last5"] = winner_df["points_last5"].fillna(0.0)
            winner_df["dnf_rate_last5"] = winner_df["dnf_rate_last5"].fillna(0.2)

            missing_feats = [f for f in feats if f not in winner_df.columns]
            if missing_feats:
                print(f"  Missing winner features for evaluation: {missing_feats}")
                print("  Skipping winner eval; retrain or align feature pipeline.")
                winner_df = pd.DataFrame()
            if len(winner_df) == 0:
                pass
            else:
                winner_df[feats] = winner_df[feats].fillna(
                    winner_df[feats].median(numeric_only=True)
                )

                X = winner_df[feats]
                y = winner_df["target_win"]
                prob = model.predict_proba(X)[:, 1]
                pred = (prob >= 0.5).astype(int)

                print(f"  precision@0.5 : {_fmt(_safe_precision(y, pred))}")
                print(f"  recall@0.5    : {_fmt(_safe_recall(y, pred))}")
                print(f"  f1@0.5        : {_fmt(_safe_f1(y, pred))}")
                print(f"  ROC AUC       : {_fmt(_safe_roc_auc(y, prob))}")
                print(f"  PR AUC        : {_fmt(_safe_pr_auc(y, prob))}")

                grp = winner_df.copy()
                grp["prob"] = prob
                race_top = (
                    grp.groupby(["event", "year", "driver"])["prob"].mean()
                    .reset_index()
                    .sort_values(["event", "year", "prob"], ascending=[True, True, False])
                )
                top_pick = race_top.groupby(["event", "year"]).head(1)
                top_pick = top_pick.merge(
                    winner_targets[["driver", "event", "year", "final_position"]],
                    on=["driver", "event", "year"],
                    how="left",
                )
                if len(top_pick):
                    acc_top1 = (top_pick["final_position"] == 1).mean()
                    print(f"  race top-1 accuracy : {acc_top1:.3f}  (n={len(top_pick)})")

                grp2 = winner_df.copy()
                grp2["prob"] = prob
                keys = ["event", "year", "LapNumber"]
                hit_counts = []
                for _, g in grp2.groupby(keys):
                    if len(g) < 10:
                        continue
                    pred_top10 = set(g.nlargest(10, "prob")["driver"].astype(str))
                    actual_top10 = set(g[g["final_position"] <= 10]["driver"].astype(str))
                    if len(actual_top10) >= 10:
                        hit_counts.append(len(pred_top10 & actual_top10))
                if hit_counts:
                    mean_hits = float(np.mean(hit_counts))
                    p80_hits = float(np.percentile(hit_counts, 80))
                    print(f"  lap-wise top-10 hits : {mean_hits:.2f}/10  (p80={p80_hits:.1f}/10)")

    if "pit_model.pkl" in artifacts and "pit_feats.pkl" in artifacts:
        print("\n[PIT MODEL]")
        model = artifacts["pit_model.pkl"]
        feats = artifacts["pit_feats.pkl"]

        pit_df = lap_df_f[~lap_df_f["is_pit_lap"].astype(bool)].copy()
        pit_df = pit_df[feats + ["pit_within_horizon"]].copy()
        pit_df = pit_df.fillna(pit_df.median(numeric_only=True))

        if len(pit_df) == 0:
            print("  No rows to evaluate; skipping.")
        else:
            X = pit_df[feats]
            y = pit_df["pit_within_horizon"]
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)

            print(f"  precision@0.5 : {_fmt(_safe_precision(y, pred))}")
            print(f"  recall@0.5    : {_fmt(_safe_recall(y, pred))}")
            print(f"  f1@0.5        : {_fmt(_safe_f1(y, pred))}")
            print(f"  ROC AUC       : {_fmt(_safe_roc_auc(y, prob))}")
            print(f"  PR AUC        : {_fmt(_safe_pr_auc(y, prob))}")

            if tune_pit_threshold:
                t, p, r, f1v = _best_threshold(
                    y.to_numpy(),
                    prob,
                    objective=pit_objective,
                    min_precision=pit_min_precision,
                )
                print("  [TUNED] best threshold:")
                print(f"    objective  : {pit_objective}")
                if pit_min_precision is not None:
                    print(f"    min_prec   : {pit_min_precision:.2f}")
                print(f"    threshold  : {t:.2f}")
                print(f"    precision  : {_fmt(p)}")
                print(f"    recall     : {_fmt(r)}")
                print(f"    f1         : {_fmt(f1v)}")
                try:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    with open(os.path.join(MODEL_DIR, "pit_threshold.pkl"), "wb") as f:
                        pickle.dump(float(t), f)
                    print(f"    saved      : pit_threshold.pkl")
                except Exception as e:
                    print(f"    save_error : {e}")

    if "tire_model.pkl" in artifacts and "tire_feats.pkl" in artifacts:
        print("\n[TIRE MODEL]")
        model = artifacts["tire_model.pkl"]
        feats = artifacts["tire_feats.pkl"]

        tire_df = lap_df_f[
            (lap_df_f["is_clean_lap"] == 1) & (lap_df_f["tire_age"] > 0)
        ][feats + ["rel_speed_delta"]].copy()
        tire_df = tire_df.fillna(tire_df.median(numeric_only=True))

        if len(tire_df) == 0:
            print("  No rows to evaluate; skipping.")
        else:
            X = tire_df[feats]
            y = tire_df["rel_speed_delta"]
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            print(f"  MAE  : {mae:.6f}")
            print(f"  RMSE : {rmse:.6f}")
            print(f"  R2   : {r2:.4f}")

    if "pace_model.pkl" in artifacts and "pace_feats.pkl" in artifacts:
        print("\n[PACE MODEL]")
        model = artifacts["pace_model.pkl"]
        feats = artifacts["pace_feats.pkl"]

        pace_df = lap_df_f[lap_df_f["is_clean_lap"] == 1][feats + ["avg_speed"]].copy()
        pace_df = pace_df.fillna(pace_df.median(numeric_only=True))

        if len(pace_df) == 0:
            print("  No rows to evaluate; skipping.")
        else:
            X = pace_df[feats]
            y = pace_df["avg_speed"]
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            print(f"  MAE  : {mae:.4f} km/h")
            print(f"  RMSE : {rmse:.4f} km/h")
            print(f"  R2   : {r2:.4f}")

    if "laptime_model.pkl" in artifacts and "laptime_feats.pkl" in artifacts:
        print("\n[LAPTIME MODEL]")
        model = artifacts["laptime_model.pkl"]
        feats = artifacts["laptime_feats.pkl"]
        circuit_lengths = artifacts.get("circuit_lengths.pkl", {})

        lt_df = lap_df_f.copy()
        lt_df = _compute_pace_drop_3(lt_df)
        lt_df["race_pct"] = lt_df["LapNumber"] / lt_df["total_laps"].clip(lower=1)
        lt_df["avg_speed_next"] = lt_df.groupby(["driver", "event", "year"])["avg_speed"].shift(-1)
        lt_df["is_clean_next"] = lt_df.groupby(["driver", "event", "year"])["is_clean_lap"].shift(-1)
        lt_df["is_pit_next"] = lt_df.groupby(["driver", "event", "year"])["is_pit_lap"].shift(-1)

        lt_df["circuit_m"] = lt_df["event"].map(circuit_lengths).fillna(
            float(np.median(list(circuit_lengths.values()))) if circuit_lengths else 5300.0
        )

        model_df = lt_df[
            lt_df["avg_speed_next"].notna()
            & (lt_df["avg_speed_next"] > 150)
            & (lt_df["is_clean_lap"] == 1)
            & (lt_df["is_clean_next"].fillna(0) == 1)
            & (lt_df["is_pit_lap"] == 0)
            & (lt_df["is_pit_next"].fillna(1) == 0)
        ].copy()

        if len(model_df) == 0:
            print("  No rows to evaluate; skipping.")
        else:
            model_df = model_df.fillna(model_df.median(numeric_only=True))
            X = model_df[feats]
            y = model_df["avg_speed_next"]
            pred = model.predict(X)
            mae_kmh = mean_absolute_error(y, pred)
            rmse_kmh = np.sqrt(mean_squared_error(y, pred))

            med_circ = float(model_df["circuit_m"].median())
            med_spd = float(y.mean())
            mae_s = med_circ / (med_spd / 3.6) ** 2 * (mae_kmh / 3.6)

            print(f"  MAE  : {mae_kmh:.4f} km/h  (≈{mae_s:.3f}s per lap)")
            print(f"  RMSE : {rmse_kmh:.4f} km/h")

    if "sc_speed_threshold.pkl" in artifacts:
        print("\n[SC DETECTOR]")
        sc_threshold = artifacts["sc_speed_threshold.pkl"]
        field_speed_by_lap = (
            lap_df_f.groupby(["event", "year", "LapNumber"])["avg_speed"].mean().reset_index()
        )
        field_speed_by_lap["sc_flag"] = (field_speed_by_lap["avg_speed"] < sc_threshold).astype(int)
        flagged = int(field_speed_by_lap["sc_flag"].sum())
        total = len(field_speed_by_lap)
        print(f"  threshold : {sc_threshold:.1f} km/h")
        if total > 0:
            print(f"  flagged   : {flagged:,} / {total:,} laps  ({flagged/total*100:.2f}%)")
        else:
            print("  No laps to evaluate.")

    timings.report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all F1 ML models.")
    parser.add_argument("--event", type=str, default="", help="Event name (exact)")
    parser.add_argument("--year", type=int, default=0, help="Event year")
    parser.add_argument(
        "--tune-pit-threshold",
        action="store_true",
        help="Tune pit model threshold on current eval slice.",
    )
    parser.add_argument(
        "--pit-objective",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Objective for pit threshold tuning.",
    )
    parser.add_argument(
        "--pit-min-precision",
        type=float,
        default=None,
        help="Optional minimum precision constraint for pit threshold tuning.",
    )
    args = parser.parse_args()

    event = args.event.strip() or None
    year = args.year or None

    evaluate_models(
        event=event,
        year=year,
        tune_pit_threshold=args.tune_pit_threshold,
        pit_objective=args.pit_objective,
        pit_min_precision=args.pit_min_precision,
    )
