"""
Train winner (race victory) prediction model with probability calibration.

Key goal: Make predictions more reliable than an LLM.
- Discrimination: Can the model rank drivers by likelihood? (ranking quality)
- Calibration: Do predicted probabilities match observed frequencies? (reliability)

KEY FIXES vs original:
  1. Safety car feature: sc_active binary + position_jump_2 (SC beneficiary signal)
  2. Actual gap_to_leader_s in seconds (not speed delta proxy)
  3. gap_laps_remaining: gap_to_leader_s × (1 / laps_remaining) — the decisive late-race feature
  4. Isotonic calibration on a dedicated cal split (not sigmoid on val set)
  5. Exponential lap weighting: later laps count ~20× more than lap 1
  6. 3-way data split: 60% train / 20% calibration / 20% validation
  7. current_position added as direct integer feature (P1=1 ... Pn=n)
  8. Removed pace_consistency / gap_advantage (they were noise — too coarse)

Expected calibration improvement:
  - Leader should be 45-65% (was ~9%)
  - P2 should be 20-30% (was ~7%)
  - P3 should be 10-15% (was ~7%)
  - Rest: <5%
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb

from _shared_utils import (
    load_reference_data,
    load_lap_telemetry,
    apply_clean_lap_filter,
    compute_speed_features,
    compute_gap_and_pace_features,
    compute_teammate_and_compound_features,
    compute_pit_features,
    apply_fewshot_filter,
    compute_fewshot_weights,
    add_track_type_features,
    save_artifact,
    MODEL_DIR,
)

# ── Safety car speed threshold ─────────────────────────────────────────────────
# A lap where field median speed drops below SC_SPEED_RATIO × race p75 speed
# is flagged as a safety car lap. Matches inference_engine.py logic.
SC_SPEED_RATIO = 0.85


def _detect_sc_laps(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 1: Add sc_active binary feature.

    For each (event, year, LapNumber), compute field median speed.
    If median < p75_speed × SC_SPEED_RATIO → sc_active=1.

    p75_speed is computed per race from laps 5+ to avoid start-lap noise.
    """
    # Per-race p75 speed (laps 5+ only, all drivers)
    p75 = (
        lap_df[lap_df["LapNumber"] >= 5]
        .groupby(["event", "year"])["avg_speed"]
        .quantile(0.75)
        .rename("race_p75_speed")
        .reset_index()
    )
    lap_df = lap_df.merge(p75, on=["event", "year"], how="left")
    lap_df["race_p75_speed"] = lap_df["race_p75_speed"].fillna(lap_df["avg_speed"].median())

    # Field median speed per lap
    lap_median = (
        lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
        .median()
        .rename("lap_field_median_speed")
        .reset_index()
    )
    lap_df = lap_df.merge(lap_median, on=["event", "year", "LapNumber"], how="left")

    lap_df["sc_active"] = (
        lap_df["lap_field_median_speed"] < lap_df["race_p75_speed"] * SC_SPEED_RATIO
    ).astype(int)

    return lap_df


def _compute_actual_position(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 2: Compute current_position (integer 1..n) from speed rank per lap.

    Lower position number = faster = winning. This replaces the noisy
    speed_rank_pct proxy when used as a direct feature.
    """
    # Prefer cumulative race-time position from _shared_utils.load_lap_telemetry()
    # because it reflects true on-track order after pit cycles.
    if "cumul_position" in lap_df.columns:
        lap_df["current_position"] = lap_df["cumul_position"].astype(float)
    else:
        lap_df["current_position"] = (
            lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
            .rank(ascending=False, method="min")
            .astype(float)
        )
    n_drivers = (
        lap_df.groupby(["event", "year", "LapNumber"])["driver"]
        .nunique()
        .rename("n_active")
        .reset_index()
    )
    lap_df = lap_df.merge(n_drivers, on=["event", "year", "LapNumber"], how="left")
    lap_df["n_active"] = lap_df["n_active"].fillna(20)

    if "cumul_position_pct" in lap_df.columns:
        lap_df["position_pct"] = lap_df["cumul_position_pct"].clip(0.0, 1.0)
    else:
        # position_pct: P1 = 1.0, Plast = 0.0
        lap_df["position_pct"] = 1.0 - (
            (lap_df["current_position"] - 1) / (lap_df["n_active"] - 1).clip(lower=1)
        )
        lap_df["position_pct"] = lap_df["position_pct"].clip(0.0, 1.0)

    return lap_df


def _compute_gap_to_leader(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 3: Compute gap_to_leader_s — actual seconds behind the race leader.

    Approximation: use lap time difference between leader's avg_speed and driver's
    avg_speed × circuit_lap_fraction. Better than speed delta alone.

    gap_to_leader_s = (1/driver_speed - 1/leader_speed) × circuit_length_m / 0.2778
    where 0.2778 = 1/3.6 (km/h → m/s conversion)

    For the actual leader, gap = 0.
    """
    # Prefer true race-time gap computed in _shared_utils.load_lap_telemetry().
    if "gap_to_leader_s" in lap_df.columns:
        lap_df["gap_to_leader_s"] = pd.to_numeric(
            lap_df["gap_to_leader_s"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        return lap_df

    # Fallback: speed-based approximation if race-time gap is unavailable.
    leader_speed = (
        lap_df.groupby(["event", "year", "LapNumber"])["avg_speed"]
        .max()
        .rename("leader_avg_speed")
        .reset_index()
    )
    lap_df = lap_df.merge(leader_speed, on=["event", "year", "LapNumber"], how="left")
    CIRCUIT_LENGTH_M = 5300.0
    lap_df["gap_to_leader_s"] = np.where(
        (lap_df["avg_speed"] > 50) & (lap_df["leader_avg_speed"] > 50),
        (1.0 / lap_df["avg_speed"] - 1.0 / lap_df["leader_avg_speed"]) * CIRCUIT_LENGTH_M * 3.6,
        0.0,
    )
    lap_df["gap_to_leader_s"] = lap_df["gap_to_leader_s"].clip(lower=0.0)

    return lap_df


def _compute_position_jump(lap_df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    FIX 1b: position_jump_N = positions gained in last N laps.

    Positive = moved toward front. This is the key SC beneficiary signal:
    when a driver pits under SC and rejoins, they often gain 3-8 positions
    in 1-2 laps. This feature directly captures that.
    """
    lap_df = lap_df.sort_values(["driver", "event", "year", "LapNumber"])
    lap_df[f"position_{n}_laps_ago"] = lap_df.groupby(
        ["driver", "event", "year"]
    )["current_position"].shift(n)

    lap_df[f"position_jump_{n}"] = (
        lap_df[f"position_{n}_laps_ago"] - lap_df["current_position"]
    ).fillna(0.0)

    return lap_df


def train_winner_model():
    print("\n[3/7] Training win probability model...")

    # ── Load data ──────────────────────────────────────────────────────────────
    race_results, features_df, qualifying = load_reference_data()

    print("Loading telemetry...")
    lap_df, _, _ = load_lap_telemetry(race_results)

    print("Cleaning laps...")
    lap_df = apply_clean_lap_filter(lap_df)

    print("Computing telemetry features...")
    lap_df = compute_speed_features(lap_df)
    lap_df = compute_gap_and_pace_features(lap_df)
    lap_df = compute_teammate_and_compound_features(lap_df, race_results)
    lap_df = compute_pit_features(lap_df)

    # ── FIX 1: Safety car detection ────────────────────────────────────────────
    print("Detecting safety car laps...")
    lap_df = _detect_sc_laps(lap_df)

    # ── FIX 2: Actual position (integer) ───────────────────────────────────────
    print("Computing actual race positions...")
    lap_df = _compute_actual_position(lap_df)

    # ── FIX 3: Actual gap to leader ────────────────────────────────────────────
    print("Computing gap to leader...")
    lap_df = _compute_gap_to_leader(lap_df)

    # ── FIX 1b: Position jump (SC beneficiary signal) ──────────────────────────
    print("Computing position jump features...")
    lap_df = _compute_position_jump(lap_df, n=2)
    lap_df = _compute_position_jump(lap_df, n=3)

    # ── Smart sampling: bias heavily toward late race ──────────────────────────
    # FIX 5: Exponential weighting means we can sample more uniformly here
    # since the weighting step will amplify late-race laps anyway.
    all_laps = []
    for (driver, event, year), group in lap_df.groupby(["driver", "event", "year"]):
        group = group.sort_values("LapNumber")

        # Early (1-15): every 5th lap
        early = group[group["LapNumber"] <= 15]
        early_samples = early[early["LapNumber"] % 5 == 0]

        # Mid (16-35): every 3rd lap
        mid = group[(group["LapNumber"] > 15) & (group["LapNumber"] <= 35)]
        mid_samples = mid[mid["LapNumber"] % 3 == 0]

        # Late (36+): every lap — most predictive
        late = group[group["LapNumber"] > 35]

        all_laps.append(pd.concat([early_samples, mid_samples, late]))

    mid_race = pd.concat(all_laps, ignore_index=True).copy()
    print(f"  Smart sampling: {len(mid_race):,} samples from {len(lap_df):,} total laps")

    # ── Merge target data ──────────────────────────────────────────────────────
    winner_targets = race_results[race_results["status"] == "Finished"][[
        "driver_code", "event", "year", "final_position", "grid_position"
    ]].rename(columns={"driver_code": "driver"})

    feats_merge = features_df[[
        "driver_code", "event", "year",
        "avg_finish_last5", "points_last5", "dnf_rate_last5"
    ]].rename(columns={"driver_code": "driver"})

    quali_merge = qualifying[[
        "driver_code", "event", "year", "best_quali_lap"
    ]].rename(columns={"driver_code": "driver"})

    winner_df = (
        mid_race[[
            "driver", "event", "year", "LapNumber", "laps_remaining",
            "avg_speed", "speed_rank_pct", "delta_vs_field",
            "tire_age", "tire_age_pct", "compound_enc", "team",
            # NEW columns from fixes
            "sc_active",
            "current_position", "position_pct", "n_active",
            "gap_to_leader_s",
            "position_jump_2", "position_jump_3",
        ]]
        .merge(winner_targets, on=["driver", "event", "year"], how="inner")
        .merge(feats_merge, on=["driver", "event", "year"], how="left")
        .merge(quali_merge, on=["driver", "event", "year"], how="left")
    )

    winner_df = add_track_type_features(winner_df, MODEL_DIR)

    assert len(winner_df) > 0, "winner_df empty after merges"

    # ── Target ─────────────────────────────────────────────────────────────────
    winner_df["target_win"] = (winner_df["final_position"] == 1).astype(int)

    # ── Grid stratification ────────────────────────────────────────────────────
    winner_df["grid_position_group"] = winner_df["grid_position"].apply(
        lambda gp: 1 if gp <= 5 else (2 if gp <= 15 else 3)
    )

    # ── Derived features ───────────────────────────────────────────────────────
    winner_df["position_gain_pct"] = (
        winner_df["speed_rank_pct"] -
        (winner_df["grid_position"] / 20.0).clip(0, 1)
    ).fillna(0)

    total_laps_map = (
        winner_df.groupby(["event", "year"])["LapNumber"].max()
        .rename("total_laps").reset_index()
    )
    winner_df = winner_df.merge(total_laps_map, on=["event", "year"], how="left")
    winner_df["total_laps"] = winner_df["total_laps"].fillna(60)

    winner_df["lap_progress"] = winner_df["LapNumber"] / winner_df["total_laps"].clip(lower=1)
    winner_df["is_late_race"] = (winner_df["lap_progress"] > 0.75).astype(int)
    winner_df["tire_freshness"] = 1.0 / (1.0 + winner_df["tire_age_pct"])

    # FIX 3b: gap_laps_remaining — the single most predictive late-race feature
    # "How many seconds behind leader per lap remaining?"
    # If you're P1 (gap=0) with 5 laps left → 0 (perfect).
    # If you're 13s behind with 5 laps left → 2.6 (very hard to recover).
    winner_df["gap_laps_remaining"] = (
        winner_df["gap_to_leader_s"] /
        winner_df["laps_remaining"].clip(lower=1)
    )

    # FIX 3c: is_leading binary — most direct signal
    winner_df["is_leading"] = (winner_df["current_position"] == 1.0).astype(int)

    # leading_and_late: position_pct × lap_progress (peaks at 1.0 for leader in final lap)
    winner_df["leading_and_late"] = winner_df["position_pct"] * winner_df["lap_progress"]

    # gap_urgency using delta_vs_field (matches inference engine)
    winner_df["gap_urgency"] = (
        winner_df["delta_vs_field"] /
        winner_df["laps_remaining"].clip(lower=1)
    )

    # FIX 1c: sc_beneficiary — driver gained positions AND SC is active
    # This captures the "free pit stop under SC" scenario exactly
    winner_df["sc_beneficiary"] = (
        (winner_df["sc_active"] == 1) &
        (winner_df["position_jump_2"] > 2)
    ).astype(int)

    # ── Encoders ───────────────────────────────────────────────────────────────
    le_team = LabelEncoder()
    winner_df["team_enc"] = le_team.fit_transform(
        winner_df["team"].fillna("Unknown")
    )

    # ── Missing value handling ─────────────────────────────────────────────────
    winner_df["best_quali_lap"] = winner_df["best_quali_lap"].fillna(
        winner_df["best_quali_lap"].median()
    )
    winner_df["grid_position"] = winner_df["grid_position"].fillna(20)
    winner_df["avg_finish_last5"] = winner_df["avg_finish_last5"].fillna(10)
    winner_df["points_last5"] = winner_df["points_last5"].fillna(0)
    winner_df["dnf_rate_last5"] = winner_df["dnf_rate_last5"].fillna(0.2)

    # ── Feature list ───────────────────────────────────────────────────────────
    WINNER_FEATS = [
        # Static / historical
        "grid_position",
        "grid_position_group",
        "avg_finish_last5",
        "points_last5",
        "dnf_rate_last5",
        "team_enc",
        "best_quali_lap",
        "track_type_enc",

        # Live race state — pace
        "laps_remaining",
        "speed_rank_pct",
        "delta_vs_field",
        "tire_age",
        "tire_age_pct",
        "compound_enc",

        # FIX 2: Actual race position (direct integer + normalized)
        "current_position",    # 1 = leading, n = last
        "position_pct",        # P1=1.0, Plast=0.0
        "is_leading",          # binary: are they P1 right now?

        # FIX 1: Safety car features
        "sc_active",           # binary: SC deployed this lap?
        "position_jump_2",     # positions gained in 2 laps (SC beneficiary)
        "position_jump_3",     # positions gained in 3 laps
        "sc_beneficiary",      # sc_active AND gained >2 positions

        # FIX 3: Actual gap features
        "gap_to_leader_s",     # seconds behind leader
        "gap_laps_remaining",  # gap_s / laps_remaining (recovery difficulty)

        # Late-race interaction features
        "lap_progress",        # 0.0 → 1.0
        "is_late_race",        # binary: last 25%
        "gap_urgency",         # delta_vs_field / laps_remaining
        "leading_and_late",    # position_pct × lap_progress

        # Supporting
        "position_gain_pct",   # gained vs qualifying grid
        "tire_freshness",      # fresher = better
    ]

    winner_df[WINNER_FEATS] = winner_df[WINNER_FEATS].fillna(
        winner_df[WINNER_FEATS].median(numeric_only=True)
    )

    # ── Few-shot filter ────────────────────────────────────────────────────────
    winner_df = apply_fewshot_filter(winner_df)

    X = winner_df[WINNER_FEATS]
    y = winner_df["target_win"]
    groups = winner_df["event"] + "_" + winner_df["year"].astype(str)

    # ── FIX 5: Exponential lap weighting ──────────────────────────────────────
    # Later laps are exponentially more predictive: e^(3×lap_progress)
    # lap 1 weight ≈ 1.0, lap final weight ≈ e^3 ≈ 20.1
    # This multiplies the few-shot weights, so both corrections compound.
    base_weights = compute_fewshot_weights(winner_df)
    lap_weight = np.exp(3.0 * winner_df["lap_progress"].values)
    lap_weight = lap_weight / lap_weight.mean()  # normalise so mean=1
    weights = base_weights * lap_weight

    # ── FIX 4: 3-way split: 60% train / 20% calibration / 20% validation ─────
    # Isotonic calibration needs its own held-out set — not the validation set.
    # Using validation for calibration = data leakage into the final metric.
    gss_train = GroupShuffleSplit(test_size=0.40, n_splits=1, random_state=42)
    train_idx, holdout_idx = next(gss_train.split(X, y, groups))

    gss_cal = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=99)
    holdout_groups = groups.iloc[holdout_idx]
    cal_local, val_local = next(
        gss_cal.split(X.iloc[holdout_idx], y.iloc[holdout_idx], holdout_groups)
    )
    cal_idx = holdout_idx[cal_local]
    val_idx = holdout_idx[val_local]

    print(f"  Split: train={len(train_idx):,}  cal={len(cal_idx):,}  val={len(val_idx):,}")

    # ── Class imbalance weighting ──────────────────────────────────────────────
    pos_ratio = int((y.iloc[train_idx] == 0).sum()) / max(
        int((y.iloc[train_idx] == 1).sum()), 1
    )
    print(f"  Class ratio (neg:pos) = {pos_ratio:.1f}:1")

    # ── Base XGBoost model ─────────────────────────────────────────────────────
    base_model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=pos_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    base_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        sample_weight=weights.iloc[train_idx],
    )

    # ── FIX 4b: Isotonic calibration on dedicated calibration split ────────────
    # Isotonic is more flexible than sigmoid and spreads probabilities further.
    # cv="prefit" means: base_model is already trained, just learn the mapping.
    model = CalibratedClassifierCV(
        base_model,
        method="isotonic",   # was "sigmoid" — isotonic spreads probs more
        cv="prefit",
    )
    model.fit(
        X.iloc[cal_idx],
        y.iloc[cal_idx],
        # Note: sample_weight not used in calibration fit (sklearn limitation)
    )

    # ── Validation ─────────────────────────────────────────────────────────────
    val_probs = model.predict_proba(X.iloc[val_idx])[:, 1]
    val_preds = (val_probs > 0.5).astype(int)

    acc   = accuracy_score(y.iloc[val_idx], val_preds)
    ll    = log_loss(y.iloc[val_idx], val_probs)
    brier = brier_score_loss(y.iloc[val_idx], val_probs)

    print(f"\n  val accuracy      : {acc:.4f}")
    print(f"  val log-loss      : {ll:.4f}")
    print(f"  val brier score   : {brier:.4f}  (lower = better calibrated)")

    # ── Top-5 accuracy ─────────────────────────────────────────────────────────
    val_winner_df = winner_df.iloc[val_idx].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "actual_position": val_winner_df["final_position"].values,
        "pred_prob":       val_probs,
        "grid_group":      val_winner_df["grid_position_group"].values,
        "is_late":         val_winner_df["is_late_race"].values,
    })

    top5_by_prob    = pred_df.nlargest(5, "pred_prob")
    top5_accuracy   = (top5_by_prob["actual_position"] <= 5).mean()
    late_mask       = pred_df["is_late"] == 1
    late_top5       = pred_df[late_mask].nlargest(5, "pred_prob")
    late_top5_acc   = (late_top5["actual_position"] <= 5).mean() if len(late_top5) else float("nan")

    print(f"  top-5 accuracy    : {top5_accuracy:.1%}  (all laps)")
    print(f"  late top-5 acc    : {late_top5_acc:.1%}  (last 25% of race)")

    # ── Calibration curve ──────────────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(
        y.iloc[val_idx], val_probs, n_bins=10, strategy="quantile"
    )
    cal_rmse = np.sqrt(np.mean((prob_true - prob_pred) ** 2))
    print(f"  calibration RMSE  : {cal_rmse:.4f}  (target: <0.03)")

    print("\n  Calibration bins (predicted vs observed):")
    for p_pred, p_true in zip(prob_pred, prob_true):
        bar = "#" * int(p_true * 40)
        print(f"    pred={p_pred:.2%}  actual={p_true:.2%}  err={abs(p_true-p_pred):.2%}  {bar}")

    # Sanity check: what's the leader's average win probability?
    leader_mask = val_winner_df["current_position"] == 1.0
    if leader_mask.any():
        leader_probs = val_probs[leader_mask.values]
        print(f"\n  Leader avg win prob : {leader_probs.mean():.1%}  "
              f"(min={leader_probs.min():.1%}, max={leader_probs.max():.1%})")
        print(f"  [Target: leader should be 45-65%]")

    # ── Grid stratified accuracy ────────────────────────────────────────────────
    print("\n  Grid stratification validation:")
    for grid_group in [1, 2, 3]:
        mask = pred_df["grid_group"] == grid_group
        if mask.sum() > 0:
            g = pred_df[mask]
            acc_g     = accuracy_score(
                (g["actual_position"] == 1).astype(int),
                (g["pred_prob"] > 0.5).astype(int)
            )
            top5_rate = (g["actual_position"] <= 5).mean()
            avg_p     = g["pred_prob"].mean()
            name      = ["Front (1-5)", "Mid (6-15)", "Back (16+)"][grid_group - 1]
            print(f"    {name:15} | win_acc={acc_g:.3f}  top5={top5_rate:.1%}  "
                  f"avg_prob={avg_p:.3f}  n={mask.sum()}")

    # ── Feature importance ─────────────────────────────────────────────────────
    importances = pd.Series(
        base_model.feature_importances_,
        index=WINNER_FEATS
    ).sort_values(ascending=False)

    print("\n  Top-15 features by importance:")
    print(importances.head(15).to_string())

    # Highlight the new fix features
    fix_features = [
        "sc_active", "position_jump_2", "position_jump_3", "sc_beneficiary",
        "gap_to_leader_s", "gap_laps_remaining", "is_leading",
        "current_position", "position_pct",
    ]
    fix_imp = importances[importances.index.isin(fix_features)]
    print(f"\n  New fix features:")
    print(fix_imp.to_string())

    # ── Save artifacts ─────────────────────────────────────────────────────────
    save_artifact(MODEL_DIR, "winner_model.pkl", model)
    save_artifact(MODEL_DIR, "winner_feats.pkl", WINNER_FEATS)
    save_artifact(MODEL_DIR, "team_encoder.pkl", le_team)

    print("\nWinner model saved.")
    print(f"  WINNER_FEATS ({len(WINNER_FEATS)} features): {WINNER_FEATS}")

    return model, WINNER_FEATS, le_team


if __name__ == "__main__":
    train_winner_model()