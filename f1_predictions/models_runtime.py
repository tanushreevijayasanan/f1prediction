import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeArtifacts:
    winner_model: Any
    tire_model: Any
    pit_model: Any
    pace_model: Any
    le_team: Any
    compound_classes: Any
    median_stint_lengths: Any
    winner_feats: Any
    tire_feats: Any
    pit_feats: Any
    sc_threshold: Any
    track_type_map: Any
    track_type_encoder: Any
    pit_horizon: Any
    pit_alert_threshold: float
    laptime_model: Any
    laptime_feats: Any
    circuit_lengths: Any
    laptime_mae_s: Any
    ranking_model: Any
    ranking_feats: Any
    has_ranking_model: bool


def _verify_artifact_hash(path: str, manifest_path: str, strict: bool, warn) -> None:
    if not manifest_path:
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected = str(manifest.get(os.path.basename(path), "")).strip().lower()
        if not expected:
            msg = f"Artifact hash missing in manifest for {os.path.basename(path)}"
            if strict:
                raise ValueError(msg)
            warn(msg)
            return
        h = hashlib.sha256()
        with open(path, "rb") as rf:
            for chunk in iter(lambda: rf.read(8192), b""):
                h.update(chunk)
        actual = h.hexdigest().lower()
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch for {os.path.basename(path)}")
    except Exception as e:
        if strict:
            raise
        warn(f"Artifact hash verification warning: {e}")


def _load_pickle(name: str, model_dir: str, manifest_path: str, strict: bool, status, fail, warn):
    path = os.path.join(model_dir, name)
    if not os.path.isfile(path):
        fail(f"âœ— Missing model file: {path}")
        raise SystemExit(1)
    _verify_artifact_hash(path, manifest_path, strict, warn)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_runtime_artifacts(model_dir: str, manifest_path: str, strict: bool, status, fail, warn) -> RuntimeArtifacts:
    winner_model = _load_pickle("winner_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
    tire_model = _load_pickle("tire_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
    pit_model = _load_pickle("pit_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
    pace_model = _load_pickle("pace_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
    le_team = _load_pickle("team_encoder.pkl", model_dir, manifest_path, strict, status, fail, warn)
    compound_classes = _load_pickle("compound_classes.pkl", model_dir, manifest_path, strict, status, fail, warn)
    median_stint_lengths = _load_pickle("median_stint_lengths.pkl", model_dir, manifest_path, strict, status, fail, warn)
    winner_feats = _load_pickle("winner_feats.pkl", model_dir, manifest_path, strict, status, fail, warn)
    tire_feats = _load_pickle("tire_feats.pkl", model_dir, manifest_path, strict, status, fail, warn)
    pit_feats = _load_pickle("pit_feats.pkl", model_dir, manifest_path, strict, status, fail, warn)
    sc_threshold = _load_pickle("sc_speed_threshold.pkl", model_dir, manifest_path, strict, status, fail, warn)
    track_type_map = _load_pickle("track_type_map.pkl", model_dir, manifest_path, strict, status, fail, warn)
    track_type_encoder = _load_pickle("track_type_encoder.pkl", model_dir, manifest_path, strict, status, fail, warn)
    pit_horizon = _load_pickle("pit_horizon.pkl", model_dir, manifest_path, strict, status, fail, warn)

    pit_alert_threshold = 0.40
    pit_thr_path = os.path.join(model_dir, "pit_threshold.pkl")
    if os.path.isfile(pit_thr_path):
        try:
            _verify_artifact_hash(pit_thr_path, manifest_path, strict, warn)
            with open(pit_thr_path, "rb") as f:
                pit_alert_threshold = float(pickle.load(f))
            status(f"âœ“ Pit threshold loaded: {pit_alert_threshold:.2f}")
        except Exception as e:
            warn(f"âš  Pit threshold load failed: {e}")

    laptime_model = _load_pickle("laptime_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
    laptime_feats = _load_pickle("laptime_feats.pkl", model_dir, manifest_path, strict, status, fail, warn)
    circuit_lengths = _load_pickle("circuit_lengths.pkl", model_dir, manifest_path, strict, status, fail, warn)
    laptime_mae_s = _load_pickle("laptime_mae_s.pkl", model_dir, manifest_path, strict, status, fail, warn)

    ranking_model = None
    ranking_feats = []
    has_ranking_model = False
    try:
        ranking_model = _load_pickle("ranking_model.pkl", model_dir, manifest_path, strict, status, fail, warn)
        ranking_feats = _load_pickle("ranking_feats.pkl", model_dir, manifest_path, strict, status, fail, warn)
        has_ranking_model = True
    except Exception as e:
        warn(f"âš  Ranking model not available: {e} (Monte Carlo disabled)")

    required = {"laps_remaining", "speed_rank_pct", "delta_vs_field"}
    missing = required - set(winner_feats)
    assert not missing, f"Retrain needed â€” missing WINNER_FEATS: {missing}"
    assert "must_change_compound" in pit_feats, "Retrain needed â€” must_change_compound not in PIT_FEATS"
    assert "hard_brake_rate" in pit_feats, "Retrain needed â€” hard_brake_rate not in PIT_FEATS"

    status(f"âœ“ Models loaded from {model_dir}  (pit horizon={pit_horizon} laps)")
    status("lap time model loaded (required)")
    if has_ranking_model:
        status("âœ“ Ranking distribution model loaded (enables Monte Carlo simulation)")

    return RuntimeArtifacts(
        winner_model=winner_model,
        tire_model=tire_model,
        pit_model=pit_model,
        pace_model=pace_model,
        le_team=le_team,
        compound_classes=compound_classes,
        median_stint_lengths=median_stint_lengths,
        winner_feats=winner_feats,
        tire_feats=tire_feats,
        pit_feats=pit_feats,
        sc_threshold=sc_threshold,
        track_type_map=track_type_map,
        track_type_encoder=track_type_encoder,
        pit_horizon=pit_horizon,
        pit_alert_threshold=pit_alert_threshold,
        laptime_model=laptime_model,
        laptime_feats=laptime_feats,
        circuit_lengths=circuit_lengths,
        laptime_mae_s=laptime_mae_s,
        ranking_model=ranking_model,
        ranking_feats=ranking_feats,
        has_ranking_model=has_ranking_model,
    )

