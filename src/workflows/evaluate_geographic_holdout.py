# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
86_open_meanstate_spatial_holdout_core_gateS9F.py

Gate S9F: ONE-TIME East-Asia core opening and final spatial-holdout evaluation.

This script is intentionally a single-shot evaluator + aggregator. It consumes
the already sealed Gate-S9E 3 x 3 formal set and opens the pre-specified
East-Asia core for the first time.

No training, checkpoint selection, hyperparameter tuning, region selection,
seed selection, or model replacement is performed here.

Predeclared evaluation
----------------------
Core:
    East Asia, y=220..290, x=448..520
    (20-55 N, 100-145 E)

Months:
    2022-11 and 2022-12

Models:
    mean_meteo_clim_only
    mean_binary_support_meteo_clim
    mean_emis_meteo_clim

Seeds:
    42, 43, 44

Primary statistics:
    per month and Nov+Dec pooled:
      log-space R2, RMSE, bias
    three-seed mean +/- sample SD
    matched-seed descriptive contrasts:
      binary - meteo
      joint - meteo
      joint - binary

Sensitivity:
    the same statistics with cos(latitude) area weighting.

Important interpretation boundary
---------------------------------
This is a pre-specified GEOGRAPHIC / SPATIAL holdout evaluation. It is NOT a
fully independent spatiotemporal test, because November-December 2022 were
used for checkpoint selection outside the East-Asia core + 64-cell exclusion.

No statistical significance or post-hoc equivalence threshold is introduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


H = 361
W = 576
LAT = np.linspace(-90.0, 90.0, H, dtype=np.float64)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float64)

EXPECTED_EXPERIMENTS = [
    "mean_meteo_clim_only",
    "mean_binary_support_meteo_clim",
    "mean_emis_meteo_clim",
]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_VAL_MONTHS = [11, 12]

EXPECTED_CLOSURE_ARTIFACT = (
    "s2_meanstate_spatial_holdout_preopening_closure_gateS9E"
)
EXPECTED_CLOSURE_STATUS = "PASS_PREOPENING_3X3_FORMAL_SET_SEALED"

EXPECTED_47_SHA256 = (
    "ff3bec0f270b1f8dd191f43230d9665761fb6f4c65baafd71bdfc56cda6a2fe8"
)
EXPECTED_77_SHA256 = (
    "b2de46481d2119a7c4bb49dc698ce27ed34390ee6ca37b47a472e5cfde41b2e1"
)
EXPECTED_83_SHA256 = (
    "e839ac22651728ce8de2eb676a7f7e8edf1a20a59300f39d09f03b8d429b1dea"
)
EXPECTED_84_SHA256 = (
    "265083d4bbd918bcd6be7d67900280e3e5dcd0d0b5697d82b433f70945a727bd"
)
EXPECTED_85_SHA256 = (
    "53fccd1a0f982dd34d8d3c4f347fea96f00da17afb9f01ce9684d1ace6ea264b"
)
EXPECTED_MONTHLY_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)

EXPECTED_CORE_BBOX = [220, 290, 448, 520]
EXPECTED_TRAIN_EXCLUSION_BBOX = [188, 322, 416, 552]
EXPECTED_SELECTION_EXCLUSION_BBOX = [156, 354, 384, 575]
EXPECTED_TRAIN_GUARD = 32
EXPECTED_SELECTION_GUARD = 64
EXPECTED_MAX_HALF_RF = 23
EXPECTED_CORE_VALID_COUNTS = {11: 3497, 12: 3488}

METEO_ORDER = ["Temp", "RH", "BLH", "Windspeed_U", "Windspeed_V"]

EXPECTED_MODEL = {
    "mean_meteo_clim_only": {
        "uses_emission": False,
        "input_channels": 5,
        "trainable_parameters": 477025,
    },
    "mean_binary_support_meteo_clim": {
        "uses_emission": True,
        "input_channels": 6,
        "trainable_parameters": 476449,
    },
    "mean_emis_meteo_clim": {
        "uses_emission": True,
        "input_channels": 13,
        "trainable_parameters": 477457,
    },
}

METRICS = ["log_r2", "log_rmse", "log_bias"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    require(isinstance(obj, dict), f"PT_ROOT_NOT_DICT: {path}")
    return obj


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    require(isinstance(obj, dict), f"CHECKPOINT_ROOT_NOT_DICT: {path}")
    return obj


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name}_IS_NONE")
    value = float(value)
    require(math.isfinite(value), f"NONFINITE_{name}: {value}")
    return value


def import_base47(path: Path):
    module_name = "gate_s9f_locked_base47"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None, "BASE47_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def make_spec(base, experiment: str):
    if experiment == "mean_meteo_clim_only":
        return base.MeanStateSpec(
            name=experiment,
            mapping="Spatial-holdout Monthly-Mean Meteo -> Monthly Mean Concentration",
            note="Gate S9: meteorology-only spatial transfer control.",
            uses_emission=False,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=0,
            meteo_channels=5,
        )
    if experiment == "mean_binary_support_meteo_clim":
        return base.MeanStateSpec(
            name=experiment,
            mapping="Spatial-holdout Binary Processed-Prior Support + Monthly Meteo -> Monthly Mean Concentration",
            note="Gate S9: binary processed-prior support spatial transfer control.",
            uses_emission=True,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=1,
            meteo_channels=5,
        )
    if experiment == "mean_emis_meteo_clim":
        return base.MeanStateSpec(
            name=experiment,
            mapping="Spatial-holdout Processed Emission Prior + Monthly Meteo -> Monthly Mean Concentration",
            note="Gate S9: sector-resolved processed-prior spatial transfer model.",
            uses_emission=True,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=8,
            meteo_channels=5,
        )
    raise RuntimeError(f"UNKNOWN_EXPERIMENT: {experiment}")


def core_mask() -> torch.Tensor:
    y0, y1, x0, x1 = EXPECTED_CORE_BBOX
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


CORE_MASK = core_mask()
CORE_Y0, CORE_Y1, CORE_X0, CORE_X1 = EXPECTED_CORE_BBOX
CORE_H = CORE_Y1 - CORE_Y0 + 1
CORE_W = CORE_X1 - CORE_X0 + 1


def build_input(
    payload: dict[str, Any],
    experiment: str,
    *,
    meteo_scale: torch.Tensor,
    meteo_offset: torch.Tensor,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    if EXPECTED_MODEL[experiment]["uses_emission"]:
        emission = payload.get("emission_log_mean")
        require(isinstance(emission, torch.Tensor), "EMISSION_INPUT_MISSING")
        emission = emission.detach().cpu().float()
        require(
            tuple(emission.shape) == (8, H, W),
            f"EMISSION_SHAPE_INVALID: {tuple(emission.shape)}",
        )
        require(bool(torch.isfinite(emission).all().item()), "EMISSION_NONFINITE")
        if experiment == "mean_binary_support_meteo_clim":
            support = torch.any(
                torch.abs(emission) > 0.0,
                dim=0,
                keepdim=True,
            ).float()
            unique = set(float(v) for v in torch.unique(support).tolist())
            require(unique.issubset({0.0, 1.0}), f"SUPPORT_NOT_BINARY: {unique}")
            parts.append(support)
        else:
            parts.append(emission)

    meteo = payload.get("meteo_mean")
    require(isinstance(meteo, torch.Tensor), "METEO_INPUT_MISSING")
    meteo = meteo.detach().cpu().float()
    require(
        tuple(meteo.shape) == (5, H, W),
        f"METEO_SHAPE_INVALID: {tuple(meteo.shape)}",
    )

    # Gate S9B locked ordering: affine first, then nan_to_num.
    meteo = meteo_scale * meteo + meteo_offset
    meteo = torch.nan_to_num(
        meteo,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    require(bool(torch.isfinite(meteo).all().item()), "TRANSFORMED_METEO_NONFINITE")
    parts.append(meteo)

    x = torch.cat(parts, dim=0)
    require(
        tuple(x.shape)
        == (EXPECTED_MODEL[experiment]["input_channels"], H, W),
        f"INPUT_SHAPE_MISMATCH: {tuple(x.shape)}",
    )
    return x


def unweighted_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred = pred.reshape(-1).double()
    target = target.reshape(-1).double()
    require(pred.numel() == target.numel() and pred.numel() > 1, "UNWEIGHTED_LENGTH_INVALID")
    require(bool(torch.isfinite(pred).all().item()), "UNWEIGHTED_PRED_NONFINITE")
    require(bool(torch.isfinite(target).all().item()), "UNWEIGHTED_TARGET_NONFINITE")

    residual = pred - target
    target_mean = torch.mean(target)
    ss_res = torch.sum(residual * residual)
    ss_tot = torch.sum((target - target_mean) ** 2)
    return {
        "log_r2": float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-18)),
        "log_rmse": float(torch.sqrt(torch.mean(residual * residual))),
        "log_bias": float(torch.mean(residual)),
    }


def weighted_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    pred = pred.reshape(-1).double()
    target = target.reshape(-1).double()
    weight = weight.reshape(-1).double()
    require(
        pred.numel() == target.numel() == weight.numel() and pred.numel() > 1,
        "WEIGHTED_LENGTH_INVALID",
    )
    require(bool(torch.isfinite(pred).all().item()), "WEIGHTED_PRED_NONFINITE")
    require(bool(torch.isfinite(target).all().item()), "WEIGHTED_TARGET_NONFINITE")
    require(bool(torch.isfinite(weight).all().item()), "WEIGHT_NONFINITE")
    require(bool((weight > 0).all().item()), "NONPOSITIVE_AREA_WEIGHT")

    wsum = torch.sum(weight)
    require(float(wsum) > 0.0, "ZERO_WEIGHT_SUM")
    residual = pred - target
    target_mean = torch.sum(weight * target) / wsum
    ss_res = torch.sum(weight * residual * residual)
    ss_tot = torch.sum(weight * (target - target_mean) ** 2)

    return {
        "log_r2": float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-18)),
        "log_rmse": float(torch.sqrt(torch.sum(weight * residual * residual) / wsum)),
        "log_bias": float(torch.sum(weight * residual) / wsum),
    }


def summarize(per_seed: dict[int, float]) -> dict[str, Any]:
    require(sorted(per_seed) == EXPECTED_SEEDS, f"SEED_SET_MISMATCH: {sorted(per_seed)}")
    values = [finite_float(per_seed[s], f"seed{s}") for s in EXPECTED_SEEDS]
    return {
        "per_seed": {str(s): float(per_seed[s]) for s in EXPECTED_SEEDS},
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "all_positive": all(v > 0 for v in values),
        "all_negative": all(v < 0 for v in values),
        "mixed_sign": any(v > 0 for v in values) and any(v < 0 for v in values),
    }


def paired_difference(
    lhs: dict[int, float],
    rhs: dict[int, float],
) -> dict[str, Any]:
    return summarize({s: float(lhs[s]) - float(rhs[s]) for s in EXPECTED_SEEDS})


def ensure_record_matrix(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    require(len(records) == 9, f"CLOSURE_RECORD_COUNT_MISMATCH: {len(records)}")
    matrix = {}
    for row in records:
        key = (str(row["experiment"]), int(row["seed"]))
        require(key not in matrix, f"DUPLICATE_CLOSURE_RECORD: {key}")
        matrix[key] = row
    expected = {
        (experiment, seed)
        for experiment in EXPECTED_EXPERIMENTS
        for seed in EXPECTED_SEEDS
    }
    require(set(matrix) == expected, "CLOSURE_3X3_MATRIX_MISMATCH")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S9F one-time East-Asia core evaluation and aggregation."
    )
    parser.add_argument("--preopening-closure", required=True)
    parser.add_argument("--base-training-script", required=True)
    parser.add_argument("--binary-training-script", required=True)
    parser.add_argument("--spatial-training-script", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--preopening-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    closure_path = Path(args.preopening_closure).expanduser().resolve()
    base47_path = Path(args.base_training_script).expanduser().resolve()
    binary77_path = Path(args.binary_training_script).expanduser().resolve()
    spatial83_path = Path(args.spatial_training_script).expanduser().resolve()
    audit84_path = Path(args.audit_script).expanduser().resolve()
    preopening85_path = Path(args.preopening_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for p in [
        closure_path,
        base47_path,
        binary77_path,
        spatial83_path,
        audit84_path,
        preopening85_path,
    ]:
        require(p.is_file(), f"REQUIRED_INPUT_NOT_FOUND: {p}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    require(sha256_file(base47_path) == EXPECTED_47_SHA256, "BASE47_SHA256_MISMATCH")
    require(sha256_file(binary77_path) == EXPECTED_77_SHA256, "BINARY77_SHA256_MISMATCH")
    require(sha256_file(spatial83_path) == EXPECTED_83_SHA256, "SPATIAL83_SHA256_MISMATCH")
    require(sha256_file(audit84_path) == EXPECTED_84_SHA256, "AUDIT84_SHA256_MISMATCH")
    require(sha256_file(preopening85_path) == EXPECTED_85_SHA256, "PREOPENING85_SHA256_MISMATCH")

    closure = load_json(closure_path)
    require(closure.get("artifact_name") == EXPECTED_CLOSURE_ARTIFACT, "CLOSURE_ARTIFACT_MISMATCH")
    require(closure.get("status") == EXPECTED_CLOSURE_STATUS, "CLOSURE_STATUS_MISMATCH")
    require(int(closure.get("artifact_version", -1)) == 1, "CLOSURE_VERSION_MISMATCH")

    verification = closure.get("verification", {})
    for key in [
        "exact_3x3_formal_set_present",
        "all_gateS9D_audits_pass",
        "all_artifacts_unchanged_since_audit",
        "all_checkpoint_strict_load_pass",
        "all_best_safe_selection_metrics_reproduced",
        "shared_monthly_manifest_hash_pass",
        "shared_s9a_hash_pass",
        "shared_s9b_hash_pass",
        "ready_for_one_time_core_evaluation",
        "overall_pass",
    ]:
        require(verification.get(key) is True, f"CLOSURE_VERIFICATION_FAIL_{key}")
    require(
        verification.get("core_holdout_performance_opened") is False,
        "CORE_ALREADY_OPENED_BEFORE_S9F",
    )

    locked = closure.get("locked_scripts", {})
    require(locked.get("base_47_sha256") == EXPECTED_47_SHA256, "CLOSURE_BASE47_HASH_MISMATCH")
    require(locked.get("binary_77_sha256") == EXPECTED_77_SHA256, "CLOSURE_BINARY77_HASH_MISMATCH")
    require(locked.get("spatial_83_sha256") == EXPECTED_83_SHA256, "CLOSURE_83_HASH_MISMATCH")
    require(locked.get("audit_84_sha256") == EXPECTED_84_SHA256, "CLOSURE_84_HASH_MISMATCH")

    shared = closure.get("shared_provenance", {})
    require(
        shared.get("monthly_manifest_sha256") == EXPECTED_MONTHLY_MANIFEST_SHA256,
        "CLOSURE_MONTHLY_MANIFEST_HASH_MISMATCH",
    )

    spatial = closure.get("spatial_contract", {})
    require(spatial.get("core_bbox_indices") == EXPECTED_CORE_BBOX, "CLOSURE_CORE_BBOX_MISMATCH")
    require(
        spatial.get("training_exclusion_bbox_indices") == EXPECTED_TRAIN_EXCLUSION_BBOX,
        "CLOSURE_TRAIN_EXCLUSION_MISMATCH",
    )
    require(
        spatial.get("selection_exclusion_bbox_indices") == EXPECTED_SELECTION_EXCLUSION_BBOX,
        "CLOSURE_SELECTION_EXCLUSION_MISMATCH",
    )
    require(int(spatial.get("training_guard_cells", -1)) == EXPECTED_TRAIN_GUARD, "TRAIN_GUARD_MISMATCH")
    require(
        int(spatial.get("selection_guard_cells", -1)) == EXPECTED_SELECTION_GUARD,
        "SELECTION_GUARD_MISMATCH",
    )
    require(int(spatial.get("max_half_receptive_field", -1)) == EXPECTED_MAX_HALF_RF, "RF_HALF_MISMATCH")
    require(spatial.get("core_holdout_performance_opened") is False, "CLOSURE_CORE_OPENED_TRUE")
    require(EXPECTED_MAX_HALF_RF <= EXPECTED_TRAIN_GUARD, "RF_EXCEEDS_TRAINING_GUARD")

    matrix = ensure_record_matrix(list(closure.get("records", [])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA_REQUIRED_BUT_UNAVAILABLE")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = import_base47(base47_path)

    lat_weights_full = torch.from_numpy(
        np.cos(np.deg2rad(LAT))
    ).double().view(H, 1).expand(H, W)
    require(bool((lat_weights_full[CORE_MASK] > 0).all().item()), "CORE_AREA_WEIGHTS_NONPOSITIVE")

    print("=" * 96)
    print("Gate S9F - ONE-TIME East-Asia core opening")
    print("=" * 96)
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print("PREOPENING_CLOSURE_ACCEPTED=True")
    print("CORE_REGION=East_Asia")
    print(f"CORE_BBOX={tuple(EXPECTED_CORE_BBOX)}")
    print("CORE_HOLDOUT_PERFORMANCE_OPENING_NOW=True")
    print("NO_FURTHER_MODEL_SELECTION_ALLOWED=True")

    run_results: dict[tuple[str, int], dict[str, Any]] = {}
    prediction_artifact: dict[str, np.ndarray] = {
        "lat_core": LAT[CORE_Y0 : CORE_Y1 + 1].astype(np.float32),
        "lon_core": LON[CORE_X0 : CORE_X1 + 1].astype(np.float32),
    }
    reference_masks: dict[int, torch.Tensor] = {}
    reference_targets: dict[int, torch.Tensor] = {}
    reference_weights: dict[int, torch.Tensor] = {}

    for experiment in EXPECTED_EXPERIMENTS:
        cfg = EXPECTED_MODEL[experiment]
        for seed in EXPECTED_SEEDS:
            sealed = matrix[(experiment, seed)]
            require(sealed.get("reproduction_pass") is True, "SEALED_REPRODUCTION_NOT_PASS")
            require(
                sealed.get("core_holdout_performance_opened") is False,
                "SEALED_RECORD_CORE_ALREADY_OPENED",
            )

            manifest_path = Path(sealed["manifest"]).expanduser().resolve()
            checkpoint_path = Path(sealed["checkpoint"]).expanduser().resolve()
            history_path = Path(sealed["history"]).expanduser().resolve()
            audit_path = Path(sealed["audit"]).expanduser().resolve()
            for p in [manifest_path, checkpoint_path, history_path, audit_path]:
                require(p.is_file(), f"SEALED_ARTIFACT_NOT_FOUND: {p}")

            require(sha256_file(manifest_path) == sealed["manifest_sha256"], "MANIFEST_CHANGED_AFTER_S9E")
            require(sha256_file(checkpoint_path) == sealed["checkpoint_sha256"], "CHECKPOINT_CHANGED_AFTER_S9E")
            require(sha256_file(history_path) == sealed["history_sha256"], "HISTORY_CHANGED_AFTER_S9E")
            require(sha256_file(audit_path) == sealed["audit_sha256"], "AUDIT_CHANGED_AFTER_S9E")

            manifest = load_json(manifest_path)
            require(manifest.get("run_mode") == "formal", "RUN_MODE_NOT_FORMAL")
            require(manifest.get("run_name") == f"{experiment}_seed{seed}", "RUN_NAME_MISMATCH")
            require(int(manifest.get("seed", -1)) == seed, "MANIFEST_SEED_MISMATCH")

            provenance = manifest.get("provenance", {})
            require(provenance.get("monthly_manifest_sha256") == EXPECTED_MONTHLY_MANIFEST_SHA256, "MANIFEST_MONTHLY_HASH_MISMATCH")
            require(provenance.get("base_47_script_sha256") == EXPECTED_47_SHA256, "MANIFEST_BASE47_HASH_MISMATCH")
            require(provenance.get("binary_77_script_sha256") == EXPECTED_77_SHA256, "MANIFEST_BINARY77_HASH_MISMATCH")
            require(provenance.get("training_script_sha256") == EXPECTED_83_SHA256, "MANIFEST_83_HASH_MISMATCH")

            m_spatial = manifest.get("spatial_contract", {})
            require(m_spatial.get("core_bbox_indices") == EXPECTED_CORE_BBOX, "MANIFEST_CORE_BBOX_MISMATCH")
            require(m_spatial.get("core_test_opened_during_training") is False, "CORE_WAS_OPENED_DURING_TRAINING")

            normalization = manifest.get("normalization_contract", {})
            require(normalization.get("spatial_train_only") is True, "NORMALIZATION_NOT_SPATIAL_TRAIN_ONLY")
            require(normalization.get("variables") == METEO_ORDER, "METEO_ORDER_MISMATCH")
            require(
                normalization.get("application_order") == "affine transform before nan_to_num",
                "NORMALIZATION_APPLICATION_ORDER_MISMATCH",
            )
            scales = [finite_float(x, "AFFINE_SCALE") for x in normalization.get("affine_scale", [])]
            offsets = [finite_float(x, "AFFINE_OFFSET") for x in normalization.get("affine_offset", [])]
            require(len(scales) == 5 and len(offsets) == 5, "AFFINE_LENGTH_MISMATCH")
            scale_t = torch.tensor(scales, dtype=torch.float32).view(5, 1, 1)
            offset_t = torch.tensor(offsets, dtype=torch.float32).view(5, 1, 1)

            val_entries = list(manifest.get("val_entries", []))
            require([int(x["month"]) for x in val_entries] == EXPECTED_VAL_MONTHS, "VAL_MONTHS_MISMATCH")

            spec = make_spec(base, experiment)
            model = base.MeanStateUNet(spec=spec, base_features=32).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            require(n_params == cfg["trainable_parameters"], "PARAMETER_COUNT_MISMATCH")

            checkpoint = load_checkpoint(checkpoint_path, device)
            require(checkpoint.get("core_holdout_performance_opened") is False, "CHECKPOINT_CORE_OPENED_TRUE")
            require(int(checkpoint.get("epoch", -1)) + 1 == int(sealed["best_epoch"]), "BEST_EPOCH_MISMATCH")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            month_rows = []
            pooled_pred = []
            pooled_target = []
            pooled_weight = []

            with torch.no_grad():
                for entry in val_entries:
                    month = int(entry["month"])
                    payload_path = Path(entry["file"]).expanduser().resolve()
                    require(payload_path.is_file(), f"VAL_PAYLOAD_NOT_FOUND: {payload_path}")
                    payload = load_pt(payload_path)

                    x = build_input(
                        payload,
                        experiment,
                        meteo_scale=scale_t,
                        meteo_offset=offset_t,
                    )
                    pred = model(
                        x.unsqueeze(0).to(device, non_blocking=True)
                    ).squeeze(0).squeeze(0).detach().cpu().float()

                    target = payload.get("target_log_mean_conc")
                    valid = payload.get("valid_mask")
                    require(isinstance(target, torch.Tensor), "TARGET_MISSING")
                    require(isinstance(valid, torch.Tensor), "VALID_MASK_MISSING")
                    target = target.detach().cpu().float()
                    valid = valid.detach().cpu().float() > 0.5
                    require(tuple(target.shape) == (H, W), "TARGET_SHAPE_INVALID")
                    require(tuple(valid.shape) == (H, W), "VALID_MASK_SHAPE_INVALID")
                    require(tuple(pred.shape) == (H, W), "PRED_SHAPE_INVALID")

                    eval_mask = (
                        CORE_MASK
                        & valid
                        & torch.isfinite(target)
                        & torch.isfinite(pred)
                    )
                    n_valid = int(eval_mask.sum().item())
                    require(
                        n_valid == EXPECTED_CORE_VALID_COUNTS[month],
                        f"CORE_VALID_COUNT_MISMATCH month={month} got={n_valid}",
                    )

                    p = pred[eval_mask]
                    t = target[eval_mask]
                    w = lat_weights_full[eval_mask].float()

                    if month not in reference_masks:
                        reference_masks[month] = eval_mask.clone()
                        reference_targets[month] = target.clone()
                        reference_weights[month] = lat_weights_full.clone()
                    else:
                        require(
                            torch.equal(eval_mask, reference_masks[month]),
                            f"CORE_MASK_DIFFERS_ACROSS_RUNS_MONTH_{month}",
                        )
                        require(
                            torch.equal(
                                torch.nan_to_num(target, nan=-999999.0),
                                torch.nan_to_num(reference_targets[month], nan=-999999.0),
                            ),
                            f"TARGET_DIFFERS_ACROSS_RUNS_MONTH_{month}",
                        )

                    uw = unweighted_metrics(p, t)
                    aw = weighted_metrics(p, t, w)
                    month_rows.append(
                        {
                            "year": 2022,
                            "month": month,
                            "n_valid_pixels": n_valid,
                            "unweighted": uw,
                            "area_weighted_cos_lat": aw,
                        }
                    )
                    pooled_pred.append(p)
                    pooled_target.append(t)
                    pooled_weight.append(w)

                    core_pred_map = pred[
                        CORE_Y0 : CORE_Y1 + 1,
                        CORE_X0 : CORE_X1 + 1,
                    ].numpy().astype(np.float32)
                    core_target_map = target[
                        CORE_Y0 : CORE_Y1 + 1,
                        CORE_X0 : CORE_X1 + 1,
                    ].numpy().astype(np.float32)
                    core_valid_map = eval_mask[
                        CORE_Y0 : CORE_Y1 + 1,
                        CORE_X0 : CORE_X1 + 1,
                    ].numpy().astype(np.uint8)

                    prediction_artifact[
                        f"pred__{experiment}__seed{seed}__2022_{month:02d}"
                    ] = core_pred_map
                    if f"target__2022_{month:02d}" not in prediction_artifact:
                        prediction_artifact[f"target__2022_{month:02d}"] = core_target_map
                        prediction_artifact[f"valid__2022_{month:02d}"] = core_valid_map

            pooled_pred_t = torch.cat(pooled_pred)
            pooled_target_t = torch.cat(pooled_target)
            pooled_weight_t = torch.cat(pooled_weight)
            pooled_uw = unweighted_metrics(pooled_pred_t, pooled_target_t)
            pooled_aw = weighted_metrics(
                pooled_pred_t,
                pooled_target_t,
                pooled_weight_t,
            )

            run_results[(experiment, seed)] = {
                "experiment": experiment,
                "seed": seed,
                "best_epoch": int(sealed["best_epoch"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sealed["checkpoint_sha256"],
                "n_total_valid_pairs": int(pooled_target_t.numel()),
                "per_month": month_rows,
                "pooled_nov_dec": {
                    "unweighted": pooled_uw,
                    "area_weighted_cos_lat": pooled_aw,
                },
            }

            print(
                f"CORE_EVAL experiment={experiment} seed={seed} "
                f"N={pooled_target_t.numel()} "
                f"R2={pooled_uw['log_r2']:.8f} "
                f"RMSE={pooled_uw['log_rmse']:.8f} "
                f"BIAS={pooled_uw['log_bias']:.8f} "
                f"AREA_R2={pooled_aw['log_r2']:.8f}",
                flush=True,
            )

    # Aggregate by model, split, weighting, metric.
    aggregate: dict[str, Any] = {}
    for experiment in EXPECTED_EXPERIMENTS:
        aggregate[experiment] = {}
        for period_name, month in [("2022_11", 11), ("2022_12", 12), ("pooled_2022_11_12", None)]:
            aggregate[experiment][period_name] = {}
            for weighting in ["unweighted", "area_weighted_cos_lat"]:
                aggregate[experiment][period_name][weighting] = {}
                for metric in METRICS:
                    per_seed = {}
                    for seed in EXPECTED_SEEDS:
                        result = run_results[(experiment, seed)]
                        if month is None:
                            value = result["pooled_nov_dec"][weighting][metric]
                        else:
                            row = next(
                                r for r in result["per_month"] if r["month"] == month
                            )
                            value = row[weighting][metric]
                        per_seed[seed] = float(value)
                    aggregate[experiment][period_name][weighting][metric] = summarize(per_seed)

    contrasts: dict[str, Any] = {}
    contrast_specs = [
        ("binary_minus_meteo", "mean_binary_support_meteo_clim", "mean_meteo_clim_only"),
        ("joint_minus_meteo", "mean_emis_meteo_clim", "mean_meteo_clim_only"),
        ("joint_minus_binary", "mean_emis_meteo_clim", "mean_binary_support_meteo_clim"),
    ]

    for contrast_name, lhs_exp, rhs_exp in contrast_specs:
        contrasts[contrast_name] = {}
        for period_name in ["2022_11", "2022_12", "pooled_2022_11_12"]:
            contrasts[contrast_name][period_name] = {}
            for weighting in ["unweighted", "area_weighted_cos_lat"]:
                contrasts[contrast_name][period_name][weighting] = {}
                for metric in METRICS:
                    lhs = {
                        s: float(
                            aggregate[lhs_exp][period_name][weighting][metric]["per_seed"][str(s)]
                        )
                        for s in EXPECTED_SEEDS
                    }
                    rhs = {
                        s: float(
                            aggregate[rhs_exp][period_name][weighting][metric]["per_seed"][str(s)]
                        )
                        for s in EXPECTED_SEEDS
                    }
                    contrasts[contrast_name][period_name][weighting][metric] = paired_difference(lhs, rhs)

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "meanstate_spatial_holdout_EastAsia_core_gateS9F.json"
    csv_path = output_dir / "meanstate_spatial_holdout_EastAsia_core_per_seed_gateS9F.csv"
    contrast_csv_path = output_dir / "meanstate_spatial_holdout_EastAsia_core_contrasts_gateS9F.csv"
    npz_path = output_dir / "meanstate_spatial_holdout_EastAsia_core_predictions_gateS9F.npz"
    txt_path = output_dir / "meanstate_spatial_holdout_EastAsia_core_gateS9F.txt"

    np.savez_compressed(npz_path, **prediction_artifact)

    records_sorted = [
        run_results[(experiment, seed)]
        for experiment in EXPECTED_EXPERIMENTS
        for seed in EXPECTED_SEEDS
    ]

    report = {
        "artifact_name": "s2_meanstate_spatial_holdout_EastAsia_core_evaluation_gateS9F",
        "artifact_version": 1,
        "status": "PASS_ONE_TIME_EAST_ASIA_CORE_EVALUATION",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "preopening_closure": {
            "path": str(closure_path),
            "sha256": sha256_file(closure_path),
            "artifact_name": closure.get("artifact_name"),
            "status": closure.get("status"),
        },
        "locked_scripts": {
            "base_47_sha256": EXPECTED_47_SHA256,
            "binary_77_sha256": EXPECTED_77_SHA256,
            "spatial_83_sha256": EXPECTED_83_SHA256,
            "audit_84_sha256": EXPECTED_84_SHA256,
            "preopening_85_sha256": EXPECTED_85_SHA256,
            "evaluator_86_path": str(Path(__file__).resolve()),
            "evaluator_86_sha256": sha256_file(__file__),
        },
        "evaluation_contract": {
            "evaluation_type": "pre-specified spatially held-out geographic transfer evaluation",
            "fully_independent_spatiotemporal_test": False,
            "reason_not_fully_spatiotemporal": (
                "November-December 2022 were used for checkpoint selection only outside "
                "the East-Asia core plus 64-cell exclusion."
            ),
            "core_region": "East_Asia",
            "core_bbox_indices": EXPECTED_CORE_BBOX,
            "core_latitude_range_deg": [20.0, 55.0],
            "core_longitude_range_deg": [100.0, 145.0],
            "months": EXPECTED_VAL_MONTHS,
            "models": EXPECTED_EXPERIMENTS,
            "seeds": EXPECTED_SEEDS,
            "primary_metrics": METRICS,
            "primary_weighting": "unweighted valid grid-cell pairs",
            "sensitivity_weighting": "cos(latitude)",
            "pooled_definition": "concatenate valid Nov and Dec core prediction-target pairs before metric computation",
            "area_weighted_pooled_definition": "concatenate Nov and Dec core pairs with cos(latitude) weights repeated by month",
            "posthoc_equivalence_threshold_used": False,
            "statistical_significance_test_used": False,
            "ensemble_prediction_used": False,
        },
        "core_valid_counts": {
            "2022_11": EXPECTED_CORE_VALID_COUNTS[11],
            "2022_12": EXPECTED_CORE_VALID_COUNTS[12],
            "pooled_pairs": EXPECTED_CORE_VALID_COUNTS[11] + EXPECTED_CORE_VALID_COUNTS[12],
        },
        "per_run": records_sorted,
        "three_seed_aggregate": aggregate,
        "matched_seed_descriptive_contrasts": contrasts,
        "metric_direction": {
            "log_r2": "higher is better",
            "log_rmse": "lower is better",
            "log_bias": "closer to zero is better; signed differences are descriptive only",
        },
        "prediction_artifact": {
            "path": str(npz_path),
            "sha256": sha256_file(npz_path),
            "description": (
                "Core-region prediction maps, target maps, validity masks and core lat/lon "
                "for all nine formal runs and both evaluation months."
            ),
        },
        "verification": {
            "preopening_closure_pass_and_ready": True,
            "preopening_core_status_was_unopened": True,
            "all_nine_sealed_artifact_hashes_unchanged": True,
            "all_nine_checkpoints_strict_loaded": True,
            "same_core_masks_used_across_all_models_and_seeds": True,
            "core_valid_counts_match_preflight": True,
            "all_primary_metrics_finite": True,
            "all_area_weighted_metrics_finite": True,
            "all_predeclared_contrasts_computed": True,
            "posthoc_equivalence_threshold_used": False,
            "statistical_significance_test_used": False,
            "core_holdout_performance_opened": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "These results assess geographic transfer to a pre-specified East-Asia core "
            "excluded from meteorological normalization fitting, training crops and "
            "checkpoint-selection targets. They do not validate native EDGAR geography, "
            "establish causal emission-concentration relationships, or constitute a fully "
            "independent spatiotemporal test. The processed emission fields remain inherited "
            "observation-collocated priors."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "experiment", "seed", "best_epoch", "period", "n_valid",
            "unweighted_log_r2", "unweighted_log_rmse", "unweighted_log_bias",
            "area_log_r2", "area_log_rmse", "area_log_bias",
            "checkpoint_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records_sorted:
            for month_row in row["per_month"]:
                writer.writerow({
                    "experiment": row["experiment"],
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "period": f"2022_{month_row['month']:02d}",
                    "n_valid": month_row["n_valid_pixels"],
                    "unweighted_log_r2": month_row["unweighted"]["log_r2"],
                    "unweighted_log_rmse": month_row["unweighted"]["log_rmse"],
                    "unweighted_log_bias": month_row["unweighted"]["log_bias"],
                    "area_log_r2": month_row["area_weighted_cos_lat"]["log_r2"],
                    "area_log_rmse": month_row["area_weighted_cos_lat"]["log_rmse"],
                    "area_log_bias": month_row["area_weighted_cos_lat"]["log_bias"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                })
            writer.writerow({
                "experiment": row["experiment"],
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "period": "pooled_2022_11_12",
                "n_valid": row["n_total_valid_pairs"],
                "unweighted_log_r2": row["pooled_nov_dec"]["unweighted"]["log_r2"],
                "unweighted_log_rmse": row["pooled_nov_dec"]["unweighted"]["log_rmse"],
                "unweighted_log_bias": row["pooled_nov_dec"]["unweighted"]["log_bias"],
                "area_log_r2": row["pooled_nov_dec"]["area_weighted_cos_lat"]["log_r2"],
                "area_log_rmse": row["pooled_nov_dec"]["area_weighted_cos_lat"]["log_rmse"],
                "area_log_bias": row["pooled_nov_dec"]["area_weighted_cos_lat"]["log_bias"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            })

    with contrast_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "contrast", "period", "weighting", "metric",
            "seed42", "seed43", "seed44", "mean", "sample_sd",
            "all_positive", "all_negative", "mixed_sign",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cname, cblock in contrasts.items():
            for period, pblock in cblock.items():
                for weighting, wblock in pblock.items():
                    for metric, summary in wblock.items():
                        writer.writerow({
                            "contrast": cname,
                            "period": period,
                            "weighting": weighting,
                            "metric": metric,
                            "seed42": summary["per_seed"]["42"],
                            "seed43": summary["per_seed"]["43"],
                            "seed44": summary["per_seed"]["44"],
                            "mean": summary["mean"],
                            "sample_sd": summary["sample_sd"],
                            "all_positive": summary["all_positive"],
                            "all_negative": summary["all_negative"],
                            "mixed_sign": summary["mixed_sign"],
                        })

    # Concise terminal seal: primary pooled R2 and contrasts only.
    lines = [
        "GATE_S9F_ONE_TIME_EAST_ASIA_CORE_EVALUATION_PASS",
        "CORE_HOLDOUT_PERFORMANCE_OPENED=True",
        "NO_FURTHER_MODEL_SELECTION_ALLOWED=True",
        "EVALUATION_TYPE=pre-specified_spatially_held_out_geographic_transfer",
        "FULLY_INDEPENDENT_SPATIOTEMPORAL_TEST=False",
        "POSTHOC_EQUIVALENCE_THRESHOLD_USED=False",
        "STATISTICAL_SIGNIFICANCE_TEST_USED=False",
    ]
    for exp in EXPECTED_EXPERIMENTS:
        s = aggregate[exp]["pooled_2022_11_12"]["unweighted"]["log_r2"]
        a = aggregate[exp]["pooled_2022_11_12"]["area_weighted_cos_lat"]["log_r2"]
        lines.extend([
            f"{exp}_POOLED_R2_MEAN={s['mean']:.8f}",
            f"{exp}_POOLED_R2_SD={s['sample_sd']:.8f}",
            f"{exp}_POOLED_AREA_R2_MEAN={a['mean']:.8f}",
            f"{exp}_POOLED_AREA_R2_SD={a['sample_sd']:.8f}",
        ])

    for cname in ["binary_minus_meteo", "joint_minus_meteo", "joint_minus_binary"]:
        s = contrasts[cname]["pooled_2022_11_12"]["unweighted"]["log_r2"]
        a = contrasts[cname]["pooled_2022_11_12"]["area_weighted_cos_lat"]["log_r2"]
        lines.extend([
            f"{cname}_POOLED_R2_MEAN={s['mean']:.8f}",
            f"{cname}_POOLED_R2_SD={s['sample_sd']:.8f}",
            f"{cname}_POOLED_R2_ALL_POSITIVE={s['all_positive']}",
            f"{cname}_POOLED_R2_ALL_NEGATIVE={s['all_negative']}",
            f"{cname}_POOLED_R2_MIXED_SIGN={s['mixed_sign']}",
            f"{cname}_POOLED_AREA_R2_MEAN={a['mean']:.8f}",
            f"{cname}_POOLED_AREA_R2_SD={a['sample_sd']:.8f}",
        ])

    lines.extend([
        f"OUTPUT_JSON={report_path}",
        f"OUTPUT_PER_SEED_CSV={csv_path}",
        f"OUTPUT_CONTRAST_CSV={contrast_csv_path}",
        f"OUTPUT_PREDICTIONS_NPZ={npz_path}",
        f"OUTPUT_TXT={txt_path}",
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
