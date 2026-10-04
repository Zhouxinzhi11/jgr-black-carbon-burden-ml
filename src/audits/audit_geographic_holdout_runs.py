# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
84_audit_s2_meanstate_spatial_holdout_formal_run_gateS9D.py

Gate S9D read-only audit for one formal spatial-holdout mean-state run
created by 83_train_s2_meanstate_spatial_holdout_gateS9C.py.

Reusable for:
- mean_meteo_clim_only
- mean_binary_support_meteo_clim
- mean_emis_meteo_clim
and seeds 42, 43, 44.

This audit never computes East-Asia core predictive performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_TRAINING_ARTIFACT = "s2_meanstate_spatial_holdout_training_gateS9C"
EXPECTED_S9A_ARTIFACT = "s2_meanstate_spatial_holdout_preflight_gateS9A"
EXPECTED_S9B_ARTIFACT = "s2_meanstate_spatial_train_only_meteo_stats_gateS9B"

EXPECTED_83_SHA256 = (
    "e839ac22651728ce8de2eb676a7f7e8edf1a20a59300f39d09f03b8d429b1dea"
)
EXPECTED_47_SHA256 = (
    "ff3bec0f270b1f8dd191f43230d9665761fb6f4c65baafd71bdfc56cda6a2fe8"
)
EXPECTED_77_SHA256 = (
    "b2de46481d2119a7c4bb49dc698ce27ed34390ee6ca37b47a472e5cfde41b2e1"
)
EXPECTED_MONTHLY_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)

ALLOWED_EXPERIMENTS = {
    "mean_meteo_clim_only": {
        "emission_channels": 0,
        "meteo_channels": 5,
        "input_channels": 5,
        "trainable_parameters": 477025,
    },
    "mean_binary_support_meteo_clim": {
        "emission_channels": 1,
        "meteo_channels": 5,
        "input_channels": 6,
        "trainable_parameters": 476449,
    },
    "mean_emis_meteo_clim": {
        "emission_channels": 8,
        "meteo_channels": 5,
        "input_channels": 13,
        "trainable_parameters": 477457,
    },
}
ALLOWED_SEEDS = {42, 43, 44}

EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
EXPECTED_CORE_BBOX = [220, 290, 448, 520]
EXPECTED_TRAIN_EXCLUSION_BBOX = [188, 322, 416, 552]
EXPECTED_SELECTION_EXCLUSION_BBOX = [156, 354, 384, 575]
EXPECTED_ELIGIBLE_PATCH_POSITIONS = 77386
EXPECTED_GUARD_CELLS = 32
EXPECTED_SELECTION_GUARD_CELLS = 64
EXPECTED_RF_DIAMETER = 46
EXPECTED_RF_HALF = 23


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


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


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    require(isinstance(obj, dict), "CHECKPOINT_ROOT_NOT_DICT")
    return obj


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name}_IS_NONE")
    value = float(value)
    require(math.isfinite(value), f"NONFINITE_{name}: {value}")
    return value


def import_base47(path: Path):
    module_name = "gate_s9d_locked_base47"
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


def build_spec(base, experiment: str):
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
    raise RuntimeError(f"EXPERIMENT_NOT_ALLOWED: {experiment}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S9D read-only audit for one formal spatial-holdout run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(ALLOWED_EXPERIMENTS))
    parser.add_argument("--expected-seed", required=True, type=int, choices=sorted(ALLOWED_SEEDS))
    parser.add_argument("--s9a-report", required=True)
    parser.add_argument("--s9b-report", required=True)
    parser.add_argument("--base-training-script", required=True)
    parser.add_argument("--binary-training-script", required=True)
    parser.add_argument("--spatial-training-script", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    s9a_path = Path(args.s9a_report).expanduser().resolve()
    s9b_path = Path(args.s9b_report).expanduser().resolve()
    base47_path = Path(args.base_training_script).expanduser().resolve()
    binary77_path = Path(args.binary_training_script).expanduser().resolve()
    spatial83_path = Path(args.spatial_training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for p in [run_dir, s9a_path, s9b_path, base47_path, binary77_path, spatial83_path]:
        require(p.exists(), f"REQUIRED_INPUT_NOT_FOUND: {p}")
    require(run_dir.is_dir(), f"RUN_DIR_NOT_DIRECTORY: {run_dir}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    require(sha256_file(base47_path) == EXPECTED_47_SHA256, "BASE47_SHA256_MISMATCH")
    require(sha256_file(binary77_path) == EXPECTED_77_SHA256, "BINARY77_SHA256_MISMATCH")
    require(sha256_file(spatial83_path) == EXPECTED_83_SHA256, "SPATIAL83_SHA256_MISMATCH")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_safe_path = run_dir / "logs" / "latest_safe_selection_report.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    for p in [manifest_path, history_path, latest_safe_path, checkpoint_path]:
        require(p.is_file() and p.stat().st_size > 0, f"MISSING_OR_EMPTY_ARTIFACT: {p}")

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    latest_safe = load_json(latest_safe_path)
    checkpoint = load_checkpoint(checkpoint_path)
    s9a = load_json(s9a_path)
    s9b = load_json(s9b_path)

    seed = int(args.expected_seed)
    experiment = args.experiment
    expected_run_name = f"{experiment}_seed{seed}"

    # Identity and formal settings.
    require(manifest.get("artifact_name") == EXPECTED_TRAINING_ARTIFACT, "MANIFEST_ARTIFACT_MISMATCH")
    require(int(manifest.get("artifact_version", -1)) == 1, "MANIFEST_VERSION_MISMATCH")
    require(manifest.get("run_mode") == "formal", "RUN_MODE_NOT_FORMAL")
    require(manifest.get("run_name") == expected_run_name, "RUN_NAME_MISMATCH")
    require(int(manifest.get("seed", -1)) == seed, "MANIFEST_SEED_MISMATCH")
    require(norm_path(manifest.get("run_dir", "")) == norm_path(run_dir), "RUN_DIR_MISMATCH")

    exp = manifest.get("experiment", {})
    require(exp.get("name") == experiment, "EXPERIMENT_NAME_MISMATCH")
    cfg = ALLOWED_EXPERIMENTS[experiment]
    require(int(exp.get("emission_channels", -1)) == cfg["emission_channels"], "EMISSION_CHANNELS_MISMATCH")
    require(int(exp.get("meteo_channels", -1)) == cfg["meteo_channels"], "METEO_CHANNELS_MISMATCH")

    settings = manifest.get("settings", {})
    exact = {
        "epochs": 160,
        "samples_per_epoch": 512,
        "batch_size": 8,
        "crop_size": 128,
        "patience": 25,
        "base_features": 32,
        "num_workers": 0,
        "seed": seed,
        "require_cuda": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in exact.items():
        require(settings.get(key) == expected, f"SETTING_MISMATCH_{key}")
    require(math.isclose(float(settings["learning_rate"]), 3e-4, rel_tol=0.0, abs_tol=1e-15), "LR_MISMATCH")
    require(math.isclose(float(settings["weight_decay"]), 1e-2, rel_tol=0.0, abs_tol=1e-15), "WEIGHT_DECAY_MISMATCH")

    # Split contract.
    train_months = [int(x["month"]) for x in manifest.get("train_entries", [])]
    val_months = [int(x["month"]) for x in manifest.get("val_entries", [])]
    ext_months = [
        int(x["month"])
        for x in manifest.get("external_2023_entries_recorded_but_not_evaluated", [])
    ]
    require(train_months == EXPECTED_TRAIN_MONTHS, f"TRAIN_MONTHS_MISMATCH: {train_months}")
    require(val_months == EXPECTED_VAL_MONTHS, f"VAL_MONTHS_MISMATCH: {val_months}")
    require(ext_months == EXPECTED_EXTERNAL_MONTHS, f"EXTERNAL_MONTHS_MISMATCH: {ext_months}")
    dropped22 = manifest.get("dropped_2022_entries", [])
    dropped23 = manifest.get("dropped_2023_entries", [])
    require(len(dropped22) == 1 and int(dropped22[0]["month"]) == 7, "DROPPED_2022_MISMATCH")
    require(not dropped23, "DROPPED_2023_MUST_BE_EMPTY")

    # Spatial contract.
    spatial = manifest.get("spatial_contract", {})
    require(spatial.get("primary_region") == "East_Asia", "PRIMARY_REGION_MISMATCH")
    require(spatial.get("core_bbox_indices") == EXPECTED_CORE_BBOX, "CORE_BBOX_MISMATCH")
    require(spatial.get("training_exclusion_bbox_indices") == EXPECTED_TRAIN_EXCLUSION_BBOX, "TRAIN_EXCLUSION_BBOX_MISMATCH")
    require(spatial.get("selection_exclusion_bbox_indices") == EXPECTED_SELECTION_EXCLUSION_BBOX, "SELECTION_EXCLUSION_BBOX_MISMATCH")
    require(int(spatial.get("guard_cells", -1)) == EXPECTED_GUARD_CELLS, "GUARD_MISMATCH")
    require(int(spatial.get("selection_guard_cells", -1)) == EXPECTED_SELECTION_GUARD_CELLS, "SELECTION_GUARD_MISMATCH")
    require(int(spatial.get("crop_size", -1)) == 128, "SPATIAL_CROP_MISMATCH")
    require(int(spatial.get("eligible_patch_positions", -1)) == EXPECTED_ELIGIBLE_PATCH_POSITIONS, "ELIGIBLE_PATCH_COUNT_MISMATCH")
    require(int(spatial.get("receptive_field_diameter", -1)) == EXPECTED_RF_DIAMETER, "RF_DIAMETER_MISMATCH")
    require(int(spatial.get("max_half_receptive_field", -1)) == EXPECTED_RF_HALF, "RF_HALF_MISMATCH")
    require(spatial.get("core_test_opened_during_training") is False, "CORE_TEST_WAS_OPENED_DURING_TRAINING")
    require(EXPECTED_RF_HALF <= EXPECTED_GUARD_CELLS, "TRAINING_GUARD_TOO_SMALL_FOR_RF")

    # Normalization contract.
    normalization = manifest.get("normalization_contract", {})
    require(normalization.get("spatial_train_only") is True, "SPATIAL_TRAIN_NORMALIZATION_FALSE")
    require(normalization.get("variables") == ["Temp", "RH", "BLH", "Windspeed_U", "Windspeed_V"], "METEO_VARIABLE_ORDER_MISMATCH")
    require(normalization.get("application_order") == "affine transform before nan_to_num", "AFFINE_APPLICATION_ORDER_MISMATCH")
    require(normalization.get("cache_rebuilt") is False, "CACHE_REBUILT_FLAG_MISMATCH")
    scales = [finite_float(x, "AFFINE_SCALE") for x in normalization.get("affine_scale", [])]
    offsets = [finite_float(x, "AFFINE_OFFSET") for x in normalization.get("affine_offset", [])]
    require(len(scales) == 5 and len(offsets) == 5, "AFFINE_LENGTH_MISMATCH")
    require(all(x > 0 for x in scales), "AFFINE_SCALE_NONPOSITIVE")

    # Provenance.
    provenance = manifest.get("provenance", {})
    require(provenance.get("monthly_manifest_sha256") == EXPECTED_MONTHLY_MANIFEST_SHA256, "MONTHLY_MANIFEST_HASH_MISMATCH")
    require(provenance.get("s9a_report_sha256") == sha256_file(s9a_path), "S9A_HASH_MISMATCH")
    require(provenance.get("s9b_report_sha256") == sha256_file(s9b_path), "S9B_HASH_MISMATCH")
    require(provenance.get("base_47_script_sha256") == EXPECTED_47_SHA256, "MANIFEST_BASE47_HASH_MISMATCH")
    require(provenance.get("binary_77_script_sha256") == EXPECTED_77_SHA256, "MANIFEST_BINARY77_HASH_MISMATCH")
    require(provenance.get("training_script_sha256") == EXPECTED_83_SHA256, "MANIFEST_83_HASH_MISMATCH")

    require(s9a.get("artifact_name") == EXPECTED_S9A_ARTIFACT, "S9A_ARTIFACT_MISMATCH")
    require(s9a.get("verification", {}).get("overall_pass") is True, "S9A_NOT_PASS")
    require(s9b.get("artifact_name") == EXPECTED_S9B_ARTIFACT, "S9B_ARTIFACT_MISMATCH")
    require(s9b.get("verification", {}).get("overall_pass") is True, "S9B_NOT_PASS")

    # History and early stopping.
    keys = [
        "train_loss",
        "train_patch_r2",
        "safe_selection_log_r2",
        "safe_selection_log_rmse",
        "safe_selection_log_bias",
        "lr",
    ]
    lengths = {}
    for key in keys:
        values = history.get(key)
        require(isinstance(values, list), f"HISTORY_KEY_NOT_LIST_{key}")
        lengths[key] = len(values)
        for i, value in enumerate(values, start=1):
            finite_float(value, f"{key}_epoch{i}")
    require(len(set(lengths.values())) == 1, f"HISTORY_LENGTH_MISMATCH: {lengths}")
    completed_epochs = next(iter(lengths.values()))
    require(1 <= completed_epochs <= 160, "COMPLETED_EPOCHS_OUT_OF_RANGE")

    r2_series = [float(x) for x in history["safe_selection_log_r2"]]
    best_index = max(range(completed_epochs), key=lambda i: r2_series[i])
    best_epoch = best_index + 1
    best_r2 = r2_series[best_index]
    epochs_after_best = completed_epochs - best_epoch
    if completed_epochs < 160:
        require(epochs_after_best == 25, "EARLY_STOP_PATIENCE_INCONSISTENT")
    else:
        require(epochs_after_best < 25, "MAX_EPOCH_RUN_SHOULD_HAVE_EARLY_STOPPED")

    # Checkpoint.
    require(checkpoint.get("experiment") == experiment, "CHECKPOINT_EXPERIMENT_MISMATCH")
    require(checkpoint.get("run_name") == expected_run_name, "CHECKPOINT_RUN_NAME_MISMATCH")
    require(checkpoint.get("run_mode") == "formal", "CHECKPOINT_RUN_MODE_MISMATCH")
    require(int(checkpoint.get("seed", -1)) == seed, "CHECKPOINT_SEED_MISMATCH")
    require(int(checkpoint.get("epoch", -1)) == best_index, "CHECKPOINT_BEST_EPOCH_MISMATCH")
    ckpt_best = finite_float(checkpoint.get("best_safe_selection_log_r2"), "CHECKPOINT_BEST_SAFE_R2")
    require(math.isclose(ckpt_best, best_r2, rel_tol=0.0, abs_tol=1e-8), "CHECKPOINT_BEST_R2_MISMATCH")
    require(checkpoint.get("core_holdout_performance_opened") is False, "CHECKPOINT_CORE_OPENED_TRUE")
    require(checkpoint.get("training_script_sha256") == EXPECTED_83_SHA256, "CHECKPOINT_83_HASH_MISMATCH")
    require(checkpoint.get("s9a_report_sha256") == sha256_file(s9a_path), "CHECKPOINT_S9A_HASH_MISMATCH")
    require(checkpoint.get("s9b_report_sha256") == sha256_file(s9b_path), "CHECKPOINT_S9B_HASH_MISMATCH")

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and state, "MODEL_STATE_DICT_MISSING")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor), f"STATE_NOT_TENSOR: {name}")
        require(bool(torch.isfinite(tensor).all().item()), f"NONFINITE_STATE_TENSOR: {name}")

    # Strict architecture reconstruction.
    base = import_base47(base47_path)
    spec = build_spec(base, experiment)
    model = base.MeanStateUNet(spec=spec, base_features=32)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(n_params == cfg["trainable_parameters"], f"PARAMETER_COUNT_MISMATCH: {n_params}")

    # Latest safe-selection artifact: must be explicitly non-core.
    require(
        latest_safe.get("evaluation_role")
        == "checkpoint_selection_only_outside_core_plus_64_cell_exclusion",
        "LATEST_SAFE_SELECTION_ROLE_MISMATCH",
    )
    require(latest_safe.get("core_holdout_performance_opened") is False, "LATEST_SAFE_CORE_OPENED_TRUE")
    require(int(latest_safe.get("n_groups", -1)) == 2, "LATEST_SAFE_GROUP_COUNT_MISMATCH")
    require(int(latest_safe.get("n_total_safe_selection_pixels", 0)) > 10000, "LATEST_SAFE_TOO_FEW_PIXELS")
    latest_global = latest_safe.get("global", {})
    for metric in ["log_r2", "log_rmse", "log_bias"]:
        finite_float(latest_global.get(metric), f"LATEST_SAFE_{metric}")

    # Input preflight.
    input_preflight = manifest.get("input_preflight", [])
    require(len(input_preflight) == 8, "INPUT_PREFLIGHT_COUNT_MISMATCH")
    require(all(row.get("all_finite") is True for row in input_preflight), "INPUT_PREFLIGHT_NONFINITE")
    require(
        [int(row["month"]) for row in input_preflight]
        == EXPECTED_TRAIN_MONTHS + EXPECTED_VAL_MONTHS,
        "INPUT_PREFLIGHT_MONTHS_MISMATCH",
    )
    require(
        all(int(row["input_shape"][0]) == cfg["input_channels"] for row in input_preflight),
        "INPUT_PREFLIGHT_CHANNEL_MISMATCH",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / f"{experiment}_seed{seed}_formal_audit_gateS9D.json"
    txt_path = output_dir / f"{experiment}_seed{seed}_formal_audit_gateS9D.txt"

    report = {
        "artifact_name": "s2_meanstate_spatial_holdout_formal_run_audit_gateS9D",
        "artifact_version": 1,
        "status": "PASS_READ_ONLY_FORMAL_AUDIT",
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(run_dir),
        "inputs": {
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "training_history": str(history_path),
            "training_history_sha256": sha256_file(history_path),
            "latest_safe_selection_report": str(latest_safe_path),
            "latest_safe_selection_report_sha256": sha256_file(latest_safe_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "s9a_report": str(s9a_path),
            "s9a_report_sha256": sha256_file(s9a_path),
            "s9b_report": str(s9b_path),
            "s9b_report_sha256": sha256_file(s9b_path),
            "base_47_script": str(base47_path),
            "base_47_script_sha256": EXPECTED_47_SHA256,
            "binary_77_script": str(binary77_path),
            "binary_77_script_sha256": EXPECTED_77_SHA256,
            "spatial_83_script": str(spatial83_path),
            "spatial_83_script_sha256": EXPECTED_83_SHA256,
        },
        "summary": {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "epochs_after_best": epochs_after_best,
            "best_safe_selection_log_r2": best_r2,
            "trainable_parameters": n_params,
        },
        "verification": {
            "formal_contract_exact": True,
            "split_contract_exact": True,
            "spatial_contract_exact": True,
            "normalization_contract_exact": True,
            "provenance_exact": True,
            "history_complete_and_finite": True,
            "early_stopping_consistent": True,
            "checkpoint_matches_history_best": True,
            "checkpoint_strict_load_pass": True,
            "latest_safe_selection_noncore_and_finite": True,
            "input_preflight_pass": True,
            "core_holdout_performance_opened": False,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This audit establishes formal training integrity only. It deliberately does "
            "not evaluate East-Asia core predictive performance. Core performance must "
            "remain unopened until the planned spatial-holdout formal run set is complete "
            "and audited."
        ),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        f"GATE_S9D_{experiment}_SEED{seed}_FORMAL_AUDIT_PASS",
        f"EXPERIMENT={experiment}",
        f"SEED={seed}",
        f"COMPLETED_EPOCHS={completed_epochs}",
        f"BEST_EPOCH={best_epoch}",
        f"EPOCHS_AFTER_BEST={epochs_after_best}",
        f"BEST_SAFE_SELECTION_LOG_R2={best_r2:.8f}",
        f"TRAINABLE_PARAMETERS={n_params}",
        "FORMAL_CONTRACT_EXACT=True",
        "SPLIT_CONTRACT_EXACT=True",
        "SPATIAL_CONTRACT_EXACT=True",
        "NORMALIZATION_CONTRACT_EXACT=True",
        "PROVENANCE_EXACT=True",
        "HISTORY_COMPLETE_AND_FINITE=True",
        "EARLY_STOPPING_CONSISTENT=True",
        "CHECKPOINT_MATCHES_HISTORY_BEST=True",
        "CHECKPOINT_STRICT_LOAD_PASS=True",
        "LATEST_SAFE_SELECTION_NONCORE_AND_FINITE=True",
        "CORE_HOLDOUT_PERFORMANCE_OPENED=False",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
