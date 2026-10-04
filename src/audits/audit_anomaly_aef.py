# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
69_audit_s2_anomaly_clean_formal_seed42.py

Independent read-only audit of one clean formal anomaly run:
anom_meteo_aef / seed 42, 43 or 44.

The audit verifies:
- exact formal contract and frozen-cache provenance;
- full 178/61/361 split records;
- history completeness and finite metrics;
- epoch-based early stopping;
- best epoch/checkpoint consistency;
- strict architecture-compatible checkpoint loading;
- finite checkpoint tensors;
- exact reproduction of the best 2022 held-out validation anomaly metric.

This script does not evaluate 2023 and does not start any new training.
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
from types import ModuleType
from typing import Any

import torch


EXPECTED_EXPERIMENT = "anom_meteo_aef"
ALLOWED_SEEDS = {42, 43, 44}
EXPECTED_RUN_MODE = "formal"
EXPECTED_TRAIN_FILES = 178
EXPECTED_VAL_FILES = 61
EXPECTED_EXTERNAL_FILES = 361
EXPECTED_PARAMETERS = 486241
EXPECTED_PROBES = 9

EXPECTED_CONTRACT = {
    "batch_size": 4,
    "num_workers": 0,
    "patch_size": 96,
    "patches_per_file": 4,
    "base_features": 32,
    "max_epochs": 140,
    "patience": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_every": 5,
    "train_limit": 0,
    "val_limit": 0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)


def finite_float(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def import_training_module(path: Path) -> ModuleType:
    require(path.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {path}")
    spec = importlib.util.spec_from_file_location(
        "clean_anomaly_formal_training_module_generic",
        path,
    )
    require(spec is not None and spec.loader is not None,
            "TRAINING_MODULE_IMPORT_SPEC_FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def verify_contract(actual: dict[str, Any]) -> None:
    require(set(actual) == set(EXPECTED_CONTRACT),
            "FORMAL_CONTRACT_KEY_SET_MISMATCH")
    for key, expected in EXPECTED_CONTRACT.items():
        value = actual[key]
        if isinstance(expected, float):
            require(
                math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12),
                f"FORMAL_CONTRACT_MISMATCH_{key}: {value} != {expected}",
            )
        else:
            require(
                value == expected,
                f"FORMAL_CONTRACT_MISMATCH_{key}: {value} != {expected}",
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit one clean formal anomaly meteo-only run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", help="Relocated restricted anomaly cache; overrides the historical manifest path.")
    parser.add_argument("--aef-source-root-2022")
    parser.add_argument("--aef-source-root-2023")
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    require(args.metric_tolerance == 1e-6,
            "METRIC_TOLERANCE_LOCKED_TO_1E-6")

    require(args.expected_seed in ALLOWED_SEEDS,
            f"EXPECTED_SEED_NOT_ALLOWED: {args.expected_seed}")
    expected_seed = int(args.expected_seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    cache_audit_path = Path(args.cache_audit).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
    require(training_script.is_file(),
            f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(cache_audit_path.is_file(),
            f"CACHE_AUDIT_NOT_FOUND: {cache_audit_path}")
    require(not output_dir.exists(),
            f"STOP_OUTPUT_DIR_EXISTS: {output_dir}")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_val_path = run_dir / "logs" / "latest_val_full_report.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    manifest = load_json(manifest_path)
    history_root = load_json(history_path)
    latest_val = load_json(latest_val_path)
    cache_audit = load_json(cache_audit_path)

    require(manifest.get("artifact_name") == "clean_anomaly_training_run",
            "RUN_MANIFEST_ARTIFACT_MISMATCH")
    require(int(manifest.get("artifact_version", -1)) == 1,
            "RUN_MANIFEST_VERSION_MISMATCH")
    require(manifest.get("task") == "s2_clean_daily_anomaly",
            "TASK_MISMATCH")
    require(manifest.get("experiment") == EXPECTED_EXPERIMENT,
            "EXPERIMENT_MISMATCH")
    require(manifest.get("run_mode") == EXPECTED_RUN_MODE,
            "RUN_MODE_MISMATCH")
    require(int(manifest.get("seed", -1)) == expected_seed,
            "SEED_MISMATCH")
    require(canonical(manifest.get("run_dir", "")) == canonical(run_dir),
            "RUN_DIR_MANIFEST_MISMATCH")
    require(manifest.get("scientific_result_interpretable") is True,
            "FORMAL_INTERPRETABILITY_FLAG_MISMATCH")
    require(manifest.get("formal_result_ready_for_paper") is False,
            "FORMAL_PAPER_READY_FLAG_INVALID")

    verify_contract(manifest.get("contract", {}))

    early = manifest.get("early_stopping", {})
    require(early.get("monitor") ==
            "held-out_2022_validation_pooled_log_r2_anom",
            "EARLY_STOP_MONITOR_MISMATCH")
    require(float(early.get("minimum_delta", -1)) == 1e-6,
            "EARLY_STOP_MIN_DELTA_MISMATCH")
    require(int(early.get("patience", -1)) == 25,
            "EARLY_STOP_PATIENCE_MISMATCH")
    require(early.get("patience_unit") == "epochs",
            "EARLY_STOP_PATIENCE_UNIT_MISMATCH")
    require(int(early.get("validation_frequency_epochs", -1)) == 5,
            "EARLY_STOP_VALIDATION_FREQUENCY_MISMATCH")
    require(early.get("stopping_check") == "validation_epochs_only",
            "EARLY_STOP_CHECK_MODE_MISMATCH")

    effective = manifest.get("effective_counts", {})
    require(int(effective.get("train_files", -1)) == EXPECTED_TRAIN_FILES,
            "EFFECTIVE_TRAIN_COUNT_MISMATCH")
    require(int(effective.get("val_files", -1)) == EXPECTED_VAL_FILES,
            "EFFECTIVE_VAL_COUNT_MISMATCH")
    require(int(effective.get("external_manifest_files", -1)) ==
            EXPECTED_EXTERNAL_FILES,
            "EXTERNAL_MANIFEST_COUNT_MISMATCH")

    require(len(manifest.get("train_entries", [])) == EXPECTED_TRAIN_FILES,
            "TRAIN_ENTRY_LIST_LENGTH_MISMATCH")
    require(len(manifest.get("val_entries", [])) == EXPECTED_VAL_FILES,
            "VAL_ENTRY_LIST_LENGTH_MISMATCH")
    require(len(manifest.get("external_entries", [])) ==
            EXPECTED_EXTERNAL_FILES,
            "EXTERNAL_ENTRY_LIST_LENGTH_MISMATCH")

    probes = manifest.get("input_mode_probes", [])
    require(len(probes) == EXPECTED_PROBES,
            "INPUT_PROBE_COUNT_MISMATCH")
    expected_probe_shapes = {
        "anom_meteo_only": [10, 361, 576],
        "anom_meteo_fire": [12, 361, 576],
        "anom_meteo_aef": [74, 361, 576],
    }
    probe_pairs = set()
    for probe in probes:
        split = str(probe["split"])
        experiment = str(probe["experiment"])
        require(split in {"2022_train", "2022_val", "2023_all"},
                f"UNEXPECTED_PROBE_SPLIT: {split}")
        require(experiment in expected_probe_shapes,
                f"UNEXPECTED_PROBE_EXPERIMENT: {experiment}")
        require(probe.get("input_shape") ==
                expected_probe_shapes[experiment],
                f"PROBE_INPUT_SHAPE_MISMATCH: {split} {experiment}")
        require(probe.get("target_shape") == [1, 361, 576],
                f"PROBE_TARGET_SHAPE_MISMATCH: {split} {experiment}")
        require(probe.get("mask_shape") == [1, 361, 576],
                f"PROBE_MASK_SHAPE_MISMATCH: {split} {experiment}")
        require(int(probe.get("n_valid_pixels", 0)) > 0,
                f"PROBE_NO_VALID_PIXELS: {split} {experiment}")
        require(probe.get("finite") is True,
                f"PROBE_FINITE_FALSE: {split} {experiment}")
        probe_pairs.add((split, experiment))
    require(len(probe_pairs) == EXPECTED_PROBES,
            "INPUT_PROBE_PAIR_SET_MISMATCH")

    require(cache_audit.get("artifact_name") ==
            "clean_anomaly_cache_independent_audit",
            "CACHE_AUDIT_ARTIFACT_MISMATCH")
    require(cache_audit.get("verification", {}).get("overall_pass") is True,
            "CACHE_AUDIT_NOT_PASS")
    require(cache_audit.get("decision") ==
            "PASS_FREEZE_CLEAN_ANOMALY_CACHE",
            "CACHE_AUDIT_DECISION_MISMATCH")
    require(
        manifest.get("cache_audit", {}).get("sha256") ==
        sha256_file(cache_audit_path),
        "CACHE_AUDIT_SHA256_MISMATCH",
    )
    require(
        canonical(manifest.get("cache_audit", {}).get("path", "")) ==
        canonical(cache_audit_path),
        "CACHE_AUDIT_PATH_MISMATCH",
    )

    training_info = manifest.get("training_script", {})
    require(canonical(training_info.get("path", "")) ==
            canonical(training_script),
            "TRAINING_SCRIPT_PATH_MISMATCH")
    require(training_info.get("sha256") ==
            sha256_file(training_script),
            "TRAINING_SCRIPT_SHA256_MISMATCH")

    build_seal_path = Path(
        manifest.get("build_seal", {}).get("path", "")
    ).resolve()
    require(build_seal_path.is_file(),
            "BUILD_SEAL_NOT_FOUND")
    require(
        sha256_file(build_seal_path) ==
        manifest.get("build_seal", {}).get("sha256"),
        "BUILD_SEAL_SHA256_MISMATCH",
    )

    build_seal = load_json(build_seal_path)
    seal_hashes = build_seal.get("manifest_sha256", {})
    require(manifest.get("manifest_sha256", {}) == seal_hashes,
            "CACHE_MANIFEST_HASH_MAP_MISMATCH")
    cache_dir = Path(manifest.get("cache_dir", "")).resolve()
    require(cache_dir.is_dir(), "CACHE_DIR_NOT_FOUND")
    for name, expected_hash in seal_hashes.items():
        path = cache_dir / "manifests" / name
        require(path.is_file(),
                f"CACHE_MANIFEST_NOT_FOUND: {name}")
        require(sha256_file(path) == expected_hash,
                f"CACHE_MANIFEST_CHANGED_AFTER_TRAINING: {name}")

    require(history_root.get("artifact_name") ==
            "clean_anomaly_training_history",
            "HISTORY_ARTIFACT_MISMATCH")
    history = history_root.get("history", [])
    completed_epochs = int(manifest.get("completed_epochs", -1))
    require(1 <= completed_epochs <= EXPECTED_CONTRACT["max_epochs"],
            f"COMPLETED_EPOCHS_OUT_OF_RANGE: {completed_epochs}")
    require(len(history) == completed_epochs,
            f"HISTORY_LENGTH_MISMATCH: {len(history)} != {completed_epochs}")

    validation_rows: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(history, start=1):
        require(int(row.get("epoch", -1)) == expected_epoch,
                f"HISTORY_EPOCH_SEQUENCE_MISMATCH: {expected_epoch}")
        finite_float(row.get("train_patch_loss"),
                     f"TRAIN_LOSS_EPOCH_{expected_epoch}")
        finite_float(row.get("train_patch_r2"),
                     f"TRAIN_R2_EPOCH_{expected_epoch}")

        did_validate = bool(row.get("did_validate"))
        expected_validate = expected_epoch == 1 or expected_epoch % 5 == 0
        require(did_validate == expected_validate,
                f"VALIDATION_SCHEDULE_MISMATCH_EPOCH_{expected_epoch}")

        validation = row.get("validation", {})
        if did_validate:
            for key in (
                "log_r2_anom",
                "log_rmse_anom",
                "log_bias_anom",
                "log_corr_anom",
                "log_r2_recon",
                "log_rmse_recon",
                "log_bias_recon",
            ):
                finite_float(validation.get(key),
                             f"{key}_EPOCH_{expected_epoch}")
            require(int(validation.get("valid_days", -1)) > 0,
                    f"NO_VALIDATION_DAYS_EPOCH_{expected_epoch}")
            require(int(validation.get("n_values", -1)) > 0,
                    f"NO_VALIDATION_VALUES_EPOCH_{expected_epoch}")
            validation_rows.append(row)
        else:
            require(validation.get("log_r2_anom") is None,
                    f"NONVALIDATION_EPOCH_HAS_R2_{expected_epoch}")

    require(validation_rows, "NO_VALIDATION_ROWS")
    validation_r2 = {
        int(row["epoch"]): float(row["validation"]["log_r2_anom"])
        for row in validation_rows
    }
    expected_best_epoch = max(validation_r2, key=validation_r2.get)
    expected_best_r2 = validation_r2[expected_best_epoch]

    best_epoch = int(manifest.get("best_epoch", -1))
    best_r2 = finite_float(
        manifest.get("best_val_log_r2_anom"),
        "MANIFEST_BEST_VAL_R2",
    )
    require(best_epoch == expected_best_epoch,
            "MANIFEST_BEST_EPOCH_HISTORY_MISMATCH")
    require(
        math.isclose(
            best_r2,
            expected_best_r2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "MANIFEST_BEST_R2_HISTORY_MISMATCH",
    )

    if completed_epochs < EXPECTED_CONTRACT["max_epochs"]:
        require(completed_epochs % 5 == 0,
                "EARLY_STOP_NOT_ON_VALIDATION_EPOCH")
        require(completed_epochs - best_epoch >= 25,
                "EARLY_STOP_PATIENCE_EPOCHS_NOT_SATISFIED")
        later_validation_epochs = [
            epoch for epoch in validation_r2
            if best_epoch < epoch <= completed_epochs
        ]
        require(later_validation_epochs,
                "NO_POST_BEST_VALIDATION_ROWS")
        require(
            all(validation_r2[epoch] <= best_r2 + 1e-6
                for epoch in later_validation_epochs),
            "POST_BEST_IMPROVEMENT_EXISTS",
        )

    require(int(latest_val.get("epoch", -1)) ==
            max(validation_r2),
            "LATEST_VAL_EPOCH_MISMATCH")
    require(latest_val.get("experiment") == EXPECTED_EXPERIMENT,
            "LATEST_VAL_EXPERIMENT_MISMATCH")
    require(latest_val.get("run_mode") == EXPECTED_RUN_MODE,
            "LATEST_VAL_RUN_MODE_MISMATCH")
    require(int(latest_val.get("seed", -1)) == expected_seed,
            "LATEST_VAL_SEED_MISMATCH")
    require(int(latest_val.get("validation_entries", -1)) ==
            EXPECTED_VAL_FILES,
            "LATEST_VAL_ENTRY_COUNT_MISMATCH")

    require(checkpoint_path.is_file(),
            "CHECKPOINT_NOT_FOUND")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(isinstance(checkpoint, dict),
            "CHECKPOINT_ROOT_NOT_DICT")
    require(checkpoint.get("artifact_name") ==
            "clean_anomaly_best_checkpoint",
            "CHECKPOINT_ARTIFACT_MISMATCH")
    require(int(checkpoint.get("artifact_version", -1)) == 1,
            "CHECKPOINT_VERSION_MISMATCH")
    require(checkpoint.get("run_mode") == EXPECTED_RUN_MODE,
            "CHECKPOINT_RUN_MODE_MISMATCH")
    require(int(checkpoint.get("seed", -1)) == expected_seed,
            "CHECKPOINT_SEED_MISMATCH")
    require(checkpoint.get("spec", {}).get("name") ==
            EXPECTED_EXPERIMENT,
            "CHECKPOINT_EXPERIMENT_MISMATCH")
    require(int(checkpoint.get("best_epoch", -1)) ==
            best_epoch,
            "CHECKPOINT_BEST_EPOCH_MISMATCH")
    require(
        math.isclose(
            float(checkpoint.get("best_val_log_r2_anom")),
            best_r2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "CHECKPOINT_BEST_R2_MISMATCH",
    )
    require(int(checkpoint.get("trainable_parameters", -1)) ==
            EXPECTED_PARAMETERS,
            "CHECKPOINT_PARAMETER_COUNT_MISMATCH")
    require(int(manifest.get("trainable_parameters", -1)) ==
            EXPECTED_PARAMETERS,
            "MANIFEST_PARAMETER_COUNT_MISMATCH")

    state = checkpoint.get("model_state")
    require(isinstance(state, dict) and state,
            "CHECKPOINT_MODEL_STATE_EMPTY")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor),
                f"CHECKPOINT_STATE_NOT_TENSOR: {name}")
        require(bool(torch.isfinite(tensor).all().item()),
                f"CHECKPOINT_STATE_NONFINITE: {name}")

    module = import_training_module(training_script)
    spec_obj = module.AnomalySpec(**checkpoint["spec"])
    model = module.AnomalyUNet(
        spec_obj,
        base_features=int(checkpoint.get("base_features", 32)),
    )
    incompatible = model.load_state_dict(state, strict=True)
    require(not incompatible.missing_keys,
            "STRICT_LOAD_MISSING_KEYS")
    require(not incompatible.unexpected_keys,
            "STRICT_LOAD_UNEXPECTED_KEYS")
    actual_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    require(actual_parameters == EXPECTED_PARAMETERS,
            f"MODEL_PARAMETER_COUNT_MISMATCH: {actual_parameters}")

    if args.require_cuda:
        require(torch.cuda.is_available(),
                "CUDA_REQUIRED_BUT_UNAVAILABLE")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    require(device.type == "cuda",
            "FORMAL_AUDIT_REPRODUCTION_REQUIRES_CUDA")
    model = model.to(device)

    reader = module.LeanAnomalyReader(
        cache_dir,
        aef_source_root_2022=(None if args.aef_source_root_2022 is None else Path(args.aef_source_root_2022).expanduser()),
        aef_source_root_2023=(None if args.aef_source_root_2023 is None else Path(args.aef_source_root_2023).expanduser()),
    )
    val_entries = list(manifest["val_entries"])
    reproduced = module.evaluate_full_image(
        model,
        val_entries,
        reader,
        spec_obj,
        device,
    )
    reproduced_r2 = finite_float(
        reproduced.get("log_r2_anom"),
        "REPRODUCED_VAL_R2",
    )
    reproduced_rmse = finite_float(
        reproduced.get("log_rmse_anom"),
        "REPRODUCED_VAL_RMSE",
    )
    require(
        math.isclose(
            reproduced_r2,
            best_r2,
            rel_tol=0.0,
            abs_tol=args.metric_tolerance,
        ),
        f"BEST_METRIC_REPRODUCTION_FAIL: "
        f"{reproduced_r2} != {best_r2}",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = (
        output_dir /
        f"anomaly_clean_formal_aef_seed{expected_seed}_audit_v01.json"
    )
    txt_path = (
        output_dir /
        f"anomaly_clean_formal_aef_seed{expected_seed}_audit_v01.txt"
    )

    report = {
        "artifact_name": "clean_anomaly_formal_run_audit",
        "artifact_version": 1,
        "run_dir": str(run_dir),
        "experiment": EXPECTED_EXPERIMENT,
        "run_mode": EXPECTED_RUN_MODE,
        "seed": expected_seed,
        "input_provenance": {
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "training_history": {
                "path": str(history_path),
                "sha256": sha256_file(history_path),
            },
            "latest_validation_report": {
                "path": str(latest_val_path),
                "sha256": sha256_file(latest_val_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "training_script": {
                "path": str(training_script),
                "sha256": sha256_file(training_script),
            },
            "cache_audit": {
                "path": str(cache_audit_path),
                "sha256": sha256_file(cache_audit_path),
            },
            "build_seal": {
                "path": str(build_seal_path),
                "sha256": sha256_file(build_seal_path),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "contract": EXPECTED_CONTRACT,
        "training_summary": {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "best_val_log_r2_anom": best_r2,
            "trainable_parameters": actual_parameters,
            "effective_train_files": EXPECTED_TRAIN_FILES,
            "effective_val_files": EXPECTED_VAL_FILES,
            "external_manifest_files": EXPECTED_EXTERNAL_FILES,
            "early_stopped": (
                completed_epochs <
                EXPECTED_CONTRACT["max_epochs"]
            ),
            "epochs_since_best_at_stop": (
                completed_epochs - best_epoch
            ),
        },
        "metric_reproduction": {
            "tolerance": args.metric_tolerance,
            "checkpoint_best_log_r2_anom": best_r2,
            "recomputed_log_r2_anom": reproduced_r2,
            "recomputed_log_rmse_anom": reproduced_rmse,
            "pass": True,
        },
        "verification": {
            "formal_contract_exact": True,
            "split_contract_exact": True,
            "cache_audit_provenance_exact": True,
            "cache_manifest_hashes_unchanged": True,
            "input_mode_probes_all_pass": True,
            "history_complete_and_finite": True,
            "validation_schedule_exact": True,
            "epoch_based_early_stopping_exact": True,
            "best_epoch_matches_history": True,
            "checkpoint_matches_best_history": True,
            "checkpoint_finite": True,
            "checkpoint_strict_load_pass": True,
            "architecture_parameter_count_exact": True,
            "best_validation_metric_reproduction_pass": True,
            "formal_result_ready_for_external_evaluation": True,
            "overall_pass": True,
        },
        "decision": f"PASS_PROCEED_TO_SEED{expected_seed}_EXTERNAL_EVALUATION",
        "reporting_boundary": {
            "2022_validation_type":
                "held-out_model-selection_validation",
            "single_seed_only": True,
            "statistical_significance_established": False,
            "paper_ready": False,
        },
    }
    save_json(report, json_path)

    lines = [
        f"GATE_A12_CLEAN_ANOMALY_AEF_SEED{expected_seed}_AUDIT_PASS",
        f"RUN_DIR={run_dir}",
        f"EXPERIMENT={EXPECTED_EXPERIMENT}",
        f"RUN_MODE={EXPECTED_RUN_MODE}",
        f"SEED={expected_seed}",
        f"COMPLETED_EPOCHS={completed_epochs}",
        f"BEST_EPOCH={best_epoch}",
        f"EPOCHS_SINCE_BEST_AT_STOP={completed_epochs - best_epoch}",
        f"BEST_VAL_ANOM_LOG_R2={best_r2:.8f}",
        f"REPRODUCED_VAL_ANOM_LOG_R2={reproduced_r2:.8f}",
        f"REPRODUCED_VAL_ANOM_LOG_RMSE={reproduced_rmse:.8f}",
        f"TRAINABLE_PARAMETERS={actual_parameters}",
        f"TRAIN_FILES_EFFECTIVE={EXPECTED_TRAIN_FILES}",
        f"VAL_FILES_EFFECTIVE={EXPECTED_VAL_FILES}",
        "FORMAL_CONTRACT_EXACT=True",
        "CACHE_AUDIT_PROVENANCE_EXACT=True",
        "CACHE_MANIFEST_HASHES_UNCHANGED=True",
        "HISTORY_COMPLETE_AND_FINITE=True",
        "VALIDATION_SCHEDULE_EXACT=True",
        "EPOCH_BASED_EARLY_STOPPING_EXACT=True",
        "CHECKPOINT_MATCHES_HISTORY_BEST=True",
        "CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True",
        "BEST_VALIDATION_METRIC_REPRODUCTION_PASS=True",
        "FORMAL_RESULT_READY_FOR_EXTERNAL_EVALUATION=True",
        "SINGLE_SEED_ONLY=True",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
        "PAPER_READY=False",
        f"DECISION=PASS_PROCEED_TO_SEED{expected_seed}_EXTERNAL_EVALUATION",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
