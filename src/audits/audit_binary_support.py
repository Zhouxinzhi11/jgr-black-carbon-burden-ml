# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
78_audit_s2_meanstate_binary_support_formal_run.py

Gate S8D read-only audit for one formal binary-support + meteorology mean-state run
created by 77_train_s2_meanstate_binary_support_gateS8.py.

Reusable for seeds 42, 43, and 44. The audit never modifies training artifacts
and never overwrites an existing audit directory.
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


EXPECTED_EXPERIMENT = "mean_binary_support_meteo_clim"
EXPECTED_ARTIFACT = "s2_meanstate_binary_support_training_gateS8"
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
ALLOWED_SEEDS = {42, 43, 44}

EXPECTED_SUPPORT_DEFINITION = "any(abs(emission_log_mean) > 0.0, dim=0)"
EXPECTED_SUPPORT_EPSILON = 0.0
EXPECTED_SUPPORT_CHANNELS = 1
EXPECTED_METEO_CHANNELS = 5
EXPECTED_INPUT_CHANNELS = 6
EXPECTED_TRAINABLE_PARAMETERS = 476449
EXPECTED_BASE_TRAINING_SHA256 = (
    "ff3bec0f270b1f8dd191f43230d9665761fb6f4c65baafd71bdfc56cda6a2fe8"
)
EXPECTED_PROXY_TENSOR_SHA256 = (
    "283867292ed0b51ebce073db35940c4b091d23cacba8ca929f5248982bb0d749"
)
EXPECTED_PROXY_SUPPORT_CELLS = 42949


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
    number = float(value)
    require(math.isfinite(number), f"NONFINITE_{name}: {value}")
    return number


def month_list(entries: list[dict[str, Any]]) -> list[int]:
    return [int(entry["month"]) for entry in entries]


def load_training_module(path: Path):
    module_name = "binary_support_training_module_for_gate_s8d_audit"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(
        spec is not None and spec.loader is not None,
        "TRAINING_IMPORT_SPEC_FAILED",
    )
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


def save_json_strict(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S8D read-only audit for one formal binary-support run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--expected-seed",
        required=True,
        type=int,
        choices=sorted(ALLOWED_SEEDS),
    )
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--s8a-preflight-report", required=True)
    parser.add_argument(
        "--training-script",
        default="./77_train_s2_meanstate_binary_support_gateS8.py",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    expected_seed = int(args.expected_seed)
    expected_run_name = f"{EXPECTED_EXPERIMENT}_seed{expected_seed}"

    run_dir = Path(args.run_dir).expanduser().resolve()
    gate11b_path = Path(args.gate11b_report).expanduser().resolve()
    s8a_path = Path(args.s8a_preflight_report).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
    require(gate11b_path.is_file(), f"GATE11B_NOT_FOUND: {gate11b_path}")
    require(s8a_path.is_file(), f"S8A_PREFLIGHT_NOT_FOUND: {s8a_path}")
    require(training_script.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(
        not output_dir.exists(),
        f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}",
    )

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_val_path = run_dir / "logs" / "latest_val_monthly_report.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    for path in [
        manifest_path,
        history_path,
        latest_val_path,
        checkpoint_path,
    ]:
        require(path.is_file(), f"MISSING_ARTIFACT: {path}")
        require(path.stat().st_size > 0, f"EMPTY_ARTIFACT: {path}")

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    latest_val = load_json(latest_val_path)
    checkpoint = load_checkpoint(checkpoint_path)
    gate11b = load_json(gate11b_path)
    s8a = load_json(s8a_path)

    # Formal run identity.
    require(
        manifest.get("artifact_name") == EXPECTED_ARTIFACT,
        "MANIFEST_ARTIFACT_NAME_MISMATCH",
    )
    require(
        int(manifest.get("artifact_version", -1)) == 1,
        "MANIFEST_VERSION_MISMATCH",
    )
    require(manifest.get("run_mode") == "formal", "RUN_MODE_NOT_FORMAL")
    require(
        manifest.get("run_name") == expected_run_name,
        f"RUN_NAME_MISMATCH: {manifest.get('run_name')}",
    )
    require(
        norm_path(manifest.get("run_dir", "")) == norm_path(run_dir),
        "RUN_DIR_MISMATCH",
    )

    experiment = manifest.get("experiment", {})
    require(
        experiment.get("name") == EXPECTED_EXPERIMENT,
        "EXPERIMENT_NAME_MISMATCH",
    )
    require(experiment.get("uses_emission") is True, "USES_EMISSION_MISMATCH")
    require(experiment.get("uses_meteo") is True, "USES_METEO_MISMATCH")
    require(
        experiment.get("shuffle_emission") is False,
        "SHUFFLE_EMISSION_FLAG_MISMATCH",
    )
    require(
        experiment.get("binary_support_emission") is True,
        "BINARY_SUPPORT_FLAG_MISMATCH",
    )
    require(
        int(experiment.get("emission_channels", -1)) == EXPECTED_SUPPORT_CHANNELS,
        "EMISSION_CHANNELS_MISMATCH",
    )
    require(
        int(experiment.get("meteo_channels", -1)) == EXPECTED_METEO_CHANNELS,
        "METEO_CHANNELS_MISMATCH",
    )

    # Exact formal settings.
    settings = manifest.get("settings", {})
    exact_settings = {
        "val_months": EXPECTED_VAL_MONTHS,
        "blacklist_months": [7],
        "min_train_days": 5,
        "crop_size": 128,
        "batch_size": 8,
        "samples_per_epoch": 512,
        "epochs": 160,
        "patience": 25,
        "base_features": 32,
        "num_workers": 0,
        "seed": expected_seed,
        "require_cuda": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in exact_settings.items():
        require(
            settings.get(key) == expected,
            f"SETTING_MISMATCH_{key}: "
            f"actual={settings.get(key)!r}, expected={expected!r}",
        )

    float_settings = {
        "min_valid_fraction": 1e-6,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
    }
    for key, expected in float_settings.items():
        actual = settings.get(key)
        require(
            actual is not None
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"SETTING_MISMATCH_{key}: actual={actual}, expected={expected}",
        )

    # Split contract.
    train_entries = list(manifest.get("train_entries", []))
    val_entries = list(manifest.get("val_entries", []))
    external_entries = list(manifest.get("external_2023_entries", []))
    dropped_2022 = list(manifest.get("dropped_2022_entries", []))
    dropped_2023 = list(manifest.get("dropped_2023_entries", []))

    require(
        month_list(train_entries) == EXPECTED_TRAIN_MONTHS,
        f"TRAIN_MONTHS_MISMATCH: {month_list(train_entries)}",
    )
    require(
        month_list(val_entries) == EXPECTED_VAL_MONTHS,
        f"VAL_MONTHS_MISMATCH: {month_list(val_entries)}",
    )
    require(
        month_list(external_entries) == EXPECTED_EXTERNAL_MONTHS,
        f"EXTERNAL_MONTHS_MISMATCH: {month_list(external_entries)}",
    )
    require(
        len(dropped_2022) == 1
        and int(dropped_2022[0].get("month", -1)) == 7,
        "DROPPED_2022_CONTRACT_MISMATCH",
    )
    require(not dropped_2023, "DROPPED_2023_MUST_BE_EMPTY")

    # Gate provenance + recorded script hashes.
    gate11b_hash = sha256_file(gate11b_path)
    s8a_hash = sha256_file(s8a_path)
    training_hash = sha256_file(training_script)

    require(
        gate11b.get("verification", {}).get("overall_pass") is True,
        "GATE11B_OVERALL_PASS_NOT_TRUE",
    )
    require(
        s8a.get("status") == "PASS_READ_ONLY_PREFLIGHT",
        "S8A_STATUS_MISMATCH",
    )
    require(
        s8a.get("verification", {}).get("overall_pass") is True,
        "S8A_OVERALL_PASS_NOT_TRUE",
    )

    provenance = manifest.get("provenance", {})
    require(
        provenance.get("gate11b_report_sha256") == gate11b_hash,
        "GATE11B_HASH_MISMATCH",
    )
    require(
        provenance.get("s8a_preflight_report_sha256") == s8a_hash,
        "S8A_HASH_MISMATCH",
    )
    require(
        provenance.get("training_script_sha256") == training_hash,
        "TRAINING_SCRIPT_HASH_MISMATCH",
    )
    require(
        provenance.get("base_training_script_sha256")
        == EXPECTED_BASE_TRAINING_SHA256,
        "BASE_47_SHA256_MISMATCH",
    )
    require(
        provenance.get("support_definition") == EXPECTED_SUPPORT_DEFINITION,
        "PROVENANCE_SUPPORT_DEFINITION_MISMATCH",
    )
    require(
        provenance.get("proxy_tensor_sha256")
        == EXPECTED_PROXY_TENSOR_SHA256,
        "PROVENANCE_PROXY_TENSOR_SHA256_MISMATCH",
    )
    require(
        int(provenance.get("proxy_support_cells", -1))
        == EXPECTED_PROXY_SUPPORT_CELLS,
        "PROVENANCE_PROXY_SUPPORT_CELLS_MISMATCH",
    )

    monthly_manifest_path = Path(provenance.get("monthly_manifest", ""))
    require(
        monthly_manifest_path.is_file(),
        "RECORDED_MONTHLY_MANIFEST_NOT_FOUND",
    )
    require(
        sha256_file(monthly_manifest_path)
        == provenance.get("monthly_manifest_sha256"),
        "MONTHLY_MANIFEST_HASH_MISMATCH",
    )

    # Re-run the training script's own provenance gates.
    module = load_training_module(training_script)
    require(
        getattr(module, "S8_EXPERIMENT", None) == EXPECTED_EXPERIMENT,
        "TRAINING_MODULE_EXPERIMENT_CONSTANT_MISMATCH",
    )
    gate11b_recheck = module.validate_gate11b_provenance(
        cache_dir=manifest.get("cache_dir", ""),
        gate11b_report_path=str(gate11b_path),
    )
    s8a_recheck = module.validate_s8a_preflight(
        cache_dir=manifest.get("cache_dir", ""),
        preflight_report_path=str(s8a_path),
    )
    require(
        gate11b_recheck.get("gate11b_report_sha256") == gate11b_hash,
        "GATE11B_RECHECK_HASH_MISMATCH",
    )
    require(
        s8a_recheck.get("s8a_preflight_report_sha256") == s8a_hash,
        "S8A_RECHECK_HASH_MISMATCH",
    )

    # Locked binary-support contract.
    support_contract = manifest.get("binary_support_contract", {})
    expected_contract = {
        "support_definition": EXPECTED_SUPPORT_DEFINITION,
        "support_epsilon": EXPECTED_SUPPORT_EPSILON,
        "support_channels": EXPECTED_SUPPORT_CHANNELS,
        "meteo_channels": EXPECTED_METEO_CHANNELS,
        "total_input_channels": EXPECTED_INPUT_CHANNELS,
        "2022_support_source": "actual monthly processed emission prior",
        "2023_support_source_for_later_eval": (
            "fixed unweighted mean of 2022 training-month processed priors, then binarized"
        ),
        "expected_2023_proxy_tensor_sha256": EXPECTED_PROXY_TENSOR_SHA256,
        "expected_2023_proxy_support_cells": EXPECTED_PROXY_SUPPORT_CELLS,
    }
    for key, expected in expected_contract.items():
        actual = support_contract.get(key)
        if isinstance(expected, float):
            require(
                actual is not None
                and math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                f"BINARY_SUPPORT_CONTRACT_MISMATCH_{key}",
            )
        else:
            require(
                actual == expected,
                f"BINARY_SUPPORT_CONTRACT_MISMATCH_{key}: "
                f"actual={actual!r}, expected={expected!r}",
            )

    # Training history / best epoch / early stopping.
    history_keys = [
        "train_loss",
        "train_patch_r2",
        "val_monthly_log_r2",
        "val_monthly_log_rmse",
        "lr",
    ]
    lengths: dict[str, int] = {}
    for key in history_keys:
        values = history.get(key)
        require(isinstance(values, list), f"HISTORY_NOT_LIST_{key}")
        lengths[key] = len(values)
        for index, value in enumerate(values, start=1):
            finite_float(value, f"{key}_epoch{index}")

    require(
        len(set(lengths.values())) == 1,
        f"HISTORY_LENGTHS_DIFFER: {lengths}",
    )
    completed_epochs = next(iter(lengths.values()))
    require(
        1 <= completed_epochs <= 160,
        f"COMPLETED_EPOCHS_OUT_OF_RANGE: {completed_epochs}",
    )

    val_r2 = [float(v) for v in history["val_monthly_log_r2"]]
    best_index = max(range(completed_epochs), key=lambda i: val_r2[i])
    best_epoch = best_index + 1
    best_val_r2 = val_r2[best_index]
    epochs_after_best = completed_epochs - best_epoch

    require(
        int(checkpoint.get("epoch", -1)) == best_index,
        "CHECKPOINT_EPOCH_MISMATCH",
    )
    require(
        math.isclose(
            finite_float(
                checkpoint.get("best_val_log_r2"),
                "checkpoint_best_val_log_r2",
            ),
            best_val_r2,
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "CHECKPOINT_BEST_METRIC_MISMATCH",
    )
    if completed_epochs < 160:
        require(
            epochs_after_best == 25,
            f"EARLY_STOPPING_INCONSISTENT: "
            f"completed={completed_epochs}, best={best_epoch}, "
            f"after_best={epochs_after_best}",
        )

    # Checkpoint identity, hashes and model reconstruction.
    require(
        checkpoint.get("experiment") == EXPECTED_EXPERIMENT,
        "CKPT_EXPERIMENT_MISMATCH",
    )
    require(
        checkpoint.get("run_name") == expected_run_name,
        "CKPT_RUN_NAME_MISMATCH",
    )
    require(checkpoint.get("run_mode") == "formal", "CKPT_RUN_MODE_MISMATCH")
    require(
        checkpoint.get("gate11b_report_sha256") == gate11b_hash,
        "CKPT_GATE11B_HASH_MISMATCH",
    )
    require(
        checkpoint.get("s8a_preflight_report_sha256") == s8a_hash,
        "CKPT_S8A_HASH_MISMATCH",
    )
    require(
        checkpoint.get("training_script_sha256") == training_hash,
        "CKPT_TRAINING_SCRIPT_HASH_MISMATCH",
    )
    require(
        checkpoint.get("binary_support_definition")
        == EXPECTED_SUPPORT_DEFINITION,
        "CKPT_SUPPORT_DEFINITION_MISMATCH",
    )
    require(
        math.isclose(
            float(checkpoint.get("binary_support_epsilon")),
            EXPECTED_SUPPORT_EPSILON,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "CKPT_SUPPORT_EPSILON_MISMATCH",
    )

    checkpoint_spec = checkpoint.get("spec", {})
    require(
        checkpoint_spec.get("binary_support_emission") is True,
        "CKPT_SPEC_BINARY_SUPPORT_FLAG_MISMATCH",
    )
    require(
        int(checkpoint_spec.get("emission_channels", -1))
        == EXPECTED_SUPPORT_CHANNELS,
        "CKPT_SPEC_EMISSION_CHANNELS_MISMATCH",
    )
    require(
        int(checkpoint_spec.get("meteo_channels", -1))
        == EXPECTED_METEO_CHANNELS,
        "CKPT_SPEC_METEO_CHANNELS_MISMATCH",
    )

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and state, "MODEL_STATE_DICT_MISSING")
    for name, tensor in state.items():
        require(
            isinstance(tensor, torch.Tensor),
            f"STATE_NOT_TENSOR: {name}",
        )
        require(
            bool(torch.isfinite(tensor).all().item()),
            f"NONFINITE_STATE_TENSOR: {name}",
        )

    spec_obj = module.EXPERIMENT_SPECS[EXPECTED_EXPERIMENT]
    require(
        spec_obj.input_channels == EXPECTED_INPUT_CHANNELS,
        "SPEC_INPUT_CHANNELS_MISMATCH",
    )
    require(
        spec_obj.emission_channels == EXPECTED_SUPPORT_CHANNELS,
        "SPEC_SUPPORT_CHANNELS_MISMATCH",
    )
    require(
        spec_obj.meteo_channels == EXPECTED_METEO_CHANNELS,
        "SPEC_METEO_CHANNELS_MISMATCH",
    )

    model = module.MeanStateUNet(
        spec=spec_obj,
        base_features=32,
    )
    model.load_state_dict(state, strict=True)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    require(
        trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS,
        f"PARAMETER_COUNT_MISMATCH: {trainable_parameters}",
    )

    # Latest full-field validation artifact.
    # This is the final executed epoch, not necessarily the best checkpoint.
    require(
        int(latest_val.get("n_groups", -1)) == 2,
        "LATEST_VAL_GROUP_COUNT_MISMATCH",
    )
    require(
        int(latest_val.get("n_total_valid_pixels", 0)) > 0,
        "LATEST_VAL_NO_PIXELS",
    )
    latest_groups = list(latest_val.get("per_group", []))
    require(
        month_list(latest_groups) == EXPECTED_VAL_MONTHS,
        f"LATEST_VAL_MONTHS_MISMATCH: {month_list(latest_groups)}",
    )
    for row in latest_groups:
        finite_float(
            row.get("log_r2"),
            f"latest_val_month_{row.get('month')}_r2",
        )
        finite_float(
            row.get("log_rmse"),
            f"latest_val_month_{row.get('month')}_rmse",
        )
        require(
            int(row.get("n_valid_pixels", 0)) > 0,
            f"LATEST_VAL_MONTH_{row.get('month')}_NO_PIXELS",
        )

    global_metrics = latest_val.get("global", {})
    latest_global_r2 = finite_float(
        global_metrics.get("log_r2"),
        "latest_val_global_log_r2",
    )
    latest_global_rmse = finite_float(
        global_metrics.get("log_rmse"),
        "latest_val_global_log_rmse",
    )
    latest_global_bias = finite_float(
        global_metrics.get("log_bias"),
        "latest_val_global_log_bias",
    )

    # Write audit artifacts.
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = (
        output_dir
        / f"binary_support_formal_seed{expected_seed}_audit_gateS8D.json"
    )
    txt_path = (
        output_dir
        / f"binary_support_formal_seed{expected_seed}_audit_gateS8D.txt"
    )

    report = {
        "artifact_name": "s2_meanstate_binary_support_formal_run_audit_gateS8D",
        "artifact_version": 1,
        "status": "PASS_READ_ONLY_FORMAL_AUDIT",
        "seed": expected_seed,
        "run_name": expected_run_name,
        "run_dir": str(run_dir),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "history": str(history_path),
            "history_sha256": sha256_file(history_path),
            "latest_validation": str(latest_val_path),
            "latest_validation_sha256": sha256_file(latest_val_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "gate11b_report": str(gate11b_path),
            "gate11b_report_sha256": gate11b_hash,
            "s8a_preflight_report": str(s8a_path),
            "s8a_preflight_report_sha256": s8a_hash,
            "training_script": str(training_script),
            "training_script_sha256": training_hash,
        },
        "summary": {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "epochs_after_best": epochs_after_best,
            "best_val_log_r2": best_val_r2,
            "latest_epoch_val_log_r2": latest_global_r2,
            "latest_epoch_val_log_rmse": latest_global_rmse,
            "latest_epoch_val_log_bias": latest_global_bias,
            "trainable_parameters": trainable_parameters,
        },
        "binary_support_contract": expected_contract,
        "verification": {
            "formal_contract_exact": True,
            "split_contract_exact": True,
            "gate11b_provenance_exact": True,
            "gate_s8a_provenance_exact": True,
            "binary_support_contract_exact": True,
            "history_complete_and_finite": True,
            "early_stopping_consistent": True,
            "checkpoint_matches_history_best": True,
            "checkpoint_provenance_exact": True,
            "checkpoint_finite_and_architecture_compatible": True,
            "latest_validation_complete_and_finite": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This read-only audit establishes provenance, implementation, checkpoint, "
            "and training-contract integrity only. The best validation metric is from "
            "November-December 2022 held-out model-selection validation and is not an "
            "independent test. Scientific interpretation requires the predeclared "
            "three-seed formal evaluation, including the fixed-2022-prior proxy contract "
            "for 2023."
        ),
    }
    save_json_strict(report, report_path)

    lines = [
        f"GATE_S8D_BINARY_SUPPORT_FORMAL_SEED{expected_seed}_AUDIT_PASS",
        f"RUN_DIR={run_dir}",
        "RUN_MODE=formal",
        f"EXPERIMENT={EXPECTED_EXPERIMENT}",
        f"SEED={expected_seed}",
        f"COMPLETED_EPOCHS={completed_epochs}",
        f"BEST_EPOCH={best_epoch}",
        f"EPOCHS_AFTER_BEST={epochs_after_best}",
        f"BEST_VAL_LOG_R2={best_val_r2:.8f}",
        f"TRAINABLE_PARAMETERS={trainable_parameters}",
        "FORMAL_CONTRACT_EXACT=True",
        "SPLIT_CONTRACT_EXACT=True",
        "GATE11B_PROVENANCE_EXACT=True",
        "GATE_S8A_PROVENANCE_EXACT=True",
        "BINARY_SUPPORT_CONTRACT_EXACT=True",
        "HISTORY_COMPLETE_AND_FINITE=True",
        "EARLY_STOPPING_CONSISTENT=True",
        "CHECKPOINT_MATCHES_HISTORY_BEST=True",
        "CHECKPOINT_PROVENANCE_EXACT=True",
        "CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True",
        "LATEST_VALIDATION_COMPLETE_AND_FINITE=True",
        "SCIENTIFIC_EVALUATION_NOT_YET_RUN=True",
        f"OUTPUT_JSON={report_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
