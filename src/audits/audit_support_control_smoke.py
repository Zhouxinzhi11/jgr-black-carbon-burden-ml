# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
58_audit_s2_meanstate_support_preserving_smoke.py

Read-only audit of the Gate-P2B support-preserving smoke run.
No training artifact is modified.
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


EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
EXPECTED_RUN_NAME = "mean_emis_meteo_clim_support_preserving_seed42_smoke"
EXPECTED_EXPERIMENT = "mean_emis_meteo_clim_support_preserving"
EXPECTED_SETTINGS = {
    "val_months": [11, 12],
    "blacklist_months": [7],
    "min_train_days": 5,
    "min_valid_fraction": 1e-6,
    "crop_size": 128,
    "batch_size": 8,
    "samples_per_epoch": 32,
    "epochs": 2,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "patience": 2,
    "base_features": 32,
    "num_workers": 0,
    "seed": 42,
    "require_cuda": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "support_epsilon": 0.0,
    "support_preserving_permutation": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    number = float(value)
    require(math.isfinite(number), f"NONFINITE_{name}: {value}")
    return number


def load_training_module(path: Path):
    module_name = "support_preserving_training_for_smoke_audit"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None, "TRAINING_IMPORT_SPEC_FAILED")
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


def month_list(entries: list[dict[str, Any]]) -> list[int]:
    return sorted(int(e["month"]) for e in entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--gatep2a-report", required=True)
    parser.add_argument(
        "--training-script",
        default="./57_train_s2_meanstate_support_preserving.py",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    gate11b = Path(args.gate11b_report).expanduser().resolve()
    gatep2a = Path(args.gatep2a_report).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
    require(gate11b.is_file(), f"GATE11B_NOT_FOUND: {gate11b}")
    require(gatep2a.is_file(), f"GATEP2A_NOT_FOUND: {gatep2a}")
    require(training_script.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_val_path = run_dir / "logs" / "latest_val_monthly_report.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    for path in [manifest_path, history_path, latest_val_path, checkpoint_path]:
        require(path.is_file(), f"MISSING_ARTIFACT: {path}")
        require(path.stat().st_size > 0, f"EMPTY_ARTIFACT: {path}")

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    latest_val = load_json(latest_val_path)
    checkpoint = load_checkpoint(checkpoint_path)

    require(
        manifest.get("artifact_name") == "s2_meanstate_support_preserving_training",
        "MANIFEST_ARTIFACT_NAME_MISMATCH",
    )
    require(int(manifest.get("artifact_version", -1)) == 1, "MANIFEST_VERSION_MISMATCH")
    require(manifest.get("run_mode") == "smoke", "RUN_MODE_NOT_SMOKE")
    require(manifest.get("run_name") == EXPECTED_RUN_NAME, "RUN_NAME_MISMATCH")
    require(
        normalized_path(manifest.get("run_dir", "")) == normalized_path(run_dir),
        "RUN_DIR_PROVENANCE_MISMATCH",
    )

    experiment = manifest.get("experiment", {})
    require(experiment.get("name") == EXPECTED_EXPERIMENT, "EXPERIMENT_NAME_MISMATCH")
    require(experiment.get("uses_emission") is True, "EXPERIMENT_EMISSION_FLAG_FAILED")
    require(experiment.get("uses_meteo") is True, "EXPERIMENT_METEO_FLAG_FAILED")
    require(
        experiment.get("support_preserving_permutation") is True,
        "EXPERIMENT_SUPPORT_PRESERVING_FLAG_FAILED",
    )
    require(int(experiment.get("emission_channels", -1)) == 8, "EMISSION_CHANNELS_MISMATCH")
    require(int(experiment.get("meteo_channels", -1)) == 5, "METEO_CHANNELS_MISMATCH")

    settings = manifest.get("settings", {})
    for key, expected in EXPECTED_SETTINGS.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            require(
                actual is not None
                and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12),
                f"SETTING_MISMATCH_{key}: actual={actual} expected={expected}",
            )
        else:
            require(actual == expected, f"SETTING_MISMATCH_{key}: actual={actual} expected={expected}")

    require(
        month_list(list(manifest.get("train_entries", []))) == EXPECTED_TRAIN_MONTHS,
        "TRAIN_MONTHS_MISMATCH",
    )
    require(
        month_list(list(manifest.get("val_entries", []))) == EXPECTED_VAL_MONTHS,
        "VAL_MONTHS_MISMATCH",
    )
    require(
        month_list(list(manifest.get("external_2023_entries", [])))
        == EXPECTED_EXTERNAL_MONTHS,
        "EXTERNAL_MONTHS_MISMATCH",
    )

    provenance = manifest.get("provenance", {})
    require(
        provenance.get("gate11b_report_sha256") == sha256_file(gate11b),
        "GATE11B_HASH_MISMATCH",
    )
    require(
        provenance.get("gatep2a_report_sha256") == sha256_file(gatep2a),
        "GATEP2A_HASH_MISMATCH",
    )
    training_hash = sha256_file(training_script)
    require(
        provenance.get("training_script_sha256") == training_hash,
        "TRAINING_SCRIPT_HASH_MISMATCH",
    )

    expected_history_keys = [
        "train_loss",
        "train_patch_r2",
        "val_monthly_log_r2",
        "val_monthly_log_rmse",
        "lr",
    ]
    for key in expected_history_keys:
        values = history.get(key)
        require(isinstance(values, list), f"HISTORY_KEY_NOT_LIST_{key}")
        require(len(values) == 2, f"HISTORY_LENGTH_MISMATCH_{key}: {len(values)}")
        for idx, value in enumerate(values, start=1):
            finite_float(value, f"{key}_epoch{idx}")

    val_r2 = [float(x) for x in history["val_monthly_log_r2"]]
    best_index = max(range(len(val_r2)), key=lambda i: val_r2[i])
    require(best_index == 1, f"UNEXPECTED_BEST_EPOCH_INDEX: {best_index}")
    require(int(checkpoint.get("epoch", -1)) == best_index, "CHECKPOINT_EPOCH_MISMATCH")
    require(
        math.isclose(
            finite_float(checkpoint.get("best_val_log_r2"), "checkpoint_best_val_log_r2"),
            val_r2[best_index],
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "CHECKPOINT_BEST_METRIC_MISMATCH",
    )

    require(checkpoint.get("experiment") == EXPECTED_EXPERIMENT, "CHECKPOINT_EXPERIMENT_MISMATCH")
    require(checkpoint.get("run_name") == EXPECTED_RUN_NAME, "CHECKPOINT_RUN_NAME_MISMATCH")
    require(checkpoint.get("run_mode") == "smoke", "CHECKPOINT_RUN_MODE_MISMATCH")
    require(checkpoint.get("support_preserving_permutation") is True, "CHECKPOINT_SUPPORT_FLAG_FAILED")
    require(
        math.isclose(float(checkpoint.get("support_epsilon")), 0.0, rel_tol=0.0, abs_tol=0.0),
        "CHECKPOINT_SUPPORT_EPSILON_MISMATCH",
    )
    require(
        checkpoint.get("gate11b_report_sha256") == sha256_file(gate11b),
        "CHECKPOINT_GATE11B_HASH_MISMATCH",
    )
    require(
        checkpoint.get("gatep2a_report_sha256") == sha256_file(gatep2a),
        "CHECKPOINT_GATEP2A_HASH_MISMATCH",
    )
    require(
        checkpoint.get("training_script_sha256") == training_hash,
        "CHECKPOINT_TRAINING_SCRIPT_HASH_MISMATCH",
    )

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and state, "MODEL_STATE_DICT_MISSING")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor), f"STATE_VALUE_NOT_TENSOR: {name}")
        require(bool(torch.isfinite(tensor).all().item()), f"NONFINITE_CHECKPOINT_TENSOR: {name}")

    module = load_training_module(training_script)
    spec_obj = module.EXPERIMENT_SPECS[EXPECTED_EXPERIMENT]
    model = module.MeanStateUNet(spec=spec_obj, base_features=32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not missing, f"MODEL_STATE_MISSING_KEYS: {missing}")
    require(not unexpected, f"MODEL_STATE_UNEXPECTED_KEYS: {unexpected}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(n_params == 477457, f"TRAINABLE_PARAMETER_COUNT_MISMATCH: {n_params}")

    require(int(latest_val.get("n_groups", -1)) == 2, "LATEST_VAL_GROUP_COUNT_MISMATCH")
    require(int(latest_val.get("n_total_valid_pixels", 0)) > 0, "LATEST_VAL_HAS_NO_PIXELS")
    global_metrics = latest_val.get("global", {})
    finite_float(global_metrics.get("log_r2"), "latest_val_log_r2")
    finite_float(global_metrics.get("log_rmse"), "latest_val_log_rmse")
    finite_float(global_metrics.get("log_bias"), "latest_val_log_bias")
    require(
        month_list(list(latest_val.get("per_group", []))) == EXPECTED_VAL_MONTHS,
        "LATEST_VAL_MONTHS_MISMATCH",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "support_preserving_smoke_audit.json"
    txt_path = output_dir / "support_preserving_smoke_audit.txt"

    report = {
        "artifact_name": "s2_meanstate_support_preserving_smoke_audit",
        "artifact_version": 1,
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
            "gate11b_report": str(gate11b),
            "gate11b_report_sha256": sha256_file(gate11b),
            "gatep2a_report": str(gatep2a),
            "gatep2a_report_sha256": sha256_file(gatep2a),
            "training_script": str(training_script),
            "training_script_sha256": training_hash,
        },
        "summary": {
            "completed_epochs": 2,
            "best_epoch": best_index + 1,
            "best_val_log_r2": val_r2[best_index],
            "trainable_parameters": n_params,
        },
        "verification": {
            "smoke_contract_exact": True,
            "split_contract_exact": True,
            "gate11b_provenance_exact": True,
            "gatep2a_provenance_exact": True,
            "training_script_hash_exact": True,
            "history_complete_and_finite": True,
            "best_checkpoint_matches_history": True,
            "checkpoint_finite_and_architecture_compatible": True,
            "latest_validation_complete_and_finite": True,
            "overall_pass": True,
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    lines = [
        "GATEP2C_SUPPORT_PRESERVING_SMOKE_AUDIT_PASS",
        f"RUN_DIR={run_dir}",
        "RUN_MODE=smoke",
        f"COMPLETED_EPOCHS=2",
        f"BEST_EPOCH={best_index + 1}",
        f"BEST_VAL_LOG_R2={val_r2[best_index]:.8f}",
        f"TRAINABLE_PARAMETERS={n_params}",
        "SMOKE_CONTRACT_EXACT=True",
        "SPLIT_CONTRACT_EXACT=True",
        "GATE11B_PROVENANCE_EXACT=True",
        "GATEP2A_PROVENANCE_EXACT=True",
        "CHECKPOINT_MATCHES_HISTORY_BEST=True",
        "CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True",
        "SCIENTIFIC_RESULT_INTERPRETABLE=False",
        f"OUTPUT_JSON={report_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
