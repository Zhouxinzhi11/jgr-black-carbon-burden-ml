# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
51_audit_s2_meanstate_clean_formal_run.py

Reusable read-only audit for any formal clean mean-state training run created by
47_train_s2_meanstate_clean.py.

Validated invariants:
- formal run mode;
- approved split: train [2,3,5,6,8,9], validation [11,12], blacklist [7],
  external 2023 [1..12];
- exact formal hyperparameters;
- Gate-11B provenance and artifact hashes;
- finite and internally consistent history;
- dynamic best epoch selection;
- early stopping consistency;
- checkpoint/history consistency;
- all checkpoint tensors finite;
- strict model reconstruction from the recorded training script.

The script never modifies training artifacts and never overwrites an audit directory.
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
EXPECTED_VALIDATION_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
ALLOWED_SEEDS = {42, 43, 44}
ALLOWED_EXPERIMENTS = {
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name} is None")
    number = float(value)
    require(math.isfinite(number), f"{name} is non-finite: {value}")
    return number


def find_nonfinite_json_values(obj: Any, prefix: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            failures.extend(
                find_nonfinite_json_values(value, f"{prefix}.{key}")
            )
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            failures.extend(
                find_nonfinite_json_values(value, f"{prefix}[{index}]")
            )
    elif isinstance(obj, float) and not math.isfinite(obj):
        failures.append(f"{prefix}={obj}")
    return failures


def load_training_module(script_path: Path):
    module_name = "meanstate_clean_training_module_for_generic_audit"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    return module


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reusable audit for a formal clean mean-state training run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    gate11b_path = Path(args.gate11b_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(args.expected_experiment in ALLOWED_EXPERIMENTS,
            f"Unsupported experiment: {args.expected_experiment}")
    require(args.expected_seed in ALLOWED_SEEDS,
            f"Unsupported seed: {args.expected_seed}")
    require(run_dir.is_dir(), f"Run directory not found: {run_dir}")
    require(gate11b_path.is_file(), f"Gate-11B report not found: {gate11b_path}")
    require(not output_dir.exists(),
            "Audit output directory already exists. This script never overwrites.")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_validation_path = run_dir / "logs" / "latest_val_monthly_report.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    for path in [manifest_path, history_path, latest_validation_path, checkpoint_path]:
        require(path.is_file() and path.stat().st_size > 0,
                f"Missing or empty artifact: {path}")

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    latest_validation = load_json(latest_validation_path)
    checkpoint = load_checkpoint(checkpoint_path)
    gate11b = load_json(gate11b_path)

    # Manifest identity and split.
    require(manifest.get("artifact_name") == "s2_meanstate_clean_training",
            "Unexpected manifest artifact_name.")
    require(int(manifest.get("artifact_version", -1)) == 1,
            "Unexpected manifest artifact_version.")
    require(manifest.get("run_mode") == "formal", "Run is not formal.")
    require(normalized_path(manifest.get("run_dir", "")) == normalized_path(run_dir),
            "Manifest run_dir mismatch.")

    experiment = manifest.get("experiment", {})
    require(experiment.get("name") == args.expected_experiment,
            "Manifest experiment mismatch.")
    settings = manifest.get("settings", {})
    require(int(settings.get("seed", -1)) == args.expected_seed,
            "Manifest seed mismatch.")

    expected_settings = {
        "val_months": [11, 12],
        "blacklist_months": [7],
        "min_train_days": 5,
        "min_valid_fraction": 1e-6,
        "crop_size": 128,
        "batch_size": 8,
        "samples_per_epoch": 512,
        "epochs": 160,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "patience": 25,
        "base_features": 32,
        "num_workers": 0,
        "require_cuda": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in expected_settings.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            require(
                actual is not None
                and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15),
                f"Setting mismatch: {key}; expected={expected}, actual={actual}",
            )
        else:
            require(actual == expected,
                    f"Setting mismatch: {key}; expected={expected}, actual={actual}")

    train_months = [int(e["month"]) for e in manifest.get("train_entries", [])]
    validation_months = [int(e["month"]) for e in manifest.get("val_entries", [])]
    external_months = [
        int(e["month"]) for e in manifest.get("external_2023_entries", [])
    ]
    dropped_2022 = manifest.get("dropped_2022_entries", [])
    dropped_2023 = manifest.get("dropped_2023_entries", [])

    require(train_months == EXPECTED_TRAIN_MONTHS,
            f"Unexpected train months: {train_months}")
    require(validation_months == EXPECTED_VALIDATION_MONTHS,
            f"Unexpected validation months: {validation_months}")
    require(external_months == EXPECTED_EXTERNAL_MONTHS,
            f"Unexpected external months: {external_months}")
    require(
        len(dropped_2022) == 1 and int(dropped_2022[0]["month"]) == 7,
        f"Only 2022-07 may be dropped: {dropped_2022}",
    )
    require(dropped_2023 == [], f"No 2023 month may be dropped: {dropped_2023}")

    # Provenance.
    require(gate11b.get("verification", {}).get("overall_pass") is True,
            "Gate-11B overall_pass is not true.")
    provenance = manifest.get("provenance", {})
    require(
        normalized_path(provenance.get("gate11b_report", ""))
        == normalized_path(gate11b_path),
        "Gate-11B report path mismatch.",
    )
    require(
        provenance.get("gate11b_report_sha256") == sha256_file(gate11b_path),
        "Gate-11B report hash mismatch.",
    )

    monthly_manifest_path = Path(provenance.get("monthly_manifest", ""))
    training_script_path = Path(provenance.get("training_script", ""))
    require(monthly_manifest_path.is_file(), "Recorded monthly manifest is missing.")
    require(training_script_path.is_file(), "Recorded training script is missing.")
    require(
        provenance.get("monthly_manifest_sha256")
        == sha256_file(monthly_manifest_path),
        "Monthly manifest changed after training.",
    )
    require(
        provenance.get("training_script_sha256")
        == sha256_file(training_script_path),
        "Training script changed after training.",
    )

    # History.
    history_keys = [
        "train_loss",
        "train_patch_r2",
        "val_monthly_log_r2",
        "val_monthly_log_rmse",
        "lr",
    ]
    lengths: dict[str, int] = {}
    for key in history_keys:
        require(isinstance(history.get(key), list),
                f"History field missing or invalid: {key}")
        lengths[key] = len(history[key])

    require(len(set(lengths.values())) == 1,
            f"History lengths differ: {lengths}")
    completed_epochs = next(iter(set(lengths.values())))
    require(1 <= completed_epochs <= 160,
            f"Invalid completed epoch count: {completed_epochs}")
    nonfinite_history = find_nonfinite_json_values(history)
    require(not nonfinite_history,
            f"History contains non-finite values: {nonfinite_history}")

    train_loss = [
        finite_float(v, f"train_loss[{i}]")
        for i, v in enumerate(history["train_loss"])
    ]
    train_patch_r2 = [
        finite_float(v, f"train_patch_r2[{i}]")
        for i, v in enumerate(history["train_patch_r2"])
    ]
    val_r2 = [
        finite_float(v, f"val_r2[{i}]")
        for i, v in enumerate(history["val_monthly_log_r2"])
    ]
    val_rmse = [
        finite_float(v, f"val_rmse[{i}]")
        for i, v in enumerate(history["val_monthly_log_rmse"])
    ]
    learning_rates = [
        finite_float(v, f"lr[{i}]")
        for i, v in enumerate(history["lr"])
    ]

    best_index = max(range(completed_epochs), key=lambda i: val_r2[i])
    best_epoch_one_based = best_index + 1
    best_history_r2 = val_r2[best_index]
    best_history_rmse = val_rmse[best_index]

    patience = int(settings["patience"])
    stopped_early = completed_epochs < int(settings["epochs"])
    epochs_without_improvement = completed_epochs - best_epoch_one_based
    if stopped_early:
        require(
            epochs_without_improvement == patience,
            "Early stopping inconsistent with patience: "
            f"completed={completed_epochs}, best={best_epoch_one_based}, "
            f"patience={patience}",
        )

    # Latest validation report must match final epoch, not best epoch.
    nonfinite_latest = find_nonfinite_json_values(latest_validation)
    require(not nonfinite_latest,
            f"Latest validation report contains non-finite values: {nonfinite_latest}")
    latest_global = latest_validation.get("global", {})
    latest_r2 = finite_float(latest_global.get("log_r2"),
                             "latest_validation.global.log_r2")
    latest_rmse = finite_float(latest_global.get("log_rmse"),
                               "latest_validation.global.log_rmse")
    require(math.isclose(latest_r2, val_r2[-1], rel_tol=0.0, abs_tol=1e-12),
            "Latest validation R2 differs from final history epoch.")
    require(math.isclose(latest_rmse, val_rmse[-1], rel_tol=0.0, abs_tol=1e-12),
            "Latest validation RMSE differs from final history epoch.")
    per_group = latest_validation.get("per_group", [])
    require(isinstance(per_group, list) and len(per_group) == 2,
            "Latest validation report must contain two groups.")
    require(sorted(int(item["month"]) for item in per_group) == [11, 12],
            "Latest validation report months are not 11 and 12.")

    # Checkpoint.
    require(checkpoint.get("experiment") == args.expected_experiment,
            "Checkpoint experiment mismatch.")
    require(checkpoint.get("run_mode") == "formal",
            "Checkpoint run_mode mismatch.")
    require(int(checkpoint.get("base_features", -1)) == 32,
            "Checkpoint base_features mismatch.")
    require(normalized_path(checkpoint.get("cache_dir", ""))
            == normalized_path(manifest.get("cache_dir", "")),
            "Checkpoint cache_dir mismatch.")
    require(checkpoint.get("gate11b_report_sha256")
            == provenance.get("gate11b_report_sha256"),
            "Checkpoint Gate-11B hash mismatch.")
    require(checkpoint.get("training_script_sha256")
            == provenance.get("training_script_sha256"),
            "Checkpoint training-script hash mismatch.")

    checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
    checkpoint_best_r2 = finite_float(
        checkpoint.get("best_val_log_r2"),
        "checkpoint.best_val_log_r2",
    )
    require(checkpoint_epoch == best_epoch_one_based,
            "Checkpoint epoch differs from history best epoch.")
    require(math.isclose(checkpoint_best_r2, best_history_r2,
                         rel_tol=0.0, abs_tol=1e-12),
            "Checkpoint best R2 differs from history maximum.")

    state_dict = checkpoint.get("model_state_dict")
    require(isinstance(state_dict, dict) and state_dict,
            "Checkpoint state_dict is missing or empty.")
    nonfinite_state: list[str] = []
    state_tensor_count = 0
    state_element_count = 0
    for name, tensor in state_dict.items():
        require(isinstance(tensor, torch.Tensor),
                f"Non-tensor state_dict value: {name}")
        state_tensor_count += 1
        state_element_count += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite_state.append(name)
    require(not nonfinite_state,
            f"Checkpoint contains non-finite tensors: {nonfinite_state}")

    # Strict architecture reconstruction.
    training_module = load_training_module(training_script_path)
    model_spec = training_module.EXPERIMENT_SPECS[args.expected_experiment]
    model = training_module.MeanStateUNet(
        spec=model_spec,
        base_features=32,
    )
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    model.load_state_dict(state_dict, strict=True)

    report = {
        "audit_name": "s2_meanstate_clean_formal_training_generic",
        "read_only": True,
        "experiment": args.expected_experiment,
        "seed": args.expected_seed,
        "inputs": {
            "run_dir": str(run_dir),
            "gate11b_report": str(gate11b_path),
        },
        "artifact_hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "history_sha256": sha256_file(history_path),
            "latest_validation_sha256": sha256_file(latest_validation_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_script_sha256": sha256_file(training_script_path),
        },
        "verified_split": {
            "train_months": train_months,
            "validation_months": validation_months,
            "external_2023_months": external_months,
            "dropped_2022_months": [int(e["month"]) for e in dropped_2022],
        },
        "training_summary": {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch_one_based,
            "best_val_log_r2": best_history_r2,
            "best_val_log_rmse_at_same_epoch": best_history_rmse,
            "final_val_log_r2": val_r2[-1],
            "final_val_log_rmse": val_rmse[-1],
            "stopped_early": stopped_early,
            "epochs_without_improvement_at_end": epochs_without_improvement,
            "patience": patience,
            "minimum_train_loss": min(train_loss),
            "maximum_train_patch_r2": max(train_patch_r2),
            "learning_rate_min": min(learning_rates),
            "learning_rate_max": max(learning_rates),
        },
        "checkpoint": {
            "selected_epoch": checkpoint_epoch,
            "best_val_log_r2": checkpoint_best_r2,
            "state_tensor_count": state_tensor_count,
            "state_element_count_including_buffers": state_element_count,
            "all_state_tensors_finite": True,
            "strict_model_load_pass": True,
            "trainable_parameters": trainable_parameters,
        },
        "verification": {
            "manifest_and_provenance_pass": True,
            "formal_settings_and_split_pass": True,
            "history_finite_and_consistent": True,
            "early_stopping_consistent": True,
            "latest_validation_is_final_epoch_report": True,
            "checkpoint_matches_history_best": True,
            "checkpoint_finite_and_architecture_compatible": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This audit validates training integrity only. The result remains a "
            "single-seed held-out model-selection validation result until formal "
            "external evaluation and three-seed aggregation are complete."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_json = output_dir / "training_audit.json"
    report_txt = output_dir / "training_audit.txt"

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(report),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    report_txt.write_text(
        "\n".join(
            [
                "Clean mean-state formal training audit",
                f"Experiment: {args.expected_experiment}",
                f"Seed: {args.expected_seed}",
                f"Completed epochs: {completed_epochs}",
                f"Best epoch: {best_epoch_one_based}",
                f"Best validation log R2: {best_history_r2}",
                f"Best validation log RMSE: {best_history_rmse}",
                f"Stopped early: {stopped_early}",
                f"Early stopping consistent: True",
                f"Trainable parameters: {trainable_parameters}",
                "Checkpoint finite and strict load pass: True",
                "Overall pass: True",
            ]
        ),
        encoding="utf-8",
    )

    print("GATE_GENERIC_CLEAN_MEANSTATE_FORMAL_TRAINING_AUDIT_PASS")
    print(f"EXPERIMENT={args.expected_experiment}")
    print(f"SEED={args.expected_seed}")
    print(f"JSON_REPORT={report_json}")
    print(f"TXT_REPORT={report_txt}")
    print("MANIFEST_AND_PROVENANCE_PASS=True")
    print("FORMAL_SETTINGS_AND_SPLIT_PASS=True")
    print("HISTORY_FINITE_AND_CONSISTENT=True")
    print(f"COMPLETED_EPOCHS={completed_epochs}")
    print(f"BEST_EPOCH={best_epoch_one_based}")
    print(f"BEST_VAL_LOG_R2={best_history_r2:.8f}")
    print(f"BEST_VAL_LOG_RMSE={best_history_rmse:.8f}")
    print(f"EARLY_STOPPING_CONSISTENT=True")
    print("CHECKPOINT_MATCHES_HISTORY_BEST=True")
    print("CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True")
    print("SINGLE_SEED_RESULT_READY_FOR_PAPER=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
