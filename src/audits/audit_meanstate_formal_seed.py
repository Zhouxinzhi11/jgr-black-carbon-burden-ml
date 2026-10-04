# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
49_audit_s2_meanstate_clean_formal_seed42_gate13b.py

Gate 13B: read-only audit of the first formal clean mean-state training run.

Expected run:
- experiment: mean_meteo_clim_only
- seed: 42
- run mode: formal
- clean Gate-11B cache
- 2022 train months: 02, 03, 05, 06, 08, 09
- 2022 validation months: 11, 12
- 2023 external months: 01-12

The audit verifies:
1. manifest, Gate-11B provenance, script hash, and exact formal settings;
2. finite and internally consistent training history;
3. best checkpoint equals the maximum validation R2 in history;
4. early stopping is consistent with patience=25;
5. latest validation report equals the final epoch, not the best epoch;
6. checkpoint metadata and all state tensors are finite;
7. checkpoint loads strictly into the recorded model architecture;
8. the model has exactly 477,025 trainable parameters.

No artifact is modified.
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


EXPECTED_EXPERIMENT = "mean_meteo_clim_only"
EXPECTED_SEED = 42
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VALIDATION_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
EXPECTED_TRAINABLE_PARAMETERS = 477025


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
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(str(path))
        )
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name} is None")
    number = float(value)
    require(math.isfinite(number), f"{name} is non-finite: {value}")
    return number


def find_nonfinite_json_values(
    obj: Any,
    prefix: str = "root",
) -> list[str]:
    failures: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            failures.extend(
                find_nonfinite_json_values(
                    value,
                    f"{prefix}.{key}",
                )
            )
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            failures.extend(
                find_nonfinite_json_values(
                    value,
                    f"{prefix}[{index}]",
                )
            )
    elif isinstance(obj, float) and not math.isfinite(obj):
        failures.append(f"{prefix}={obj}")
    return failures


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "meanstate_clean_training_module_for_audit",
        str(script_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import training script: {script_path}"
        )
    module = importlib.util.module_from_spec(spec)

    # Python 3.8 dataclasses with postponed annotations require the
    # dynamically created module to be visible in sys.modules while
    # class decorators are executed.
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        raise

    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 13B read-only audit of formal clean "
            "mean-state meteo-only seed-42 training."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    gate11b_path = Path(
        args.gate11b_report
    ).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"Run directory not found: {run_dir}")
    require(
        gate11b_path.is_file(),
        f"Gate-11B report not found: {gate11b_path}",
    )

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    latest_validation_path = (
        run_dir / "logs" / "latest_val_monthly_report.json"
    )
    checkpoint_path = (
        run_dir / "checkpoints" / "best_model.pt"
    )

    for path in [
        manifest_path,
        history_path,
        latest_validation_path,
        checkpoint_path,
    ]:
        require(path.is_file(), f"Missing artifact: {path}")
        require(path.stat().st_size > 0, f"Empty artifact: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_report = (
        output_dir
        / "meanstate_meteo_only_seed42_training_audit_gate13b.json"
    )
    txt_report = (
        output_dir
        / "meanstate_meteo_only_seed42_training_audit_gate13b.txt"
    )
    require(
        not json_report.exists()
        and not txt_report.exists(),
        "Gate-13B report already exists. "
        "Use a new output directory.",
    )

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    latest_validation = load_json(latest_validation_path)
    checkpoint = load_checkpoint(checkpoint_path)
    gate11b = load_json(gate11b_path)

    # ---------------------------------------------------------------
    # Manifest and provenance
    # ---------------------------------------------------------------
    require(
        manifest.get("artifact_name")
        == "s2_meanstate_clean_training",
        "Unexpected manifest artifact_name.",
    )
    require(
        int(manifest.get("artifact_version", -1)) == 1,
        "Unexpected manifest artifact_version.",
    )
    require(
        manifest.get("run_mode") == "formal",
        "Run is not marked formal.",
    )
    require(
        manifest.get("run_name")
        == "mean_meteo_clim_only_seed42",
        "Unexpected run_name.",
    )
    require(
        normalized_path(manifest.get("run_dir", ""))
        == normalized_path(run_dir),
        "Manifest run_dir mismatch.",
    )

    experiment = manifest.get("experiment", {})
    require(
        experiment.get("name") == EXPECTED_EXPERIMENT,
        "Unexpected experiment.",
    )
    require(
        experiment.get("uses_emission") is False,
        "Meteo-only run unexpectedly uses emission.",
    )
    require(
        experiment.get("uses_meteo") is True,
        "Meteo-only run does not use meteorology.",
    )
    require(
        experiment.get("shuffle_emission") is False,
        "Meteo-only run unexpectedly shuffles emission.",
    )
    require(
        int(experiment.get("emission_channels", -1)) == 0,
        "Unexpected emission channel count.",
    )
    require(
        int(experiment.get("meteo_channels", -1)) == 5,
        "Unexpected meteorology channel count.",
    )

    settings = manifest.get("settings", {})
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
        "seed": EXPECTED_SEED,
        "require_cuda": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in expected_settings.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            require(
                actual is not None
                and math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                f"Setting mismatch: {key}; "
                f"expected={expected}, actual={actual}",
            )
        else:
            require(
                actual == expected,
                f"Setting mismatch: {key}; "
                f"expected={expected}, actual={actual}",
            )

    train_months = [
        int(entry["month"])
        for entry in manifest.get("train_entries", [])
    ]
    validation_months = [
        int(entry["month"])
        for entry in manifest.get("val_entries", [])
    ]
    external_months = [
        int(entry["month"])
        for entry in manifest.get(
            "external_2023_entries",
            [],
        )
    ]
    dropped_2022 = manifest.get(
        "dropped_2022_entries",
        [],
    )
    dropped_2023 = manifest.get(
        "dropped_2023_entries",
        [],
    )

    require(
        train_months == EXPECTED_TRAIN_MONTHS,
        f"Unexpected training months: {train_months}",
    )
    require(
        validation_months == EXPECTED_VALIDATION_MONTHS,
        f"Unexpected validation months: {validation_months}",
    )
    require(
        external_months == EXPECTED_EXTERNAL_MONTHS,
        f"Unexpected 2023 months: {external_months}",
    )
    require(
        len(dropped_2022) == 1
        and int(dropped_2022[0]["month"]) == 7,
        f"Only 2022-07 may be dropped: {dropped_2022}",
    )
    require(
        dropped_2023 == [],
        f"No 2023 month may be dropped: {dropped_2023}",
    )

    require(
        gate11b.get("verification", {}).get("overall_pass")
        is True,
        "Gate-11B overall_pass is not true.",
    )
    provenance = manifest.get("provenance", {})
    require(
        provenance.get("gate11b_report_sha256")
        == sha256_file(gate11b_path),
        "Gate-11B report hash mismatch.",
    )
    require(
        normalized_path(
            provenance.get("gate11b_report", "")
        )
        == normalized_path(gate11b_path),
        "Gate-11B report path mismatch.",
    )

    monthly_manifest_path = Path(
        provenance.get("monthly_manifest", "")
    )
    require(
        monthly_manifest_path.is_file(),
        "Recorded monthly manifest is missing.",
    )
    require(
        provenance.get("monthly_manifest_sha256")
        == sha256_file(monthly_manifest_path),
        "Monthly manifest changed after training.",
    )

    training_script_path = Path(
        provenance.get("training_script", "")
    )
    require(
        training_script_path.is_file(),
        "Recorded training script is missing.",
    )
    require(
        provenance.get("training_script_sha256")
        == sha256_file(training_script_path),
        "Training script changed after the formal run.",
    )

    # ---------------------------------------------------------------
    # History and early stopping
    # ---------------------------------------------------------------
    expected_keys = [
        "train_loss",
        "train_patch_r2",
        "val_monthly_log_r2",
        "val_monthly_log_rmse",
        "lr",
    ]
    lengths: dict[str, int] = {}
    for key in expected_keys:
        values = history.get(key)
        require(
            isinstance(values, list),
            f"History field missing or not list: {key}",
        )
        lengths[key] = len(values)

    unique_lengths = set(lengths.values())
    require(
        len(unique_lengths) == 1,
        f"History lengths differ: {lengths}",
    )
    completed_epochs = next(iter(unique_lengths))
    require(
        1 <= completed_epochs <= int(settings["epochs"]),
        f"Invalid completed epoch count: {completed_epochs}",
    )

    nonfinite_history = find_nonfinite_json_values(history)
    require(
        not nonfinite_history,
        f"History contains non-finite values: "
        f"{nonfinite_history}",
    )

    train_loss = [
        finite_float(value, f"train_loss[{index}]")
        for index, value in enumerate(history["train_loss"])
    ]
    train_patch_r2 = [
        finite_float(
            value,
            f"train_patch_r2[{index}]",
        )
        for index, value in enumerate(
            history["train_patch_r2"]
        )
    ]
    val_r2 = [
        finite_float(
            value,
            f"val_r2[{index}]",
        )
        for index, value in enumerate(
            history["val_monthly_log_r2"]
        )
    ]
    val_rmse = [
        finite_float(
            value,
            f"val_rmse[{index}]",
        )
        for index, value in enumerate(
            history["val_monthly_log_rmse"]
        )
    ]
    learning_rates = [
        finite_float(value, f"lr[{index}]")
        for index, value in enumerate(history["lr"])
    ]

    best_index = max(
        range(completed_epochs),
        key=lambda index: val_r2[index],
    )
    best_epoch_one_based = best_index + 1
    best_history_r2 = val_r2[best_index]
    best_history_rmse = val_rmse[best_index]

    patience = int(settings["patience"])
    stopped_early = (
        completed_epochs < int(settings["epochs"])
    )
    epochs_without_improvement_at_end = (
        completed_epochs - best_epoch_one_based
    )
    require(
        stopped_early,
        "The run did not stop early as recorded by the terminal log.",
    )
    require(
        epochs_without_improvement_at_end == patience,
        "Early stopping is inconsistent with the recorded "
        f"patience: completed={completed_epochs}, "
        f"best_epoch={best_epoch_one_based}, "
        f"patience={patience}.",
    )

    require(
        best_epoch_one_based == 49,
        f"Expected best epoch 49, found {best_epoch_one_based}.",
    )
    require(
        completed_epochs == 74,
        f"Expected 74 completed epochs, found {completed_epochs}.",
    )
    require(
        math.isclose(
            best_history_r2,
            0.54608762,
            rel_tol=0.0,
            abs_tol=5e-8,
        ),
        "Best validation R2 differs from terminal result.",
    )

    # The restart at epoch 72 is expected from T0=24, Tmult=2:
    # cycle boundaries occur after epochs 24 and 72.
    require(
        learning_rates[70] < 5e-6,
        "Epoch-71 learning rate is not near the cycle minimum.",
    )
    require(
        math.isclose(
            learning_rates[71],
            3e-4,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Epoch-72 learning rate does not show the expected restart.",
    )

    # ---------------------------------------------------------------
    # Latest validation report is the final epoch report
    # ---------------------------------------------------------------
    nonfinite_latest = find_nonfinite_json_values(
        latest_validation
    )
    require(
        not nonfinite_latest,
        "Latest validation report contains non-finite values: "
        f"{nonfinite_latest}",
    )
    latest_global = latest_validation.get("global", {})
    latest_r2 = finite_float(
        latest_global.get("log_r2"),
        "latest_validation.global.log_r2",
    )
    latest_rmse = finite_float(
        latest_global.get("log_rmse"),
        "latest_validation.global.log_rmse",
    )
    require(
        math.isclose(
            latest_r2,
            val_r2[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Latest validation R2 differs from final history epoch.",
    )
    require(
        math.isclose(
            latest_rmse,
            val_rmse[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Latest validation RMSE differs from final history epoch.",
    )
    per_group = latest_validation.get("per_group", [])
    require(
        isinstance(per_group, list)
        and len(per_group) == 2,
        "Latest validation report must contain two groups.",
    )
    require(
        sorted(int(item["month"]) for item in per_group)
        == EXPECTED_VALIDATION_MONTHS,
        "Latest validation report months are not 11 and 12.",
    )

    # ---------------------------------------------------------------
    # Checkpoint and architecture
    # ---------------------------------------------------------------
    require(
        checkpoint.get("experiment") == EXPECTED_EXPERIMENT,
        "Checkpoint experiment mismatch.",
    )
    require(
        checkpoint.get("run_mode") == "formal",
        "Checkpoint run_mode mismatch.",
    )
    require(
        checkpoint.get("run_name")
        == manifest.get("run_name"),
        "Checkpoint run_name mismatch.",
    )
    require(
        int(checkpoint.get("base_features", -1)) == 32,
        "Checkpoint base_features mismatch.",
    )
    require(
        normalized_path(checkpoint.get("cache_dir", ""))
        == normalized_path(manifest.get("cache_dir", "")),
        "Checkpoint cache_dir mismatch.",
    )
    require(
        checkpoint.get("gate11b_report_sha256")
        == provenance.get("gate11b_report_sha256"),
        "Checkpoint Gate-11B hash mismatch.",
    )
    require(
        checkpoint.get("training_script_sha256")
        == provenance.get("training_script_sha256"),
        "Checkpoint training-script hash mismatch.",
    )

    checkpoint_epoch_zero_based = int(
        checkpoint.get("epoch", -1)
    )
    checkpoint_best_r2 = finite_float(
        checkpoint.get("best_val_log_r2"),
        "checkpoint.best_val_log_r2",
    )
    require(
        checkpoint_epoch_zero_based + 1
        == best_epoch_one_based,
        "Checkpoint epoch differs from history best epoch.",
    )
    require(
        math.isclose(
            checkpoint_best_r2,
            best_history_r2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Checkpoint best R2 differs from history maximum.",
    )

    state_dict = checkpoint.get("model_state_dict")
    require(
        isinstance(state_dict, dict)
        and len(state_dict) > 0,
        "Checkpoint state_dict is missing or empty.",
    )

    nonfinite_state_tensors: list[str] = []
    state_tensor_count = 0
    state_element_count = 0
    for name, tensor in state_dict.items():
        require(
            isinstance(tensor, torch.Tensor),
            f"Non-tensor state_dict value: {name}",
        )
        state_tensor_count += 1
        state_element_count += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite_state_tensors.append(name)

    require(
        not nonfinite_state_tensors,
        "Checkpoint contains non-finite tensors: "
        f"{nonfinite_state_tensors}",
    )

    training_module = load_training_module(
        training_script_path
    )
    require(
        hasattr(training_module, "EXPERIMENT_SPECS"),
        "Training module lacks EXPERIMENT_SPECS.",
    )
    require(
        hasattr(training_module, "MeanStateUNet"),
        "Training module lacks MeanStateUNet.",
    )
    model_spec = training_module.EXPERIMENT_SPECS[
        EXPECTED_EXPERIMENT
    ]
    model = training_module.MeanStateUNet(
        spec=model_spec,
        base_features=32,
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    require(
        trainable_parameters
        == EXPECTED_TRAINABLE_PARAMETERS,
        "Unexpected model parameter count: "
        f"{trainable_parameters}",
    )
    model.load_state_dict(state_dict, strict=True)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    report = {
        "audit_name": (
            "s2_meanstate_clean_meteo_only_seed42_gate13b"
        ),
        "read_only": True,
        "inputs": {
            "run_dir": str(run_dir),
            "gate11b_report": str(gate11b_path),
        },
        "artifact_hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "history_sha256": sha256_file(history_path),
            "latest_validation_sha256": sha256_file(
                latest_validation_path
            ),
            "checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
            "training_script_sha256": sha256_file(
                training_script_path
            ),
        },
        "verified_configuration": {
            "experiment": EXPECTED_EXPERIMENT,
            "seed": EXPECTED_SEED,
            "train_months": train_months,
            "validation_months": validation_months,
            "external_2023_months": external_months,
            "base_features": 32,
            "trainable_parameters": trainable_parameters,
        },
        "training_summary": {
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch_one_based,
            "best_val_log_r2": best_history_r2,
            "best_val_log_rmse_at_same_epoch": (
                best_history_rmse
            ),
            "final_epoch_val_log_r2": val_r2[-1],
            "final_epoch_val_log_rmse": val_rmse[-1],
            "epochs_without_improvement_at_end": (
                epochs_without_improvement_at_end
            ),
            "early_stopping_patience": patience,
            "early_stopping_consistent": True,
            "learning_rate_restart_at_epoch_72": True,
            "minimum_train_loss": min(train_loss),
            "maximum_train_patch_r2": max(train_patch_r2),
        },
        "checkpoint": {
            "selected_epoch": checkpoint_epoch_zero_based + 1,
            "best_val_log_r2": checkpoint_best_r2,
            "state_tensor_count": state_tensor_count,
            "state_element_count_including_buffers": (
                state_element_count
            ),
            "all_state_tensors_finite": True,
            "strict_model_load_pass": True,
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
            "This audit validates training integrity only. "
            "The best 2022 held-out validation metric is a "
            "model-selection result from one random seed, "
            "not an independent test and not yet a final "
            "paper result. Formal 2023 evaluation and "
            "multi-seed replication remain required."
        ),
    }

    json_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    txt_report.write_text(
        "\n".join(
            [
                "Clean mean-state meteo-only seed-42 training audit - Gate 13B",
                "Manifest and provenance pass: True",
                "Formal settings and split pass: True",
                "History finite and consistent: True",
                f"Completed epochs: {completed_epochs}",
                f"Best epoch: {best_epoch_one_based}",
                f"Best validation log R2: {best_history_r2}",
                f"Best validation log RMSE: {best_history_rmse}",
                f"Final validation log R2: {val_r2[-1]}",
                f"Final validation log RMSE: {val_rmse[-1]}",
                "Early stopping consistent: True",
                "Learning-rate restart at epoch 72: True",
                "Checkpoint matches history best: True",
                "Checkpoint finite and strict model load pass: True",
                "Scientific status: single-seed model-selection validation only",
                "Overall pass: True",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "GATE13B_CLEAN_MEANSTATE_METEO_SEED42_TRAINING_AUDIT_PASS"
    )
    print(f"JSON_REPORT={json_report}")
    print(f"TXT_REPORT={txt_report}")
    print("MANIFEST_AND_PROVENANCE_PASS=True")
    print("FORMAL_SETTINGS_AND_SPLIT_PASS=True")
    print("HISTORY_FINITE_AND_CONSISTENT=True")
    print(f"COMPLETED_EPOCHS={completed_epochs}")
    print(f"BEST_EPOCH={best_epoch_one_based}")
    print(
        f"BEST_VAL_LOG_R2={best_history_r2:.8f}"
    )
    print(
        f"BEST_VAL_LOG_RMSE={best_history_rmse:.8f}"
    )
    print(
        "EARLY_STOPPING_CONSISTENT=True"
    )
    print(
        "LEARNING_RATE_RESTART_AT_EPOCH72_EXPECTED=True"
    )
    print("CHECKPOINT_MATCHES_HISTORY_BEST=True")
    print(
        "CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True"
    )
    print(
        "SINGLE_SEED_RESULT_READY_FOR_PAPER=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
