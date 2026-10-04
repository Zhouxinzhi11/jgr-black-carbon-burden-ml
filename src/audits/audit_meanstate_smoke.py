# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
48_audit_s2_meanstate_clean_smoke_gate12b.py

Gate 12B: read-only audit of the Gate-12A clean mean-state training smoke run.

This audit verifies:
- run_manifest provenance, clean cache path, approved split, and exact smoke settings;
- history length and finite numerical values;
- latest validation report structure and finite global metrics;
- checkpoint metadata, finite state_dict tensors, and consistency with manifest/history;
- no scientific interpretation is attached to the two-epoch smoke metrics.

No artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


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


def require_finite_number(value: Any, name: str) -> float:
    if value is None:
        raise RuntimeError(f"{name} is None")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is non-finite: {value}")
    return number


def collect_finite_values(obj: Any, prefix: str = "root") -> list[str]:
    errors: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            errors.extend(
                collect_finite_values(
                    value,
                    f"{prefix}.{key}",
                )
            )
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            errors.extend(
                collect_finite_values(
                    value,
                    f"{prefix}[{index}]",
                )
            )
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            errors.append(f"{prefix}={obj}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 12B read-only audit of the clean mean-state "
            "two-epoch smoke run."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-experiment",
        default="mean_meteo_clim_only",
    )
    parser.add_argument("--expected-seed", type=int, default=42)
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
    validation_path = (
        run_dir / "logs" / "latest_val_monthly_report.json"
    )
    checkpoint_path = (
        run_dir / "checkpoints" / "best_model.pt"
    )

    required_files = [
        manifest_path,
        history_path,
        validation_path,
        checkpoint_path,
    ]
    for path in required_files:
        require(path.is_file(), f"Missing artifact: {path}")
        require(
            path.stat().st_size > 0,
            f"Empty artifact: {path}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_report = (
        output_dir
        / "meanstate_clean_smoke_audit_gate12b.json"
    )
    txt_report = (
        output_dir
        / "meanstate_clean_smoke_audit_gate12b.txt"
    )
    require(
        not json_report.exists()
        and not txt_report.exists(),
        "Gate-12B report already exists. "
        "Use a new output directory.",
    )

    manifest = load_json(manifest_path)
    history = load_json(history_path)
    validation = load_json(validation_path)
    gate11b = load_json(gate11b_path)
    checkpoint = load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # Manifest and provenance
    # ------------------------------------------------------------------
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
        manifest.get("run_mode") == "smoke",
        "Run is not marked as smoke.",
    )
    require(
        manifest.get("run_name")
        == "mean_meteo_clim_only_seed42_smoke",
        "Unexpected smoke run_name.",
    )
    require(
        normalized_path(manifest.get("run_dir", ""))
        == normalized_path(run_dir),
        "Manifest run_dir mismatch.",
    )

    experiment = manifest.get("experiment", {})
    require(
        experiment.get("name") == args.expected_experiment,
        "Unexpected experiment in manifest.",
    )
    require(
        experiment.get("uses_emission") is False,
        "Smoke experiment unexpectedly uses emission.",
    )
    require(
        experiment.get("uses_meteo") is True,
        "Smoke experiment does not use meteorology.",
    )
    require(
        experiment.get("shuffle_emission") is False,
        "Smoke experiment unexpectedly shuffles emission.",
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
        "samples_per_epoch": 32,
        "epochs": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "patience": 2,
        "base_features": 32,
        "num_workers": 0,
        "seed": args.expected_seed,
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
                f"Smoke setting mismatch: {key}, "
                f"expected={expected}, actual={actual}",
            )
        else:
            require(
                actual == expected,
                f"Smoke setting mismatch: {key}, "
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
        train_months == [2, 3, 5, 6, 8, 9],
        f"Unexpected train months: {train_months}",
    )
    require(
        validation_months == [11, 12],
        f"Unexpected validation months: {validation_months}",
    )
    require(
        external_months == list(range(1, 13)),
        f"Unexpected 2023 months: {external_months}",
    )
    require(
        len(dropped_2022) == 1
        and int(dropped_2022[0]["month"]) == 7,
        f"Only 2022-07 may be dropped: {dropped_2022}",
    )
    require(
        dropped_2023 == [],
        f"No 2023 group may be dropped: {dropped_2023}",
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
        "Manifest Gate-11B report hash mismatch.",
    )
    require(
        normalized_path(
            provenance.get("gate11b_report", "")
        )
        == normalized_path(gate11b_path),
        "Manifest Gate-11B report path mismatch.",
    )

    monthly_manifest_path = Path(
        provenance.get("monthly_manifest", "")
    )
    require(
        monthly_manifest_path.is_file(),
        "Monthly manifest recorded in provenance is missing.",
    )
    require(
        provenance.get("monthly_manifest_sha256")
        == sha256_file(monthly_manifest_path),
        "Monthly manifest changed after training smoke.",
    )

    training_script_path = Path(
        provenance.get("training_script", "")
    )
    require(
        training_script_path.is_file(),
        "Training script recorded in provenance is missing.",
    )
    require(
        provenance.get("training_script_sha256")
        == sha256_file(training_script_path),
        "Training script changed after the smoke run.",
    )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    expected_history_keys = [
        "train_loss",
        "train_patch_r2",
        "val_monthly_log_r2",
        "val_monthly_log_rmse",
        "lr",
    ]
    for key in expected_history_keys:
        require(
            isinstance(history.get(key), list),
            f"History key is missing or not a list: {key}",
        )
        require(
            len(history[key]) == 2,
            f"History {key} length is not 2.",
        )

    history_nonfinite = collect_finite_values(history)
    require(
        not history_nonfinite,
        f"History contains non-finite values: "
        f"{history_nonfinite}",
    )

    train_loss = [
        require_finite_number(value, f"train_loss[{i}]")
        for i, value in enumerate(history["train_loss"])
    ]
    val_r2 = [
        require_finite_number(
            value,
            f"val_monthly_log_r2[{i}]",
        )
        for i, value in enumerate(
            history["val_monthly_log_r2"]
        )
    ]
    val_rmse = [
        require_finite_number(
            value,
            f"val_monthly_log_rmse[{i}]",
        )
        for i, value in enumerate(
            history["val_monthly_log_rmse"]
        )
    ]

    require(
        train_loss[1] < train_loss[0],
        "Smoke training loss did not decrease. "
        "This is not always a formal requirement, but it "
        "is expected for this recorded run.",
    )
    require(
        val_r2[1] > val_r2[0],
        "Smoke validation R2 did not improve in the "
        "recorded two-epoch run.",
    )
    require(
        val_rmse[1] < val_rmse[0],
        "Smoke validation RMSE did not improve in the "
        "recorded two-epoch run.",
    )

    # ------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------
    validation_nonfinite = collect_finite_values(validation)
    require(
        not validation_nonfinite,
        "Validation report contains non-finite values: "
        f"{validation_nonfinite}",
    )

    global_metrics = validation.get("global", {})
    latest_r2 = require_finite_number(
        global_metrics.get("log_r2"),
        "validation.global.log_r2",
    )
    latest_rmse = require_finite_number(
        global_metrics.get("log_rmse"),
        "validation.global.log_rmse",
    )
    require(
        math.isclose(
            latest_r2,
            val_r2[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Latest validation R2 differs from history.",
    )
    require(
        math.isclose(
            latest_rmse,
            val_rmse[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Latest validation RMSE differs from history.",
    )

    per_group = validation.get("per_group", [])
    require(
        isinstance(per_group, list)
        and len(per_group) == 2,
        "Validation report must contain two monthly groups.",
    )
    report_months = sorted(
        int(item["month"])
        for item in per_group
    )
    require(
        report_months == [11, 12],
        f"Unexpected validation report months: "
        f"{report_months}",
    )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    require(
        checkpoint.get("experiment")
        == args.expected_experiment,
        "Checkpoint experiment mismatch.",
    )
    require(
        checkpoint.get("run_mode") == "smoke",
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
        == normalized_path(
            manifest.get("cache_dir", "")
        ),
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

    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_best_r2 = require_finite_number(
        checkpoint.get("best_val_log_r2"),
        "checkpoint.best_val_log_r2",
    )
    require(
        checkpoint_epoch == 1,
        "Expected the second epoch to be the selected "
        "best smoke checkpoint.",
    )
    require(
        math.isclose(
            checkpoint_best_r2,
            max(val_r2),
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
    parameter_element_count = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Non-tensor state_dict value: {name}"
            )
        state_tensor_count += 1
        parameter_element_count += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite_state_tensors.append(name)

    require(
        not nonfinite_state_tensors,
        "Checkpoint contains non-finite tensors: "
        f"{nonfinite_state_tensors}",
    )

    overall_pass = True
    report = {
        "audit_name": (
            "s2_meanstate_clean_training_smoke_gate12b"
        ),
        "read_only": True,
        "inputs": {
            "run_dir": str(run_dir),
            "gate11b_report": str(gate11b_path),
        },
        "artifact_hashes": {
            "run_manifest_sha256": sha256_file(
                manifest_path
            ),
            "training_history_sha256": sha256_file(
                history_path
            ),
            "validation_report_sha256": sha256_file(
                validation_path
            ),
            "checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
        },
        "verified_split": {
            "train_months": train_months,
            "validation_months": validation_months,
            "external_2023_months": external_months,
            "dropped_2022_months": [
                int(entry["month"])
                for entry in dropped_2022
            ],
        },
        "history": {
            "train_loss": train_loss,
            "val_log_r2": val_r2,
            "val_log_rmse": val_rmse,
            "loss_decreased": train_loss[1] < train_loss[0],
            "validation_r2_improved": val_r2[1] > val_r2[0],
            "validation_rmse_improved": (
                val_rmse[1] < val_rmse[0]
            ),
        },
        "checkpoint": {
            "selected_epoch_one_based": checkpoint_epoch + 1,
            "best_val_log_r2": checkpoint_best_r2,
            "state_tensor_count": state_tensor_count,
            "state_element_count_including_buffers": (
                parameter_element_count
            ),
            "all_state_tensors_finite": True,
        },
        "verification": {
            "manifest_and_provenance_pass": True,
            "split_and_settings_pass": True,
            "history_finite_and_consistent": True,
            "validation_report_finite_and_consistent": True,
            "checkpoint_finite_and_consistent": True,
            "overall_pass": overall_pass,
        },
        "scientific_boundary": (
            "The two-epoch smoke metrics are engineering "
            "diagnostics only and must not be used in the "
            "manuscript or interpreted as model skill."
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
                "Clean mean-state training smoke audit - Gate 12B",
                "Manifest and provenance pass: True",
                "Split and settings pass: True",
                "History finite and consistent: True",
                "Validation report finite and consistent: True",
                "Checkpoint finite and consistent: True",
                f"Selected best epoch: {checkpoint_epoch + 1}",
                f"Best smoke validation log R2: {checkpoint_best_r2}",
                "Scientific interpretation allowed: False",
                "Overall pass: True",
            ]
        ),
        encoding="utf-8",
    )

    print("GATE12B_CLEAN_MEANSTATE_SMOKE_AUDIT_PASS")
    print(f"JSON_REPORT={json_report}")
    print(f"TXT_REPORT={txt_report}")
    print("MANIFEST_AND_PROVENANCE_PASS=True")
    print("SPLIT_AND_SETTINGS_PASS=True")
    print("HISTORY_FINITE_AND_CONSISTENT=True")
    print(
        "VALIDATION_REPORT_FINITE_AND_CONSISTENT=True"
    )
    print("CHECKPOINT_FINITE_AND_CONSISTENT=True")
    print(f"BEST_SMOKE_EPOCH={checkpoint_epoch + 1}")
    print(
        f"BEST_SMOKE_VAL_LOG_R2={checkpoint_best_r2:.8f}"
    )
    print("SMOKE_METRICS_SCIENTIFICALLY_INTERPRETABLE=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
