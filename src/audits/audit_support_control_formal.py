# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
61_audit_s2_meanstate_support_preserving_formal_run.py

Read-only audit for one formal support-preserving mean-state run.
Reusable for seeds 42, 43, and 44. No training artifact is modified.
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

EXPECTED_EXPERIMENT = "mean_emis_meteo_clim_support_preserving"
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_path(path: str | Path) -> str:
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
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{name}: {value}")
    return x


def month_list(entries: list[dict[str, Any]]) -> list[int]:
    return sorted(int(e["month"]) for e in entries)


def load_training_module(path: Path):
    module_name = "support_preserving_training_formal_run_audit"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-seed", required=True, type=int, choices=[42, 43, 44])
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--gatep2a-report", required=True)
    parser.add_argument("--training-script", default="./57_train_s2_meanstate_support_preserving.py")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    expected_seed = int(args.expected_seed)
    expected_run_name = f"{EXPECTED_EXPERIMENT}_seed{expected_seed}"

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

    require(manifest.get("artifact_name") == "s2_meanstate_support_preserving_training", "MANIFEST_ARTIFACT_NAME_MISMATCH")
    require(int(manifest.get("artifact_version", -1)) == 1, "MANIFEST_VERSION_MISMATCH")
    require(manifest.get("run_mode") == "formal", "RUN_MODE_NOT_FORMAL")
    require(manifest.get("run_name") == expected_run_name, "RUN_NAME_MISMATCH")
    require(norm_path(manifest.get("run_dir", "")) == norm_path(run_dir), "RUN_DIR_MISMATCH")

    experiment = manifest.get("experiment", {})
    require(experiment.get("name") == EXPECTED_EXPERIMENT, "EXPERIMENT_NAME_MISMATCH")
    require(experiment.get("uses_emission") is True, "USES_EMISSION_MISMATCH")
    require(experiment.get("uses_meteo") is True, "USES_METEO_MISMATCH")
    require(experiment.get("support_preserving_permutation") is True, "SUPPORT_PRESERVING_FLAG_MISMATCH")
    require(int(experiment.get("emission_channels", -1)) == 8, "EMISSION_CHANNELS_MISMATCH")
    require(int(experiment.get("meteo_channels", -1)) == 5, "METEO_CHANNELS_MISMATCH")

    settings = manifest.get("settings", {})
    exact_settings = {
        "val_months": [11, 12], "blacklist_months": [7], "min_train_days": 5,
        "crop_size": 128, "batch_size": 8, "samples_per_epoch": 512,
        "epochs": 160, "patience": 25, "base_features": 32, "num_workers": 0,
        "seed": expected_seed, "require_cuda": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "support_preserving_permutation": True,
    }
    for key, expected in exact_settings.items():
        require(settings.get(key) == expected, f"SETTING_MISMATCH_{key}: {settings.get(key)}")

    float_settings = {"min_valid_fraction": 1e-6, "learning_rate": 3e-4, "weight_decay": 1e-2, "support_epsilon": 0.0}
    for key, expected in float_settings.items():
        actual = settings.get(key)
        require(actual is not None and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12),
                f"SETTING_MISMATCH_{key}: actual={actual}, expected={expected}")

    require(month_list(list(manifest.get("train_entries", []))) == EXPECTED_TRAIN_MONTHS, "TRAIN_MONTHS_MISMATCH")
    require(month_list(list(manifest.get("val_entries", []))) == EXPECTED_VAL_MONTHS, "VAL_MONTHS_MISMATCH")
    require(month_list(list(manifest.get("external_2023_entries", []))) == EXPECTED_EXTERNAL_MONTHS, "EXTERNAL_MONTHS_MISMATCH")

    dropped_2022 = list(manifest.get("dropped_2022_entries", []))
    dropped_2023 = list(manifest.get("dropped_2023_entries", []))
    require(len(dropped_2022) == 1 and int(dropped_2022[0]["month"]) == 7, "DROPPED_2022_CONTRACT_MISMATCH")
    require(not dropped_2023, "DROPPED_2023_MUST_BE_EMPTY")

    gate11b_hash = sha256_file(gate11b)
    gatep2a_hash = sha256_file(gatep2a)
    training_hash = sha256_file(training_script)
    provenance = manifest.get("provenance", {})
    require(provenance.get("gate11b_report_sha256") == gate11b_hash, "GATE11B_HASH_MISMATCH")
    require(provenance.get("gatep2a_report_sha256") == gatep2a_hash, "GATEP2A_HASH_MISMATCH")
    require(provenance.get("training_script_sha256") == training_hash, "TRAINING_HASH_MISMATCH")

    history_keys = ["train_loss", "train_patch_r2", "val_monthly_log_r2", "val_monthly_log_rmse", "lr"]
    lengths = {}
    for key in history_keys:
        values = history.get(key)
        require(isinstance(values, list), f"HISTORY_NOT_LIST_{key}")
        lengths[key] = len(values)
        for i, value in enumerate(values, start=1):
            finite_float(value, f"{key}_epoch{i}")
    require(len(set(lengths.values())) == 1, f"HISTORY_LENGTHS_DIFFER: {lengths}")
    completed_epochs = next(iter(lengths.values()))
    require(1 <= completed_epochs <= 160, "COMPLETED_EPOCHS_OUT_OF_RANGE")

    val_r2 = [float(v) for v in history["val_monthly_log_r2"]]
    best_index = max(range(completed_epochs), key=lambda i: val_r2[i])
    best_epoch = best_index + 1
    best_val_r2 = val_r2[best_index]
    epochs_after_best = completed_epochs - best_epoch

    require(int(checkpoint.get("epoch", -1)) == best_index, "CHECKPOINT_EPOCH_MISMATCH")
    require(math.isclose(finite_float(checkpoint.get("best_val_log_r2"), "checkpoint_best_val_log_r2"), best_val_r2,
                         rel_tol=0.0, abs_tol=1e-8), "CHECKPOINT_BEST_METRIC_MISMATCH")
    if completed_epochs < 160:
        require(epochs_after_best == 25, "EARLY_STOPPING_INCONSISTENT")

    require(checkpoint.get("experiment") == EXPECTED_EXPERIMENT, "CKPT_EXPERIMENT_MISMATCH")
    require(checkpoint.get("run_name") == expected_run_name, "CKPT_RUN_NAME_MISMATCH")
    require(checkpoint.get("run_mode") == "formal", "CKPT_RUN_MODE_MISMATCH")
    require(checkpoint.get("support_preserving_permutation") is True, "CKPT_SUPPORT_FLAG_MISMATCH")
    if "seed" in checkpoint:
        require(int(checkpoint["seed"]) == expected_seed, "CKPT_SEED_MISMATCH")
    require(checkpoint.get("gate11b_report_sha256") == gate11b_hash, "CKPT_GATE11B_HASH_MISMATCH")
    require(checkpoint.get("gatep2a_report_sha256") == gatep2a_hash, "CKPT_GATEP2A_HASH_MISMATCH")
    require(checkpoint.get("training_script_sha256") == training_hash, "CKPT_TRAINING_HASH_MISMATCH")

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and state, "MODEL_STATE_DICT_MISSING")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor), f"STATE_NOT_TENSOR: {name}")
        require(bool(torch.isfinite(tensor).all().item()), f"NONFINITE_STATE_TENSOR: {name}")

    module = load_training_module(training_script)
    spec_obj = module.EXPERIMENT_SPECS[EXPECTED_EXPERIMENT]
    model = module.MeanStateUNet(spec=spec_obj, base_features=32)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(n_params == 477457, f"PARAMETER_COUNT_MISMATCH: {n_params}")

    require(int(latest_val.get("n_groups", -1)) == 2, "LATEST_VAL_GROUP_COUNT_MISMATCH")
    require(int(latest_val.get("n_total_valid_pixels", 0)) > 0, "LATEST_VAL_NO_PIXELS")
    require(month_list(list(latest_val.get("per_group", []))) == EXPECTED_VAL_MONTHS, "LATEST_VAL_MONTHS_MISMATCH")
    global_metrics = latest_val.get("global", {})
    finite_float(global_metrics.get("log_r2"), "latest_val_log_r2")
    finite_float(global_metrics.get("log_rmse"), "latest_val_log_rmse")
    finite_float(global_metrics.get("log_bias"), "latest_val_log_bias")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / f"support_preserving_formal_seed{expected_seed}_audit.json"
    txt_path = output_dir / f"support_preserving_formal_seed{expected_seed}_audit.txt"

    report = {
        "artifact_name": "s2_meanstate_support_preserving_formal_run_audit",
        "artifact_version": 1,
        "seed": expected_seed,
        "run_dir": str(run_dir),
        "inputs": {
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "history": str(history_path), "history_sha256": sha256_file(history_path),
            "latest_validation": str(latest_val_path), "latest_validation_sha256": sha256_file(latest_val_path),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "gate11b_report": str(gate11b), "gate11b_report_sha256": gate11b_hash,
            "gatep2a_report": str(gatep2a), "gatep2a_report_sha256": gatep2a_hash,
            "training_script": str(training_script), "training_script_sha256": training_hash,
        },
        "summary": {
            "completed_epochs": completed_epochs, "best_epoch": best_epoch,
            "epochs_after_best": epochs_after_best, "best_val_log_r2": best_val_r2,
            "trainable_parameters": n_params,
        },
        "verification": {
            "formal_contract_exact": True, "split_contract_exact": True,
            "provenance_exact": True, "history_complete_and_finite": True,
            "early_stopping_consistent": True, "checkpoint_matches_history_best": True,
            "checkpoint_finite_and_architecture_compatible": True,
            "latest_validation_complete_and_finite": True, "overall_pass": True,
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    lines = [
        f"GATE_SUPPORT_PRESERVING_FORMAL_SEED{expected_seed}_AUDIT_PASS",
        f"RUN_DIR={run_dir}", "RUN_MODE=formal", f"SEED={expected_seed}",
        f"COMPLETED_EPOCHS={completed_epochs}", f"BEST_EPOCH={best_epoch}",
        f"EPOCHS_AFTER_BEST={epochs_after_best}", f"BEST_VAL_LOG_R2={best_val_r2:.8f}",
        f"TRAINABLE_PARAMETERS={n_params}", "FORMAL_CONTRACT_EXACT=True",
        "SPLIT_CONTRACT_EXACT=True", "PROVENANCE_EXACT=True",
        "HISTORY_COMPLETE_AND_FINITE=True", "EARLY_STOPPING_CONSISTENT=True",
        "CHECKPOINT_MATCHES_HISTORY_BEST=True",
        "CHECKPOINT_FINITE_AND_ARCHITECTURE_COMPATIBLE=True",
        "SCIENTIFIC_EVALUATION_NOT_YET_RUN=True",
        f"OUTPUT_JSON={report_path}", f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
