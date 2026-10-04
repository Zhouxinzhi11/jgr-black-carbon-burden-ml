# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
85_seal_meanstate_spatial_holdout_preopening_gateS9E.py

Gate S9E: final pre-opening closure for the 3 x 3 spatial-holdout formal run set.

Purpose
-------
Before East-Asia core performance is opened for the first time, independently
re-evaluate the BEST checkpoint of every formal run on the already-approved
NON-CORE checkpoint-selection domain and seal the full 3 models x 3 seeds
artifact/hash matrix.

This script MUST NOT compute East-Asia core performance.

Expected formal set
-------------------
Experiments:
- mean_meteo_clim_only
- mean_binary_support_meteo_clim
- mean_emis_meteo_clim

Seeds:
- 42, 43, 44

Reproduction target
-------------------
For each run:
1) verify the Gate-S9D audit is PASS;
2) verify manifest/history/checkpoint files are unchanged since that audit;
3) reconstruct the locked model and strict-load the best checkpoint;
4) recompute Nov-Dec 2022 predictions;
5) evaluate ONLY valid pixels outside the East-Asia core + 64-cell exclusion;
6) reproduce history[best_epoch] R2 / RMSE / bias within a fixed tolerance.

Only after all nine runs pass is the formal set considered sealed and ready
for one-time East-Asia core evaluation.

Scientific boundary
-------------------
This is an integrity/provenance closure, not a scientific result. Safe-selection
metrics are checkpoint-selection metrics and must not be reported as spatial
test performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


H = 361
W = 576

EXPECTED_EXPERIMENTS = [
    "mean_meteo_clim_only",
    "mean_binary_support_meteo_clim",
    "mean_emis_meteo_clim",
]
EXPECTED_SEEDS = [42, 43, 44]

EXPECTED_AUDIT_ARTIFACT = "s2_meanstate_spatial_holdout_formal_run_audit_gateS9D"
EXPECTED_TRAINING_ARTIFACT = "s2_meanstate_spatial_holdout_training_gateS9C"

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
EXPECTED_MONTHLY_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)

EXPECTED_CORE_BBOX = [220, 290, 448, 520]
EXPECTED_TRAIN_EXCLUSION_BBOX = [188, 322, 416, 552]
EXPECTED_SELECTION_EXCLUSION_BBOX = [156, 354, 384, 575]
EXPECTED_GUARD_CELLS = 32
EXPECTED_SELECTION_GUARD_CELLS = 64
EXPECTED_RF_HALF = 23
EXPECTED_ELIGIBLE_PATCH_POSITIONS = 77386
EXPECTED_VAL_MONTHS = [11, 12]

EXPECTED_MODEL = {
    "mean_meteo_clim_only": {
        "uses_emission": False,
        "emission_channels": 0,
        "meteo_channels": 5,
        "input_channels": 5,
        "trainable_parameters": 477025,
    },
    "mean_binary_support_meteo_clim": {
        "uses_emission": True,
        "emission_channels": 1,
        "meteo_channels": 5,
        "input_channels": 6,
        "trainable_parameters": 476449,
    },
    "mean_emis_meteo_clim": {
        "uses_emission": True,
        "emission_channels": 8,
        "meteo_channels": 5,
        "input_channels": 13,
        "trainable_parameters": 477457,
    },
}

METEO_ORDER = ["Temp", "RH", "BLH", "Windspeed_U", "Windspeed_V"]
DEFAULT_REPRO_TOLERANCE = 1e-6


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
    module_name = "gate_s9e_locked_base47"
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


def selection_safe_mask() -> torch.Tensor:
    y0, y1, x0, x1 = EXPECTED_SELECTION_EXCLUSION_BBOX
    mask = torch.ones((H, W), dtype=torch.bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = False
    return mask


SAFE_SELECTION_MASK = selection_safe_mask()


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
        require(tuple(emission.shape) == (8, H, W), f"EMISSION_SHAPE_INVALID: {tuple(emission.shape)}")
        require(bool(torch.isfinite(emission).all().item()), "EMISSION_NONFINITE")
        if experiment == "mean_binary_support_meteo_clim":
            support = torch.any(
                torch.abs(emission) > 0.0,
                dim=0,
                keepdim=True,
            ).float()
            unique = set(float(x) for x in torch.unique(support).tolist())
            require(unique.issubset({0.0, 1.0}), f"SUPPORT_NOT_BINARY: {unique}")
            parts.append(support)
        else:
            parts.append(emission)

    meteo = payload.get("meteo_mean")
    require(isinstance(meteo, torch.Tensor), "METEO_INPUT_MISSING")
    meteo = meteo.detach().cpu().float()
    require(tuple(meteo.shape) == (5, H, W), f"METEO_SHAPE_INVALID: {tuple(meteo.shape)}")

    # Gate S9B contract: transform before nan_to_num.
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


def pooled_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    require(pred.numel() == target.numel() and pred.numel() > 1, "METRIC_LENGTH_INVALID")
    require(bool(torch.isfinite(pred).all().item()), "PRED_NONFINITE")
    require(bool(torch.isfinite(target).all().item()), "TARGET_NONFINITE")
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return {
        "log_r2": float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)),
        "log_rmse": float(torch.sqrt(torch.mean((pred - target) ** 2))),
        "log_bias": float(torch.mean(pred - target)),
    }


@torch.no_grad()
def recompute_safe_selection(
    *,
    model,
    val_entries: list[dict[str, Any]],
    experiment: str,
    meteo_scale: torch.Tensor,
    meteo_offset: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_pred = []
    all_target = []
    rows = []

    for entry in val_entries:
        month = int(entry["month"])
        require(month in EXPECTED_VAL_MONTHS, f"UNEXPECTED_VAL_MONTH: {month}")
        path = Path(entry["file"]).expanduser().resolve()
        require(path.is_file(), f"VAL_PAYLOAD_NOT_FOUND: {path}")
        payload = load_pt(path)

        x = build_input(
            payload,
            experiment,
            meteo_scale=meteo_scale,
            meteo_offset=meteo_offset,
        )
        pred = model(
            x.unsqueeze(0).to(device, non_blocking=True)
        ).squeeze(0).squeeze(0).detach().cpu().float()

        target = payload.get("target_log_mean_conc")
        valid = payload.get("valid_mask")
        require(isinstance(target, torch.Tensor), f"TARGET_MISSING: {path}")
        require(isinstance(valid, torch.Tensor), f"VALID_MASK_MISSING: {path}")
        target = target.detach().cpu().float()
        valid = valid.detach().cpu().float() > 0.5
        require(tuple(target.shape) == (H, W), "TARGET_SHAPE_INVALID")
        require(tuple(valid.shape) == (H, W), "VALID_SHAPE_INVALID")
        require(tuple(pred.shape) == (H, W), "PRED_SHAPE_INVALID")

        safe = (
            valid
            & SAFE_SELECTION_MASK
            & torch.isfinite(target)
            & torch.isfinite(pred)
        )
        n = int(safe.sum().item())
        require(n > 10000, f"SAFE_SELECTION_TOO_FEW_PIXELS_MONTH_{month}: {n}")

        p = pred[safe]
        t = target[safe]
        metrics = pooled_metrics(p, t)
        all_pred.append(p)
        all_target.append(t)
        rows.append(
            {
                "year": int(entry["year"]),
                "month": month,
                "n_safe_selection_pixels": n,
                **metrics,
            }
        )

    require([row["month"] for row in rows] == EXPECTED_VAL_MONTHS, "VAL_MONTH_ORDER_MISMATCH")
    global_metrics = pooled_metrics(
        torch.cat(all_pred),
        torch.cat(all_target),
    )
    return {
        "evaluation_role": "preopening_checkpoint_reproduction_on_noncore_safe_selection_only",
        "core_holdout_performance_opened": False,
        "n_groups": len(rows),
        "n_total_safe_selection_pixels": int(sum(row["n_safe_selection_pixels"] for row in rows)),
        "global": global_metrics,
        "per_group": rows,
    }


def expected_audit_path(
    audit_root: Path,
    experiment: str,
    seed: int,
) -> Path:
    return (
        audit_root
        / f"{experiment}_seed{seed}"
        / f"{experiment}_seed{seed}_formal_audit_gateS9D.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S9E seal 3x3 spatial-holdout formal set before opening East-Asia core."
    )
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--base-training-script", required=True)
    parser.add_argument("--binary-training-script", required=True)
    parser.add_argument("--spatial-training-script", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=DEFAULT_REPRO_TOLERANCE,
    )
    args = parser.parse_args()

    audit_root = Path(args.audit_root).expanduser().resolve()
    base47_path = Path(args.base_training_script).expanduser().resolve()
    binary77_path = Path(args.binary_training_script).expanduser().resolve()
    spatial83_path = Path(args.spatial_training_script).expanduser().resolve()
    audit84_path = Path(args.audit_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(audit_root.is_dir(), f"AUDIT_ROOT_NOT_FOUND: {audit_root}")
    for p in [base47_path, binary77_path, spatial83_path, audit84_path]:
        require(p.is_file(), f"LOCKED_SCRIPT_NOT_FOUND: {p}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")
    require(args.reproduction_tolerance > 0, "REPRODUCTION_TOLERANCE_MUST_BE_POSITIVE")

    require(sha256_file(base47_path) == EXPECTED_47_SHA256, "BASE47_SHA256_MISMATCH")
    require(sha256_file(binary77_path) == EXPECTED_77_SHA256, "BINARY77_SHA256_MISMATCH")
    require(sha256_file(spatial83_path) == EXPECTED_83_SHA256, "SPATIAL83_SHA256_MISMATCH")
    require(sha256_file(audit84_path) == EXPECTED_84_SHA256, "AUDIT84_SHA256_MISMATCH")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA_REQUIRED_BUT_UNAVAILABLE")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base = import_base47(base47_path)

    all_records: list[dict[str, Any]] = []
    common_s9a_hashes = set()
    common_s9b_hashes = set()
    common_manifest_hashes = set()

    print("=" * 92)
    print("Gate S9E - pre-opening spatial-holdout 3x3 closure")
    print("=" * 92)
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print("CORE_HOLDOUT_PERFORMANCE_OPENED=False")

    for experiment in EXPECTED_EXPERIMENTS:
        cfg = EXPECTED_MODEL[experiment]
        for seed in EXPECTED_SEEDS:
            audit_path = expected_audit_path(
                audit_root,
                experiment,
                seed,
            )
            require(audit_path.is_file(), f"AUDIT_JSON_NOT_FOUND: {audit_path}")
            audit = load_json(audit_path)

            require(audit.get("artifact_name") == EXPECTED_AUDIT_ARTIFACT, "AUDIT_ARTIFACT_MISMATCH")
            require(audit.get("status") == "PASS_READ_ONLY_FORMAL_AUDIT", "AUDIT_STATUS_MISMATCH")
            require(audit.get("experiment") == experiment, "AUDIT_EXPERIMENT_MISMATCH")
            require(int(audit.get("seed", -1)) == seed, "AUDIT_SEED_MISMATCH")
            verification = audit.get("verification", {})
            require(verification.get("overall_pass") is True, "AUDIT_NOT_PASS")
            require(verification.get("core_holdout_performance_opened") is False, "AUDIT_CORE_ALREADY_OPENED")
            for key in [
                "formal_contract_exact",
                "split_contract_exact",
                "spatial_contract_exact",
                "normalization_contract_exact",
                "provenance_exact",
                "history_complete_and_finite",
                "early_stopping_consistent",
                "checkpoint_matches_history_best",
                "checkpoint_strict_load_pass",
                "latest_safe_selection_noncore_and_finite",
                "input_preflight_pass",
            ]:
                require(verification.get(key) is True, f"AUDIT_VERIFICATION_FAIL_{key}")

            run_dir = Path(audit["run_dir"]).expanduser().resolve()
            require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
            inputs = audit["inputs"]

            manifest_path = Path(inputs["run_manifest"]).expanduser().resolve()
            history_path = Path(inputs["training_history"]).expanduser().resolve()
            checkpoint_path = Path(inputs["checkpoint"]).expanduser().resolve()
            latest_safe_path = Path(inputs["latest_safe_selection_report"]).expanduser().resolve()
            for p in [manifest_path, history_path, checkpoint_path, latest_safe_path]:
                require(p.is_file(), f"RUN_ARTIFACT_NOT_FOUND: {p}")

            require(sha256_file(manifest_path) == inputs["run_manifest_sha256"], "MANIFEST_CHANGED_AFTER_AUDIT")
            require(sha256_file(history_path) == inputs["training_history_sha256"], "HISTORY_CHANGED_AFTER_AUDIT")
            require(sha256_file(checkpoint_path) == inputs["checkpoint_sha256"], "CHECKPOINT_CHANGED_AFTER_AUDIT")
            require(
                sha256_file(latest_safe_path)
                == inputs["latest_safe_selection_report_sha256"],
                "LATEST_SAFE_CHANGED_AFTER_AUDIT",
            )

            manifest = load_json(manifest_path)
            history = load_json(history_path)
            checkpoint = load_checkpoint(checkpoint_path, device)

            require(manifest.get("artifact_name") == EXPECTED_TRAINING_ARTIFACT, "TRAINING_ARTIFACT_MISMATCH")
            require(manifest.get("run_mode") == "formal", "RUN_MODE_NOT_FORMAL")
            require(manifest.get("run_name") == f"{experiment}_seed{seed}", "RUN_NAME_MISMATCH")
            require(int(manifest.get("seed", -1)) == seed, "MANIFEST_SEED_MISMATCH")

            provenance = manifest.get("provenance", {})
            require(provenance.get("base_47_script_sha256") == EXPECTED_47_SHA256, "MANIFEST_BASE47_HASH_MISMATCH")
            require(provenance.get("binary_77_script_sha256") == EXPECTED_77_SHA256, "MANIFEST_BINARY77_HASH_MISMATCH")
            require(provenance.get("training_script_sha256") == EXPECTED_83_SHA256, "MANIFEST_83_HASH_MISMATCH")
            require(provenance.get("monthly_manifest_sha256") == EXPECTED_MONTHLY_MANIFEST_SHA256, "MONTHLY_MANIFEST_HASH_MISMATCH")
            common_manifest_hashes.add(provenance["monthly_manifest_sha256"])
            common_s9a_hashes.add(provenance["s9a_report_sha256"])
            common_s9b_hashes.add(provenance["s9b_report_sha256"])

            spatial = manifest.get("spatial_contract", {})
            require(spatial.get("core_bbox_indices") == EXPECTED_CORE_BBOX, "CORE_BBOX_MISMATCH")
            require(spatial.get("training_exclusion_bbox_indices") == EXPECTED_TRAIN_EXCLUSION_BBOX, "TRAIN_EXCLUSION_BBOX_MISMATCH")
            require(spatial.get("selection_exclusion_bbox_indices") == EXPECTED_SELECTION_EXCLUSION_BBOX, "SELECTION_EXCLUSION_BBOX_MISMATCH")
            require(int(spatial.get("guard_cells", -1)) == EXPECTED_GUARD_CELLS, "GUARD_MISMATCH")
            require(int(spatial.get("selection_guard_cells", -1)) == EXPECTED_SELECTION_GUARD_CELLS, "SELECTION_GUARD_MISMATCH")
            require(int(spatial.get("max_half_receptive_field", -1)) == EXPECTED_RF_HALF, "RF_HALF_MISMATCH")
            require(int(spatial.get("eligible_patch_positions", -1)) == EXPECTED_ELIGIBLE_PATCH_POSITIONS, "ELIGIBLE_PATCH_COUNT_MISMATCH")
            require(spatial.get("core_test_opened_during_training") is False, "CORE_TEST_OPENED_DURING_TRAINING")
            require(EXPECTED_RF_HALF <= EXPECTED_GUARD_CELLS, "RF_EXCEEDS_TRAIN_GUARD")

            normalization = manifest.get("normalization_contract", {})
            require(normalization.get("spatial_train_only") is True, "NORMALIZATION_NOT_SPATIAL_TRAIN_ONLY")
            require(normalization.get("variables") == METEO_ORDER, "METEO_ORDER_MISMATCH")
            require(normalization.get("application_order") == "affine transform before nan_to_num", "NORMALIZATION_ORDER_MISMATCH")
            require(normalization.get("cache_rebuilt") is False, "CACHE_REBUILT_FLAG_MISMATCH")
            scales = [
                finite_float(x, "AFFINE_SCALE")
                for x in normalization.get("affine_scale", [])
            ]
            offsets = [
                finite_float(x, "AFFINE_OFFSET")
                for x in normalization.get("affine_offset", [])
            ]
            require(len(scales) == 5 and len(offsets) == 5, "AFFINE_LENGTH_MISMATCH")
            require(all(x > 0 for x in scales), "AFFINE_SCALE_NONPOSITIVE")
            scale_t = torch.tensor(scales, dtype=torch.float32).view(5, 1, 1)
            offset_t = torch.tensor(offsets, dtype=torch.float32).view(5, 1, 1)

            val_entries = list(manifest.get("val_entries", []))
            require([int(x["month"]) for x in val_entries] == EXPECTED_VAL_MONTHS, "VAL_MONTHS_MISMATCH")

            # History best target.
            r2_history = [
                finite_float(x, f"HISTORY_R2_{experiment}_{seed}")
                for x in history.get("safe_selection_log_r2", [])
            ]
            rmse_history = [
                finite_float(x, f"HISTORY_RMSE_{experiment}_{seed}")
                for x in history.get("safe_selection_log_rmse", [])
            ]
            bias_history = [
                finite_float(x, f"HISTORY_BIAS_{experiment}_{seed}")
                for x in history.get("safe_selection_log_bias", [])
            ]
            require(
                r2_history
                and len(r2_history) == len(rmse_history) == len(bias_history),
                "HISTORY_SELECTION_SERIES_INVALID",
            )
            best_idx = max(range(len(r2_history)), key=lambda i: r2_history[i])
            best_epoch = best_idx + 1

            require(
                int(audit["summary"]["best_epoch"]) == best_epoch,
                "AUDIT_BEST_EPOCH_MISMATCH",
            )
            require(
                int(checkpoint.get("epoch", -1)) + 1 == best_epoch,
                "CHECKPOINT_BEST_EPOCH_MISMATCH",
            )
            require(
                checkpoint.get("core_holdout_performance_opened") is False,
                "CHECKPOINT_CORE_ALREADY_OPENED",
            )

            # Strict model reconstruction.
            model_spec = make_spec(base, experiment)
            model = base.MeanStateUNet(
                spec=model_spec,
                base_features=32,
            ).to(device)
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            n_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
            require(
                n_params == cfg["trainable_parameters"],
                f"PARAMETER_COUNT_MISMATCH_{experiment}_{seed}: {n_params}",
            )

            reproduced = recompute_safe_selection(
                model=model,
                val_entries=val_entries,
                experiment=experiment,
                meteo_scale=scale_t,
                meteo_offset=offset_t,
                device=device,
            )
            got = reproduced["global"]
            expected = {
                "log_r2": r2_history[best_idx],
                "log_rmse": rmse_history[best_idx],
                "log_bias": bias_history[best_idx],
            }

            diffs = {}
            for metric in ["log_r2", "log_rmse", "log_bias"]:
                diff = abs(float(got[metric]) - float(expected[metric]))
                diffs[metric] = diff
                require(
                    diff <= args.reproduction_tolerance,
                    f"BEST_CHECKPOINT_REPRODUCTION_FAIL "
                    f"experiment={experiment} seed={seed} metric={metric} "
                    f"expected={expected[metric]} got={got[metric]} diff={diff}",
                )

            print(
                f"REPRO_PASS experiment={experiment} seed={seed} "
                f"BEST_EPOCH={best_epoch} "
                f"SAFE_R2={got['log_r2']:.8f} "
                f"SAFE_RMSE={got['log_rmse']:.8f} "
                f"MAX_ABS_DIFF={max(diffs.values()):.3e}",
                flush=True,
            )

            all_records.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "audit": str(audit_path),
                    "audit_sha256": sha256_file(audit_path),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "history": str(history_path),
                    "history_sha256": sha256_file(history_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "best_epoch": best_epoch,
                    "trainable_parameters": n_params,
                    "expected_best_safe_selection": expected,
                    "recomputed_best_safe_selection": got,
                    "absolute_differences": diffs,
                    "n_total_safe_selection_pixels": reproduced[
                        "n_total_safe_selection_pixels"
                    ],
                    "per_month_noncore_safe_selection": reproduced["per_group"],
                    "core_holdout_performance_opened": False,
                    "reproduction_pass": True,
                }
            )

    require(len(all_records) == 9, f"FORMAL_SET_SIZE_MISMATCH: {len(all_records)}")
    require(len(common_manifest_hashes) == 1, "MONTHLY_MANIFEST_HASHES_DIFFER")
    require(len(common_s9a_hashes) == 1, "S9A_HASHES_DIFFER")
    require(len(common_s9b_hashes) == 1, "S9B_HASHES_DIFFER")

    pair_set = {(x["experiment"], x["seed"]) for x in all_records}
    expected_pair_set = {
        (experiment, seed)
        for experiment in EXPECTED_EXPERIMENTS
        for seed in EXPECTED_SEEDS
    }
    require(pair_set == expected_pair_set, "FORMAL_3X3_MATRIX_INCOMPLETE")

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "meanstate_spatial_holdout_preopening_closure_gateS9E.json"
    csv_path = output_dir / "meanstate_spatial_holdout_preopening_matrix_gateS9E.csv"
    txt_path = output_dir / "meanstate_spatial_holdout_preopening_closure_gateS9E.txt"

    report = {
        "artifact_name": "s2_meanstate_spatial_holdout_preopening_closure_gateS9E",
        "artifact_version": 1,
        "status": "PASS_PREOPENING_3X3_FORMAL_SET_SEALED",
        "formal_set": {
            "experiments": EXPECTED_EXPERIMENTS,
            "seeds": EXPECTED_SEEDS,
            "n_runs": 9,
        },
        "locked_scripts": {
            "base_47_sha256": EXPECTED_47_SHA256,
            "binary_77_sha256": EXPECTED_77_SHA256,
            "spatial_83_sha256": EXPECTED_83_SHA256,
            "audit_84_sha256": EXPECTED_84_SHA256,
        },
        "shared_provenance": {
            "monthly_manifest_sha256": next(iter(common_manifest_hashes)),
            "s9a_report_sha256": next(iter(common_s9a_hashes)),
            "s9b_report_sha256": next(iter(common_s9b_hashes)),
        },
        "spatial_contract": {
            "core_bbox_indices": EXPECTED_CORE_BBOX,
            "training_exclusion_bbox_indices": EXPECTED_TRAIN_EXCLUSION_BBOX,
            "selection_exclusion_bbox_indices": EXPECTED_SELECTION_EXCLUSION_BBOX,
            "training_guard_cells": EXPECTED_GUARD_CELLS,
            "selection_guard_cells": EXPECTED_SELECTION_GUARD_CELLS,
            "max_half_receptive_field": EXPECTED_RF_HALF,
            "core_holdout_performance_opened": False,
        },
        "reproduction_tolerance": args.reproduction_tolerance,
        "records": all_records,
        "verification": {
            "exact_3x3_formal_set_present": True,
            "all_gateS9D_audits_pass": True,
            "all_artifacts_unchanged_since_audit": True,
            "all_checkpoint_strict_load_pass": True,
            "all_best_safe_selection_metrics_reproduced": True,
            "shared_monthly_manifest_hash_pass": True,
            "shared_s9a_hash_pass": True,
            "shared_s9b_hash_pass": True,
            "core_holdout_performance_opened": False,
            "ready_for_one_time_core_evaluation": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This artifact seals the nine-run spatial-holdout formal set before "
            "the East-Asia core is opened. Reproduced metrics are non-core "
            "checkpoint-selection metrics only and are not spatial test results. "
            "No model, seed, hyperparameter, region, guard width or checkpoint "
            "may be changed after this closure based on future East-Asia core performance."
        ),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "seed",
                "best_epoch",
                "trainable_parameters",
                "checkpoint_sha256",
                "expected_safe_r2",
                "recomputed_safe_r2",
                "expected_safe_rmse",
                "recomputed_safe_rmse",
                "expected_safe_bias",
                "recomputed_safe_bias",
                "max_abs_diff",
                "core_holdout_performance_opened",
            ],
        )
        writer.writeheader()
        for row in all_records:
            expected = row["expected_best_safe_selection"]
            got = row["recomputed_best_safe_selection"]
            writer.writerow(
                {
                    "experiment": row["experiment"],
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "trainable_parameters": row["trainable_parameters"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "expected_safe_r2": expected["log_r2"],
                    "recomputed_safe_r2": got["log_r2"],
                    "expected_safe_rmse": expected["log_rmse"],
                    "recomputed_safe_rmse": got["log_rmse"],
                    "expected_safe_bias": expected["log_bias"],
                    "recomputed_safe_bias": got["log_bias"],
                    "max_abs_diff": max(row["absolute_differences"].values()),
                    "core_holdout_performance_opened": False,
                }
            )

    lines = [
        "GATE_S9E_PREOPENING_3X3_FORMAL_SET_SEALED_PASS",
        "EXACT_3X3_FORMAL_SET_PRESENT=True",
        "ALL_GATE_S9D_AUDITS_PASS=True",
        "ALL_ARTIFACTS_UNCHANGED_SINCE_AUDIT=True",
        "ALL_CHECKPOINT_STRICT_LOAD_PASS=True",
        "ALL_BEST_SAFE_SELECTION_METRICS_REPRODUCED=True",
        f"REPRODUCTION_TOLERANCE={args.reproduction_tolerance}",
        "CORE_HOLDOUT_PERFORMANCE_OPENED=False",
        "READY_FOR_ONE_TIME_CORE_EVALUATION=True",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_CSV={csv_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
