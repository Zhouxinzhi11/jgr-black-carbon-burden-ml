# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
79_eval_s2_meanstate_binary_support_formal_run.py

Gate S8E v2 provenance-locked formal evaluation for one audited binary-support
 meteorology mean-state run created by
77_train_s2_meanstate_binary_support_gateS8.py and audited by
78_audit_s2_meanstate_binary_support_formal_run.py.

2022:
- held-out model-selection validation;
- months 11 and 12;
- support is derived from the actual monthly processed prior.

2023:
- months 01-12;
- proxy external evaluation;
- support is derived from the fixed unweighted arithmetic mean of the six
  2022 training-month processed priors, then binarized with the locked
  zero threshold.

v2 fixes only the proxy-hash semantic mismatch in the first 79 script: Gate P3/S8A
used raw tensor bytes, whereas the failed v1 evaluator compared a dtype+shape+bytes
hash against that raw-byte digest. No scientific or model-setting change is made.

Outputs pooled, monthly, seasonal and fixed-region log-space metrics.
No training artifact is modified and existing output directories are never
overwritten.
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
from typing import Any, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

H = 361
W = 576
EXPECTED_EXPERIMENT = "mean_binary_support_meteo_clim"
EXPECTED_AUDIT_ARTIFACT = "s2_meanstate_binary_support_formal_run_audit_gateS8D"
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
ALLOWED_SEEDS = {42, 43, 44}
EXPECTED_SUPPORT_DEFINITION = "any(abs(emission_log_mean) > 0.0, dim=0)"
EXPECTED_PROXY_TENSOR_SHA256 = (
    "283867292ed0b51ebce073db35940c4b091d23cacba8ca929f5248982bb0d749"
)
EXPECTED_PROXY_SUPPORT_CELLS = 42949
EXPECTED_TRAINABLE_PARAMETERS = 476449

REGIONS = {
    "North_America": {"lat": (15.0, 75.0), "lon": (-170.0, -50.0)},
    "Europe": {"lat": (35.0, 72.0), "lon": (-10.0, 40.0)},
    "East_Asia": {"lat": (20.0, 55.0), "lon": (100.0, 145.0)},
    "South_Asia": {"lat": (5.0, 35.0), "lon": (65.0, 100.0)},
    "Africa": {"lat": (-35.0, 37.0), "lon": (-20.0, 55.0)},
}

LAT = np.linspace(-90.0, 90.0, H, dtype=np.float32)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float32)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    require(isinstance(obj, dict), "CHECKPOINT_ROOT_NOT_DICT")
    return obj


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor_raw_bytes(tensor: torch.Tensor) -> str:
    """Gate-P3 / Gate-S8A hash semantics: SHA256 over contiguous tensor bytes only."""
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sha256_tensor_canonical(tensor: torch.Tensor) -> str:
    """52/62 evaluator-style diagnostic hash: dtype + shape + contiguous bytes."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name}_IS_NONE")
    number = float(value)
    require(math.isfinite(number), f"NONFINITE_{name}: {value}")
    return number


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json_strict(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(obj),
            f,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(month)


def region_mask(region_name: str) -> torch.Tensor:
    cfg = REGIONS[region_name]
    lat0, lat1 = cfg["lat"]
    lon0, lon1 = cfg["lon"]
    mask = (
        (LAT_GRID >= lat0)
        & (LAT_GRID <= lat1)
        & (LON_GRID >= lon0)
        & (LON_GRID <= lon1)
    )
    return torch.from_numpy(mask).bool()


def load_training_module(path: Path):
    module_name = "binary_support_training_module_for_gate_s8e_eval"
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


def pooled_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Optional[float]]:
    prediction = prediction.reshape(-1).float()
    target = target.reshape(-1).float()
    if prediction.numel() <= 1:
        return {"log_r2": None, "log_rmse": None, "log_bias": None}
    require(prediction.numel() == target.numel(), "PREDICTION_TARGET_LENGTH_MISMATCH")
    require(bool(torch.isfinite(prediction).all().item()), "PREDICTION_NONFINITE")
    require(bool(torch.isfinite(target).all().item()), "TARGET_NONFINITE")
    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))
    rmse = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    bias = float(torch.mean(prediction - target))
    return {"log_r2": r2, "log_rmse": rmse, "log_bias": bias}


def build_proxy_emission_baseline(
    train_entries: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    months = [int(entry["month"]) for entry in train_entries]
    require(months == EXPECTED_TRAIN_MONTHS, f"PROXY_SOURCE_MONTHS_MISMATCH: {months}")
    tensors: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for entry in train_entries:
        path = Path(entry["file"]).expanduser().resolve()
        require(path.is_file(), f"PROXY_SOURCE_NOT_FOUND: {path}")
        payload = load_pt(path)
        emission = payload.get("emission_log_mean")
        require(isinstance(emission, torch.Tensor), f"PROXY_SOURCE_EMISSION_MISSING: {path}")
        emission = emission.detach().cpu().float()
        require(tuple(emission.shape) == (8, H, W), f"PROXY_SOURCE_SHAPE_INVALID: {path}")
        require(bool(torch.isfinite(emission).all().item()), f"PROXY_SOURCE_NONFINITE: {path}")
        tensors.append(emission)
        records.append(
            {
                "month": int(entry["month"]),
                "file": str(path),
                "file_sha256": sha256_file(path),
                "tensor_sha256_canonical": sha256_tensor_canonical(emission),
            }
        )
    proxy = torch.stack(tensors, dim=0).mean(dim=0)
    require(bool(torch.isfinite(proxy).all().item()), "PROXY_NONFINITE")
    # IMPORTANT: Gate P3 / Gate S8A locked the proxy with a raw-byte SHA256.
    # The earlier 79 implementation accidentally used the 52/62 evaluator-style
    # dtype+shape+bytes hash while comparing against that raw-byte hash.
    # Keep both hashes explicitly and use the raw-byte hash for the locked contract.
    proxy_hash_raw = sha256_tensor_raw_bytes(proxy)
    proxy_hash_canonical = sha256_tensor_canonical(proxy)
    support = torch.any(torch.abs(proxy) > 0.0, dim=0)
    support_cells = int(support.sum().item())
    require(
        proxy_hash_raw == EXPECTED_PROXY_TENSOR_SHA256,
        "PROXY_TENSOR_SHA256_GATEP3_RAW_BYTES_MISMATCH",
    )
    require(support_cells == EXPECTED_PROXY_SUPPORT_CELLS, "PROXY_SUPPORT_CELLS_MISMATCH")
    return proxy, {
        "construction": "Unweighted arithmetic mean of six 2022 training-month processed priors.",
        "source_months": months,
        "source_records": records,
        "proxy_tensor_sha256": proxy_hash_raw,
        "proxy_tensor_sha256_semantics": "raw_contiguous_numpy_bytes_only (Gate P3 / Gate S8A contract)",
        "proxy_tensor_sha256_canonical": proxy_hash_canonical,
        "proxy_tensor_sha256_canonical_semantics": "dtype + shape + raw bytes (52/62 evaluator-style diagnostic)",
        "proxy_support_cells": support_cells,
        "shape": list(proxy.shape),
        "dtype": str(proxy.dtype),
    }


def build_eval_input(
    *,
    payload: dict[str, Any],
    module,
    spec,
    key: str,
    proxy_emission: Optional[torch.Tensor],
) -> torch.Tensor:
    if proxy_emission is None:
        payload_for_input = payload
    else:
        payload_for_input = dict(payload)
        payload_for_input["emission_log_mean"] = proxy_emission
    x = module.build_input_from_payload(
        payload_for_input,
        spec,
        key=key,
    )
    require(tuple(x.shape) == (6, H, W), f"EVAL_INPUT_SHAPE_MISMATCH: {tuple(x.shape)}")
    require(bool(torch.isfinite(x).all().item()), "EVAL_INPUT_NONFINITE")
    support = x[0]
    unique_values = set(float(v) for v in torch.unique(support).tolist())
    require(unique_values.issubset({0.0, 1.0}), f"SUPPORT_NOT_BINARY: {sorted(unique_values)}")
    return x


@torch.no_grad()
def evaluate_entries(
    *,
    model,
    entries: Sequence[dict[str, Any]],
    module,
    spec,
    device: torch.device,
    proxy_emission: Optional[torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    all_prediction: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    season_buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "DJF": [], "MAM": [], "JJA": [], "SON": []
    }
    region_buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        name: [] for name in REGIONS
    }
    per_group: list[dict[str, Any]] = []

    region_masks = {name: region_mask(name) for name in REGIONS}

    for entry in entries:
        path = Path(entry["file"]).expanduser().resolve()
        payload = load_pt(path)
        x = build_eval_input(
            payload=payload,
            module=module,
            spec=spec,
            key=path.name,
            proxy_emission=proxy_emission,
        )
        target = payload.get("target_log_mean_conc")
        valid_mask = payload.get("valid_mask")
        require(isinstance(target, torch.Tensor), f"TARGET_MISSING: {path}")
        require(isinstance(valid_mask, torch.Tensor), f"VALID_MASK_MISSING: {path}")
        target = target.detach().cpu().float()
        valid = valid_mask.detach().cpu().float() > 0.5
        require(tuple(target.shape) == (H, W), f"TARGET_SHAPE_INVALID: {path}")
        require(tuple(valid.shape) == (H, W), f"VALID_MASK_SHAPE_INVALID: {path}")

        pred = model(x.unsqueeze(0).to(device, non_blocking=True)).squeeze(0).squeeze(0)
        pred = pred.detach().cpu().float()
        require(tuple(pred.shape) == (H, W), f"PREDICTION_SHAPE_INVALID: {tuple(pred.shape)}")

        eval_mask = valid & torch.isfinite(target) & torch.isfinite(pred)
        n_valid = int(eval_mask.sum().item())
        require(n_valid > 1, f"NO_VALID_PIXELS: {path}")

        pred_vec = pred[eval_mask]
        target_vec = target[eval_mask]
        metrics = pooled_metrics(pred_vec, target_vec)
        month = int(entry["month"])
        season = month_to_season(month)

        per_region: dict[str, Any] = {}
        for region_name, rmask in region_masks.items():
            mask = eval_mask & rmask
            n_region = int(mask.sum().item())
            if n_region > 1:
                rpred = pred[mask]
                rtgt = target[mask]
                rmetrics = pooled_metrics(rpred, rtgt)
                region_buckets[region_name].append((rpred, rtgt))
                per_region[region_name] = {
                    "n_valid_pixels": n_region,
                    **rmetrics,
                }
            else:
                per_region[region_name] = {
                    "n_valid_pixels": n_region,
                    "log_r2": None,
                    "log_rmse": None,
                    "log_bias": None,
                }

        all_prediction.append(pred_vec)
        all_target.append(target_vec)
        season_buckets[season].append((pred_vec, target_vec))
        per_group.append(
            {
                "year": int(entry["year"]),
                "month": month,
                "season": season,
                "file": str(path),
                "n_valid_pixels": n_valid,
                **metrics,
                "regions": per_region,
            }
        )
        print(
            f"EVAL_GROUP={int(entry['year'])}-{month:02d} "
            f"N_VALID={n_valid} "
            f"LOG_R2={metrics['log_r2']:.8f} "
            f"LOG_RMSE={metrics['log_rmse']:.8f} "
            f"LOG_BIAS={metrics['log_bias']:.8f}",
            flush=True,
        )

    global_metrics = pooled_metrics(torch.cat(all_prediction), torch.cat(all_target))

    season_summary: dict[str, Any] = {}
    for season, pairs in season_buckets.items():
        if pairs:
            season_summary[season] = {
                "n_groups": len(pairs),
                **pooled_metrics(
                    torch.cat([pred for pred, _ in pairs]),
                    torch.cat([target for _, target in pairs]),
                ),
            }
        else:
            season_summary[season] = {
                "n_groups": 0,
                "log_r2": None,
                "log_rmse": None,
                "log_bias": None,
            }

    region_summary: dict[str, Any] = {}
    for region_name, pairs in region_buckets.items():
        if pairs:
            region_summary[region_name] = {
                "n_groups_with_valid_pixels": len(pairs),
                **pooled_metrics(
                    torch.cat([pred for pred, _ in pairs]),
                    torch.cat([target for _, target in pairs]),
                ),
            }
        else:
            region_summary[region_name] = {
                "n_groups_with_valid_pixels": 0,
                "log_r2": None,
                "log_rmse": None,
                "log_bias": None,
            }

    return {
        "n_groups": len(entries),
        "n_total_valid_pixels": int(sum(row["n_valid_pixels"] for row in per_group)),
        "global": global_metrics,
        "season_summary": season_summary,
        "region_summary": region_summary,
        "per_group": per_group,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate S8E formal evaluation for one binary-support run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seed", required=True, type=int, choices=sorted(ALLOWED_SEEDS))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--metric-reproduction-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    seed = int(args.expected_seed)
    expected_run_name = f"{EXPECTED_EXPERIMENT}_seed{seed}"
    run_dir = Path(args.run_dir).expanduser().resolve()
    audit_path = Path(args.training_audit).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
    require(audit_path.is_file(), f"TRAINING_AUDIT_NOT_FOUND: {audit_path}")
    require(training_script.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    for path in [manifest_path, history_path, checkpoint_path]:
        require(path.is_file() and path.stat().st_size > 0, f"MISSING_TRAINING_ARTIFACT: {path}")

    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    history = load_json(history_path)

    require(audit.get("artifact_name") == EXPECTED_AUDIT_ARTIFACT, "AUDIT_ARTIFACT_NAME_MISMATCH")
    require(int(audit.get("seed", -1)) == seed, "AUDIT_SEED_MISMATCH")
    require(audit.get("status") == "PASS_READ_ONLY_FORMAL_AUDIT", "AUDIT_STATUS_MISMATCH")
    require(audit.get("verification", {}).get("overall_pass") is True, "AUDIT_OVERALL_PASS_NOT_TRUE")
    require(norm_path(audit.get("run_dir", "")) == norm_path(run_dir), "AUDIT_RUN_DIR_MISMATCH")

    audit_inputs = audit.get("inputs", {})
    require(audit_inputs.get("manifest_sha256") == sha256_file(manifest_path), "MANIFEST_CHANGED_AFTER_AUDIT")
    require(audit_inputs.get("history_sha256") == sha256_file(history_path), "HISTORY_CHANGED_AFTER_AUDIT")
    require(audit_inputs.get("checkpoint_sha256") == sha256_file(checkpoint_path), "CHECKPOINT_CHANGED_AFTER_AUDIT")
    require(audit_inputs.get("training_script_sha256") == sha256_file(training_script), "TRAINING_SCRIPT_CHANGED_AFTER_AUDIT")

    require(manifest.get("run_mode") == "formal", "MANIFEST_RUN_MODE_NOT_FORMAL")
    require(manifest.get("run_name") == expected_run_name, "MANIFEST_RUN_NAME_MISMATCH")
    experiment = manifest.get("experiment", {})
    require(experiment.get("name") == EXPECTED_EXPERIMENT, "MANIFEST_EXPERIMENT_MISMATCH")
    require(experiment.get("binary_support_emission") is True, "BINARY_SUPPORT_FLAG_NOT_TRUE")
    require(experiment.get("shuffle_emission") is False, "SHUFFLE_FLAG_NOT_FALSE")
    require(int(experiment.get("emission_channels", -1)) == 1, "SUPPORT_CHANNEL_COUNT_MISMATCH")
    require(int(experiment.get("meteo_channels", -1)) == 5, "METEO_CHANNEL_COUNT_MISMATCH")

    support_contract = manifest.get("binary_support_contract", {})
    require(support_contract.get("support_definition") == EXPECTED_SUPPORT_DEFINITION, "SUPPORT_DEFINITION_MISMATCH")
    require(float(support_contract.get("support_epsilon", -1.0)) == 0.0, "SUPPORT_EPSILON_MISMATCH")
    require(support_contract.get("expected_2023_proxy_tensor_sha256") == EXPECTED_PROXY_TENSOR_SHA256, "EXPECTED_PROXY_HASH_MISMATCH")
    require(int(support_contract.get("expected_2023_proxy_support_cells", -1)) == EXPECTED_PROXY_SUPPORT_CELLS, "EXPECTED_PROXY_SUPPORT_CELLS_MISMATCH")

    train_entries = list(manifest.get("train_entries", []))
    validation_entries = list(manifest.get("val_entries", []))
    external_entries = list(manifest.get("external_2023_entries", []))
    require([int(x["month"]) for x in train_entries] == EXPECTED_TRAIN_MONTHS, "TRAIN_MONTHS_MISMATCH")
    require([int(x["month"]) for x in validation_entries] == EXPECTED_VAL_MONTHS, "VAL_MONTHS_MISMATCH")
    require([int(x["month"]) for x in external_entries] == EXPECTED_EXTERNAL_MONTHS, "EXTERNAL_MONTHS_MISMATCH")

    module = load_training_module(training_script)
    require(getattr(module, "S8_EXPERIMENT", None) == EXPECTED_EXPERIMENT, "TRAINING_MODULE_EXPERIMENT_MISMATCH")
    spec = module.EXPERIMENT_SPECS[EXPECTED_EXPERIMENT]
    require(spec.input_channels == 6, "TRAINING_SPEC_INPUT_CHANNELS_MISMATCH")

    proxy_emission, proxy_diag = build_proxy_emission_baseline(train_entries)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA_REQUIRED_BUT_UNAVAILABLE")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("=" * 88)
    print("Gate S8E - binary-support + meteorology formal evaluation")
    print("=" * 88)
    print(f"EXPERIMENT={EXPECTED_EXPERIMENT}")
    print(f"SEED={seed}")
    print(f"RUN_DIR={run_dir}")
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print(f"PROXY_TENSOR_SHA256={proxy_diag['proxy_tensor_sha256']}")
    print(f"PROXY_TENSOR_SHA256_CANONICAL={proxy_diag['proxy_tensor_sha256_canonical']}")
    print(f"PROXY_SUPPORT_CELLS={proxy_diag['proxy_support_cells']}")
    print("TRAINING_AUDIT_PROVENANCE_ACCEPTED=True")

    model = module.MeanStateUNet(
        spec=spec,
        base_features=int(manifest["settings"]["base_features"]),
    ).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS, "TRAINABLE_PARAMETER_COUNT_MISMATCH")

    checkpoint = load_checkpoint(checkpoint_path, device)
    require(checkpoint.get("experiment") == EXPECTED_EXPERIMENT, "CHECKPOINT_EXPERIMENT_MISMATCH")
    require(checkpoint.get("run_name") == expected_run_name, "CHECKPOINT_RUN_NAME_MISMATCH")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected_epoch = int(checkpoint.get("epoch", -1)) + 1
    checkpoint_best_r2 = finite_float(checkpoint.get("best_val_log_r2"), "CHECKPOINT_BEST_VAL_LOG_R2")

    audit_summary = audit.get("summary", {})
    audit_best_epoch = int(audit_summary.get("best_epoch", -1))
    audit_best_r2 = finite_float(audit_summary.get("best_val_log_r2"), "AUDIT_BEST_VAL_LOG_R2")
    require(selected_epoch == audit_best_epoch, "SELECTED_EPOCH_AUDIT_MISMATCH")
    require(math.isclose(checkpoint_best_r2, audit_best_r2, rel_tol=0.0, abs_tol=1e-12), "BEST_R2_AUDIT_MISMATCH")

    history_r2 = [finite_float(v, f"HISTORY_R2_{i}") for i, v in enumerate(history.get("val_monthly_log_r2", []), start=1)]
    history_rmse = [finite_float(v, f"HISTORY_RMSE_{i}") for i, v in enumerate(history.get("val_monthly_log_rmse", []), start=1)]
    require(history_r2 and len(history_r2) == len(history_rmse), "HISTORY_VALIDATION_SERIES_INVALID")
    best_index = max(range(len(history_r2)), key=lambda i: history_r2[i])
    require(best_index + 1 == audit_best_epoch, "HISTORY_BEST_EPOCH_AUDIT_MISMATCH")
    audit_best_rmse = history_rmse[best_index]

    print("\n[1/2] 2022 held-out model-selection validation")
    evaluation_2022 = evaluate_entries(
        model=model,
        entries=validation_entries,
        module=module,
        spec=spec,
        device=device,
        proxy_emission=None,
    )
    val_r2 = finite_float(evaluation_2022["global"]["log_r2"], "VAL_2022_LOG_R2")
    val_rmse = finite_float(evaluation_2022["global"]["log_rmse"], "VAL_2022_LOG_RMSE")
    tolerance = float(args.metric_reproduction_tolerance)
    require(abs(val_r2 - audit_best_r2) <= tolerance, "VAL_R2_BEST_CHECKPOINT_REPRODUCTION_FAIL")
    require(abs(val_rmse - audit_best_rmse) <= tolerance, "VAL_RMSE_BEST_CHECKPOINT_REPRODUCTION_FAIL")

    print("\n[2/2] 2023 proxy external evaluation")
    evaluation_2023 = evaluate_entries(
        model=model,
        entries=external_entries,
        module=module,
        spec=spec,
        device=device,
        proxy_emission=proxy_emission,
    )
    ext_r2 = finite_float(evaluation_2023["global"]["log_r2"], "EXTERNAL_2023_LOG_R2")
    ext_rmse = finite_float(evaluation_2023["global"]["log_rmse"], "EXTERNAL_2023_LOG_RMSE")
    ext_bias = finite_float(evaluation_2023["global"]["log_bias"], "EXTERNAL_2023_LOG_BIAS")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "eval_report.json"
    summary_path = output_dir / "eval_summary.txt"

    report = {
        "artifact_name": "s2_meanstate_binary_support_formal_evaluation_gateS8E",
        "artifact_version": 1,
        "experiment": manifest["experiment"],
        "seed": seed,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selected_epoch": selected_epoch,
            "best_val_log_r2": checkpoint_best_r2,
            "trainable_parameters": trainable_parameters,
        },
        "provenance": {
            "training_audit": str(audit_path),
            "training_audit_sha256": sha256_file(audit_path),
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "training_history": str(history_path),
            "training_history_sha256": sha256_file(history_path),
            "cache_dir": manifest["cache_dir"],
            "monthly_manifest": manifest["provenance"]["monthly_manifest"],
            "monthly_manifest_sha256": manifest["provenance"]["monthly_manifest_sha256"],
            "gate11b_report": manifest["provenance"]["gate11b_report"],
            "gate11b_report_sha256": manifest["provenance"]["gate11b_report_sha256"],
            "s8a_preflight_report": manifest["provenance"]["s8a_preflight_report"],
            "s8a_preflight_report_sha256": manifest["provenance"]["s8a_preflight_report_sha256"],
            "training_script": str(training_script),
            "training_script_sha256": sha256_file(training_script),
        },
        "evaluation_design": {
            "2022_validation_months": EXPECTED_VAL_MONTHS,
            "2022_status": "held-out model-selection validation",
            "2023_external_months": EXPECTED_EXTERNAL_MONTHS,
            "2023_status": "proxy_external",
            "emission_proxy_used": True,
            "binary_support_emission": True,
            "support_definition": EXPECTED_SUPPORT_DEFINITION,
            "support_epsilon": 0.0,
            "2022_support_source": "actual monthly processed emission prior",
            "2023_support_source": "fixed 2022 train-mean processed-prior proxy, then binarized",
            "metric_space": "processed log-BC monthly mean-state",
            "pooling": "all valid prediction-target pairs pooled within each reported split",
            "regions": REGIONS,
        },
        "proxy_emission": proxy_diag,
        "metric_reproduction": {
            "tolerance": tolerance,
            "audit_best_epoch": audit_best_epoch,
            "audit_best_log_r2": audit_best_r2,
            "audit_best_log_rmse": audit_best_rmse,
            "recomputed_2022_log_r2": val_r2,
            "recomputed_2022_log_rmse": val_rmse,
            "pass": True,
        },
        "eval_2022_validation": evaluation_2022,
        "eval_2023_external": evaluation_2023,
        "verification": {
            "training_audit_provenance_pass": True,
            "binary_support_contract_pass": True,
            "checkpoint_strict_load_pass": True,
            "2022_best_metric_reproduction_pass": True,
            "2023_all_12_months_evaluated": len(evaluation_2023["per_group"]) == 12,
            "2023_emission_proxy_used": True,
            "proxy_hash_exact": proxy_diag["proxy_tensor_sha256"] == EXPECTED_PROXY_TENSOR_SHA256,
            "proxy_support_count_exact": proxy_diag["proxy_support_cells"] == EXPECTED_PROXY_SUPPORT_CELLS,
            "all_reported_global_metrics_finite": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "The 2022 result is held-out model-selection validation. The 2023 result is a proxy "
            "external consistency test because contemporaneous 2023 processed sector channels are "
            "unavailable. The binary support channel is a property of the inherited processed-prior "
            "construction and is not verified physical emission geography. A single seed is not a "
            "final manuscript estimate."
        ),
    }
    save_json_strict(report, report_path)
    summary_path.write_text(
        "\n".join(
            [
                "Gate S8E binary-support formal evaluation",
                f"Experiment: {EXPECTED_EXPERIMENT}",
                f"Seed: {seed}",
                f"2022 pooled log R2: {val_r2:.8f}",
                f"2022 pooled log RMSE: {val_rmse:.8f}",
                f"2022 pooled log bias: {evaluation_2022['global']['log_bias']:.8f}",
                "2023 status: proxy_external",
                "2023 emission proxy used: True",
                f"2023 pooled log R2: {ext_r2:.8f}",
                f"2023 pooled log RMSE: {ext_rmse:.8f}",
                f"2023 pooled log bias: {ext_bias:.8f}",
                "Single-seed result ready for paper: False",
                "Overall pass: True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reloaded = load_json(report_path)
    require(reloaded.get("verification", {}).get("overall_pass") is True, "WRITTEN_REPORT_RELOAD_FAIL")

    print(f"\nGATE_S8E_BINARY_SUPPORT_FORMAL_SEED{seed}_EVALUATION_PASS")
    print(f"EXPERIMENT={EXPECTED_EXPERIMENT}")
    print(f"SEED={seed}")
    print(f"EVAL_REPORT={report_path}")
    print(f"EVAL_SUMMARY={summary_path}")
    print("TRAINING_AUDIT_PROVENANCE_PASS=True")
    print("BINARY_SUPPORT_CONTRACT_PASS=True")
    print("CHECKPOINT_STRICT_LOAD_PASS=True")
    print("BEST_CHECKPOINT_METRIC_REPRODUCTION_PASS=True")
    print(f"VAL_2022_LOG_R2={val_r2:.8f}")
    print(f"VAL_2022_LOG_RMSE={val_rmse:.8f}")
    print("EXTERNAL_2023_TYPE=proxy_external")
    print("EXTERNAL_2023_EMISSION_PROXY_USED=True")
    print(f"EXTERNAL_2023_LOG_R2={ext_r2:.8f}")
    print(f"EXTERNAL_2023_LOG_RMSE={ext_rmse:.8f}")
    print("SINGLE_SEED_RESULT_READY_FOR_PAPER=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
