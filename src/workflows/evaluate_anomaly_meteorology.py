# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
70_eval_s2_anomaly_clean_seed42.py

Unified best-checkpoint evaluation for one clean formal anomaly run:
anom_meteo_only / seed 42, 43 or 44.

Evaluation scope
----------------
1. 2022 held-out model-selection validation (Nov-Dec)
2. 2023 ordinary external anomaly evaluation

The script reports:
- anomaly metrics;
- reconstruction metrics after adding the monthly background;
- a zero-anomaly baseline;
- a monthly-background-only reconstruction baseline;
- month, season and fixed-region diagnostics.

The baseline comparison is essential: a high reconstruction R2 can largely
come from the monthly background. The anomaly skill relative to a zero
prediction is the relevant incremental quantity.

This script is read-only with respect to cache and run artifacts and never
overwrites an existing evaluation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


H = 361
W = 576
EXPECTED_EXPERIMENT = "anom_meteo_only"
ALLOWED_SEEDS = {42, 43, 44}
EXPECTED_PARAMETERS = 477025
EXPECTED_VAL_FILES = 61
EXPECTED_EXTERNAL_FILES = 361

LAT = np.linspace(-90.0, 90.0, H, dtype=np.float32)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float32)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)

REGIONS = {
    "North_America": {"lat": (15.0, 75.0), "lon": (-170.0, -50.0)},
    "Europe": {"lat": (35.0, 72.0), "lon": (-10.0, 40.0)},
    "East_Asia": {"lat": (20.0, 55.0), "lon": (100.0, 145.0)},
    "South_Asia": {"lat": (5.0, 35.0), "lon": (65.0, 100.0)},
    "Africa": {"lat": (-35.0, 37.0), "lon": (-20.0, 55.0)},
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


def import_training_module(path: Path) -> ModuleType:
    require(path.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {path}")
    spec = importlib.util.spec_from_file_location(
        "clean_anomaly_formal_eval_training_module",
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


def finite_float(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise RuntimeError(f"MONTH_OUT_OF_RANGE: {month}")


class RegressionMoments:
    def __init__(self) -> None:
        self.n = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_error = 0.0
        self.sum_squared_error = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        require(pred.shape == true.shape,
                f"REGRESSION_SHAPE_MISMATCH: {pred.shape} vs {true.shape}")
        if pred.size == 0:
            return
        p = pred.astype(np.float64, copy=False).ravel()
        t = true.astype(np.float64, copy=False).ravel()
        error = p - t

        self.n += int(p.size)
        self.sum_pred += float(p.sum())
        self.sum_true += float(t.sum())
        self.sum_pred2 += float(np.dot(p, p))
        self.sum_true2 += float(np.dot(t, t))
        self.sum_pred_true += float(np.dot(p, t))
        self.sum_error += float(error.sum())
        self.sum_squared_error += float(np.dot(error, error))

    def finalize(self) -> dict[str, Any]:
        require(self.n > 0, "EMPTY_REGRESSION_MOMENTS")
        n = float(self.n)
        true_mean = self.sum_true / n
        pred_mean = self.sum_pred / n
        true_sst = self.sum_true2 - (self.sum_true * self.sum_true) / n
        pred_ss = self.sum_pred2 - (self.sum_pred * self.sum_pred) / n
        covariance = self.sum_pred_true - (
            self.sum_pred * self.sum_true
        ) / n

        mse = self.sum_squared_error / n
        rmse = math.sqrt(max(mse, 0.0))
        bias = self.sum_error / n
        r2 = (
            0.0
            if true_sst <= 1e-12
            else 1.0 - self.sum_squared_error / true_sst
        )
        corr_denom = math.sqrt(max(pred_ss, 0.0) * max(true_sst, 0.0))
        corr_defined = corr_denom > 1e-12
        corr = covariance / corr_denom if corr_defined else 0.0
        true_std = math.sqrt(max(true_sst / n, 0.0))
        pred_std = math.sqrt(max(pred_ss / n, 0.0))

        values = {
            "n_values": self.n,
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "true_std": true_std,
            "pred_std": pred_std,
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "corr": corr,
            "corr_defined": corr_defined,
        }
        for key, value in values.items():
            if key not in {"n_values", "corr_defined"}:
                finite_float(value, f"REGRESSION_{key}")
        return values


class EvaluationBundle:
    def __init__(self) -> None:
        self.model_anomaly = RegressionMoments()
        self.zero_anomaly = RegressionMoments()
        self.model_reconstruction = RegressionMoments()
        self.month_background = RegressionMoments()
        self.valid_days = 0
        self.coverage_sum = 0.0

    def update(
        self,
        pred_anomaly: np.ndarray,
        true_anomaly: np.ndarray,
        pred_daily: np.ndarray,
        true_daily: np.ndarray,
        month_background: np.ndarray,
        coverage_fraction: float,
    ) -> None:
        require(pred_anomaly.size > 0, "EMPTY_BUNDLE_UPDATE")
        self.model_anomaly.update(pred_anomaly, true_anomaly)
        self.zero_anomaly.update(np.zeros_like(true_anomaly), true_anomaly)
        self.model_reconstruction.update(pred_daily, true_daily)
        self.month_background.update(month_background, true_daily)
        self.valid_days += 1
        self.coverage_sum += float(coverage_fraction)

    def finalize(self) -> dict[str, Any]:
        require(self.valid_days > 0, "EMPTY_EVALUATION_BUNDLE")
        model_anom = self.model_anomaly.finalize()
        zero_anom = self.zero_anomaly.finalize()
        model_recon = self.model_reconstruction.finalize()
        background = self.month_background.finalize()

        zero_mse = zero_anom["mse"]
        background_mse = background["mse"]
        require(zero_mse > 0.0, "ZERO_ANOMALY_BASELINE_MSE_ZERO")
        require(background_mse > 0.0, "MONTH_BACKGROUND_BASELINE_MSE_ZERO")

        anomaly_skill = 1.0 - model_anom["mse"] / zero_mse
        reconstruction_skill = 1.0 - model_recon["mse"] / background_mse
        r2_delta = model_recon["r2"] - background["r2"]

        return {
            "valid_days": self.valid_days,
            "obs_coverage_mean": self.coverage_sum / self.valid_days,
            "model_anomaly": model_anom,
            "zero_anomaly_baseline": zero_anom,
            "anomaly_mse_skill_vs_zero": finite_float(
                anomaly_skill,
                "ANOMALY_SKILL_VS_ZERO",
            ),
            "anomaly_rmse_reduction_vs_zero": finite_float(
                zero_anom["rmse"] - model_anom["rmse"],
                "ANOMALY_RMSE_REDUCTION",
            ),
            "model_reconstruction": model_recon,
            "month_background_baseline": background,
            "reconstruction_mse_skill_vs_month_background": finite_float(
                reconstruction_skill,
                "RECONSTRUCTION_SKILL_VS_BACKGROUND",
            ),
            "reconstruction_r2_delta_vs_month_background": finite_float(
                r2_delta,
                "RECONSTRUCTION_R2_DELTA",
            ),
        }


def build_region_masks() -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, definition in REGIONS.items():
        lat_min, lat_max = definition["lat"]
        lon_min, lon_max = definition["lon"]
        mask = (
            (LAT_GRID >= lat_min)
            & (LAT_GRID <= lat_max)
            & (LON_GRID >= lon_min)
            & (LON_GRID <= lon_max)
        )
        require(mask.shape == (H, W), f"REGION_MASK_SHAPE_FAIL: {name}")
        require(bool(mask.any()), f"REGION_MASK_EMPTY: {name}")
        masks[name] = mask
    return masks


@torch.no_grad()
def evaluate_split(
    *,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    reader: Any,
    spec_obj: Any,
    device: torch.device,
    split_label: str,
) -> dict[str, Any]:
    require(entries, f"EMPTY_EVALUATION_SPLIT: {split_label}")
    model.eval()

    global_bundle = EvaluationBundle()
    month_bundles: dict[int, EvaluationBundle] = defaultdict(EvaluationBundle)
    season_bundles: dict[str, EvaluationBundle] = defaultdict(EvaluationBundle)
    region_bundles: dict[str, EvaluationBundle] = {
        name: EvaluationBundle()
        for name in REGIONS
    }
    region_masks = build_region_masks()

    for index, entry in enumerate(entries, start=1):
        example = reader.load_example(entry, spec_obj)
        x = example["x"].unsqueeze(0).to(device, non_blocking=True)
        true_anomaly_t = example["target"]
        mask_t = example["mask"]
        background_t = example["target_background"]

        pred_anomaly_t = model(x).detach().cpu().squeeze(0)
        valid = (mask_t > 0.5).squeeze(0).numpy()
        require(int(valid.sum()) >= 16,
                f"TOO_FEW_VALID_PIXELS: {entry['file']}")

        true_anomaly_grid = true_anomaly_t.squeeze(0).numpy()
        pred_anomaly_grid = pred_anomaly_t.squeeze(0).numpy()
        background_grid = background_t.squeeze(0).numpy()
        true_daily_grid = true_anomaly_grid + background_grid
        pred_daily_grid = pred_anomaly_grid + background_grid

        coverage = float(valid.mean())
        month = int(entry["month"])
        season = str(entry.get("season", month_to_season(month)))
        require(season == month_to_season(month),
                f"SEASON_MONTH_MISMATCH: {entry['file']}")

        def update_bundle(bundle: EvaluationBundle, subset: np.ndarray) -> None:
            require(subset.shape == (H, W), "SUBSET_MASK_SHAPE_FAIL")
            if int(subset.sum()) < 16:
                return
            bundle.update(
                pred_anomaly_grid[subset],
                true_anomaly_grid[subset],
                pred_daily_grid[subset],
                true_daily_grid[subset],
                background_grid[subset],
                float(subset.mean()),
            )

        update_bundle(global_bundle, valid)
        update_bundle(month_bundles[month], valid)
        update_bundle(season_bundles[season], valid)

        for region_name, region_mask in region_masks.items():
            update_bundle(region_bundles[region_name], valid & region_mask)

        if index == len(entries) or index % 20 == 0:
            print(
                f"{split_label}_EVALUATED_FILES={index}/{len(entries)}",
                flush=True,
            )

        del x, pred_anomaly_t

    return {
        "split": split_label,
        "n_manifest_files": len(entries),
        "global": global_bundle.finalize(),
        "by_month": {
            str(month): month_bundles[month].finalize()
            for month in sorted(month_bundles)
            if month_bundles[month].valid_days > 0
        },
        "by_season": {
            season: season_bundles[season].finalize()
            for season in sorted(season_bundles)
            if season_bundles[season].valid_days > 0
        },
        "by_region": {
            name: region_bundles[name].finalize()
            for name in REGIONS
            if region_bundles[name].valid_days > 0
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one clean anomaly meteo-only formal run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", help="Relocated restricted anomaly cache; overrides the historical manifest path.")
    parser.add_argument("--aef-source-root-2022")
    parser.add_argument("--aef-source-root-2023")
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--formal-audit", required=True)
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
    formal_audit_path = Path(args.formal_audit).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND: {run_dir}")
    require(training_script.is_file(),
            f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(formal_audit_path.is_file(),
            f"FORMAL_AUDIT_NOT_FOUND: {formal_audit_path}")
    require(not output_dir.exists(),
            f"STOP_OUTPUT_DIR_EXISTS: {output_dir}")

    manifest_path = run_dir / "logs" / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    manifest = load_json(manifest_path)
    formal_audit = load_json(formal_audit_path)

    require(formal_audit.get("artifact_name") ==
            "clean_anomaly_formal_run_audit",
            "FORMAL_AUDIT_ARTIFACT_MISMATCH")
    require(formal_audit.get("verification", {}).get("overall_pass") is True,
            "FORMAL_AUDIT_NOT_PASS")
    require(formal_audit.get("decision") ==
            f"PASS_PROCEED_TO_SEED{expected_seed}_EXTERNAL_EVALUATION",
            "FORMAL_AUDIT_DECISION_MISMATCH")
    require(canonical(formal_audit.get("run_dir", "")) == canonical(run_dir),
            "FORMAL_AUDIT_RUN_DIR_MISMATCH")
    require(formal_audit.get("experiment") == EXPECTED_EXPERIMENT,
            "FORMAL_AUDIT_EXPERIMENT_MISMATCH")
    require(int(formal_audit.get("seed", -1)) == expected_seed,
            "FORMAL_AUDIT_SEED_MISMATCH")

    audit_provenance = formal_audit.get("input_provenance", {})
    require(
        audit_provenance.get("run_manifest", {}).get("sha256") ==
        sha256_file(manifest_path),
        "RUN_MANIFEST_CHANGED_AFTER_FORMAL_AUDIT",
    )
    require(
        audit_provenance.get("checkpoint", {}).get("sha256") ==
        sha256_file(checkpoint_path),
        "CHECKPOINT_CHANGED_AFTER_FORMAL_AUDIT",
    )
    require(
        audit_provenance.get("training_script", {}).get("sha256") ==
        sha256_file(training_script),
        "TRAINING_SCRIPT_CHANGED_AFTER_FORMAL_AUDIT",
    )

    require(manifest.get("experiment") == EXPECTED_EXPERIMENT,
            "RUN_MANIFEST_EXPERIMENT_MISMATCH")
    require(manifest.get("run_mode") == "formal",
            "RUN_MANIFEST_MODE_MISMATCH")
    require(int(manifest.get("seed", -1)) == expected_seed,
            "RUN_MANIFEST_SEED_MISMATCH")
    require(len(manifest.get("val_entries", [])) == EXPECTED_VAL_FILES,
            "VAL_ENTRY_COUNT_MISMATCH")
    require(len(manifest.get("external_entries", [])) ==
            EXPECTED_EXTERNAL_FILES,
            "EXTERNAL_ENTRY_COUNT_MISMATCH")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(checkpoint.get("artifact_name") ==
            "clean_anomaly_best_checkpoint",
            "CHECKPOINT_ARTIFACT_MISMATCH")
    require(int(checkpoint.get("seed", -1)) == expected_seed,
            "CHECKPOINT_SEED_MISMATCH")
    require(checkpoint.get("spec", {}).get("name") ==
            EXPECTED_EXPERIMENT,
            "CHECKPOINT_EXPERIMENT_MISMATCH")
    require(int(checkpoint.get("trainable_parameters", -1)) ==
            EXPECTED_PARAMETERS,
            "CHECKPOINT_PARAMETER_COUNT_MISMATCH")

    module = import_training_module(training_script)
    spec_obj = module.AnomalySpec(**checkpoint["spec"])
    model = module.AnomalyUNet(
        spec_obj,
        base_features=int(checkpoint.get("base_features", 32)),
    )
    incompatible = model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
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
            "SEED42_EVALUATION_REQUIRES_CUDA")
    model = model.to(device)

    cache_dir = (Path(args.cache_dir).expanduser().resolve() if args.cache_dir else Path(manifest["cache_dir"]).resolve())
    reader = module.LeanAnomalyReader(
        cache_dir,
        aef_source_root_2022=(None if args.aef_source_root_2022 is None else Path(args.aef_source_root_2022).expanduser()),
        aef_source_root_2023=(None if args.aef_source_root_2023 is None else Path(args.aef_source_root_2023).expanduser()),
    )

    print("=" * 96, flush=True)
    print("Clean anomaly best-checkpoint evaluation", flush=True)
    print("=" * 96, flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"EXPERIMENT={EXPECTED_EXPERIMENT}", flush=True)
    print(f"SEED={expected_seed}", flush=True)
    print(f"BEST_EPOCH={checkpoint['best_epoch']}", flush=True)
    print(f"DEVICE={device.type}", flush=True)
    print(f"VAL_FILES={EXPECTED_VAL_FILES}", flush=True)
    print(f"EXTERNAL_FILES={EXPECTED_EXTERNAL_FILES}", flush=True)
    print("EMISSION_INPUT_USED=False", flush=True)
    print("EXTERNAL_2023_TYPE=ordinary_external_anomaly", flush=True)
    print("RECONSTRUCTION_BACKGROUND_BASELINE_REQUIRED=True", flush=True)
    print(flush=True)

    val_result = evaluate_split(
        model=model,
        entries=list(manifest["val_entries"]),
        reader=reader,
        spec_obj=spec_obj,
        device=device,
        split_label="2022_val",
    )
    external_result = evaluate_split(
        model=model,
        entries=list(manifest["external_entries"]),
        reader=reader,
        spec_obj=spec_obj,
        device=device,
        split_label="2023_external",
    )

    audited_val_r2 = finite_float(
        formal_audit["metric_reproduction"]["recomputed_log_r2_anom"],
        "AUDITED_VAL_R2",
    )
    evaluated_val_r2 = finite_float(
        val_result["global"]["model_anomaly"]["r2"],
        "EVALUATED_VAL_R2",
    )
    require(
        math.isclose(
            audited_val_r2,
            evaluated_val_r2,
            rel_tol=0.0,
            abs_tol=args.metric_tolerance,
        ),
        f"VALIDATION_METRIC_MISMATCH: "
        f"{evaluated_val_r2} != {audited_val_r2}",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "eval_report.json"
    txt_path = output_dir / "eval_summary.txt"

    report = {
        "artifact_name": "clean_anomaly_formal_run_evaluation",
        "artifact_version": 1,
        "run_dir": str(run_dir),
        "experiment": EXPECTED_EXPERIMENT,
        "seed": expected_seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "input_provenance": {
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "formal_audit": {
                "path": str(formal_audit_path),
                "sha256": sha256_file(formal_audit_path),
            },
            "training_script": {
                "path": str(training_script),
                "sha256": sha256_file(training_script),
            },
            "evaluation_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "evaluation_contract": {
            "2022_validation_type":
                "held-out_model-selection_validation",
            "2023_evaluation_type":
                "ordinary_external_anomaly_evaluation",
            "emission_input_used": False,
            "primary_metrics": [
                "model_anomaly.r2",
                "model_anomaly.rmse",
                "model_anomaly.corr",
                "anomaly_mse_skill_vs_zero",
            ],
            "reconstruction_interpretation":
                "must_be_compared_with_month_background_baseline",
            "grouped_diagnostics": [
                "month",
                "season",
                "fixed_region",
            ],
        },
        "results": {
            "2022_val": val_result,
            "2023_external": external_result,
        },
        "verification": {
            "formal_audit_provenance_pass": True,
            "checkpoint_strict_load_pass": True,
            "validation_metric_matches_formal_audit": True,
            "zero_anomaly_baseline_reported": True,
            "month_background_reconstruction_baseline_reported": True,
            "grouped_diagnostics_reported": True,
            "single_seed_only": True,
            "statistical_significance_established": False,
            "paper_ready": False,
            "overall_pass": True,
        },
        "decision_boundary": (
            "Use this result only to decide whether the clean meteo-only "
            "configuration is technically viable for completing seeds 43/44. "
            "Do not make a final scientific claim from one seed."
        ),
    }
    save_json(report, json_path)

    val_global = val_result["global"]
    ext_global = external_result["global"]

    lines = [
        f"GATE_A8_CLEAN_ANOMALY_METEO_SEED{expected_seed}_EVALUATION_PASS",
        f"RUN_DIR={run_dir}",
        f"EXPERIMENT={EXPECTED_EXPERIMENT}",
        f"SEED={expected_seed}",
        f"BEST_EPOCH={checkpoint['best_epoch']}",
        "2022_VALIDATION_TYPE=held_out_model_selection_validation",
        f"VAL_2022_ANOM_LOG_R2={val_global['model_anomaly']['r2']:.8f}",
        f"VAL_2022_ANOM_LOG_RMSE={val_global['model_anomaly']['rmse']:.8f}",
        f"VAL_2022_ANOM_LOG_BIAS={val_global['model_anomaly']['bias']:.8f}",
        f"VAL_2022_ANOM_LOG_CORR={val_global['model_anomaly']['corr']:.8f}",
        f"VAL_2022_ZERO_ANOM_RMSE={val_global['zero_anomaly_baseline']['rmse']:.8f}",
        f"VAL_2022_ANOM_MSE_SKILL_VS_ZERO={val_global['anomaly_mse_skill_vs_zero']:.8f}",
        f"VAL_2022_RECON_LOG_R2={val_global['model_reconstruction']['r2']:.8f}",
        f"VAL_2022_MONTH_BACKGROUND_LOG_R2={val_global['month_background_baseline']['r2']:.8f}",
        f"VAL_2022_RECON_R2_DELTA_VS_BACKGROUND={val_global['reconstruction_r2_delta_vs_month_background']:.8f}",
        "EXTERNAL_2023_TYPE=ordinary_external_anomaly",
        "EXTERNAL_2023_EMISSION_INPUT_USED=False",
        f"EXTERNAL_2023_ANOM_LOG_R2={ext_global['model_anomaly']['r2']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_RMSE={ext_global['model_anomaly']['rmse']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_BIAS={ext_global['model_anomaly']['bias']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_CORR={ext_global['model_anomaly']['corr']:.8f}",
        f"EXTERNAL_2023_ZERO_ANOM_RMSE={ext_global['zero_anomaly_baseline']['rmse']:.8f}",
        f"EXTERNAL_2023_ANOM_MSE_SKILL_VS_ZERO={ext_global['anomaly_mse_skill_vs_zero']:.8f}",
        f"EXTERNAL_2023_RECON_LOG_R2={ext_global['model_reconstruction']['r2']:.8f}",
        f"EXTERNAL_2023_MONTH_BACKGROUND_LOG_R2={ext_global['month_background_baseline']['r2']:.8f}",
        f"EXTERNAL_2023_RECON_R2_DELTA_VS_BACKGROUND={ext_global['reconstruction_r2_delta_vs_month_background']:.8f}",
        "VALIDATION_METRIC_MATCHES_FORMAL_AUDIT=True",
        "ZERO_ANOMALY_BASELINE_REPORTED=True",
        "MONTH_BACKGROUND_RECONSTRUCTION_BASELINE_REPORTED=True",
        "GROUPED_MONTH_SEASON_REGION_DIAGNOSTICS_REPORTED=True",
        "SINGLE_SEED_ONLY=True",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
        "PAPER_READY=False",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print()
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
