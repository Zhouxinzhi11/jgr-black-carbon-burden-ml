# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
Read-only physical-scale sensitivity audit for the formal clean mean-state runs.

The script reuses the frozen checkpoints, cached monthly payloads, input builder,
and model class from 52_eval_s2_meanstate_clean_formal_run.py. It first verifies
the provenance hashes and reproduces the existing pooled log-space metrics. It
then applies

    C_star = C0 * expm1(Y),  C0 = 1e-9 kg m^-2

to paired predictions and targets on the unchanged valid mask and reports
physical-scale RMSE, bias, MAE, and median absolute error. C_star is the
back-transformed monthly mean-state target; because the cached target is the
monthly mean of daily log-transformed burdens, C_star is not the arithmetic
monthly mean of daily burden.

No checkpoint, cache, manifest, audit, or earlier evaluation report is modified.
The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch


ARTIFACT_NAME = "meanstate_physical_scale_sensitivity_audit"
ARTIFACT_VERSION = 1
C0_KG_M2 = 1.0e-9
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_EXPERIMENTS = (
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
)
RUN_NAME_TEMPLATES = {
    "mean_emis_only": "mean_emis_only_seed{seed}",
    "mean_meteo_clim_only": "mean_meteo_clim_only_seed{seed}",
    "mean_emis_meteo_clim": "mean_emis_meteo_clim_seed{seed}",
    "mean_emis_meteo_clim_shuffled": (
        "mean_emis_meteo_clim_shuffled_seed{seed}"
    ),
}
SPLIT_DEFINITIONS = (
    (
        "2022_validation",
        "val_entries",
        "eval_2022_validation",
        "held-out model-selection validation",
    ),
    (
        "2023_external",
        "external_2023_entries",
        "eval_2023_external",
        "cross-year external evaluation",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_strict(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def import_formal_evaluator(path: Path) -> Any:
    require(path.is_file(), f"Formal evaluator not found: {path}")
    module_name = "formal_meanstate_evaluator_52"
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "Import spec failed.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required_names = (
        "MeanStateSpec",
        "MeanStateUNet",
        "build_input_from_payload",
        "build_proxy_emission_baseline",
        "load_checkpoint",
        "load_pt",
        "pooled_metrics",
    )
    missing = [name for name in required_names if not hasattr(module, name)]
    require(not missing, f"Formal evaluator lacks required objects: {missing}")
    return module


def find_reference_report(run_dir: Path) -> Path:
    preferred = run_dir / "eval_clean_generic" / "eval_report.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(run_dir.glob("eval*/eval_report.json"))
    accepted: list[Path] = []
    for candidate in candidates:
        try:
            report = load_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            report.get("artifact_name")
            == "s2_meanstate_clean_formal_evaluation_generic"
            and report.get("verification", {}).get("overall_pass") is True
        ):
            accepted.append(candidate)
    require(
        len(accepted) == 1,
        "Could not resolve exactly one generic formal evaluation report in "
        f"{run_dir}; accepted candidates={accepted}",
    )
    return accepted[0]


def verify_file_hash(
    path: Path,
    expected_sha256: str,
    label: str,
) -> str:
    require(path.is_file() and path.stat().st_size > 0, f"Missing {label}: {path}")
    actual = sha256_file(path)
    require(
        actual == expected_sha256,
        f"{label} hash differs from the sealed reference report: {path}",
    )
    return actual


def resolve_and_verify_run(
    project_root: Path,
    experiment: str,
    seed: int,
) -> dict[str, Any]:
    run_name = RUN_NAME_TEMPLATES[experiment].format(seed=seed)
    run_dir = (
        project_root
        / "runs_s2_meanstate_clean"
        / "formal"
        / f"seed{seed}"
        / run_name
    ).resolve()
    require(run_dir.is_dir(), f"Formal run directory not found: {run_dir}")

    reference_report_path = find_reference_report(run_dir)
    reference = load_json(reference_report_path)
    require(
        reference.get("verification", {}).get("overall_pass") is True,
        f"Reference evaluation did not pass: {reference_report_path}",
    )
    require(reference.get("seed") == seed, "Reference seed mismatch.")
    require(
        reference.get("experiment", {}).get("name") == experiment,
        "Reference experiment mismatch.",
    )
    require(
        normalized_path(reference.get("run_dir", ""))
        == normalized_path(run_dir),
        "Reference run_dir mismatch.",
    )

    manifest_path = run_dir / "logs" / "run_manifest.json"
    history_path = run_dir / "logs" / "training_history.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    provenance = reference.get("provenance", {})
    verify_file_hash(
        manifest_path,
        str(provenance.get("run_manifest_sha256", "")),
        "run manifest",
    )
    verify_file_hash(
        history_path,
        str(provenance.get("training_history_sha256", "")),
        "training history",
    )
    verify_file_hash(
        checkpoint_path,
        str(reference.get("checkpoint", {}).get("sha256", "")),
        "checkpoint",
    )

    audit_path = Path(str(provenance.get("training_audit", ""))).expanduser()
    if not audit_path.is_file():
        audit_path = (
            project_root
            / "sanity_reports"
            / "meanstate_clean_formal_audits"
            / run_name
            / "training_audit.json"
        )
    verify_file_hash(
        audit_path,
        str(provenance.get("training_audit_sha256", "")),
        "training audit",
    )

    manifest = load_json(manifest_path)
    require(manifest.get("run_mode") == "formal", "Manifest is not formal.")
    require(
        manifest.get("experiment", {}).get("name") == experiment,
        "Manifest experiment mismatch.",
    )
    require(
        int(manifest.get("settings", {}).get("seed", -1)) == seed,
        "Manifest seed mismatch.",
    )
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "history_path": history_path,
        "checkpoint_path": checkpoint_path,
        "audit_path": audit_path,
        "reference_report_path": reference_report_path,
        "manifest": manifest,
        "reference": reference,
    }


def physical_metrics(
    prediction_log: torch.Tensor,
    target_log: torch.Tensor,
) -> dict[str, float | int]:
    pred_log = prediction_log.reshape(-1).detach().cpu().to(torch.float64)
    targ_log = target_log.reshape(-1).detach().cpu().to(torch.float64)
    require(pred_log.numel() > 1, "Metric vector has <=1 element.")
    require(pred_log.numel() == targ_log.numel(), "Metric lengths differ.")
    require(bool(torch.isfinite(pred_log).all()), "Prediction log vector non-finite.")
    require(bool(torch.isfinite(targ_log).all()), "Target log vector non-finite.")

    pred = torch.expm1(pred_log) * C0_KG_M2
    targ = torch.expm1(targ_log) * C0_KG_M2
    require(bool(torch.isfinite(pred).all()), "Back-transformed prediction non-finite.")
    require(bool(torch.isfinite(targ).all()), "Back-transformed target non-finite.")

    error = pred - targ
    abs_error = torch.abs(error)
    ss_res = torch.sum(error * error)
    ss_tot = torch.sum((targ - torch.mean(targ)) ** 2)
    r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1.0e-300))
    rmse = float(torch.sqrt(torch.mean(error * error)))
    bias = float(torch.mean(error))
    mae = float(torch.mean(abs_error))
    median_ae = float(torch.median(abs_error))
    values = (r2, rmse, bias, mae, median_ae)
    require(all(math.isfinite(value) for value in values), "Physical metric non-finite.")

    return {
        "n": int(pred.numel()),
        "physical_r2": r2,
        "physical_rmse_kg_m2": rmse,
        "physical_bias_kg_m2": bias,
        "physical_mae_kg_m2": mae,
        "physical_median_ae_kg_m2": median_ae,
        "physical_rmse_ug_m2": rmse * 1.0e9,
        "physical_bias_ug_m2": bias * 1.0e9,
        "physical_mae_ug_m2": mae * 1.0e9,
        "physical_median_ae_ug_m2": median_ae * 1.0e9,
        "prediction_log_min": float(torch.min(pred_log)),
        "prediction_log_max": float(torch.max(pred_log)),
        "target_log_min": float(torch.min(targ_log)),
        "target_log_max": float(torch.max(targ_log)),
        "negative_back_transformed_predictions": int(torch.count_nonzero(pred < 0)),
        "negative_back_transformed_targets": int(torch.count_nonzero(targ < 0)),
    }


def compare_log_metrics(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float,
    context: str,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in ("log_r2", "log_rmse", "log_bias"):
        actual_value = float(actual[key])
        expected_value = float(expected[key])
        delta = abs(actual_value - expected_value)
        require(
            delta <= tolerance,
            f"{context}: {key} reproduction failed; actual={actual_value}, "
            f"expected={expected_value}, abs_delta={delta}, tolerance={tolerance}",
        )
        deltas[key] = delta
    return deltas


@torch.no_grad()
def evaluate_split(
    formal: Any,
    model: torch.nn.Module,
    entries: Sequence[dict[str, Any]],
    spec: Any,
    device: torch.device,
    proxy_emission: Optional[torch.Tensor],
    expected_split: dict[str, Any],
    reproduction_tolerance: float,
    experiment: str,
    seed: int,
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    pooled_predictions: list[torch.Tensor] = []
    pooled_targets: list[torch.Tensor] = []
    month_rows: list[dict[str, Any]] = []
    expected_groups = {
        int(item["month"]): item
        for item in expected_split.get("per_group", [])
    }
    require(
        len(expected_groups) == len(entries),
        f"{experiment}/seed{seed}/{split_name}: reference month count mismatch.",
    )

    for entry in entries:
        payload_path = Path(str(entry["file"])).expanduser().resolve()
        require(payload_path.is_file(), f"Monthly payload not found: {payload_path}")
        payload = formal.load_pt(payload_path)
        x = formal.build_input_from_payload(
            payload,
            spec,
            key=payload_path.name,
            proxy_emission_log_mean=proxy_emission,
        ).unsqueeze(0).to(device, non_blocking=True)
        prediction = (
            model(x).squeeze(0).squeeze(0).detach().cpu().float()
        )
        target = payload["target_log_mean_conc"].detach().cpu().float()
        mask = payload["valid_mask"].detach().cpu().float() > 0.5
        require(prediction.shape == target.shape == mask.shape, "Grid shape mismatch.")
        require(bool(torch.isfinite(prediction).all()), "Prediction grid non-finite.")
        require(bool(torch.isfinite(target).all()), "Target grid non-finite.")

        pred_valid = prediction[mask]
        targ_valid = target[mask]
        month = int(entry["month"])
        expected_group = expected_groups.get(month)
        require(expected_group is not None, f"Missing reference month {month}.")
        require(
            int(pred_valid.numel()) == int(expected_group["n_valid_pixels"]),
            f"{experiment}/seed{seed}/{split_name}/{month:02d}: valid-count mismatch.",
        )
        log_result = formal.pooled_metrics(pred_valid, targ_valid)
        log_deltas = compare_log_metrics(
            log_result,
            expected_group,
            reproduction_tolerance,
            f"{experiment}/seed{seed}/{split_name}/{month:02d}",
        )
        physical = physical_metrics(pred_valid, targ_valid)
        month_rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "split": split_name,
                "year": int(entry["year"]),
                "month": month,
                "n_valid": int(pred_valid.numel()),
                **log_result,
                **physical,
                "max_log_metric_reproduction_abs_delta": max(log_deltas.values()),
            }
        )
        pooled_predictions.append(pred_valid)
        pooled_targets.append(targ_valid)
        print(
            f"MONTH_PASS experiment={experiment} seed={seed} split={split_name} "
            f"month={month:02d} n={pred_valid.numel()}",
            flush=True,
        )

    all_prediction = torch.cat(pooled_predictions)
    all_target = torch.cat(pooled_targets)
    log_result = formal.pooled_metrics(all_prediction, all_target)
    expected_n = int(expected_split["n_total_valid_pixels"])
    require(
        int(all_prediction.numel()) == expected_n,
        f"{experiment}/seed{seed}/{split_name}: pooled valid-count mismatch.",
    )
    log_deltas = compare_log_metrics(
        log_result,
        expected_split["global"],
        reproduction_tolerance,
        f"{experiment}/seed{seed}/{split_name}/pooled",
    )
    pooled_result = {
        "experiment": experiment,
        "seed": seed,
        "split": split_name,
        "n_groups": len(entries),
        "n_valid": int(all_prediction.numel()),
        **log_result,
        **physical_metrics(all_prediction, all_target),
        "max_log_metric_reproduction_abs_delta": max(log_deltas.values()),
        "log_metric_reproduction_pass": True,
    }
    return pooled_result, month_rows


def sample_mean_sd(values: Iterable[float]) -> tuple[float, float]:
    clean = [float(value) for value in values]
    require(len(clean) >= 2, "At least two seeds are required for sample SD.")
    return statistics.mean(clean), statistics.stdev(clean)


def build_multiseed_summary(
    per_seed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        buckets[(str(row["experiment"]), str(row["split"]))].append(row)

    metric_names = (
        "log_r2",
        "log_rmse",
        "log_bias",
        "physical_r2",
        "physical_rmse_kg_m2",
        "physical_bias_kg_m2",
        "physical_mae_kg_m2",
        "physical_median_ae_kg_m2",
        "physical_rmse_ug_m2",
        "physical_bias_ug_m2",
        "physical_mae_ug_m2",
        "physical_median_ae_ug_m2",
    )
    summary: list[dict[str, Any]] = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in rows]
        require(len(set(seeds)) == len(DEFAULT_SEEDS), f"Incomplete seeds for {key}.")
        require(
            len({int(row["n_valid"]) for row in rows}) == 1,
            f"Valid sample counts differ by seed for {key}.",
        )
        item: dict[str, Any] = {
            "experiment": key[0],
            "split": key[1],
            "seeds": ",".join(str(seed) for seed in seeds),
            "n_seeds": len(rows),
            "n_valid_per_seed": int(rows[0]["n_valid"]),
        }
        for metric in metric_names:
            mean, sample_sd = sample_mean_sd(row[metric] for row in rows)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sample_sd"] = sample_sd
        item["negative_back_transformed_predictions_total_across_seeds"] = sum(
            int(row["negative_back_transformed_predictions"]) for row in rows
        )
        item["negative_back_transformed_targets_total_across_seeds"] = sum(
            int(row["negative_back_transformed_targets"]) for row in rows
        )
        summary.append(item)
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows for CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_text_summary(
    per_seed: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    device: torch.device,
) -> str:
    lines = [
        "Formal mean-state physical-scale sensitivity audit",
        f"Status: PASS",
        f"Device: {device}",
        f"C0: {C0_KG_M2:.1e} kg m^-2",
        (
            "Physical target: C_star = C0 * expm1(monthly mean of daily "
            "log-transformed burden); not arithmetic monthly mean burden."
        ),
        "Mask/pooling: unchanged formal valid masks; all valid pairs pooled per split.",
        "Checkpoint selection: unchanged; no retraining or physical-scale reselection.",
        f"Formal runs evaluated: {len(per_seed)}",
        "",
        "Three-seed pooled summaries:",
    ]
    for row in summary:
        lines.extend(
            [
                (
                    f"- {row['experiment']} | {row['split']} | "
                    f"N={row['n_valid_per_seed']}"
                ),
                (
                    "  physical RMSE (kg m^-2): "
                    f"{row['physical_rmse_kg_m2_mean']:.8e} ± "
                    f"{row['physical_rmse_kg_m2_sample_sd']:.8e}"
                ),
                (
                    "  physical bias (kg m^-2): "
                    f"{row['physical_bias_kg_m2_mean']:.8e} ± "
                    f"{row['physical_bias_kg_m2_sample_sd']:.8e}"
                ),
                (
                    "  physical MAE (kg m^-2): "
                    f"{row['physical_mae_kg_m2_mean']:.8e} ± "
                    f"{row['physical_mae_kg_m2_sample_sd']:.8e}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "LOG_METRIC_REPRODUCTION_PASS=True",
            "PROVENANCE_HASH_CHECKS_PASS=True",
            "NO_TRAINING_OR_ARTIFACT_MODIFICATION=True",
            "OVERALL_PASS=True",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce formal log-space metrics and compute paired "
            "back-transformed physical-scale sensitivity metrics."
        )
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Directory containing 52_eval...py and runs_s2_meanstate_clean.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--formal-evaluator",
        default=None,
        help="Optional explicit path to 52_eval_s2_meanstate_clean_formal_run.py.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        choices=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        choices=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--metric-reproduction-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require(project_root.is_dir(), f"Project root not found: {project_root}")
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    require(
        tuple(args.seeds) == DEFAULT_SEEDS,
        "Final audit requires exactly seeds 42, 43, and 44 in this order.",
    )
    require(
        tuple(args.experiments) == DEFAULT_EXPERIMENTS,
        "Final audit requires all four formal configurations in the default order.",
    )
    require(args.metric_reproduction_tolerance > 0, "Tolerance must be positive.")

    evaluator_path = (
        Path(args.formal_evaluator).expanduser().resolve()
        if args.formal_evaluator
        else project_root / "52_eval_s2_meanstate_clean_formal_run.py"
    )
    formal = import_formal_evaluator(evaluator_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA was required but is unavailable.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    per_seed_rows: list[dict[str, Any]] = []
    per_month_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    print(f"DEVICE={device}", flush=True)
    print(f"FORMAL_EVALUATOR={evaluator_path}", flush=True)
    for experiment in args.experiments:
        for seed in args.seeds:
            print(
                f"RUN_START experiment={experiment} seed={seed}",
                flush=True,
            )
            resolved = resolve_and_verify_run(project_root, experiment, seed)
            manifest = resolved["manifest"]
            reference = resolved["reference"]
            spec = formal.MeanStateSpec(**manifest["experiment"])
            model = formal.MeanStateUNet(
                spec=spec,
                base_features=int(manifest["settings"]["base_features"]),
            ).to(device)
            checkpoint = formal.load_checkpoint(resolved["checkpoint_path"], device)
            require(
                checkpoint.get("experiment") == experiment,
                "Checkpoint experiment mismatch.",
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            train_entries = list(manifest["train_entries"])
            if spec.uses_emission:
                proxy_emission, proxy_diagnostics = (
                    formal.build_proxy_emission_baseline(train_entries)
                )
            else:
                proxy_emission = None
                proxy_diagnostics = None

            for (
                split_name,
                manifest_entry_key,
                reference_entry_key,
                split_status,
            ) in SPLIT_DEFINITIONS:
                split_proxy = proxy_emission if split_name == "2023_external" else None
                pooled, months = evaluate_split(
                    formal=formal,
                    model=model,
                    entries=list(manifest[manifest_entry_key]),
                    spec=spec,
                    device=device,
                    proxy_emission=split_proxy,
                    expected_split=reference[reference_entry_key],
                    reproduction_tolerance=float(args.metric_reproduction_tolerance),
                    experiment=experiment,
                    seed=seed,
                    split_name=split_name,
                )
                pooled["split_status"] = split_status
                pooled["emission_proxy_used"] = bool(
                    split_name == "2023_external" and spec.uses_emission
                )
                per_seed_rows.append(pooled)
                per_month_rows.extend(months)

            run_records.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "run_dir": str(resolved["run_dir"]),
                    "reference_report": str(resolved["reference_report_path"]),
                    "reference_report_sha256": sha256_file(
                        resolved["reference_report_path"]
                    ),
                    "checkpoint": str(resolved["checkpoint_path"]),
                    "checkpoint_sha256": sha256_file(resolved["checkpoint_path"]),
                    "run_manifest": str(resolved["manifest_path"]),
                    "run_manifest_sha256": sha256_file(resolved["manifest_path"]),
                    "training_audit": str(resolved["audit_path"]),
                    "training_audit_sha256": sha256_file(resolved["audit_path"]),
                    "proxy_emission": proxy_diagnostics,
                    "pass": True,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"RUN_PASS experiment={experiment} seed={seed}", flush=True)

    multiseed_summary = build_multiseed_summary(per_seed_rows)
    require(len(per_seed_rows) == 24, "Expected 24 pooled per-seed rows.")
    require(len(multiseed_summary) == 8, "Expected 8 multiseed summary rows.")
    require(
        all(bool(row["log_metric_reproduction_pass"]) for row in per_seed_rows),
        "At least one log-space reproduction check failed.",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    per_seed_csv = output_dir / "physical_scale_metrics_per_seed.csv"
    per_month_csv = output_dir / "physical_scale_metrics_per_month.csv"
    summary_csv = output_dir / "physical_scale_metrics_multiseed_summary.csv"
    report_json = output_dir / "physical_scale_sensitivity_audit.json"
    summary_txt = output_dir / "physical_scale_sensitivity_summary.txt"
    write_csv(per_seed_csv, per_seed_rows)
    write_csv(per_month_csv, per_month_rows)
    write_csv(summary_csv, multiseed_summary)

    report = {
        "artifact_name": ARTIFACT_NAME,
        "artifact_version": ARTIFACT_VERSION,
        "status": "PASS_EXACT_FORMAL_LINEAGE_AND_PHYSICAL_SCALE_CHARACTERIZED",
        "definition": {
            "c0_kg_m2": C0_KG_M2,
            "forward_transform": "Y = ln(1 + C/C0)",
            "inverse_transform": "C_star = C0 * expm1(Y)",
            "target_interpretation": (
                "Back-transformed monthly mean of daily log-transformed BC "
                "burden; not the arithmetic monthly mean of daily BC burden."
            ),
            "bias_sign": "prediction minus target",
            "primary_physical_units": "kg m^-2",
            "secondary_display_units": "microgram m^-2",
            "prediction_clipping": "none",
        },
        "evaluation_design": {
            "experiments": list(args.experiments),
            "seeds": list(args.seeds),
            "splits": [item[0] for item in SPLIT_DEFINITIONS],
            "mask": "unchanged valid_mask from each formal monthly payload",
            "pooling": "all valid prediction-target pairs pooled per split",
            "checkpoint_selection": "unchanged log-space-selected best_model.pt",
            "training_performed": False,
            "metric_reproduction_tolerance": float(
                args.metric_reproduction_tolerance
            ),
        },
        "runtime": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "formal_evaluator": str(evaluator_path),
            "formal_evaluator_sha256": sha256_file(evaluator_path),
        },
        "runs": run_records,
        "pooled_metrics_per_seed": per_seed_rows,
        "per_month_metrics": per_month_rows,
        "multiseed_summary": multiseed_summary,
        "verification": {
            "all_12_formal_runs_evaluated": True,
            "provenance_hash_checks_pass": True,
            "checkpoint_strict_load_pass": True,
            "all_reference_log_metrics_reproduced": True,
            "all_physical_metrics_finite": True,
            "no_training_or_artifact_modification": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "These physical-scale metrics evaluate the paired, back-transformed "
            "monthly mean-state fields against the same MERRA-2-derived target. "
            "They do not constitute an independent observational validation, an "
            "arithmetic monthly-mean burden evaluation, or an emission-accuracy estimate."
        ),
    }
    write_json_strict(report_json, report)
    summary_txt.write_text(
        make_text_summary(per_seed_rows, multiseed_summary, device),
        encoding="utf-8",
    )

    reloaded = load_json(report_json)
    require(
        reloaded.get("verification", {}).get("overall_pass") is True,
        "Written report failed strict reload verification.",
    )
    print(f"JSON_REPORT={report_json}", flush=True)
    print(f"TEXT_SUMMARY={summary_txt}", flush=True)
    print(f"PER_SEED_CSV={per_seed_csv}", flush=True)
    print(f"PER_MONTH_CSV={per_month_csv}", flush=True)
    print(f"MULTISEED_SUMMARY_CSV={summary_csv}", flush=True)
    print("STATUS=PASS_EXACT_FORMAL_LINEAGE_AND_PHYSICAL_SCALE_CHARACTERIZED")
    print("MEANSTATE_PHYSICAL_SCALE_SENSITIVITY_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
