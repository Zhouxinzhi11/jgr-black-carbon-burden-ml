# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
64_eval_meanstate_area_weighted_gateP4_v5.py

Read-only latitude-area-weighted evaluation for the existing formal clean
mean-state experiments. This script performs inference only; it never trains a
model and never modifies a checkpoint, manifest, cache file, or prior report.

The primary weighting is w(lat) = cos(latitude). Metrics are computed after
pooling all valid month-cell pairs within each split, using the same formal
checkpoints, monthly payloads, target, valid mask, proxy-emission construction,
and geometry-control transformations as the locked evaluation scripts.

Outputs (new directory; overwrite is prohibited):
- meanstate_area_weighted_gateP4_v5.json
- meanstate_area_weighted_per_seed.csv
- meanstate_area_weighted_aggregate.csv
- meanstate_area_weighted_paired_contrasts.csv
- meanstate_area_weighted_summary.md

Expected project root:
.
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
import traceback
import faulthandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

faulthandler.enable()


SEEDS = (42, 43, 44)
H = 361
W = 576
EXPECTED_VALIDATION_MONTHS = (11, 12)
EXPECTED_EXTERNAL_MONTHS = tuple(range(1, 13))

EXPERIMENTS: Tuple[str, ...] = (
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
    "mean_emis_meteo_clim_support_preserving",
)

DISPLAY_NAMES: Mapping[str, str] = {
    "mean_emis_only": "Emission-only",
    "mean_meteo_clim_only": "Meteorology-only",
    "mean_emis_meteo_clim": "Joint emission + meteorology",
    "mean_emis_meteo_clim_shuffled": "Cyclic-shift emission + meteorology",
    "mean_emis_meteo_clim_support_preserving": (
        "Support-preserving emission + meteorology"
    ),
}

SPLITS = ("2022_validation", "2023_external")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def save_json_strict(payload: Mapping[str, Any], path: Path) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module_from_path(name: str, path: Path):
    require(path.is_file(), f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def manifest_experiment_name(manifest: Mapping[str, Any]) -> str:
    experiment = manifest.get("experiment")
    if isinstance(experiment, dict):
        return str(experiment.get("name", ""))
    return str(experiment or "")


def manifest_seed(manifest: Mapping[str, Any]) -> int:
    settings = manifest.get("settings", {})
    if isinstance(settings, dict) and "seed" in settings:
        return int(settings["seed"])
    return int(manifest.get("seed", -1))


def discover_run(root: Path, experiment: str, seed: int) -> Path:
    """Resolve the canonical formal run and ignore archived/interrupted copies.

    The project intentionally preserves interrupted/archived run directories for
    provenance. Those directories may contain valid-looking manifests and
    checkpoints, so recursive discovery alone is ambiguous. The primary rule is
    therefore the canonical formal run path used by the sealed three-seed results.
    A conservative fallback is retained only for legacy layouts and explicitly
    excludes archived/interrupted/backup-like directory names.
    """
    if experiment == "mean_emis_meteo_clim_support_preserving":
        canonical = (
            root
            / "runs_s2_meanstate_support_preserving"
            / "formal"
            / f"seed{seed}"
            / f"{experiment}_seed{seed}"
        )
        search_roots = [root / "runs_s2_meanstate_support_preserving" / "formal"]
    else:
        canonical = (
            root
            / "runs_s2_meanstate_clean"
            / "formal"
            / f"seed{seed}"
            / f"{experiment}_seed{seed}"
        )
        search_roots = [root / "runs_s2_meanstate_clean" / "formal"]

    manifest_path = canonical / "logs" / "run_manifest.json"
    checkpoint_path = canonical / "checkpoints" / "best_model.pt"
    if manifest_path.is_file() and checkpoint_path.is_file():
        manifest = load_json(manifest_path)
        require(str(manifest.get("run_mode", "")) == "formal", f"Canonical run is not formal: {canonical}")
        require(manifest_experiment_name(manifest) == experiment, f"Canonical experiment mismatch: {canonical}")
        require(manifest_seed(manifest) == seed, f"Canonical seed mismatch: {canonical}")
        return canonical.resolve()

    excluded_tokens = ("interrupted", "archive", "archived", "backup", "bak", "old")
    candidates: List[Path] = []
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for mp in search_root.rglob("run_manifest.json"):
            if mp.parent.name != "logs":
                continue
            run_dir = mp.parent.parent.resolve()
            lower = str(run_dir).lower()
            if any(token in lower for token in excluded_tokens):
                continue
            try:
                manifest = load_json(mp)
            except Exception:
                continue
            if str(manifest.get("run_mode", "")) != "formal":
                continue
            if manifest_experiment_name(manifest) != experiment:
                continue
            if manifest_seed(manifest) != seed:
                continue
            checkpoint = run_dir / "checkpoints" / "best_model.pt"
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                candidates.append(run_dir)

    unique = sorted({normalized(p): p for p in candidates}.values(), key=str)
    require(
        len(unique) == 1,
        "Could not resolve one canonical formal run for "
        f"experiment={experiment}, seed={seed}; found {len(unique)}:\n"
        + "\n".join(str(p) for p in unique),
    )
    return unique[0]


def find_locked_eval_report(
    run_dir: Path,
    experiment: str,
    seed: int,
    checkpoint_sha256: str,
) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[int, Path, Dict[str, Any]]] = []
    for path in run_dir.rglob("eval_report.json"):
        try:
            report = load_json(path)
        except Exception:
            continue
        if int(report.get("seed", -1)) != seed:
            continue
        report_experiment = report.get("experiment", {})
        if isinstance(report_experiment, dict):
            report_experiment = report_experiment.get("name", "")
        if str(report_experiment) != experiment:
            continue
        verification = report.get("verification", {})
        if isinstance(verification, dict) and verification.get("overall_pass") is not True:
            continue
        checkpoint = report.get("checkpoint", {})
        if not isinstance(checkpoint, dict):
            continue
        if str(checkpoint.get("sha256", "")) != checkpoint_sha256:
            continue
        if "eval_2022_validation" not in report or "eval_2023_external" not in report:
            continue
        score = 0
        path_text = normalized(path)
        if "eval_clean_generic" in path_text:
            score += 20
        if "eval_support_preserving" in path_text:
            score += 20
        if "gate" in path_text:
            score -= 2
        candidates.append((score, path.resolve(), report))

    require(
        len(candidates) >= 1,
        f"No locked formal evaluation report matched {run_dir}",
    )
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1], candidates[0][2]


def report_metrics(report: Mapping[str, Any], split: str) -> Dict[str, float]:
    key = "eval_2022_validation" if split == "2022_validation" else "eval_2023_external"
    section = report.get(key, {})
    require(isinstance(section, dict), f"Missing report section: {key}")
    global_metrics = section.get("global", {})
    require(isinstance(global_metrics, dict), f"Missing report global metrics: {key}")
    result = {}
    for name in ("log_r2", "log_rmse", "log_bias"):
        value = global_metrics.get(name)
        require(value is not None and math.isfinite(float(value)), f"Invalid {key}.{name}")
        result[name] = float(value)
    return result


def area_weight_grid() -> torch.Tensor:
    lat = torch.linspace(-90.0, 90.0, H, dtype=torch.float64)
    weights = torch.cos(torch.deg2rad(lat)).clamp(min=0.0)
    grid = weights[:, None].expand(H, W).contiguous()
    require(tuple(grid.shape) == (H, W), "Area-weight grid shape mismatch")
    require(bool(torch.isfinite(grid).all().item()), "Area weights contain NaN/Inf")
    require(float(grid.sum().item()) > 0.0, "Area weights sum to zero")
    return grid


def metrics_unweighted(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = prediction.reshape(-1).double()
    target = target.reshape(-1).double()
    require(prediction.numel() == target.numel() and prediction.numel() > 1, "Bad vectors")
    require(bool(torch.isfinite(prediction).all()), "Prediction has NaN/Inf")
    require(bool(torch.isfinite(target).all()), "Target has NaN/Inf")
    residual = prediction - target
    target_mean = torch.mean(target)
    ss_res = torch.sum(residual ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    require(float(ss_tot.item()) > 0.0, "Unweighted target variance is zero")
    return {
        "log_r2": float((1.0 - ss_res / ss_tot).item()),
        "log_rmse": float(torch.sqrt(torch.mean(residual ** 2)).item()),
        "log_bias": float(torch.mean(residual).item()),
    }


def metrics_weighted(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.reshape(-1).double()
    target = target.reshape(-1).double()
    weight = weight.reshape(-1).double()
    require(
        prediction.numel() == target.numel() == weight.numel() and prediction.numel() > 1,
        "Weighted vectors have incompatible lengths",
    )
    require(bool(torch.isfinite(prediction).all()), "Prediction has NaN/Inf")
    require(bool(torch.isfinite(target).all()), "Target has NaN/Inf")
    require(bool(torch.isfinite(weight).all()), "Weight has NaN/Inf")
    require(bool((weight >= 0).all()), "Weight contains negative values")
    weight_sum = torch.sum(weight)
    require(float(weight_sum.item()) > 0.0, "Weight sum is zero")
    target_mean = torch.sum(weight * target) / weight_sum
    residual = prediction - target
    ss_res = torch.sum(weight * residual ** 2)
    ss_tot = torch.sum(weight * (target - target_mean) ** 2)
    require(float(ss_tot.item()) > 0.0, "Weighted target variance is zero")
    return {
        "log_r2": float((1.0 - ss_res / ss_tot).item()),
        "log_rmse": float(torch.sqrt(torch.sum(weight * residual ** 2) / weight_sum).item()),
        "log_bias": float((torch.sum(weight * residual) / weight_sum).item()),
        "weight_sum": float(weight_sum.item()),
    }


def finite_record(record: Mapping[str, Any]) -> None:
    for key, value in record.items():
        if isinstance(value, float):
            require(math.isfinite(value), f"Non-finite output: {key}={value}")



def md5_int_local(text: str) -> int:
    """Match the deterministic key conversion used by the locked evaluators."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def support_preserving_permute_local(
    emission: torch.Tensor,
    *,
    key: str,
    support_epsilon: float = 0.0,
) -> torch.Tensor:
    """Exact local reproduction of the locked 62 evaluator transform.

    Complete eight-dimensional processed-sector vectors are reassigned only
    among cells in the original non-zero support. Unsupported cells remain
    unchanged and the support mask must be identical before and after.
    """
    require(
        emission.ndim == 3 and tuple(emission.shape) == (8, H, W),
        f"Support-preserving permutation expects (8,{H},{W}), "
        f"got {tuple(emission.shape)}",
    )
    require(support_epsilon >= 0.0, "support_epsilon must be non-negative")

    x = torch.nan_to_num(
        emission.detach().cpu().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    support = torch.any(torch.abs(x) > float(support_epsilon), dim=0)
    support_indices = torch.nonzero(
        support.reshape(-1), as_tuple=False
    ).squeeze(1)
    n_support = int(support_indices.numel())
    require(
        n_support >= 2,
        f"Insufficient support cells for permutation: {n_support}",
    )

    seed = md5_int_local(f"supportperm::{key}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(n_support, generator=generator)
    identity = torch.arange(n_support)
    if bool(torch.equal(permutation, identity)):
        permutation = torch.roll(permutation, shifts=1)

    flat = x.reshape(8, -1)
    source_vectors = flat[:, support_indices].clone()
    transformed = flat.clone()
    transformed[:, support_indices] = source_vectors[:, permutation]
    transformed = transformed.reshape(8, H, W)

    support_after = torch.any(
        torch.abs(transformed) > float(support_epsilon), dim=0
    )
    require(
        torch.equal(support, support_after),
        f"Support mask changed under support-preserving permutation: {key}",
    )
    unsupported = ~support
    require(
        torch.equal(x[:, unsupported], transformed[:, unsupported]),
        f"Unsupported values changed under support-preserving permutation: {key}",
    )
    return transformed


def make_runtime_spec(experiment_payload: Mapping[str, Any]) -> SimpleNamespace:
    """Create a runtime spec without importing the second PyTorch evaluator."""
    d = dict(experiment_payload)
    d.setdefault("support_preserving_permutation", False)
    emission_channels = int(d.get("emission_channels", 0))
    meteo_channels = int(d.get("meteo_channels", 0))
    d["input_channels"] = emission_channels + meteo_channels
    return SimpleNamespace(**d)


def build_input_gate(
    base_module,
    payload: Mapping[str, Any],
    spec,
    *,
    key: str,
    proxy_emission_log_mean: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build formal mean-state input using one imported evaluator only."""
    if not bool(getattr(spec, "support_preserving_permutation", False)):
        return base_module.build_input_from_payload(
            payload,
            spec,
            key=key,
            proxy_emission_log_mean=proxy_emission_log_mean,
        )

    require(
        bool(getattr(spec, "uses_emission", False)),
        "Support-preserving run must use emission input.",
    )
    require(
        not bool(getattr(spec, "shuffle_emission", False)),
        "Support-preserving run cannot also use cyclic shift.",
    )

    parts: List[torch.Tensor] = []

    emission = payload.get("emission_log_mean")
    if emission is None:
        require(
            proxy_emission_log_mean is not None,
            "Emission input requested but neither actual nor proxy emission is available.",
        )
        emission = proxy_emission_log_mean.clone()

    require(isinstance(emission, torch.Tensor), "Emission input is not a tensor.")
    emission = emission.detach().cpu().float()
    require(
        tuple(emission.shape) == (8, H, W),
        f"Invalid emission input shape: {tuple(emission.shape)}",
    )
    require(bool(torch.isfinite(emission).all().item()), "Emission contains NaN/Inf.")
    emission = torch.nan_to_num(emission, nan=0.0, posinf=0.0, neginf=0.0)
    emission = support_preserving_permute_local(
        emission,
        key=key,
        support_epsilon=0.0,
    )
    parts.append(emission)

    if bool(getattr(spec, "uses_meteo", False)):
        meteorology = payload["meteo_mean"].detach().cpu().float()
        require(
            tuple(meteorology.shape) == (5, H, W),
            f"Invalid meteorology shape: {tuple(meteorology.shape)}",
        )
        require(
            bool(torch.isfinite(meteorology).all().item()),
            "Meteorology contains NaN/Inf.",
        )
        meteorology = torch.nan_to_num(
            meteorology, nan=0.0, posinf=0.0, neginf=0.0
        )
        parts.append(meteorology)

    require(parts, "No input branch selected.")
    x = torch.cat(parts, dim=0)
    require(
        tuple(x.shape) == (int(spec.input_channels), H, W),
        f"Input shape mismatch: {tuple(x.shape)}",
    )
    return x


def evaluate_split(
    module,
    model: torch.nn.Module,
    entries: Sequence[Mapping[str, Any]],
    spec,
    device: torch.device,
    proxy_emission: Optional[torch.Tensor],
    weight_grid: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    all_pred: List[torch.Tensor] = []
    all_target: List[torch.Tensor] = []
    all_weight: List[torch.Tensor] = []
    per_month: List[Dict[str, Any]] = []

    with torch.no_grad():
        for entry in entries:
            payload_path = Path(str(entry["file"])).expanduser().resolve()
            require(payload_path.is_file(), f"Monthly payload missing: {payload_path}")
            payload = module.load_pt(payload_path)
            x = build_input_gate(
                module,
                payload,
                spec,
                key=payload_path.name,
                proxy_emission_log_mean=proxy_emission,
            ).unsqueeze(0).to(device, non_blocking=True)
            prediction = (
                model(x).squeeze(0).squeeze(0).detach().cpu().double()
            )
            target = payload["target_log_mean_conc"].detach().cpu().double()
            valid_mask = payload["valid_mask"].detach().cpu().float() > 0.5
            require(tuple(prediction.shape) == (H, W), "Prediction shape mismatch")
            require(tuple(target.shape) == (H, W), "Target shape mismatch")
            require(tuple(valid_mask.shape) == (H, W), "Mask shape mismatch")
            require(bool(torch.isfinite(prediction).all()), f"Bad prediction: {payload_path}")
            require(bool(torch.isfinite(target).all()), f"Bad target: {payload_path}")
            require(int(valid_mask.sum().item()) > 1, f"Too few valid pixels: {payload_path}")

            pred_v = prediction[valid_mask]
            target_v = target[valid_mask]
            weight_v = weight_grid[valid_mask]
            unweighted = metrics_unweighted(pred_v, target_v)
            weighted = metrics_weighted(pred_v, target_v, weight_v)
            row: Dict[str, Any] = {
                "year": int(entry["year"]),
                "month": int(entry["month"]),
                "n_valid_pixels": int(valid_mask.sum().item()),
                "payload": str(payload_path),
                "payload_sha256": sha256_file(payload_path),
                "unweighted": unweighted,
                "area_weighted": weighted,
            }
            per_month.append(row)
            all_pred.append(pred_v)
            all_target.append(target_v)
            all_weight.append(weight_v)

    pooled_unweighted = metrics_unweighted(torch.cat(all_pred), torch.cat(all_target))
    pooled_weighted = metrics_weighted(
        torch.cat(all_pred), torch.cat(all_target), torch.cat(all_weight)
    )
    return {
        "n_months": len(entries),
        "n_total_valid_month_cells": int(sum(r["n_valid_pixels"] for r in per_month)),
        "unweighted": pooled_unweighted,
        "area_weighted": pooled_weighted,
        "per_month": per_month,
    }


def sample_sd(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def aggregate_seed_records(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for split in SPLITS:
            selected = [
                r for r in per_seed
                if r["experiment"] == experiment and r["split"] == split
            ]
            require(len(selected) == len(SEEDS), f"Missing seed records: {experiment} {split}")
            row: Dict[str, Any] = {
                "experiment": experiment,
                "display_name": DISPLAY_NAMES[experiment],
                "split": split,
                "n_seeds": len(selected),
            }
            for metric in ("log_r2", "log_rmse", "log_bias"):
                uw = [float(r[f"unweighted_{metric}"]) for r in selected]
                aw = [float(r[f"area_weighted_{metric}"]) for r in selected]
                row[f"unweighted_{metric}_mean"] = float(statistics.mean(uw))
                row[f"unweighted_{metric}_sample_sd"] = sample_sd(uw)
                row[f"area_weighted_{metric}_mean"] = float(statistics.mean(aw))
                row[f"area_weighted_{metric}_sample_sd"] = sample_sd(aw)
            rows.append(row)
    return rows


def paired_contrasts(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = (
        ("mean_emis_meteo_clim", "mean_meteo_clim_only", "joint_minus_meteorology"),
        ("mean_emis_only", "mean_meteo_clim_only", "emission_minus_meteorology"),
        ("mean_emis_meteo_clim", "mean_emis_meteo_clim_shuffled", "joint_minus_cyclic"),
        (
            "mean_emis_meteo_clim",
            "mean_emis_meteo_clim_support_preserving",
            "joint_minus_support_preserving",
        ),
    )
    index = {
        (str(r["experiment"]), int(r["seed"]), str(r["split"])): r
        for r in per_seed
    }
    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        for left, right, label in comparisons:
            deltas: Dict[str, List[float]] = {
                "delta_r2": [],
                "delta_rmse": [],
                "delta_bias": [],
            }
            seed_payload: Dict[str, Dict[str, float]] = {}
            for seed in SEEDS:
                a = index[(left, seed, split)]
                b = index[(right, seed, split)]
                d_r2 = float(a["area_weighted_log_r2"]) - float(b["area_weighted_log_r2"])
                d_rmse = float(a["area_weighted_log_rmse"]) - float(b["area_weighted_log_rmse"])
                d_bias = float(a["area_weighted_log_bias"]) - float(b["area_weighted_log_bias"])
                deltas["delta_r2"].append(d_r2)
                deltas["delta_rmse"].append(d_rmse)
                deltas["delta_bias"].append(d_bias)
                seed_payload[str(seed)] = {
                    "delta_r2": d_r2,
                    "delta_rmse": d_rmse,
                    "delta_bias": d_bias,
                }
            row: Dict[str, Any] = {
                "split": split,
                "contrast": label,
                "left_experiment": left,
                "right_experiment": right,
                "seed_values": seed_payload,
            }
            for name, values in deltas.items():
                row[f"{name}_mean"] = float(statistics.mean(values))
                row[f"{name}_sample_sd"] = sample_sd(values)
                row[f"{name}_all_positive"] = all(v > 0 for v in values)
                row[f"{name}_all_negative"] = all(v < 0 for v in values)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only cos(latitude)-weighted evaluation of formal mean-state runs."
    )
    parser.add_argument(
        "--root",
        default=r".",
        help="Project code root containing formal runs and evaluation scripts.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    print("=" * 88, flush=True)
    print("MEANSTATE_AREA_WEIGHTED_GATEP4_V5_START", flush=True)
    print(f"PYTHON={sys.executable}", flush=True)
    print(f"SCRIPT={Path(__file__).resolve()}", flush=True)
    print(f"ROOT_ARG={args.root}", flush=True)
    print(f"OUTPUT_ARG={args.output_dir}", flush=True)
    print(f"CUDA_REQUIRED={args.require_cuda}", flush=True)
    print("TRAINING_PERFORMED=False", flush=True)
    print("=" * 88, flush=True)

    print("[STAGE 1/7] Resolve paths and preflight checks", flush=True)
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"  ROOT={root}", flush=True)
    print(f"  ROOT_EXISTS={root.is_dir()}", flush=True)
    print(f"  OUTPUT={output_dir}", flush=True)
    print(f"  OUTPUT_EXISTS={output_dir.exists()}", flush=True)
    require(root.is_dir(), f"Project root not found: {root}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")
    require(args.reproduction_tolerance > 0.0, "Tolerance must be positive")
    print("[STAGE 1/7] PASS", flush=True)

    print("[STAGE 2/7] Locate locked evaluation scripts", flush=True)
    base_script = root / "52_eval_s2_meanstate_clean_formal_run.py"
    support_script = root / "62_eval_s2_meanstate_support_preserving_formal_run.py"
    print(f"  BASE_SCRIPT={base_script} exists={base_script.is_file()}", flush=True)
    print(f"  SUPPORT_CONTRACT_SCRIPT={support_script} exists={support_script.is_file()}", flush=True)
    require(base_script.is_file(), f"Missing locked evaluator: {base_script}")
    require(
        support_script.is_file(),
        f"Missing locked support-preserving contract: {support_script}",
    )
    print("[STAGE 2/7] PASS", flush=True)

    print("[STAGE 3/7] Import one PyTorch evaluator only", flush=True)
    base_module = import_module_from_path("meanstate_eval_base_gatep4_v4", base_script)
    print("  imported 52_eval_s2_meanstate_clean_formal_run.py", flush=True)
    print(
        "  62 evaluator NOT imported; exact support-preserving transform is "
        "reproduced locally to avoid duplicate native PyTorch/OpenMP initialization.",
        flush=True,
    )
    print(f"  support_contract_sha256={sha256_file(support_script)}", flush=True)
    print("[STAGE 3/7] PASS", flush=True)

    print("[STAGE 4/7] CUDA and latitude-weight preflight", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  TORCH={torch.__version__}", flush=True)
    print(f"  CUDA_AVAILABLE={torch.cuda.is_available()}", flush=True)
    print(f"  DEVICE={device}", flush=True)
    if device.type == "cuda":
        try:
            print(f"  GPU={torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            pass
    if args.require_cuda:
        require(device.type == "cuda", "CUDA is required but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    weight_grid = area_weight_grid()
    print(
        f"  WEIGHT=cos(latitude), shape={tuple(weight_grid.shape)}, "
        f"sum={float(weight_grid.sum().item()):.6f}",
        flush=True,
    )
    print("[STAGE 4/7] PASS", flush=True)

    print("[RUNTIME SELF-TEST] Validate local runtime spec and support transform", flush=True)
    _test_spec = make_runtime_spec(
        {
            "name": "runtime_self_test",
            "mapping": "runtime_self_test",
            "note": "runtime_self_test",
            "uses_emission": True,
            "uses_meteo": True,
            "shuffle_emission": False,
            "support_preserving_permutation": True,
            "emission_channels": 8,
            "meteo_channels": 5,
        }
    )
    require(
        int(_test_spec.input_channels) == 13,
        "Runtime spec self-test failed.",
    )
    _test_emission = torch.zeros((8, H, W), dtype=torch.float32)
    _test_emission[:, 10, 10] = 1.0
    _test_emission[:, 20, 20] = 2.0
    _test_transformed = support_preserving_permute_local(
        _test_emission,
        key="runtime_self_test.pt",
        support_epsilon=0.0,
    )
    _before_support = torch.any(_test_emission != 0, dim=0)
    _after_support = torch.any(_test_transformed != 0, dim=0)
    require(
        torch.equal(_before_support, _after_support),
        "Support transform self-test failed.",
    )
    del _test_emission, _test_transformed
    print("[RUNTIME SELF-TEST] PASS", flush=True)

    print("[STAGE 5/7] Resolve and evaluate 15 locked formal runs", flush=True)

    detailed_runs: List[Dict[str, Any]] = []
    flat_seed_rows: List[Dict[str, Any]] = []

    for experiment in EXPERIMENTS:
        module = base_module
        for seed in SEEDS:
            print(
                f"  RUN_START experiment={experiment} seed={seed}",
                flush=True,
            )
            run_dir = discover_run(root, experiment, seed)
            print(f"    run_dir={run_dir}", flush=True)
            manifest_path = run_dir / "logs" / "run_manifest.json"
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            manifest = load_json(manifest_path)
            checkpoint_sha = sha256_file(checkpoint_path)
            report_path, locked_report = find_locked_eval_report(
                run_dir, experiment, seed, checkpoint_sha
            )
            print(f"    checkpoint_sha256={checkpoint_sha}", flush=True)
            print(f"    locked_eval_report={report_path}", flush=True)

            spec = make_runtime_spec(manifest["experiment"])
            model = module.MeanStateUNet(
                spec=spec,
                base_features=int(manifest["settings"]["base_features"]),
            ).to(device)
            checkpoint = module.load_checkpoint(checkpoint_path, device)
            require(
                str(checkpoint.get("experiment", "")) == experiment,
                f"Checkpoint experiment mismatch: {checkpoint_path}",
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            train_entries = list(manifest["train_entries"])
            validation_entries = list(manifest["val_entries"])
            external_entries = list(manifest["external_2023_entries"])
            require(
                tuple(int(e["month"]) for e in validation_entries)
                == EXPECTED_VALIDATION_MONTHS,
                f"Validation months mismatch: {run_dir}",
            )
            require(
                tuple(int(e["month"]) for e in external_entries)
                == EXPECTED_EXTERNAL_MONTHS,
                f"External months mismatch: {run_dir}",
            )

            if bool(spec.uses_emission):
                proxy_emission, proxy_diag = module.build_proxy_emission_baseline(train_entries)
            else:
                proxy_emission, proxy_diag = None, None

            print("    inference 2022 validation ...", flush=True)
            eval_2022 = evaluate_split(
                module,
                model,
                validation_entries,
                spec,
                device,
                proxy_emission=None,
                weight_grid=weight_grid,
            )
            print(
                "    2022 done: "
                f"unweighted_R2={eval_2022['unweighted']['log_r2']:.8f}, "
                f"area_R2={eval_2022['area_weighted']['log_r2']:.8f}",
                flush=True,
            )
            print("    inference 2023 external ...", flush=True)
            eval_2023 = evaluate_split(
                module,
                model,
                external_entries,
                spec,
                device,
                proxy_emission=proxy_emission,
                weight_grid=weight_grid,
            )
            print(
                "    2023 done: "
                f"unweighted_R2={eval_2023['unweighted']['log_r2']:.8f}, "
                f"area_R2={eval_2023['area_weighted']['log_r2']:.8f}",
                flush=True,
            )

            reproduction = {}
            for split, evaluated in (
                ("2022_validation", eval_2022),
                ("2023_external", eval_2023),
            ):
                locked = report_metrics(locked_report, split)
                current = evaluated["unweighted"]
                errors = {
                    metric: abs(float(current[metric]) - float(locked[metric]))
                    for metric in ("log_r2", "log_rmse", "log_bias")
                }
                passed = all(
                    value <= float(args.reproduction_tolerance)
                    for value in errors.values()
                )
                require(
                    passed,
                    f"Unweighted formal metric reproduction failed: "
                    f"{experiment} seed={seed} split={split} errors={errors}",
                )
                reproduction[split] = {
                    "locked": locked,
                    "recomputed": current,
                    "absolute_error": errors,
                    "pass": True,
                }

            run_record: Dict[str, Any] = {
                "experiment": experiment,
                "display_name": DISPLAY_NAMES[experiment],
                "seed": seed,
                "run_dir": str(run_dir),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "locked_eval_report": str(report_path),
                "locked_eval_report_sha256": sha256_file(report_path),
                "proxy_emission": proxy_diag,
                "metric_reproduction": reproduction,
                "2022_validation": eval_2022,
                "2023_external": eval_2023,
            }
            detailed_runs.append(run_record)

            for split, evaluated in (
                ("2022_validation", eval_2022),
                ("2023_external", eval_2023),
            ):
                row: Dict[str, Any] = {
                    "experiment": experiment,
                    "display_name": DISPLAY_NAMES[experiment],
                    "seed": seed,
                    "split": split,
                    "run_dir": str(run_dir),
                    "locked_eval_report": str(report_path),
                    "n_months": int(evaluated["n_months"]),
                    "n_total_valid_month_cells": int(
                        evaluated["n_total_valid_month_cells"]
                    ),
                }
                for metric in ("log_r2", "log_rmse", "log_bias"):
                    row[f"unweighted_{metric}"] = float(evaluated["unweighted"][metric])
                    row[f"area_weighted_{metric}"] = float(
                        evaluated["area_weighted"][metric]
                    )
                    row[f"area_minus_unweighted_{metric}"] = (
                        row[f"area_weighted_{metric}"]
                        - row[f"unweighted_{metric}"]
                    )
                finite_record(row)
                flat_seed_rows.append(row)

            print(
                f"PASS experiment={experiment} seed={seed} "
                f"2022_aw_R2={eval_2022['area_weighted']['log_r2']:.8f} "
                f"2023_aw_R2={eval_2023['area_weighted']['log_r2']:.8f}",
                flush=True,
            )

            print(
                f"  RUN_PASS experiment={experiment} seed={seed}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("[STAGE 5/7] PASS: all 15 formal runs evaluated", flush=True)
    print("[STAGE 6/7] Aggregate weighted results and paired contrasts", flush=True)
    aggregate_rows = aggregate_seed_records(flat_seed_rows)
    contrast_rows = paired_contrasts(flat_seed_rows)
    print(
        f"  aggregate_rows={len(aggregate_rows)} contrast_rows={len(contrast_rows)}",
        flush=True,
    )
    print("[STAGE 6/7] PASS", flush=True)

    print("[STAGE 7/7] Write new read-only evaluation artifacts", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "meanstate_area_weighted_gateP4_v5.json"
    per_seed_csv = output_dir / "meanstate_area_weighted_per_seed.csv"
    aggregate_csv = output_dir / "meanstate_area_weighted_aggregate.csv"
    contrasts_csv = output_dir / "meanstate_area_weighted_paired_contrasts.csv"
    summary_md = output_dir / "meanstate_area_weighted_summary.md"

    payload = {
        "artifact_name": "meanstate_area_weighted_gateP4_v5",
        "artifact_version": 5,
        "root": str(root),
        "device": str(device),
        "weighting": {
            "formula": "cos(latitude)",
            "latitude_centres_degrees": {
                "minimum": -90.0,
                "maximum": 90.0,
                "count": H,
            },
            "longitude_count": W,
            "interpretation": (
                "Approximate grid-cell area weighting for a regular latitude-longitude grid. "
                "The constant longitude-width factor cancels in normalized metrics."
            ),
        },
        "evaluation_contract": {
            "training_performed": False,
        "support_preserving_runtime": {
            "second_evaluator_imported": False,
            "contract_script": str(support_script),
            "contract_script_sha256": sha256_file(support_script),
            "transform": (
                "Local exact reproduction of locked 62 support-preserving "
                "within-support complete-vector permutation."
            ),
        },
            "checkpoint_modified": False,
            "cache_modified": False,
            "existing_report_modified": False,
            "unweighted_metrics_reproduced_before_accepting_weighted_metrics": True,
            "reproduction_tolerance": float(args.reproduction_tolerance),
            "2022_status": "held-out model-selection validation",
            "2023_meteorology_only_status": "ordinary external evaluation",
            "2023_emission_input_status": "proxy external test",
        },
        "runs": detailed_runs,
        "per_seed": flat_seed_rows,
        "aggregate": aggregate_rows,
        "paired_area_weighted_contrasts": contrast_rows,
        "verification": {
            "all_fifteen_formal_runs_found": len(detailed_runs) == 15,
            "all_unweighted_metrics_reproduced": True,
            "all_outputs_finite": True,
            "overall_pass": True,
        },
    }
    save_json_strict(payload, json_path)

    per_seed_fields = [
        "experiment", "display_name", "seed", "split", "n_months",
        "n_total_valid_month_cells", "unweighted_log_r2", "area_weighted_log_r2",
        "area_minus_unweighted_log_r2", "unweighted_log_rmse",
        "area_weighted_log_rmse", "area_minus_unweighted_log_rmse",
        "unweighted_log_bias", "area_weighted_log_bias",
        "area_minus_unweighted_log_bias", "run_dir", "locked_eval_report",
    ]
    write_csv(per_seed_csv, flat_seed_rows, per_seed_fields)

    aggregate_fields = [
        "experiment", "display_name", "split", "n_seeds",
        "unweighted_log_r2_mean", "unweighted_log_r2_sample_sd",
        "area_weighted_log_r2_mean", "area_weighted_log_r2_sample_sd",
        "unweighted_log_rmse_mean", "unweighted_log_rmse_sample_sd",
        "area_weighted_log_rmse_mean", "area_weighted_log_rmse_sample_sd",
        "unweighted_log_bias_mean", "unweighted_log_bias_sample_sd",
        "area_weighted_log_bias_mean", "area_weighted_log_bias_sample_sd",
    ]
    write_csv(aggregate_csv, aggregate_rows, aggregate_fields)

    contrast_flat: List[Dict[str, Any]] = []
    for row in contrast_rows:
        flat = {k: v for k, v in row.items() if k != "seed_values"}
        for seed, values in row["seed_values"].items():
            for metric, value in values.items():
                flat[f"seed{seed}_{metric}"] = value
        contrast_flat.append(flat)
    contrast_fields = [
        "split", "contrast", "left_experiment", "right_experiment",
        "seed42_delta_r2", "seed43_delta_r2", "seed44_delta_r2",
        "delta_r2_mean", "delta_r2_sample_sd", "delta_r2_all_positive",
        "delta_r2_all_negative", "seed42_delta_rmse", "seed43_delta_rmse",
        "seed44_delta_rmse", "delta_rmse_mean", "delta_rmse_sample_sd",
        "delta_rmse_all_positive", "delta_rmse_all_negative",
        "seed42_delta_bias", "seed43_delta_bias", "seed44_delta_bias",
        "delta_bias_mean", "delta_bias_sample_sd",
    ]
    write_csv(contrasts_csv, contrast_flat, contrast_fields)

    md: List[str] = []
    md.append("# Mean-state latitude-area-weighted evaluation")
    md.append("")
    md.append("- Gate: `MEANSTATE_AREA_WEIGHTED_GATEP4_V5_PASS`")
    md.append("- Weight: `cos(latitude)`")
    md.append("- Training performed: `False`")
    md.append("- Existing checkpoints modified: `False`")
    md.append("- Existing formal unweighted metrics reproduced: `True`")
    md.append("")
    md.append("## Three-seed aggregate")
    md.append("")
    md.append("| Split | Configuration | Area-weighted R² | Area-weighted RMSE | Area-weighted bias |")
    md.append("|---|---|---:|---:|---:|")
    for row in aggregate_rows:
        md.append(
            f"| {row['split']} | {row['display_name']} | "
            f"{row['area_weighted_log_r2_mean']:.4f} ± {row['area_weighted_log_r2_sample_sd']:.4f} | "
            f"{row['area_weighted_log_rmse_mean']:.4f} ± {row['area_weighted_log_rmse_sample_sd']:.4f} | "
            f"{row['area_weighted_log_bias_mean']:.4f} ± {row['area_weighted_log_bias_sample_sd']:.4f} |"
        )
    md.append("")
    md.append("## Paired area-weighted contrasts")
    md.append("")
    md.append("| Split | Contrast | ΔR² | ΔRMSE |")
    md.append("|---|---|---:|---:|")
    for row in contrast_rows:
        md.append(
            f"| {row['split']} | {row['contrast']} | "
            f"{row['delta_r2_mean']:+.4f} ± {row['delta_r2_sample_sd']:.4f} | "
            f"{row['delta_rmse_mean']:+.4f} ± {row['delta_rmse_sample_sd']:.4f} |"
        )
    md.append("")
    md.append("## Interpretation boundary")
    md.append("")
    md.append(
        "Area weighting changes only the evaluation aggregation. It does not create an "
        "independent test set, recover missing 2023 sectoral priors, validate native inventory "
        "geography, or convert the reanalysis-derived burden target into observational ground truth."
    )
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[STAGE 7/7] PASS", flush=True)
    print("=" * 88)
    print("MEANSTATE_AREA_WEIGHTED_GATEP4_V5_PASS")
    print(f"json       = {json_path}")
    print(f"per_seed   = {per_seed_csv}")
    print(f"aggregate  = {aggregate_csv}")
    print(f"contrasts  = {contrasts_csv}")
    print(f"summary    = {summary_md}")
    print("TRAINING_PERFORMED=False")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("=" * 88, file=sys.stderr, flush=True)
        print("MEANSTATE_AREA_WEIGHTED_GATEP4_V5_INTERRUPTED", file=sys.stderr, flush=True)
        print("No training was performed and no locked artifact was modified.", file=sys.stderr, flush=True)
        print("=" * 88, file=sys.stderr, flush=True)
        raise
    except BaseException as exc:
        print("=" * 88, file=sys.stderr, flush=True)
        print("MEANSTATE_AREA_WEIGHTED_GATEP4_V5_FAIL", file=sys.stderr, flush=True)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("=" * 88, file=sys.stderr, flush=True)
        raise SystemExit(2)
