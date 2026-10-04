# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
46_audit_phase2_meanstate_fullmonth_clean_gate11b.py

Gate 11B: read-only, independent audit of the clean full-month mean-state cache.

This audit is intentionally stricter than the inherited 12b script:
1. It verifies the Gate-11A provenance chain against the exact Gate-8B and Gate-10C artifacts.
2. It verifies complete source-file coverage and exact month/season membership.
3. It reconstructs ALL 21 monthly groups from the clean daily tensors using torch.float64
   accumulators. Gate 11A used NumPy float64 accumulators.
4. It reconstructs all 8 seasonal groups independently from the already audited monthly
   payloads using observation-count and day-count weighting.
5. It enforces the approved scientific split:
   - 2022 training months: 02, 03, 05, 06, 08, 09
   - 2022 validation months: 11, 12
   - 2022 month 07 is retained only as a known weak source-coverage group and must be excluded
     from training/evaluation.
   - 2023 must contain months 01-12.
6. It does not modify any tensor, cache payload, or manifest.

Only after this script reports GATE11B_CLEAN_MEANSTATE_CACHE_AUDIT_PASS may
mean-state training proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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
X_CHANNELS = 112
Y_CHANNELS = 10
METEO_SLICE = slice(43, 48)
AEF_SLICE = slice(48, 112)

EXPECTED_TRAIN_MONTHS_2022 = [2, 3, 5, 6, 8, 9]
EXPECTED_VALIDATION_MONTHS_2022 = [11, 12]
EXPECTED_BLACKLIST_MONTHS_2022 = [7]
EXPECTED_ALL_MONTHS_2022 = sorted(
    EXPECTED_TRAIN_MONTHS_2022
    + EXPECTED_VALIDATION_MONTHS_2022
    + EXPECTED_BLACKLIST_MONTHS_2022
)
EXPECTED_ALL_MONTHS_2023 = list(range(1, 13))
EXPECTED_SEASONS = ["DJF", "JJA", "MAM", "SON"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pt(path: Path) -> dict[str, Any]:
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


def parse_doy_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 2 or parts[0] != "day":
        raise ValueError(f"Unexpected daily tensor filename: {filename}")
    doy = int(parts[1])
    if not 1 <= doy <= 365:
        raise ValueError(f"DOY out of range in filename: {filename}")
    return doy


def doy_to_month_nonleap(doy: int) -> int:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative = 0
    for month, count in enumerate(month_days, start=1):
        if doy <= cumulative + count:
            return month
        cumulative += count
    raise ValueError(f"DOY out of non-leap-year range: {doy}")


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid month: {month}")


def ensure_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            f"{name} shape mismatch: got={tuple(tensor.shape)}, expected={expected}"
        )


def ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains NaN or Inf")


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("Boolean tensor flag is not scalar")
        return bool(value.item())
    return bool(value)


def tensor_diff_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    a = actual.detach().cpu().float().reshape(-1)
    b = expected.detach().cpu().float().reshape(-1)
    if a.numel() != b.numel():
        raise RuntimeError("Tensor element counts differ")
    diff = torch.abs(a - b)
    n = int(diff.numel())
    return {
        "count": n,
        "unequal_count": int(torch.count_nonzero(diff != 0).item()),
        "exact_fraction": float(torch.count_nonzero(diff == 0).item() / n),
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
    }


def exact_tensor_match(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        tuple(actual.shape) == tuple(expected.shape)
        and actual.dtype == expected.dtype
        and torch.equal(actual, expected)
    )


class TorchMonthlyAccumulator:
    """Independent monthly reconstruction using torch.float64 accumulation."""

    def __init__(self) -> None:
        self.sum_target = torch.zeros((H, W), dtype=torch.float64)
        self.count_target = torch.zeros((H, W), dtype=torch.int32)
        self.sum_meteo = torch.zeros((5, H, W), dtype=torch.float64)
        self.sum_fire = torch.zeros((1, H, W), dtype=torch.float64)
        self.sum_aef = torch.zeros((64, H, W), dtype=torch.float64)
        self.sum_emission = torch.zeros((8, H, W), dtype=torch.float64)
        self.n_days = 0
        self.n_days_with_emission = 0

    def update(self, payload: dict[str, Any]) -> None:
        x = payload["x"].detach().cpu().float()
        y = payload["y"].detach().cpu().float()
        mask = payload["obs_mask"].detach().cpu().bool()

        ensure_shape("daily x", x, (X_CHANNELS, H, W))
        ensure_shape("daily y", y, (Y_CHANNELS, H, W))
        ensure_shape("daily obs_mask", mask, (H, W))
        ensure_finite("daily x", x)
        ensure_finite("daily y", y)

        target = y[0]
        self.sum_target[mask] += target[mask].double()
        self.count_target[mask] += 1
        self.sum_meteo += x[METEO_SLICE].double()
        self.sum_fire += y[9:10].double()
        self.sum_aef += x[AEF_SLICE].double()
        self.n_days += 1

        if scalar_bool(payload.get("has_emission", False)):
            self.sum_emission += y[1:9].double()
            self.n_days_with_emission += 1

    def finalize(self, min_obs_days_per_pixel: int) -> dict[str, Any]:
        if self.n_days <= 0:
            raise RuntimeError("Cannot finalize empty monthly accumulator")

        target = torch.zeros((H, W), dtype=torch.float32)
        observed = self.count_target > 0
        target[observed] = (
            self.sum_target[observed]
            / self.count_target[observed].double()
        ).float()

        valid_mask = (
            self.count_target >= int(min_obs_days_per_pixel)
        ).float()

        meteo = (self.sum_meteo / float(self.n_days)).float()
        fire = (self.sum_fire / float(self.n_days)).float()
        aef = (self.sum_aef / float(self.n_days)).float()

        if self.n_days_with_emission > 0:
            emission = (
                self.sum_emission / float(self.n_days_with_emission)
            ).float()
            emission_available = True
        else:
            emission = None
            emission_available = False

        return {
            "target_log_mean_conc": target,
            "obs_count_map": self.count_target.clone(),
            "valid_mask": valid_mask,
            "meteo_mean": meteo,
            "fire_mean": fire,
            "aef_mean": aef,
            "emission_log_mean": emission,
            "emission_available": emission_available,
            "n_days": self.n_days,
            "n_days_with_emission": self.n_days_with_emission,
        }


def validate_saved_payload(
    payload: dict[str, Any],
    *,
    year: int,
    split: str,
    month: int | None,
    season: str,
    source_files: list[str],
    min_obs_days_per_pixel: int,
) -> dict[str, Any]:
    required = {
        "target_log_mean_conc",
        "obs_count_map",
        "valid_mask",
        "meteo_mean",
        "fire_mean",
        "aef_mean",
        "emission_log_mean",
        "emission_available",
        "n_days",
        "n_days_with_emission",
        "year",
        "split",
        "month",
        "season",
        "source_files",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Cache payload missing fields: {missing}")

    target = payload["target_log_mean_conc"].detach().cpu()
    count = payload["obs_count_map"].detach().cpu()
    valid = payload["valid_mask"].detach().cpu()
    meteo = payload["meteo_mean"].detach().cpu()
    fire = payload["fire_mean"].detach().cpu()
    aef = payload["aef_mean"].detach().cpu()

    ensure_shape("target_log_mean_conc", target, (H, W))
    ensure_shape("obs_count_map", count, (H, W))
    ensure_shape("valid_mask", valid, (H, W))
    ensure_shape("meteo_mean", meteo, (5, H, W))
    ensure_shape("fire_mean", fire, (1, H, W))
    ensure_shape("aef_mean", aef, (64, H, W))

    for name, tensor in [
        ("target_log_mean_conc", target),
        ("meteo_mean", meteo),
        ("fire_mean", fire),
        ("aef_mean", aef),
    ]:
        ensure_finite(name, tensor.float())

    if torch.any(count < 0):
        raise RuntimeError("obs_count_map contains negative counts")
    valid_expected = (
        count >= int(min_obs_days_per_pixel)
    ).to(dtype=valid.dtype)
    if not torch.equal(valid, valid_expected):
        raise RuntimeError("valid_mask does not equal obs_count_map >= threshold")

    valid_unique = set(float(v) for v in torch.unique(valid.float()).tolist())
    if not valid_unique.issubset({0.0, 1.0}):
        raise RuntimeError(f"valid_mask is not binary: {sorted(valid_unique)}")

    if int(payload["year"]) != year:
        raise RuntimeError("Payload year differs from manifest")
    if str(payload["split"]) != split:
        raise RuntimeError("Payload split differs from manifest")
    payload_month = payload["month"]
    if month is None:
        if payload_month is not None:
            # Seasonal payloads inherit the last month seen in the season in the old builder.
            # The field is not scientifically used, so it is checked for type only below.
            int(payload_month)
    elif int(payload_month) != month:
        raise RuntimeError("Payload month differs from monthly manifest")
    if str(payload["season"]) != season:
        raise RuntimeError("Payload season differs from manifest")
    if list(payload["source_files"]) != list(source_files):
        raise RuntimeError("Payload source_files differ from manifest")
    if int(payload["n_days"]) != len(source_files):
        raise RuntimeError("Payload n_days differs from source_files length")

    emission_available = scalar_bool(payload["emission_available"])
    if year == 2022:
        if not emission_available:
            raise RuntimeError("2022 cache group lacks emission data")
        emission = payload["emission_log_mean"]
        if not isinstance(emission, torch.Tensor):
            raise RuntimeError("2022 emission_log_mean is not a tensor")
        ensure_shape("emission_log_mean", emission, (8, H, W))
        ensure_finite("emission_log_mean", emission.float())
        if int(payload["n_days_with_emission"]) != len(source_files):
            raise RuntimeError("2022 n_days_with_emission differs from n_days")
    elif year == 2023:
        if emission_available:
            raise RuntimeError("2023 cache falsely reports real emission availability")
        if payload["emission_log_mean"] is not None:
            raise RuntimeError("2023 emission_log_mean must be None")
        if int(payload["n_days_with_emission"]) != 0:
            raise RuntimeError("2023 n_days_with_emission must be zero")
    else:
        raise RuntimeError(f"Unexpected year: {year}")

    return {
        "valid_pixel_fraction": float(valid.float().mean().item()),
        "max_obs_count": int(count.max().item()),
        "positive_obs_pixels": int(torch.count_nonzero(count > 0).item()),
        "valid_pixels": int(torch.count_nonzero(valid > 0.5).item()),
    }


def compare_saved_to_rebuilt(
    saved: dict[str, Any],
    rebuilt: dict[str, Any],
    *,
    value_tolerance: float,
) -> tuple[dict[str, Any], bool]:
    comparisons = {
        "target": tensor_diff_metrics(
            saved["target_log_mean_conc"],
            rebuilt["target_log_mean_conc"],
        ),
        "meteo": tensor_diff_metrics(
            saved["meteo_mean"],
            rebuilt["meteo_mean"],
        ),
        "fire": tensor_diff_metrics(
            saved["fire_mean"],
            rebuilt["fire_mean"],
        ),
        "aef": tensor_diff_metrics(
            saved["aef_mean"],
            rebuilt["aef_mean"],
        ),
    }

    count_exact = exact_tensor_match(
        saved["obs_count_map"],
        rebuilt["obs_count_map"],
    )
    valid_exact = exact_tensor_match(
        saved["valid_mask"],
        rebuilt["valid_mask"],
    )
    emission_available_match = (
        scalar_bool(saved["emission_available"])
        == bool(rebuilt["emission_available"])
    )
    n_days_match = int(saved["n_days"]) == int(rebuilt["n_days"])
    n_days_with_emission_match = (
        int(saved["n_days_with_emission"])
        == int(rebuilt["n_days_with_emission"])
    )

    if bool(rebuilt["emission_available"]):
        comparisons["emission"] = tensor_diff_metrics(
            saved["emission_log_mean"],
            rebuilt["emission_log_mean"],
        )
    else:
        comparisons["emission"] = {
            "count": 0,
            "unequal_count": 0,
            "exact_fraction": 1.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "rmse": 0.0,
        }

    numerical_pass = all(
        float(comparisons[name]["max_abs"]) <= value_tolerance
        for name in ["target", "meteo", "fire", "aef", "emission"]
    )

    overall = (
        numerical_pass
        and count_exact
        and valid_exact
        and emission_available_match
        and n_days_match
        and n_days_with_emission_match
    )

    result = {
        "fields": comparisons,
        "count_exact": count_exact,
        "valid_mask_exact": valid_exact,
        "emission_available_match": emission_available_match,
        "n_days_match": n_days_match,
        "n_days_with_emission_match": n_days_with_emission_match,
        "pass": overall,
    }
    return result, overall


def reconstruct_season_from_months(
    monthly_payloads: list[dict[str, Any]],
    min_obs_days_per_pixel: int,
) -> dict[str, Any]:
    if not monthly_payloads:
        raise RuntimeError("No monthly payloads supplied for seasonal reconstruction")

    total_count = torch.zeros((H, W), dtype=torch.int64)
    target_weighted_sum = torch.zeros((H, W), dtype=torch.float64)
    meteo_weighted_sum = torch.zeros((5, H, W), dtype=torch.float64)
    fire_weighted_sum = torch.zeros((1, H, W), dtype=torch.float64)
    aef_weighted_sum = torch.zeros((64, H, W), dtype=torch.float64)
    emission_weighted_sum = torch.zeros((8, H, W), dtype=torch.float64)

    total_days = 0
    total_emission_days = 0

    for payload in monthly_payloads:
        count = payload["obs_count_map"].detach().cpu().to(torch.int64)
        target = payload["target_log_mean_conc"].detach().cpu().double()
        n_days = int(payload["n_days"])
        n_emission_days = int(payload["n_days_with_emission"])

        total_count += count
        target_weighted_sum += target * count.double()
        meteo_weighted_sum += payload["meteo_mean"].detach().cpu().double() * n_days
        fire_weighted_sum += payload["fire_mean"].detach().cpu().double() * n_days
        aef_weighted_sum += payload["aef_mean"].detach().cpu().double() * n_days
        total_days += n_days

        if n_emission_days > 0:
            emission = payload["emission_log_mean"]
            if emission is None:
                raise RuntimeError("Monthly emission payload is missing")
            emission_weighted_sum += emission.detach().cpu().double() * n_emission_days
            total_emission_days += n_emission_days

    target = torch.zeros((H, W), dtype=torch.float32)
    observed = total_count > 0
    target[observed] = (
        target_weighted_sum[observed]
        / total_count[observed].double()
    ).float()

    result = {
        "target_log_mean_conc": target,
        "obs_count_map": total_count.to(torch.int32),
        "valid_mask": (
            total_count >= int(min_obs_days_per_pixel)
        ).float(),
        "meteo_mean": (meteo_weighted_sum / float(total_days)).float(),
        "fire_mean": (fire_weighted_sum / float(total_days)).float(),
        "aef_mean": (aef_weighted_sum / float(total_days)).float(),
        "n_days": total_days,
        "n_days_with_emission": total_emission_days,
        "emission_available": total_emission_days > 0,
    }

    if total_emission_days > 0:
        result["emission_log_mean"] = (
            emission_weighted_sum / float(total_emission_days)
        ).float()
    else:
        result["emission_log_mean"] = None

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 11B independent audit of clean full-month mean-state cache."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tensor-dir-2022", required=True)
    parser.add_argument("--tensor-dir-2023", required=True)
    parser.add_argument("--gate8b-manifest", required=True)
    parser.add_argument("--gate10c-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-min-obs-days", type=int, default=5)
    parser.add_argument("--value-tolerance", type=float, default=1e-5)
    parser.add_argument("--primary-min-days", type=int, default=5)
    parser.add_argument("--primary-min-valid-fraction", type=float, default=0.01)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    tensor_dirs = {
        2022: Path(args.tensor_dir_2022).expanduser().resolve(),
        2023: Path(args.tensor_dir_2023).expanduser().resolve(),
    }
    gate8b_path = Path(args.gate8b_manifest).expanduser().resolve()
    gate10c_path = Path(args.gate10c_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for directory in [cache_dir, tensor_dirs[2022], tensor_dirs[2023]]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for path in [gate8b_path, gate10c_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "meanstate_fullmonth_clean_audit_gate11b.json"
    report_txt = output_dir / "meanstate_fullmonth_clean_audit_gate11b.txt"
    if report_json.exists() or report_txt.exists():
        raise RuntimeError(
            "Gate-11B report already exists. Use a new output directory or archive it."
        )

    manifests_dir = cache_dir / "manifests"
    monthly_manifest_path = manifests_dir / "monthly_manifest.json"
    seasonal_manifest_path = manifests_dir / "seasonal_manifest.json"
    build_report_path = manifests_dir / "meanstate_build_report.json"
    for path in [
        monthly_manifest_path,
        seasonal_manifest_path,
        build_report_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    monthly_manifest = load_json(monthly_manifest_path)
    seasonal_manifest = load_json(seasonal_manifest_path)
    build_report = load_json(build_report_path)
    gate8b = load_json(gate8b_path)
    gate10c = load_json(gate10c_path)

    if build_report.get("artifact_name") != (
        "phase2_meanstate_fullmonth_trainonly_union_gate11a"
    ):
        raise RuntimeError("Unexpected Gate-11A artifact_name")
    if int(build_report.get("version", -1)) != 1:
        raise RuntimeError("Unexpected Gate-11A artifact version")
    if int(build_report.get("min_obs_days_per_pixel", -1)) != int(
        args.expected_min_obs_days
    ):
        raise RuntimeError("Gate-11A min_obs_days_per_pixel differs from expected")
    if int(build_report.get("source_file_count_2022", -1)) != 240:
        raise RuntimeError("Gate-11A 2022 source file count is not 240")
    if int(build_report.get("source_file_count_2023", -1)) != 361:
        raise RuntimeError("Gate-11A 2023 source file count is not 361")
    if int(build_report.get("n_monthly_groups", -1)) != 21:
        raise RuntimeError("Gate-11A monthly group count is not 21")
    if int(build_report.get("n_seasonal_groups", -1)) != 8:
        raise RuntimeError("Gate-11A seasonal group count is not 8")
    if normalized_path(build_report.get("tensor_dir_2022", "")) != normalized_path(
        tensor_dirs[2022]
    ):
        raise RuntimeError("Build report 2022 tensor path mismatch")
    if normalized_path(build_report.get("tensor_dir_2023", "")) != normalized_path(
        tensor_dirs[2023]
    ):
        raise RuntimeError("Build report 2023 tensor path mismatch")
    if normalized_path(build_report.get("out_dir", "")) != normalized_path(cache_dir):
        raise RuntimeError("Build report cache output path mismatch")

    provenance = build_report.get("provenance", {})
    if normalized_path(provenance.get("gate8b_manifest", "")) != normalized_path(
        gate8b_path
    ):
        raise RuntimeError("Gate-11A Gate-8B provenance path mismatch")
    if normalized_path(provenance.get("gate10c_report", "")) != normalized_path(
        gate10c_path
    ):
        raise RuntimeError("Gate-11A Gate-10C provenance path mismatch")
    if provenance.get("gate8b_manifest_sha256") != sha256_file(gate8b_path):
        raise RuntimeError("Gate-8B provenance hash mismatch")
    if provenance.get("gate10c_report_sha256") != sha256_file(gate10c_path):
        raise RuntimeError("Gate-10C provenance hash mismatch")

    if gate8b.get("artifact_name") != "daily_tensor_meteo_trainonly_direct_raw_gate8b":
        raise RuntimeError("Unexpected Gate-8B artifact")
    if normalized_path(gate8b.get("output_dir", "")) != normalized_path(tensor_dirs[2022]):
        raise RuntimeError("Gate-8B output path does not match clean 2022 tensor directory")
    if gate10c.get("verification", {}).get("overall_pass") is not True:
        raise RuntimeError("Gate-10C overall_pass is not true")
    if normalized_path(gate10c.get("inputs", {}).get("new_dir", "")) != normalized_path(
        tensor_dirs[2023]
    ):
        raise RuntimeError("Gate-10C new_dir does not match clean 2023 tensor directory")

    monthly_entries = list(monthly_manifest.get("groups", []))
    seasonal_entries = list(seasonal_manifest.get("groups", []))
    if len(monthly_entries) != 21:
        raise RuntimeError(f"Expected 21 monthly manifest entries, found {len(monthly_entries)}")
    if len(seasonal_entries) != 8:
        raise RuntimeError(f"Expected 8 seasonal manifest entries, found {len(seasonal_entries)}")

    daily_files: dict[int, dict[str, Path]] = {}
    for year, directory in tensor_dirs.items():
        mapping = {
            path.name: path
            for path in sorted(directory.glob("day_*.pt"))
        }
        expected = 240 if year == 2022 else 361
        if len(mapping) != expected:
            raise RuntimeError(
                f"Expected {expected} clean daily tensors for {year}, found {len(mapping)}"
            )
        daily_files[year] = mapping

    # Manifest membership and complete source coverage.
    monthly_source_counter: dict[int, Counter[str]] = {
        2022: Counter(),
        2023: Counter(),
    }
    seasonal_source_counter: dict[int, Counter[str]] = {
        2022: Counter(),
        2023: Counter(),
    }
    monthly_key_seen: set[tuple[int, str, int]] = set()
    seasonal_key_seen: set[tuple[int, str, str]] = set()

    for entry in monthly_entries:
        year = int(entry["year"])
        split = str(entry["split"])
        month = int(entry["month"])
        expected_split = "2022_full" if year == 2022 else "2023_all"
        if split != expected_split:
            raise RuntimeError(f"Unexpected monthly split: {split}")
        key = (year, split, month)
        if key in monthly_key_seen:
            raise RuntimeError(f"Duplicate monthly manifest key: {key}")
        monthly_key_seen.add(key)

        sources = list(entry["source_files"])
        if len(sources) != int(entry["n_days"]):
            raise RuntimeError(f"{key}: manifest n_days/source_files mismatch")
        for filename in sources:
            if filename not in daily_files[year]:
                raise RuntimeError(f"{key}: source file not found: {filename}")
            if doy_to_month_nonleap(parse_doy_from_filename(filename)) != month:
                raise RuntimeError(f"{key}: source file belongs to another month: {filename}")
            monthly_source_counter[year][filename] += 1

    for entry in seasonal_entries:
        year = int(entry["year"])
        split = str(entry["split"])
        season = str(entry["season"])
        expected_split = "2022_full" if year == 2022 else "2023_all"
        if split != expected_split:
            raise RuntimeError(f"Unexpected seasonal split: {split}")
        key = (year, split, season)
        if key in seasonal_key_seen:
            raise RuntimeError(f"Duplicate seasonal manifest key: {key}")
        seasonal_key_seen.add(key)

        sources = list(entry["source_files"])
        if len(sources) != int(entry["n_days"]):
            raise RuntimeError(f"{key}: manifest n_days/source_files mismatch")
        for filename in sources:
            if filename not in daily_files[year]:
                raise RuntimeError(f"{key}: source file not found: {filename}")
            source_month = doy_to_month_nonleap(parse_doy_from_filename(filename))
            if month_to_season(source_month) != season:
                raise RuntimeError(f"{key}: source file belongs to another season: {filename}")
            seasonal_source_counter[year][filename] += 1

    for year in [2022, 2023]:
        expected_names = set(daily_files[year])
        if set(monthly_source_counter[year]) != expected_names:
            missing = sorted(expected_names - set(monthly_source_counter[year]))
            extra = sorted(set(monthly_source_counter[year]) - expected_names)
            raise RuntimeError(
                f"{year} monthly source coverage mismatch: missing={missing}, extra={extra}"
            )
        if set(seasonal_source_counter[year]) != expected_names:
            missing = sorted(expected_names - set(seasonal_source_counter[year]))
            extra = sorted(set(seasonal_source_counter[year]) - expected_names)
            raise RuntimeError(
                f"{year} seasonal source coverage mismatch: missing={missing}, extra={extra}"
            )
        duplicated_monthly = [
            name for name, count in monthly_source_counter[year].items() if count != 1
        ]
        duplicated_seasonal = [
            name for name, count in seasonal_source_counter[year].items() if count != 1
        ]
        if duplicated_monthly:
            raise RuntimeError(
                f"{year} daily tensors do not appear exactly once in monthly groups: "
                f"{duplicated_monthly[:10]}"
            )
        if duplicated_seasonal:
            raise RuntimeError(
                f"{year} daily tensors do not appear exactly once in seasonal groups: "
                f"{duplicated_seasonal[:10]}"
            )

    months_by_year = defaultdict(list)
    for entry in monthly_entries:
        months_by_year[int(entry["year"])].append(int(entry["month"]))
    months_by_year = {
        year: sorted(months)
        for year, months in months_by_year.items()
    }
    if months_by_year.get(2022) != EXPECTED_ALL_MONTHS_2022:
        raise RuntimeError(
            f"Unexpected 2022 month coverage: {months_by_year.get(2022)}"
        )
    if months_by_year.get(2023) != EXPECTED_ALL_MONTHS_2023:
        raise RuntimeError(
            f"Unexpected 2023 month coverage: {months_by_year.get(2023)}"
        )

    seasons_by_year = defaultdict(list)
    for entry in seasonal_entries:
        seasons_by_year[int(entry["year"])].append(str(entry["season"]))
    for year in [2022, 2023]:
        if sorted(seasons_by_year[year]) != EXPECTED_SEASONS:
            raise RuntimeError(
                f"Unexpected {year} season coverage: {sorted(seasons_by_year[year])}"
            )

    print("=" * 88)
    print("Gate 11B - clean mean-state full-month cache audit")
    print("=" * 88)
    print(f"CACHE_DIR={cache_dir}")
    print(f"MONTHLY_GROUPS={len(monthly_entries)}")
    print(f"SEASONAL_GROUPS={len(seasonal_entries)}")
    print(f"MONTHS_2022={months_by_year[2022]}")
    print(f"MONTHS_2023={months_by_year[2023]}")
    print("SOURCE_COVERAGE_AND_MEMBERSHIP_PASS=True")

    monthly_reports: list[dict[str, Any]] = []
    monthly_payload_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    monthly_all_pass = True
    primary_groups_pass = True
    weak_groups: list[dict[str, Any]] = []

    sorted_monthly_entries = sorted(
        monthly_entries,
        key=lambda e: (int(e["year"]), int(e["month"])),
    )

    for index, entry in enumerate(sorted_monthly_entries, start=1):
        year = int(entry["year"])
        split = str(entry["split"])
        month = int(entry["month"])
        season = str(entry["season"])
        sources = list(entry["source_files"])
        cache_file = Path(entry["file"]).expanduser().resolve()
        if not cache_file.is_file():
            raise FileNotFoundError(cache_file)
        if normalized_path(cache_file).find(normalized_path(cache_dir)) != 0:
            raise RuntimeError(f"Monthly cache file is outside cache directory: {cache_file}")

        saved = load_pt(cache_file)
        payload_summary = validate_saved_payload(
            saved,
            year=year,
            split=split,
            month=month,
            season=season,
            source_files=sources,
            min_obs_days_per_pixel=args.expected_min_obs_days,
        )

        manifest_valid_fraction = float(entry["valid_pixel_fraction"])
        if not math.isclose(
            manifest_valid_fraction,
            payload_summary["valid_pixel_fraction"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{year}-{month:02d}: manifest valid_pixel_fraction mismatch"
            )
        if int(entry["n_days_with_emission"]) != int(saved["n_days_with_emission"]):
            raise RuntimeError(
                f"{year}-{month:02d}: manifest n_days_with_emission mismatch"
            )
        if bool(entry["emission_available"]) != scalar_bool(
            saved["emission_available"]
        ):
            raise RuntimeError(
                f"{year}-{month:02d}: manifest emission_available mismatch"
            )

        accumulator = TorchMonthlyAccumulator()
        for filename in sources:
            daily = load_pt(daily_files[year][filename])
            internal_day = int(daily.get("day", -1))
            filename_day = parse_doy_from_filename(filename)
            if internal_day != filename_day:
                raise RuntimeError(
                    f"{filename}: internal day {internal_day} differs from filename DOY "
                    f"{filename_day}"
                )
            accumulator.update(daily)

        rebuilt = accumulator.finalize(args.expected_min_obs_days)
        comparison, group_pass = compare_saved_to_rebuilt(
            saved,
            rebuilt,
            value_tolerance=args.value_tolerance,
        )
        monthly_all_pass = monthly_all_pass and group_pass

        is_primary_2022 = (
            year == 2022
            and month in (
                EXPECTED_TRAIN_MONTHS_2022
                + EXPECTED_VALIDATION_MONTHS_2022
            )
        )
        if is_primary_2022:
            primary_quality_pass = (
                len(sources) >= args.primary_min_days
                and payload_summary["valid_pixel_fraction"]
                >= args.primary_min_valid_fraction
            )
            primary_groups_pass = primary_groups_pass and primary_quality_pass
        else:
            primary_quality_pass = None

        if (
            len(sources) < args.primary_min_days
            or payload_summary["valid_pixel_fraction"]
            < args.primary_min_valid_fraction
        ):
            weak_groups.append(
                {
                    "year": year,
                    "month": month,
                    "n_days": len(sources),
                    "valid_pixel_fraction": payload_summary[
                        "valid_pixel_fraction"
                    ],
                    "expected_blacklisted": (
                        year == 2022
                        and month in EXPECTED_BLACKLIST_MONTHS_2022
                    ),
                }
            )

        monthly_payload_by_key[(year, month)] = saved
        monthly_reports.append(
            {
                "year": year,
                "split": split,
                "month": month,
                "season": season,
                "n_days": len(sources),
                "n_days_with_emission": int(saved["n_days_with_emission"]),
                "emission_available": scalar_bool(saved["emission_available"]),
                "payload_summary": payload_summary,
                "primary_2022_group": is_primary_2022,
                "primary_quality_pass": primary_quality_pass,
                "reconstruction": comparison,
            }
        )

        print(
            f"MONTHLY_AUDIT={index}/{len(sorted_monthly_entries)} "
            f"GROUP={year}-{month:02d} "
            f"N_DAYS={len(sources)} "
            f"VALID_FRAC={payload_summary['valid_pixel_fraction']:.6f} "
            f"RECON_PASS={group_pass}",
            flush=True,
        )

    # Weak groups are allowed only for the predeclared 2022-07 exclusion.
    unexpected_weak_groups = [
        group for group in weak_groups
        if not group["expected_blacklisted"]
    ]
    expected_blacklisted_group_present = any(
        group["year"] == 2022
        and group["month"] == 7
        for group in weak_groups
    )

    seasonal_reports: list[dict[str, Any]] = []
    seasonal_all_pass = True
    sorted_seasonal_entries = sorted(
        seasonal_entries,
        key=lambda e: (int(e["year"]), str(e["season"])),
    )

    for index, entry in enumerate(sorted_seasonal_entries, start=1):
        year = int(entry["year"])
        split = str(entry["split"])
        season = str(entry["season"])
        sources = list(entry["source_files"])
        cache_file = Path(entry["file"]).expanduser().resolve()
        if not cache_file.is_file():
            raise FileNotFoundError(cache_file)

        saved = load_pt(cache_file)
        payload_summary = validate_saved_payload(
            saved,
            year=year,
            split=split,
            month=None,
            season=season,
            source_files=sources,
            min_obs_days_per_pixel=args.expected_min_obs_days,
        )

        season_months = sorted(
            month
            for (payload_year, month), payload in monthly_payload_by_key.items()
            if payload_year == year and month_to_season(month) == season
        )
        monthly_payloads = [
            monthly_payload_by_key[(year, month)]
            for month in season_months
        ]
        rebuilt = reconstruct_season_from_months(
            monthly_payloads,
            args.expected_min_obs_days,
        )
        comparison, group_pass = compare_saved_to_rebuilt(
            saved,
            rebuilt,
            value_tolerance=args.value_tolerance,
        )
        seasonal_all_pass = seasonal_all_pass and group_pass

        expected_sources: list[str] = []
        for month in season_months:
            expected_sources.extend(
                list(monthly_payload_by_key[(year, month)]["source_files"])
            )
        expected_sources = sorted(
            expected_sources,
            key=parse_doy_from_filename,
        )
        if sources != expected_sources:
            raise RuntimeError(
                f"{year}-{season}: seasonal source_files are not the ordered union "
                "of constituent monthly groups"
            )

        seasonal_reports.append(
            {
                "year": year,
                "split": split,
                "season": season,
                "constituent_months": season_months,
                "n_days": len(sources),
                "payload_summary": payload_summary,
                "reconstruction_from_audited_months": comparison,
            }
        )
        print(
            f"SEASONAL_AUDIT={index}/{len(sorted_seasonal_entries)} "
            f"GROUP={year}-{season} "
            f"N_DAYS={len(sources)} "
            f"RECON_PASS={group_pass}",
            flush=True,
        )

    overall_pass = (
        monthly_all_pass
        and seasonal_all_pass
        and primary_groups_pass
        and not unexpected_weak_groups
        and expected_blacklisted_group_present
    )

    report = {
        "audit_name": "phase2_meanstate_fullmonth_clean_gate11b",
        "read_only": True,
        "inputs": {
            "cache_dir": str(cache_dir),
            "tensor_dir_2022": str(tensor_dirs[2022]),
            "tensor_dir_2023": str(tensor_dirs[2023]),
            "gate8b_manifest": str(gate8b_path),
            "gate10c_report": str(gate10c_path),
            "gate11a_build_report": str(build_report_path),
        },
        "hashes": {
            "gate8b_manifest_sha256": sha256_file(gate8b_path),
            "gate10c_report_sha256": sha256_file(gate10c_path),
            "gate11a_build_report_sha256": sha256_file(build_report_path),
            "monthly_manifest_sha256": sha256_file(monthly_manifest_path),
            "seasonal_manifest_sha256": sha256_file(seasonal_manifest_path),
        },
        "approved_split": {
            "train_months_2022": EXPECTED_TRAIN_MONTHS_2022,
            "validation_months_2022": EXPECTED_VALIDATION_MONTHS_2022,
            "blacklist_months_2022": EXPECTED_BLACKLIST_MONTHS_2022,
            "external_months_2023": EXPECTED_ALL_MONTHS_2023,
        },
        "coverage": {
            "source_files_2022": len(daily_files[2022]),
            "source_files_2023": len(daily_files[2023]),
            "monthly_groups": len(monthly_entries),
            "seasonal_groups": len(seasonal_entries),
            "months_2022": months_by_year[2022],
            "months_2023": months_by_year[2023],
            "source_coverage_and_membership_pass": True,
        },
        "thresholds": {
            "min_obs_days_per_pixel": args.expected_min_obs_days,
            "value_tolerance": args.value_tolerance,
            "primary_min_days": args.primary_min_days,
            "primary_min_valid_fraction": args.primary_min_valid_fraction,
        },
        "monthly_reports": monthly_reports,
        "seasonal_reports": seasonal_reports,
        "weak_groups": weak_groups,
        "unexpected_weak_groups": unexpected_weak_groups,
        "verification": {
            "all_21_monthly_groups_reconstructed": monthly_all_pass,
            "all_8_seasonal_groups_reconstructed": seasonal_all_pass,
            "primary_2022_train_validation_groups_usable": primary_groups_pass,
            "only_expected_2022_07_group_is_weak": (
                not unexpected_weak_groups
                and expected_blacklisted_group_present
            ),
            "overall_pass": overall_pass,
        },
        "next_gate_boundary": (
            "Only after overall_pass is true may the clean mean-state training "
            "experiments be launched."
        ),
    }

    report_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    txt_lines = [
        "Clean mean-state full-month cache audit - Gate 11B",
        f"Monthly groups audited: {len(monthly_entries)}",
        f"Seasonal groups audited: {len(seasonal_entries)}",
        f"2022 months: {months_by_year[2022]}",
        f"2023 months: {months_by_year[2023]}",
        f"All monthly reconstructions pass: {monthly_all_pass}",
        f"All seasonal reconstructions pass: {seasonal_all_pass}",
        f"Primary 2022 groups usable: {primary_groups_pass}",
        f"Weak groups: {weak_groups}",
        f"Unexpected weak groups: {unexpected_weak_groups}",
        f"Only expected 2022-07 group is weak: {not unexpected_weak_groups and expected_blacklisted_group_present}",
        f"Overall pass: {overall_pass}",
    ]
    report_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    if not overall_pass:
        print("GATE11B_CLEAN_MEANSTATE_CACHE_AUDIT_FAILED")
        print(f"JSON_REPORT={report_json}")
        print(f"TXT_REPORT={report_txt}")
        print(f"ALL_MONTHLY_RECONSTRUCTIONS_PASS={monthly_all_pass}")
        print(f"ALL_SEASONAL_RECONSTRUCTIONS_PASS={seasonal_all_pass}")
        print(f"PRIMARY_2022_GROUPS_USABLE={primary_groups_pass}")
        print(f"UNEXPECTED_WEAK_GROUPS={unexpected_weak_groups}")
        raise RuntimeError(
            "Gate 11B failed. Do not launch mean-state training."
        )

    print("GATE11B_CLEAN_MEANSTATE_CACHE_AUDIT_PASS")
    print(f"JSON_REPORT={report_json}")
    print(f"TXT_REPORT={report_txt}")
    print("MONTHLY_GROUPS_AUDITED=21")
    print("SEASONAL_GROUPS_AUDITED=8")
    print("ALL_MONTHLY_RECONSTRUCTIONS_PASS=True")
    print("ALL_SEASONAL_RECONSTRUCTIONS_PASS=True")
    print("PRIMARY_2022_GROUPS_USABLE=True")
    print("ONLY_EXPECTED_2022_07_GROUP_IS_WEAK=True")
    print("SOURCE_COVERAGE_AND_MEMBERSHIP_PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
