# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
64_preflight_clean_anomaly_cache.py

Read-only preflight for the clean daily-anomaly cache rebuild.

It does not build or modify the cache. It verifies the locked source directories,
calendar-month split, representative tensor payloads, output-directory safety,
and available disk space. A small JSON/TXT report is written to a new sanity
report directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


H = 361
W = 576
X_CHANNELS = 112
Y_CHANNELS = 10

TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
VAL_MONTHS = [11, 12]
BLACKLIST_MONTHS = [7]
EXPECTED_2022_MONTHS = sorted(TRAIN_MONTHS + VAL_MONTHS + BLACKLIST_MONTHS)
EXPECTED_2023_MONTHS = list(range(1, 13))

DEFAULT_2022_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\05_tensor_ready_meteo_trainonly_direct\2022"
DEFAULT_2023_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\05_tensor_ready_trainonly_union\2023"
DEFAULT_CACHE_OUT = r"data/RESTRICTED_INPUT_NOT_INCLUDED\08_phase2_anomaly_clean_trainonly_union"
DEFAULT_REPORT_OUT = (
    r".\sanity_reports"
    r"\anomaly_clean_preflight_v01"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_doy(path: Path) -> int:
    parts = path.stem.split("_")
    require(len(parts) == 2 and parts[0] == "day", f"BAD_FILENAME: {path.name}")
    try:
        doy = int(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"BAD_DOY_FILENAME: {path.name}") from exc
    require(1 <= doy <= 365, f"DOY_OUT_OF_RANGE: {path.name}")
    return doy


def doy_to_month(doy: int) -> int:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative = 0
    for month, n_days in enumerate(month_days, start=1):
        cumulative += n_days
        if doy <= cumulative:
            return month
    raise RuntimeError(f"DOY_OUT_OF_RANGE: {doy}")


def scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, "EXPECTED_SCALAR_TENSOR")
        return int(value.item())
    return int(value)


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, "EXPECTED_SCALAR_BOOL_TENSOR")
        return bool(value.item())
    return bool(value)


def tensor_finite(tensor: torch.Tensor, label: str) -> None:
    require(bool(torch.isfinite(tensor).all().item()), f"NONFINITE_TENSOR: {label}")


def inspect_sample(path: Path, expected_year: int, expected_month: int) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(data, dict), f"PAYLOAD_NOT_DICT: {path}")

    required_keys = ["x", "y", "obs_mask"]
    for key in required_keys:
        require(key in data, f"MISSING_KEY_{key}: {path}")

    x = data["x"]
    y = data["y"]
    obs = data["obs_mask"]

    require(isinstance(x, torch.Tensor), f"X_NOT_TENSOR: {path}")
    require(isinstance(y, torch.Tensor), f"Y_NOT_TENSOR: {path}")
    require(isinstance(obs, torch.Tensor), f"OBS_NOT_TENSOR: {path}")
    require(tuple(x.shape) == (X_CHANNELS, H, W), f"X_SHAPE_FAIL: {path} {tuple(x.shape)}")
    require(tuple(y.shape) == (Y_CHANNELS, H, W), f"Y_SHAPE_FAIL: {path} {tuple(y.shape)}")
    require(tuple(obs.shape) == (H, W), f"OBS_SHAPE_FAIL: {path} {tuple(obs.shape)}")

    tensor_finite(x, f"x:{path.name}")
    tensor_finite(y, f"y:{path.name}")
    tensor_finite(obs.float(), f"obs_mask:{path.name}")

    doy = parse_doy(path)
    month = doy_to_month(doy)
    require(month == expected_month, f"MONTH_MISMATCH: {path}")
    if "day" in data:
        require(scalar_int(data["day"]) == doy, f"INTERNAL_DAY_MISMATCH: {path}")
    if "year" in data:
        require(scalar_int(data["year"]) == expected_year, f"INTERNAL_YEAR_MISMATCH: {path}")

    has_emission = scalar_bool(data.get("has_emission", expected_year == 2022))
    emission = y[1:9].float()
    if expected_year == 2022:
        require(has_emission, f"2022_EMISSION_FLAG_FALSE: {path}")
        require(float(emission.min().item()) > -0.5, f"2022_EMISSION_PLACEHOLDER_FOUND: {path}")
    else:
        require(not has_emission, f"2023_EMISSION_FLAG_TRUE: {path}")
        require(
            bool(torch.allclose(emission, torch.full_like(emission, -1.0), atol=0.0, rtol=0.0)),
            f"2023_EMISSION_PLACEHOLDER_NOT_MINUS_ONE: {path}",
        )

    obs_values = torch.unique(obs)
    require(
        all(float(v.item()) in (0.0, 1.0) for v in obs_values),
        f"OBS_MASK_NOT_BINARY: {path}",
    )

    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "year": expected_year,
        "doy": doy,
        "month": month,
        "size_bytes": path.stat().st_size,
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "obs_dtype": str(obs.dtype),
        "has_emission": has_emission,
        "normalized": scalar_bool(data.get("normalized", False)),
        "keys": sorted(str(k) for k in data.keys()),
        "payload_shape_pass": True,
        "finite_pass": True,
        "emission_contract_pass": True,
    }


def classify_2022(month: int) -> str:
    if month in TRAIN_MONTHS:
        return "2022_train"
    if month in VAL_MONTHS:
        return "2022_val"
    if month in BLACKLIST_MONTHS:
        return "2022_blacklist"
    raise RuntimeError(f"UNEXPECTED_2022_MONTH: {month}")


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight clean anomaly cache rebuild.")
    parser.add_argument("--tensor-dir-2022", default=DEFAULT_2022_DIR)
    parser.add_argument("--tensor-dir-2023", default=DEFAULT_2023_DIR)
    parser.add_argument("--prospective-cache-dir", default=DEFAULT_CACHE_OUT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_OUT)
    parser.add_argument("--expected-2022-files", type=int, default=240)
    parser.add_argument("--expected-2023-files", type=int, default=361)
    parser.add_argument("--expected-blacklist-files", type=int, default=1)
    args = parser.parse_args()

    dir_2022 = Path(args.tensor_dir_2022).expanduser().resolve()
    dir_2023 = Path(args.tensor_dir_2023).expanduser().resolve()
    cache_dir = Path(args.prospective_cache_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()

    require(dir_2022.is_dir(), f"TENSOR_DIR_2022_NOT_FOUND: {dir_2022}")
    require(dir_2023.is_dir(), f"TENSOR_DIR_2023_NOT_FOUND: {dir_2023}")
    require(not cache_dir.exists(), f"STOP_PROSPECTIVE_CACHE_DIR_EXISTS: {cache_dir}")
    require(not report_dir.exists(), f"STOP_REPORT_DIR_EXISTS: {report_dir}")

    files_2022 = sorted(dir_2022.glob("day_*.pt"))
    files_2023 = sorted(dir_2023.glob("day_*.pt"))

    require(
        len(files_2022) == args.expected_2022_files,
        f"2022_FILE_COUNT_MISMATCH: actual={len(files_2022)} expected={args.expected_2022_files}",
    )
    require(
        len(files_2023) == args.expected_2023_files,
        f"2023_FILE_COUNT_MISMATCH: actual={len(files_2023)} expected={args.expected_2023_files}",
    )
    require(len({p.name for p in files_2022}) == len(files_2022), "2022_DUPLICATE_FILENAMES")
    require(len({p.name for p in files_2023}) == len(files_2023), "2023_DUPLICATE_FILENAMES")

    month_files_2022: dict[int, list[Path]] = defaultdict(list)
    month_files_2023: dict[int, list[Path]] = defaultdict(list)

    for path in files_2022:
        month_files_2022[doy_to_month(parse_doy(path))].append(path)
    for path in files_2023:
        month_files_2023[doy_to_month(parse_doy(path))].append(path)

    require(sorted(month_files_2022) == EXPECTED_2022_MONTHS,
            f"2022_MONTH_SET_MISMATCH: {sorted(month_files_2022)}")
    require(sorted(month_files_2023) == EXPECTED_2023_MONTHS,
            f"2023_MONTH_SET_MISMATCH: {sorted(month_files_2023)}")

    split_counts = Counter()
    for month, group in month_files_2022.items():
        split_counts[classify_2022(month)] += len(group)
    split_counts["2023_all"] = len(files_2023)

    require(
        split_counts["2022_blacklist"] == args.expected_blacklist_files,
        "2022_BLACKLIST_FILE_COUNT_MISMATCH: "
        f"actual={split_counts['2022_blacklist']} expected={args.expected_blacklist_files}",
    )
    require(split_counts["2022_train"] > 0, "EMPTY_2022_TRAIN")
    require(split_counts["2022_val"] > 0, "EMPTY_2022_VAL")

    # One representative payload per available month.
    sample_reports: list[dict[str, Any]] = []
    for month in sorted(month_files_2022):
        sample_reports.append(inspect_sample(month_files_2022[month][0], 2022, month))
    for month in sorted(month_files_2023):
        sample_reports.append(inspect_sample(month_files_2023[month][0], 2023, month))

    included_daily_files = (
        split_counts["2022_train"] + split_counts["2022_val"] + split_counts["2023_all"]
    )

    # Proposed lean payload:
    # 15 float32 raster channels:
    #   target daily/background/anomaly (3)
    #   meteo anomaly/background (10)
    #   fire anomaly/background (2)
    # Plus three boolean masks and a conservative serialization allowance.
    float_bytes_per_day = 15 * H * W * 4
    bool_bytes_per_day = 3 * H * W
    lean_raw_bytes = included_daily_files * (float_bytes_per_day + bool_bytes_per_day)
    monthly_raw_bytes = 20 * ((1 + 5 + 1) * H * W * 4 + 2 * H * W)
    lean_estimated_bytes = int((lean_raw_bytes + monthly_raw_bytes) * 1.30)

    # Approximate legacy duplicated payload, including AEF and emission fields.
    legacy_float_channels = 93
    legacy_estimated_bytes = int(
        included_daily_files * legacy_float_channels * H * W * 4 * 1.15
    )

    cache_parent = cache_dir.parent
    require(cache_parent.exists(), f"CACHE_PARENT_NOT_FOUND: {cache_parent}")
    disk = shutil.disk_usage(cache_parent)
    minimum_free_bytes = lean_estimated_bytes + 10 * 1024 ** 3
    require(
        disk.free >= minimum_free_bytes,
        "INSUFFICIENT_FREE_DISK_FOR_LEAN_CACHE: "
        f"free_gib={bytes_to_gib(disk.free):.2f} "
        f"required_gib={bytes_to_gib(minimum_free_bytes):.2f}",
    )

    source_bytes_2022 = sum(p.stat().st_size for p in files_2022)
    source_bytes_2023 = sum(p.stat().st_size for p in files_2023)

    report_dir.mkdir(parents=True, exist_ok=False)
    json_path = report_dir / "anomaly_clean_preflight_v01.json"
    txt_path = report_dir / "anomaly_clean_preflight_v01.txt"

    report = {
        "artifact_name": "anomaly_clean_cache_preflight",
        "artifact_version": 1,
        "locked_contract": {
            "tensor_dir_2022": str(dir_2022),
            "tensor_dir_2023": str(dir_2023),
            "prospective_cache_dir": str(cache_dir),
            "train_months_2022": TRAIN_MONTHS,
            "validation_months_2022": VAL_MONTHS,
            "blacklist_months_2022": BLACKLIST_MONTHS,
            "external_months_2023": EXPECTED_2023_MONTHS,
            "expected_2022_files": args.expected_2022_files,
            "expected_2023_files": args.expected_2023_files,
            "expected_blacklist_files": args.expected_blacklist_files,
            "cache_schema": "lean_daily_anomaly_with_source_reference",
            "aef_storage": "source_file_reference_only",
            "emission_storage": "not_duplicated_in_anomaly_cache",
        },
        "source_inventory": {
            "2022_file_count": len(files_2022),
            "2023_file_count": len(files_2023),
            "2022_month_counts": {
                str(month): len(month_files_2022[month]) for month in sorted(month_files_2022)
            },
            "2023_month_counts": {
                str(month): len(month_files_2023[month]) for month in sorted(month_files_2023)
            },
            "split_counts": dict(sorted(split_counts.items())),
            "2022_source_size_bytes": source_bytes_2022,
            "2023_source_size_bytes": source_bytes_2023,
            "2022_source_size_gib": bytes_to_gib(source_bytes_2022),
            "2023_source_size_gib": bytes_to_gib(source_bytes_2023),
        },
        "sample_payload_audit": {
            "n_samples": len(sample_reports),
            "one_sample_per_month": True,
            "reports": sample_reports,
            "overall_pass": True,
        },
        "disk_and_size_estimate": {
            "drive_total_gib": bytes_to_gib(disk.total),
            "drive_used_gib": bytes_to_gib(disk.used),
            "drive_free_gib": bytes_to_gib(disk.free),
            "included_daily_files": included_daily_files,
            "lean_cache_estimated_gib": bytes_to_gib(lean_estimated_bytes),
            "legacy_duplicated_cache_estimated_gib": bytes_to_gib(legacy_estimated_bytes),
            "minimum_free_required_gib": bytes_to_gib(minimum_free_bytes),
            "free_disk_pass": True,
        },
        "provenance": {
            "preflight_script": str(Path(__file__).resolve()),
            "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "verification": {
            "source_directories_exist": True,
            "exact_source_file_counts": True,
            "exact_month_sets": True,
            "locked_month_split_nonoverlap": True,
            "blacklist_count_exact": True,
            "representative_payload_contract_pass": True,
            "prospective_cache_dir_absent": True,
            "free_disk_pass": True,
            "overall_pass": True,
        },
        "next_action": (
            "Build a new lean clean anomaly cache in the prospective cache directory. "
            "Do not modify or overwrite 06_phase2_cache."
        ),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    lines = [
        "ANOMALY_CLEAN_PREFLIGHT_PASS",
        f"TENSOR_DIR_2022={dir_2022}",
        f"TENSOR_DIR_2023={dir_2023}",
        f"PROSPECTIVE_CACHE_DIR={cache_dir}",
        f"SOURCE_2022_FILES={len(files_2022)}",
        f"SOURCE_2023_FILES={len(files_2023)}",
        f"2022_TRAIN_FILES={split_counts['2022_train']}",
        f"2022_VAL_FILES={split_counts['2022_val']}",
        f"2022_BLACKLIST_FILES={split_counts['2022_blacklist']}",
        f"2023_EXTERNAL_FILES={split_counts['2023_all']}",
        f"2022_MONTH_COUNTS={dict(sorted((m, len(v)) for m, v in month_files_2022.items()))}",
        f"2023_MONTH_COUNTS={dict(sorted((m, len(v)) for m, v in month_files_2023.items()))}",
        f"REPRESENTATIVE_PAYLOADS_CHECKED={len(sample_reports)}",
        f"DRIVE_FREE_GIB={bytes_to_gib(disk.free):.2f}",
        f"LEAN_CACHE_ESTIMATED_GIB={bytes_to_gib(lean_estimated_bytes):.2f}",
        f"LEGACY_DUPLICATED_CACHE_ESTIMATED_GIB={bytes_to_gib(legacy_estimated_bytes):.2f}",
        "CACHE_SCHEMA=lean_daily_anomaly_with_source_reference",
        "AEF_STORAGE=source_file_reference_only",
        "OLD_PHASE2_CACHE_MODIFIED=False",
        "MODEL_TRAINING_STARTED=False",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
