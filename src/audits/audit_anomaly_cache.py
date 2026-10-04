# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
66_audit_clean_anomaly_cache_gateA3.py

Independent, read-only audit of the clean daily-anomaly cache produced by
65_build_clean_anomaly_cache.py.

The audit verifies:
1. build-report, build-seal and manifest provenance;
2. exact split/month/file contracts;
3. SHA256 of every monthly cache, daily cache and referenced source tensor;
4. payload schema, shape, dtype and metadata;
5. independent reconstruction of monthly backgrounds from source tensors;
6. daily target/meteo/fire anomaly reconstruction;
7. groupwise zero-mean anomaly conditions;
8. exclusion of the 2022 July blacklist day;
9. absence of duplicated AEF and emission tensors in the lean daily cache.

No cache file is modified. A new JSON/TXT audit report is written only after
all checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


H = 361
W = 576
X_CHANNELS = 112
Y_CHANNELS = 10
METEO_SLICE = slice(43, 48)

TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
VAL_MONTHS = [11, 12]
BLACKLIST_MONTHS = [7]
EXTERNAL_MONTHS = list(range(1, 13))

EXPECTED_COUNTS = {
    "monthly_groups": 20,
    "daily_files_total": 600,
    "2022_train": 178,
    "2022_val": 61,
    "2022_blacklist": 1,
    "2023_all": 361,
}
EXPECTED_MANIFEST_NAMES = [
    "split_manifest.json",
    "monthly_manifest.json",
    "anomaly_manifest.json",
    "build_report.json",
]

DEFAULT_CACHE_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\08_phase2_anomaly_clean_trainonly_union"
DEFAULT_REPORT_DIR = (
    r".\sanity_reports"
    r"\anomaly_clean_cache_audit_gateA3_v01"
)


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


def load_pt(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"PT_NOT_FOUND: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(obj, dict), f"PT_ROOT_NOT_DICT: {path}")
    return obj


def parse_doy(path: Path) -> int:
    parts = path.stem.split("_")
    require(len(parts) == 2 and parts[0] == "day", f"BAD_DAILY_FILENAME: {path.name}")
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


def finite_tensor(t: torch.Tensor, label: str) -> None:
    require(bool(torch.isfinite(t).all().item()), f"NONFINITE_{label}")


def safe_absmax(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def compare_array(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
    label: str,
) -> float:
    require(actual.shape == expected.shape, f"SHAPE_MISMATCH_{label}: {actual.shape} vs {expected.shape}")
    diff = safe_absmax(actual.astype(np.float64) - expected.astype(np.float64))
    require(diff <= tolerance, f"VALUE_MISMATCH_{label}: absmax={diff} tol={tolerance}")
    return diff


def resolve_declared_path(path_value: str, cache_dir: Path, label: str) -> Path:
    p = Path(path_value)
    require(p.is_absolute(), f"DECLARED_PATH_NOT_ABSOLUTE_{label}: {path_value}")
    resolved = p.resolve()
    require(resolved.is_file(), f"DECLARED_FILE_NOT_FOUND_{label}: {resolved}")
    return resolved


def validate_source_payload(
    source: dict[str, Any],
    source_path: Path,
    year: int,
    doy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("x", "y", "obs_mask"):
        require(key in source, f"SOURCE_MISSING_{key}: {source_path}")

    x = source["x"]
    y = source["y"]
    obs = source["obs_mask"]

    require(isinstance(x, torch.Tensor), f"SOURCE_X_NOT_TENSOR: {source_path}")
    require(isinstance(y, torch.Tensor), f"SOURCE_Y_NOT_TENSOR: {source_path}")
    require(isinstance(obs, torch.Tensor), f"SOURCE_OBS_NOT_TENSOR: {source_path}")
    require(tuple(x.shape) == (X_CHANNELS, H, W), f"SOURCE_X_SHAPE_FAIL: {source_path}")
    require(tuple(y.shape) == (Y_CHANNELS, H, W), f"SOURCE_Y_SHAPE_FAIL: {source_path}")
    require(tuple(obs.shape) == (H, W), f"SOURCE_OBS_SHAPE_FAIL: {source_path}")

    finite_tensor(x, f"SOURCE_X:{source_path.name}")
    finite_tensor(y, f"SOURCE_Y:{source_path.name}")
    finite_tensor(obs.float(), f"SOURCE_OBS:{source_path.name}")

    if "day" in source:
        value = source["day"]
        internal_doy = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        require(internal_doy == doy, f"SOURCE_INTERNAL_DAY_MISMATCH: {source_path}")

    has_emission_value = source.get("has_emission", year == 2022)
    has_emission = (
        bool(has_emission_value.item())
        if isinstance(has_emission_value, torch.Tensor)
        else bool(has_emission_value)
    )
    if year == 2022:
        require(has_emission, f"SOURCE_2022_EMISSION_FLAG_FALSE: {source_path}")
    else:
        require(not has_emission, f"SOURCE_2023_EMISSION_FLAG_TRUE: {source_path}")

    return (
        y[0:1].float(),
        x[METEO_SLICE].float(),
        y[9:10].float(),
        obs.bool(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit clean anomaly cache Gate A3.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--value-tolerance", type=float, default=1e-6)
    parser.add_argument("--zero-mean-tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    require(args.value_tolerance == 1e-6, "VALUE_TOLERANCE_LOCKED_TO_1E-6")
    require(args.zero_mean_tolerance == 2e-6, "ZERO_MEAN_TOLERANCE_LOCKED_TO_2E-6")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    manifests_dir = cache_dir / "manifests"

    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(manifests_dir.is_dir(), f"MANIFEST_DIR_NOT_FOUND: {manifests_dir}")
    require(not report_dir.exists(), f"STOP_REPORT_DIR_EXISTS: {report_dir}")
    require(not cache_dir.with_name(cache_dir.name + ".building").exists(),
            "INCOMPLETE_BUILDING_DIR_STILL_EXISTS")

    build_report_path = manifests_dir / "build_report.json"
    build_seal_path = manifests_dir / "build_seal.json"
    split_manifest_path = manifests_dir / "split_manifest.json"
    monthly_manifest_path = manifests_dir / "monthly_manifest.json"
    anomaly_manifest_path = manifests_dir / "anomaly_manifest.json"

    build_report = load_json(build_report_path)
    build_seal = load_json(build_seal_path)
    split_manifest = load_json(split_manifest_path)
    monthly_manifest = load_json(monthly_manifest_path)
    anomaly_manifest = load_json(anomaly_manifest_path)

    require(build_report.get("artifact_name") == "clean_anomaly_cache_build",
            "BUILD_REPORT_ARTIFACT_MISMATCH")
    require(build_report.get("verification", {}).get("overall_pass") is True,
            "BUILD_REPORT_NOT_PASS")
    require(build_seal.get("artifact_name") == "clean_anomaly_cache_build_seal",
            "BUILD_SEAL_ARTIFACT_MISMATCH")
    require(build_seal.get("verification", {}).get("overall_pass") is True,
            "BUILD_SEAL_NOT_PASS")
    require(build_seal.get("verification", {}).get("ready_for_independent_audit") is True,
            "BUILD_SEAL_NOT_READY_FOR_AUDIT")

    require(canonical(build_report.get("output_dir", "")) == canonical(cache_dir),
            "BUILD_REPORT_CACHE_DIR_MISMATCH")
    require(canonical(build_seal.get("output_dir", "")) == canonical(cache_dir),
            "BUILD_SEAL_CACHE_DIR_MISMATCH")

    seal_hashes = build_seal.get("manifest_sha256", {})
    for name in EXPECTED_MANIFEST_NAMES:
        path = manifests_dir / name
        actual = sha256_file(path)
        expected = seal_hashes.get(name)
        require(actual == expected, f"MANIFEST_SHA256_MISMATCH_{name}: {actual} != {expected}")

    preflight_info = build_report.get("preflight", {})
    preflight_path = Path(preflight_info.get("path", "")).resolve()
    require(preflight_path.is_file(), f"PREFLIGHT_NOT_FOUND: {preflight_path}")
    require(sha256_file(preflight_path) == preflight_info.get("sha256"),
            "PREFLIGHT_SHA256_MISMATCH")

    builder_info = build_report.get("builder", {})
    builder_path = Path(builder_info.get("path", "")).resolve()
    require(builder_path.is_file(), f"BUILDER_NOT_FOUND: {builder_path}")
    require(sha256_file(builder_path) == builder_info.get("sha256"),
            "BUILDER_SHA256_MISMATCH")

    contract = build_report.get("contract", {})
    require(contract.get("train_months_2022") == TRAIN_MONTHS, "TRAIN_MONTHS_MISMATCH")
    require(contract.get("validation_months_2022") == VAL_MONTHS, "VAL_MONTHS_MISMATCH")
    require(contract.get("blacklist_months_2022") == BLACKLIST_MONTHS, "BLACKLIST_MONTHS_MISMATCH")
    require(contract.get("external_months_2023") == EXTERNAL_MONTHS, "EXTERNAL_MONTHS_MISMATCH")
    require(int(contract.get("min_obs_days_per_pixel", -1)) == 5, "MIN_OBS_DAYS_MISMATCH")
    require(contract.get("cache_schema") == "lean_daily_anomaly_with_source_reference",
            "CACHE_SCHEMA_MISMATCH")
    require(contract.get("aef_storage") == "source_file_reference_only",
            "AEF_STORAGE_MISMATCH")
    require(contract.get("emission_storage") == "not_duplicated_in_anomaly_cache",
            "EMISSION_STORAGE_MISMATCH")

    counts = build_report.get("counts", {})
    require(int(counts.get("monthly_groups", -1)) == EXPECTED_COUNTS["monthly_groups"],
            "BUILD_MONTHLY_GROUP_COUNT_MISMATCH")
    require(int(counts.get("daily_files_total", -1)) == EXPECTED_COUNTS["daily_files_total"],
            "BUILD_DAILY_FILE_COUNT_MISMATCH")
    require(counts.get("daily_files_by_split") == {
        "2022_train": EXPECTED_COUNTS["2022_train"],
        "2022_val": EXPECTED_COUNTS["2022_val"],
        "2023_all": EXPECTED_COUNTS["2023_all"],
    }, "BUILD_SPLIT_COUNTS_MISMATCH")
    require(int(counts.get("blacklist_files_excluded", -1)) == EXPECTED_COUNTS["2022_blacklist"],
            "BUILD_BLACKLIST_COUNT_MISMATCH")

    require(split_manifest.get("artifact_name") == "clean_anomaly_split_manifest",
            "SPLIT_MANIFEST_ARTIFACT_MISMATCH")
    require(monthly_manifest.get("artifact_name") == "clean_anomaly_monthly_manifest",
            "MONTHLY_MANIFEST_ARTIFACT_MISMATCH")
    require(anomaly_manifest.get("artifact_name") == "clean_anomaly_daily_manifest",
            "ANOMALY_MANIFEST_ARTIFACT_MISMATCH")

    split_counts = split_manifest.get("counts", {})
    require(split_counts == {
        "2022_train": EXPECTED_COUNTS["2022_train"],
        "2022_val": EXPECTED_COUNTS["2022_val"],
        "2022_blacklist": EXPECTED_COUNTS["2022_blacklist"],
        "2023_all": EXPECTED_COUNTS["2023_all"],
    }, f"SPLIT_MANIFEST_COUNTS_MISMATCH: {split_counts}")

    source_records = split_manifest.get("source_records", [])
    require(len(source_records) == EXPECTED_COUNTS["daily_files_total"],
            f"SOURCE_RECORD_COUNT_MISMATCH: {len(source_records)}")

    source_by_path: dict[str, dict[str, Any]] = {}
    source_keys: set[tuple[int, int]] = set()
    for record in source_records:
        path = Path(record["file"]).resolve()
        key = canonical(path)
        require(key not in source_by_path, f"DUPLICATE_SOURCE_PATH: {path}")
        year = int(record["year"])
        doy = int(record["doy"])
        month = int(record["month"])
        split = str(record["split"])
        require(doy_to_month(doy) == month, f"SOURCE_MONTH_DOY_MISMATCH: {path}")
        require(month_to_season(month) == record["season"], f"SOURCE_SEASON_MISMATCH: {path}")
        require((year, doy) not in source_keys, f"DUPLICATE_SOURCE_DAY: {(year, doy)}")
        require(not (year == 2022 and month == 7), f"BLACKLIST_DAY_PRESENT_IN_SOURCE_RECORDS: {path}")
        source_keys.add((year, doy))
        source_by_path[key] = record

    require((2022, 182) not in source_keys, "BLACKLIST_DOY_182_PRESENT")
    require(sum(1 for y, _ in source_keys if y == 2022) == 239, "2022_INCLUDED_SOURCE_COUNT_MISMATCH")
    require(sum(1 for y, _ in source_keys if y == 2023) == 361, "2023_INCLUDED_SOURCE_COUNT_MISMATCH")

    monthly_groups = monthly_manifest.get("groups", [])
    require(int(monthly_manifest.get("n_groups", -1)) == EXPECTED_COUNTS["monthly_groups"],
            "MONTHLY_MANIFEST_N_GROUPS_MISMATCH")
    require(len(monthly_groups) == EXPECTED_COUNTS["monthly_groups"],
            "MONTHLY_GROUP_LIST_LENGTH_MISMATCH")

    monthly_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for entry in monthly_groups:
        key = (int(entry["year"]), str(entry["split"]), int(entry["month"]))
        require(key not in monthly_by_key, f"DUPLICATE_MONTHLY_GROUP: {key}")
        require(month_to_season(key[2]) == entry["season"], f"MONTHLY_SEASON_MISMATCH: {key}")
        monthly_by_key[key] = entry

    expected_monthly_keys = {
        *[(2022, "2022_train", m) for m in TRAIN_MONTHS],
        *[(2022, "2022_val", m) for m in VAL_MONTHS],
        *[(2023, "2023_all", m) for m in EXTERNAL_MONTHS],
    }
    require(set(monthly_by_key) == expected_monthly_keys,
            f"MONTHLY_KEY_SET_MISMATCH: {sorted(monthly_by_key)}")

    anomaly_groups = anomaly_manifest.get("groups", [])
    require(int(anomaly_manifest.get("n_files_total", -1)) == EXPECTED_COUNTS["daily_files_total"],
            "ANOMALY_MANIFEST_TOTAL_MISMATCH")

    daily_by_key: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    daily_paths_seen: set[str] = set()
    manifest_split_counts: dict[str, int] = {}

    for group in anomaly_groups:
        split = str(group["split"])
        files = group.get("files", [])
        require(int(group.get("n_files", -1)) == len(files), f"ANOMALY_GROUP_COUNT_MISMATCH_{split}")
        manifest_split_counts[split] = len(files)
        for entry in files:
            path_key = canonical(entry["file"])
            require(path_key not in daily_paths_seen, f"DUPLICATE_DAILY_FILE: {entry['file']}")
            daily_paths_seen.add(path_key)
            key = (int(entry["year"]), str(entry["split"]), int(entry["month"]))
            require(key[1] == split, f"DAILY_SPLIT_GROUP_MISMATCH: {entry['file']}")
            daily_by_key[key].append(entry)

    require(manifest_split_counts == {
        "2022_train": EXPECTED_COUNTS["2022_train"],
        "2022_val": EXPECTED_COUNTS["2022_val"],
        "2023_all": EXPECTED_COUNTS["2023_all"],
    }, f"ANOMALY_SPLIT_COUNTS_MISMATCH: {manifest_split_counts}")
    require(set(daily_by_key) == expected_monthly_keys, "DAILY_MONTHLY_KEY_SET_MISMATCH")
    require(len(daily_paths_seen) == EXPECTED_COUNTS["daily_files_total"],
            "UNIQUE_DAILY_FILE_COUNT_MISMATCH")

    actual_monthly_files = {
        canonical(p) for p in (cache_dir / "monthly_background").rglob("*.pt")
    }
    actual_daily_files = {
        canonical(p) for p in (cache_dir / "daily_anomaly").rglob("*.pt")
    }
    declared_monthly_files = {canonical(e["file"]) for e in monthly_groups}
    require(actual_monthly_files == declared_monthly_files,
            "MONTHLY_FILESYSTEM_MANIFEST_MISMATCH")
    require(actual_daily_files == daily_paths_seen,
            "DAILY_FILESYSTEM_MANIFEST_MISMATCH")

    print("=" * 96)
    print("Clean anomaly cache independent audit — Gate A3")
    print("=" * 96)
    print(f"CACHE_DIR={cache_dir}")
    print(f"MONTHLY_GROUPS={len(monthly_by_key)}")
    print(f"DAILY_FILES={len(daily_paths_seen)}")
    print(f"SOURCE_TENSORS={len(source_by_path)}")
    print("AUDIT_MODE=all_files_all_hashes_full_reconstruction")
    print("CACHE_FILES_MODIFIED=False")
    print("MODEL_TRAINING_STARTED=False")
    print()

    audit_started = time.time()
    group_results: list[dict[str, Any]] = []
    global_max = defaultdict(float)
    source_hash_verified: set[str] = set()
    daily_hash_verified = 0
    monthly_hash_verified = 0

    for group_index, key in enumerate(sorted(expected_monthly_keys), start=1):
        year, split, month = key
        monthly_entry = monthly_by_key[key]
        daily_entries = sorted(daily_by_key[key], key=lambda e: int(e["doy"]))
        n_days = len(daily_entries)

        print(
            f"[AUDIT GROUP {group_index:02d}/{len(expected_monthly_keys):02d}] "
            f"year={year} split={split} month={month:02d} n_days={n_days}"
        )

        monthly_path = resolve_declared_path(monthly_entry["file"], cache_dir, "MONTHLY")
        require(canonical(monthly_path).startswith(canonical(cache_dir)),
                f"MONTHLY_FILE_OUTSIDE_CACHE: {monthly_path}")
        monthly_actual_sha = sha256_file(monthly_path)
        require(monthly_actual_sha == monthly_entry["sha256"],
                f"MONTHLY_SHA256_MISMATCH: {monthly_path}")
        monthly_hash_verified += 1

        monthly_payload = load_pt(monthly_path)
        require(monthly_payload.get("artifact_name") == "clean_anomaly_monthly_background",
                f"MONTHLY_PAYLOAD_ARTIFACT_MISMATCH: {monthly_path}")
        require(int(monthly_payload.get("artifact_version", -1)) == 1,
                f"MONTHLY_PAYLOAD_VERSION_MISMATCH: {monthly_path}")
        require(int(monthly_payload["year"]) == year, f"MONTHLY_YEAR_MISMATCH: {monthly_path}")
        require(str(monthly_payload["split"]) == split, f"MONTHLY_SPLIT_MISMATCH: {monthly_path}")
        require(int(monthly_payload["month"]) == month, f"MONTHLY_MONTH_MISMATCH: {monthly_path}")
        require(str(monthly_payload["season"]) == month_to_season(month),
                f"MONTHLY_SEASON_MISMATCH: {monthly_path}")
        require(int(monthly_payload["n_days"]) == n_days, f"MONTHLY_N_DAYS_MISMATCH: {monthly_path}")
        require(int(monthly_payload["min_obs_days_per_pixel"]) == 5,
                f"MONTHLY_MIN_OBS_MISMATCH: {monthly_path}")

        target_bg = monthly_payload["target_log_month_mean"]
        obs_count = monthly_payload["obs_count_map"]
        month_valid = monthly_payload["month_valid_mask"]
        meteo_bg = monthly_payload["meteo_month_mean"]
        fire_bg = monthly_payload["fire_month_mean"]

        require(tuple(target_bg.shape) == (1, H, W), f"TARGET_BG_SHAPE_FAIL: {monthly_path}")
        require(tuple(obs_count.shape) == (H, W), f"OBS_COUNT_SHAPE_FAIL: {monthly_path}")
        require(tuple(month_valid.shape) == (H, W), f"MONTH_VALID_SHAPE_FAIL: {monthly_path}")
        require(tuple(meteo_bg.shape) == (5, H, W), f"METEO_BG_SHAPE_FAIL: {monthly_path}")
        require(tuple(fire_bg.shape) == (1, H, W), f"FIRE_BG_SHAPE_FAIL: {monthly_path}")

        finite_tensor(target_bg.float(), f"TARGET_BG:{monthly_path.name}")
        finite_tensor(obs_count.float(), f"OBS_COUNT:{monthly_path.name}")
        finite_tensor(meteo_bg.float(), f"METEO_BG:{monthly_path.name}")
        finite_tensor(fire_bg.float(), f"FIRE_BG:{monthly_path.name}")
        require(month_valid.dtype == torch.bool, f"MONTH_VALID_NOT_BOOL: {monthly_path}")
        require(torch.equal(month_valid, obs_count >= 5), f"MONTH_VALID_COUNT_RULE_FAIL: {monthly_path}")

        target_sum = np.zeros((H, W), dtype=np.float64)
        target_count = np.zeros((H, W), dtype=np.int16)
        meteo_sum = np.zeros((5, H, W), dtype=np.float64)
        fire_sum = np.zeros((1, H, W), dtype=np.float64)

        target_anom_sum = np.zeros((H, W), dtype=np.float64)
        target_anom_count = np.zeros((H, W), dtype=np.int16)
        meteo_anom_sum = np.zeros((5, H, W), dtype=np.float64)
        fire_anom_sum = np.zeros((1, H, W), dtype=np.float64)

        group_source_paths: list[str] = []
        group_max = defaultdict(float)

        for index, entry in enumerate(daily_entries, start=1):
            daily_path = resolve_declared_path(entry["file"], cache_dir, "DAILY")
            require(canonical(daily_path).startswith(canonical(cache_dir)),
                    f"DAILY_FILE_OUTSIDE_CACHE: {daily_path}")
            daily_sha = sha256_file(daily_path)
            require(daily_sha == entry["sha256"], f"DAILY_SHA256_MISMATCH: {daily_path}")
            daily_hash_verified += 1

            payload = load_pt(daily_path)
            require(payload.get("artifact_name") == "clean_daily_anomaly",
                    f"DAILY_ARTIFACT_MISMATCH: {daily_path}")
            require(int(payload.get("artifact_version", -1)) == 1,
                    f"DAILY_VERSION_MISMATCH: {daily_path}")
            require(int(payload["year"]) == year, f"DAILY_YEAR_MISMATCH: {daily_path}")
            require(str(payload["split"]) == split, f"DAILY_SPLIT_MISMATCH: {daily_path}")
            require(int(payload["month"]) == month, f"DAILY_MONTH_MISMATCH: {daily_path}")
            require(str(payload["season"]) == month_to_season(month),
                    f"DAILY_SEASON_MISMATCH: {daily_path}")

            doy = int(payload["doy"])
            require(doy == int(entry["doy"]), f"DAILY_DOY_MANIFEST_MISMATCH: {daily_path}")
            require(parse_doy(daily_path) == doy, f"DAILY_FILENAME_DOY_MISMATCH: {daily_path}")
            require(doy_to_month(doy) == month, f"DAILY_DOY_MONTH_MISMATCH: {daily_path}")

            require(payload.get("aef_storage") == "source_file_reference_only",
                    f"DAILY_AEF_STORAGE_MISMATCH: {daily_path}")
            require(payload.get("emission_stored") is False,
                    f"DAILY_EMISSION_STORED_TRUE: {daily_path}")
            forbidden_keys = {"aef", "emission_log", "meteo_daily", "fire_daily", "target_log_daily"}
            require(not forbidden_keys.intersection(payload.keys()),
                    f"LEAN_SCHEMA_FORBIDDEN_KEYS: {daily_path} "
                    f"{sorted(forbidden_keys.intersection(payload.keys()))}")

            target_anom = payload["target_log_anomaly"]
            anomaly_valid = payload["anomaly_valid_mask"]
            meteo_anom = payload["meteo_anomaly"]
            fire_anom = payload["fire_anomaly"]

            require(tuple(target_anom.shape) == (1, H, W), f"TARGET_ANOM_SHAPE_FAIL: {daily_path}")
            require(tuple(anomaly_valid.shape) == (H, W), f"ANOM_MASK_SHAPE_FAIL: {daily_path}")
            require(tuple(meteo_anom.shape) == (5, H, W), f"METEO_ANOM_SHAPE_FAIL: {daily_path}")
            require(tuple(fire_anom.shape) == (1, H, W), f"FIRE_ANOM_SHAPE_FAIL: {daily_path}")
            require(anomaly_valid.dtype == torch.bool, f"ANOM_MASK_NOT_BOOL: {daily_path}")
            finite_tensor(target_anom.float(), f"TARGET_ANOM:{daily_path.name}")
            finite_tensor(meteo_anom.float(), f"METEO_ANOM:{daily_path.name}")
            finite_tensor(fire_anom.float(), f"FIRE_ANOM:{daily_path.name}")

            require(canonical(payload["background_month_file"]) == canonical(monthly_path),
                    f"DAILY_BACKGROUND_PATH_MISMATCH: {daily_path}")
            require(payload["background_month_sha256"] == monthly_actual_sha,
                    f"DAILY_BACKGROUND_HASH_MISMATCH: {daily_path}")
            require(canonical(entry["background_month_file"]) == canonical(monthly_path),
                    f"MANIFEST_BACKGROUND_PATH_MISMATCH: {daily_path}")
            require(entry["background_month_sha256"] == monthly_actual_sha,
                    f"MANIFEST_BACKGROUND_HASH_MISMATCH: {daily_path}")

            source_path = Path(payload["source_file"]).resolve()
            source_key = canonical(source_path)
            require(source_key in source_by_path, f"DAILY_SOURCE_NOT_IN_SPLIT_MANIFEST: {source_path}")
            source_record = source_by_path[source_key]
            require(canonical(entry["source_file"]) == source_key,
                    f"DAILY_MANIFEST_SOURCE_PATH_MISMATCH: {daily_path}")
            require(payload["source_file_name"] == source_path.name,
                    f"DAILY_SOURCE_NAME_MISMATCH: {daily_path}")
            require(int(source_record["year"]) == year and int(source_record["doy"]) == doy,
                    f"SOURCE_RECORD_METADATA_MISMATCH: {source_path}")

            source_sha = sha256_file(source_path)
            source_hash_verified.add(source_key)
            require(source_sha == source_record["sha256"],
                    f"SOURCE_SHA256_SPLIT_MANIFEST_MISMATCH: {source_path}")
            require(source_sha == payload["source_file_sha256"],
                    f"SOURCE_SHA256_DAILY_PAYLOAD_MISMATCH: {source_path}")
            require(source_sha == entry["source_file_sha256"],
                    f"SOURCE_SHA256_DAILY_MANIFEST_MISMATCH: {source_path}")

            source = load_pt(source_path)
            target_daily, meteo_daily, fire_daily, obs = validate_source_payload(
                source, source_path, year, doy
            )

            expected_mask = obs & month_valid
            require(torch.equal(anomaly_valid, expected_mask),
                    f"ANOMALY_VALID_MASK_MISMATCH: {daily_path}")
            require(int(entry["n_valid_pixels"]) == int(expected_mask.sum().item()),
                    f"MANIFEST_VALID_PIXEL_COUNT_MISMATCH: {daily_path}")

            valid_np = expected_mask.numpy()
            target_daily_np = target_daily[0].numpy()
            meteo_daily_np = meteo_daily.numpy()
            fire_daily_np = fire_daily.numpy()

            target_sum[obs.numpy()] += target_daily_np[obs.numpy()]
            target_count[obs.numpy()] += 1
            meteo_sum += meteo_daily_np
            fire_sum += fire_daily_np

            target_reconstructed = target_bg + target_anom
            target_diff = (
                (target_reconstructed[0] - target_daily[0])[expected_mask]
                .numpy()
                .astype(np.float64)
            )
            target_recon_absmax = safe_absmax(target_diff)
            require(target_recon_absmax <= args.value_tolerance,
                    f"TARGET_RECONSTRUCTION_FAIL: {daily_path} {target_recon_absmax}")
            group_max["target_reconstruction_absmax"] = max(
                group_max["target_reconstruction_absmax"], target_recon_absmax
            )

            outside_absmax = safe_absmax(
                target_anom[0][~expected_mask].numpy().astype(np.float64)
            )
            require(outside_absmax == 0.0,
                    f"TARGET_ANOMALY_NONZERO_OUTSIDE_MASK: {daily_path} {outside_absmax}")
            group_max["target_outside_mask_absmax"] = max(
                group_max["target_outside_mask_absmax"], outside_absmax
            )

            meteo_recon_absmax = safe_absmax(
                (meteo_bg + meteo_anom - meteo_daily).numpy().astype(np.float64)
            )
            fire_recon_absmax = safe_absmax(
                (fire_bg + fire_anom - fire_daily).numpy().astype(np.float64)
            )
            require(meteo_recon_absmax <= args.value_tolerance,
                    f"METEO_RECONSTRUCTION_FAIL: {daily_path} {meteo_recon_absmax}")
            require(fire_recon_absmax <= args.value_tolerance,
                    f"FIRE_RECONSTRUCTION_FAIL: {daily_path} {fire_recon_absmax}")
            group_max["meteo_reconstruction_absmax"] = max(
                group_max["meteo_reconstruction_absmax"], meteo_recon_absmax
            )
            group_max["fire_reconstruction_absmax"] = max(
                group_max["fire_reconstruction_absmax"], fire_recon_absmax
            )

            target_anom_np = target_anom[0].numpy()
            target_anom_sum[valid_np] += target_anom_np[valid_np]
            target_anom_count[valid_np] += 1
            meteo_anom_sum += meteo_anom.numpy()
            fire_anom_sum += fire_anom.numpy()
            group_source_paths.append(source_key)

            if index == n_days or index % 10 == 0:
                print(f"  audited_daily={index}/{n_days}")

            del payload, source
            del target_daily, meteo_daily, fire_daily, obs
            del target_anom, meteo_anom, fire_anom, anomaly_valid

        expected_target_bg = np.zeros((H, W), dtype=np.float32)
        observed = target_count > 0
        expected_target_bg[observed] = (
            target_sum[observed] / target_count[observed].astype(np.float64)
        ).astype(np.float32)
        expected_month_valid = target_count >= 5
        expected_meteo_bg = (meteo_sum / float(n_days)).astype(np.float32)
        expected_fire_bg = (fire_sum / float(n_days)).astype(np.float32)

        monthly_rebuild = {
            "target_month_mean_absmax": compare_array(
                target_bg[0].numpy(), expected_target_bg, args.value_tolerance,
                f"TARGET_MONTH_MEAN_{key}"
            ),
            "obs_count_exact": bool(np.array_equal(obs_count.numpy(), target_count)),
            "month_valid_exact": bool(np.array_equal(month_valid.numpy(), expected_month_valid)),
            "meteo_month_mean_absmax": compare_array(
                meteo_bg.numpy(), expected_meteo_bg, args.value_tolerance,
                f"METEO_MONTH_MEAN_{key}"
            ),
            "fire_month_mean_absmax": compare_array(
                fire_bg.numpy(), expected_fire_bg, args.value_tolerance,
                f"FIRE_MONTH_MEAN_{key}"
            ),
        }
        require(monthly_rebuild["obs_count_exact"], f"OBS_COUNT_REBUILD_FAIL: {key}")
        require(monthly_rebuild["month_valid_exact"], f"MONTH_VALID_REBUILD_FAIL: {key}")

        declared_group_sources = {
            canonical(record["file"]) for record in monthly_entry["source_files"]
        }
        require(declared_group_sources == set(group_source_paths),
                f"MONTHLY_SOURCE_SET_MISMATCH: {key}")
        require(int(monthly_entry["n_days"]) == n_days, f"MONTHLY_ENTRY_N_DAYS_MISMATCH: {key}")
        require(int(monthly_entry["n_valid_pixels"]) == int(expected_month_valid.sum()),
                f"MONTHLY_ENTRY_VALID_COUNT_MISMATCH: {key}")

        target_group_mean = np.zeros((H, W), dtype=np.float64)
        valid_group = target_anom_count > 0
        target_group_mean[valid_group] = (
            target_anom_sum[valid_group] / target_anom_count[valid_group]
        )
        target_zero_absmax = safe_absmax(target_group_mean[valid_group])
        meteo_zero_absmax = safe_absmax(meteo_anom_sum / float(n_days))
        fire_zero_absmax = safe_absmax(fire_anom_sum / float(n_days))

        require(target_zero_absmax <= args.zero_mean_tolerance,
                f"TARGET_ZERO_MEAN_FAIL: {key} {target_zero_absmax}")
        require(meteo_zero_absmax <= args.zero_mean_tolerance,
                f"METEO_ZERO_MEAN_FAIL: {key} {meteo_zero_absmax}")
        require(fire_zero_absmax <= args.zero_mean_tolerance,
                f"FIRE_ZERO_MEAN_FAIL: {key} {fire_zero_absmax}")

        group_result = {
            "year": year,
            "split": split,
            "month": month,
            "season": month_to_season(month),
            "n_days": n_days,
            "n_valid_pixels": int(expected_month_valid.sum()),
            "valid_pixel_fraction": float(expected_month_valid.mean()),
            "monthly_file": str(monthly_path),
            "monthly_sha256": monthly_actual_sha,
            "monthly_reconstruction": monthly_rebuild,
            "daily_reconstruction_max_abs": dict(group_max),
            "zero_mean_diagnostic": {
                "target_max_abs_group_mean": target_zero_absmax,
                "meteo_max_abs_group_mean": meteo_zero_absmax,
                "fire_max_abs_group_mean": fire_zero_absmax,
            },
            "overall_pass": True,
        }
        group_results.append(group_result)

        for metric, value in monthly_rebuild.items():
            if isinstance(value, float):
                global_max[f"monthly_{metric}"] = max(global_max[f"monthly_{metric}"], value)
        for metric, value in group_max.items():
            global_max[metric] = max(global_max[metric], value)
        global_max["target_zero_mean_absmax"] = max(
            global_max["target_zero_mean_absmax"], target_zero_absmax
        )
        global_max["meteo_zero_mean_absmax"] = max(
            global_max["meteo_zero_mean_absmax"], meteo_zero_absmax
        )
        global_max["fire_zero_mean_absmax"] = max(
            global_max["fire_zero_mean_absmax"], fire_zero_absmax
        )

        print(
            "  group_pass "
            f"target_recon={group_max['target_reconstruction_absmax']:.3e} "
            f"target_zero_mean={target_zero_absmax:.3e}"
        )

    require(monthly_hash_verified == EXPECTED_COUNTS["monthly_groups"],
            "MONTHLY_HASH_VERIFIED_COUNT_MISMATCH")
    require(daily_hash_verified == EXPECTED_COUNTS["daily_files_total"],
            "DAILY_HASH_VERIFIED_COUNT_MISMATCH")
    require(len(source_hash_verified) == EXPECTED_COUNTS["daily_files_total"],
            "SOURCE_HASH_VERIFIED_COUNT_MISMATCH")

    elapsed = time.time() - audit_started
    report_dir.mkdir(parents=True, exist_ok=False)
    json_path = report_dir / "anomaly_clean_cache_audit_gateA3_v01.json"
    txt_path = report_dir / "anomaly_clean_cache_audit_gateA3_v01.txt"

    report = {
        "artifact_name": "clean_anomaly_cache_independent_audit",
        "artifact_version": 1,
        "cache_dir": str(cache_dir),
        "input_provenance": {
            "build_report": {
                "path": str(build_report_path),
                "sha256": sha256_file(build_report_path),
            },
            "build_seal": {
                "path": str(build_seal_path),
                "sha256": sha256_file(build_seal_path),
            },
            "preflight": {
                "path": str(preflight_path),
                "sha256": sha256_file(preflight_path),
            },
            "builder": {
                "path": str(builder_path),
                "sha256": sha256_file(builder_path),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "contract": {
            "train_months_2022": TRAIN_MONTHS,
            "validation_months_2022": VAL_MONTHS,
            "blacklist_months_2022": BLACKLIST_MONTHS,
            "external_months_2023": EXTERNAL_MONTHS,
            "min_obs_days_per_pixel": 5,
            "value_tolerance": args.value_tolerance,
            "zero_mean_tolerance": args.zero_mean_tolerance,
            "cache_schema": "lean_daily_anomaly_with_source_reference",
        },
        "counts": {
            "monthly_groups_audited": len(group_results),
            "daily_files_audited": daily_hash_verified,
            "source_files_sha256_verified": len(source_hash_verified),
            "2022_train": manifest_split_counts["2022_train"],
            "2022_val": manifest_split_counts["2022_val"],
            "2022_blacklist_excluded": split_counts["2022_blacklist"],
            "2023_external": manifest_split_counts["2023_all"],
        },
        "global_max_abs_diagnostics": dict(sorted(global_max.items())),
        "group_results": group_results,
        "runtime": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
        },
        "verification": {
            "build_report_pass": True,
            "build_seal_pass": True,
            "all_manifest_hashes_match_build_seal": True,
            "preflight_hash_match": True,
            "builder_hash_match": True,
            "exact_split_contract_pass": True,
            "blacklist_exclusion_pass": True,
            "filesystem_manifest_exact_match": True,
            "all_monthly_cache_hashes_verified": True,
            "all_daily_cache_hashes_verified": True,
            "all_source_tensor_hashes_verified": True,
            "all_payload_schema_shape_metadata_pass": True,
            "lean_schema_no_aef_or_emission_duplication": True,
            "all_monthly_backgrounds_independently_reconstructed": True,
            "all_daily_anomalies_independently_reconstructed": True,
            "all_group_zero_mean_checks_pass": True,
            "old_06_phase2_cache_modified": False,
            "model_training_started": False,
            "overall_pass": True,
        },
        "decision": "PASS_FREEZE_CLEAN_ANOMALY_CACHE",
        "next_action": (
            "Freeze this cache and create an engineering smoke test for the clean "
            "anomaly training/evaluation dataflow. Do not start formal multi-seed "
            "training before the smoke run and its audit pass."
        ),
    }
    save_json(report, json_path)

    lines = [
        "GATE_A3_CLEAN_ANOMALY_CACHE_AUDIT_PASS",
        f"CACHE_DIR={cache_dir}",
        f"MONTHLY_GROUPS_AUDITED={len(group_results)}",
        f"DAILY_FILES_AUDITED={daily_hash_verified}",
        f"SOURCE_FILES_SHA256_VERIFIED={len(source_hash_verified)}",
        f"2022_TRAIN_FILES={manifest_split_counts['2022_train']}",
        f"2022_VAL_FILES={manifest_split_counts['2022_val']}",
        f"2022_BLACKLIST_FILES_EXCLUDED={split_counts['2022_blacklist']}",
        f"2023_EXTERNAL_FILES={manifest_split_counts['2023_all']}",
        f"MAX_TARGET_MONTH_REBUILD_ABSMAX={global_max['monthly_target_month_mean_absmax']:.8e}",
        f"MAX_METEO_MONTH_REBUILD_ABSMAX={global_max['monthly_meteo_month_mean_absmax']:.8e}",
        f"MAX_FIRE_MONTH_REBUILD_ABSMAX={global_max['monthly_fire_month_mean_absmax']:.8e}",
        f"MAX_TARGET_DAILY_RECONSTRUCTION_ABSMAX={global_max['target_reconstruction_absmax']:.8e}",
        f"MAX_METEO_DAILY_RECONSTRUCTION_ABSMAX={global_max['meteo_reconstruction_absmax']:.8e}",
        f"MAX_FIRE_DAILY_RECONSTRUCTION_ABSMAX={global_max['fire_reconstruction_absmax']:.8e}",
        f"MAX_TARGET_GROUP_ZERO_MEAN_ABSMAX={global_max['target_zero_mean_absmax']:.8e}",
        f"MAX_METEO_GROUP_ZERO_MEAN_ABSMAX={global_max['meteo_zero_mean_absmax']:.8e}",
        f"MAX_FIRE_GROUP_ZERO_MEAN_ABSMAX={global_max['fire_zero_mean_absmax']:.8e}",
        "ALL_MANIFEST_HASHES_MATCH_BUILD_SEAL=True",
        "ALL_MONTHLY_CACHE_HASHES_VERIFIED=True",
        "ALL_DAILY_CACHE_HASHES_VERIFIED=True",
        "ALL_SOURCE_TENSOR_HASHES_VERIFIED=True",
        "ALL_MONTHLY_BACKGROUNDS_INDEPENDENTLY_RECONSTRUCTED=True",
        "ALL_DAILY_ANOMALIES_INDEPENDENTLY_RECONSTRUCTED=True",
        "LEAN_SCHEMA_NO_AEF_OR_EMISSION_DUPLICATION=True",
        "OLD_06_PHASE2_CACHE_MODIFIED=False",
        "MODEL_TRAINING_STARTED=False",
        "DECISION=PASS_FREEZE_CLEAN_ANOMALY_CACHE",
        f"ELAPSED_MINUTES={elapsed / 60.0:.2f}",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
