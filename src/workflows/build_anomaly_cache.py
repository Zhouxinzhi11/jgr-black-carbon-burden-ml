# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""
65_build_clean_anomaly_cache.py

Build the final clean, split-aware daily-anomaly cache from the locked 2022
direct tensors and the Gate-10B/10C 2023 union tensors.

Scientific contract
-------------------
2022 training months   : 2, 3, 5, 6, 8, 9
2022 validation months : 11, 12
2022 blacklist months  : 7
2023 external months   : 1..12

The cache is deliberately lean:
- daily files store only target/meteo/fire anomalies and the anomaly-valid mask;
- monthly background fields are stored once per month;
- AEF is not duplicated and can later be loaded from source_file;
- emission is not part of the anomaly branch and is not stored.

The script never overwrites. It builds into "<output>.building" and atomically
renames that directory only after all artifacts and manifests are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
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
EXPECTED_SPLIT_COUNTS = {
    "2022_train": 178,
    "2022_val": 61,
    "2022_blacklist": 1,
    "2023_all": 361,
}
EXPECTED_MONTHLY_GROUPS = 20
EXPECTED_DAILY_FILES = 600

DEFAULT_2022_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\05_tensor_ready_meteo_trainonly_direct\2022"
DEFAULT_2023_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\05_tensor_ready_trainonly_union\2023"
DEFAULT_PREFLIGHT = (
    r".\sanity_reports"
    r"\anomaly_clean_preflight_v01\anomaly_clean_preflight_v01.json"
)
DEFAULT_OUTPUT = r"data/RESTRICTED_INPUT_NOT_INCLUDED\08_phase2_anomaly_clean_trainonly_union"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


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


def classify_2022(month: int) -> str:
    if month in TRAIN_MONTHS:
        return "2022_train"
    if month in VAL_MONTHS:
        return "2022_val"
    if month in BLACKLIST_MONTHS:
        return "2022_blacklist"
    raise RuntimeError(f"UNEXPECTED_2022_MONTH: {month}")


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, "EXPECTED_SCALAR_BOOL_TENSOR")
        return bool(value.item())
    return bool(value)


def scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, "EXPECTED_SCALAR_INT_TENSOR")
        return int(value.item())
    return int(value)


def load_pt_from_bytes(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    require(isinstance(obj, dict), f"PAYLOAD_NOT_DICT: {path}")
    return obj, digest


def load_pt(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(obj, dict), f"PAYLOAD_NOT_DICT: {path}")
    return obj


def validate_and_extract(
    data: dict[str, Any],
    path: Path,
    expected_year: int,
    expected_doy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("x", "y", "obs_mask"):
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

    require(bool(torch.isfinite(x).all().item()), f"NONFINITE_X: {path}")
    require(bool(torch.isfinite(y).all().item()), f"NONFINITE_Y: {path}")
    require(bool(torch.isfinite(obs.float()).all().item()), f"NONFINITE_OBS: {path}")

    if "day" in data:
        require(scalar_int(data["day"]) == expected_doy, f"INTERNAL_DAY_MISMATCH: {path}")
    if "year" in data:
        require(scalar_int(data["year"]) == expected_year, f"INTERNAL_YEAR_MISMATCH: {path}")

    normalized = scalar_bool(data.get("normalized", False))
    require(normalized, f"NORMALIZED_FLAG_FALSE: {path}")

    has_emission = scalar_bool(data.get("has_emission", expected_year == 2022))
    emission = y[1:9]
    if expected_year == 2022:
        require(has_emission, f"2022_EMISSION_FLAG_FALSE: {path}")
        require(float(emission.min().item()) > -0.5, f"2022_EMISSION_PLACEHOLDER: {path}")
    else:
        require(not has_emission, f"2023_EMISSION_FLAG_TRUE: {path}")
        require(
            bool(torch.allclose(
                emission.float(),
                torch.full_like(emission.float(), -1.0),
                atol=0.0,
                rtol=0.0,
            )),
            f"2023_EMISSION_PLACEHOLDER_NOT_MINUS_ONE: {path}",
        )

    obs_bool = obs.bool()
    target = y[0:1].float()
    meteo = x[METEO_SLICE].float()
    fire = y[9:10].float()

    return target, meteo, fire, obs_bool


def build_source_groups(
    dir_2022: Path,
    dir_2023: Path,
) -> tuple[dict[tuple[int, str, int], list[Path]], dict[str, int]]:
    groups: dict[tuple[int, str, int], list[Path]] = defaultdict(list)
    counts = defaultdict(int)

    files_2022 = sorted(dir_2022.glob("day_*.pt"))
    files_2023 = sorted(dir_2023.glob("day_*.pt"))

    require(len(files_2022) == 240, f"2022_FILE_COUNT_MISMATCH: {len(files_2022)}")
    require(len(files_2023) == 361, f"2023_FILE_COUNT_MISMATCH: {len(files_2023)}")

    for path in files_2022:
        doy = parse_doy(path)
        month = doy_to_month(doy)
        split = classify_2022(month)
        counts[split] += 1
        if split != "2022_blacklist":
            groups[(2022, split, month)].append(path)

    for path in files_2023:
        doy = parse_doy(path)
        month = doy_to_month(doy)
        groups[(2023, "2023_all", month)].append(path)
        counts["2023_all"] += 1

    require(dict(counts) == EXPECTED_SPLIT_COUNTS, f"SPLIT_COUNTS_MISMATCH: {dict(counts)}")
    require(len(groups) == EXPECTED_MONTHLY_GROUPS, f"MONTHLY_GROUP_COUNT_MISMATCH: {len(groups)}")

    expected_keys = {
        *[(2022, "2022_train", month) for month in TRAIN_MONTHS],
        *[(2022, "2022_val", month) for month in VAL_MONTHS],
        *[(2023, "2023_all", month) for month in EXTERNAL_MONTHS],
    }
    require(set(groups) == expected_keys, f"GROUP_KEYS_MISMATCH: {sorted(groups)}")
    return groups, dict(counts)


def verify_preflight(
    preflight: dict[str, Any],
    preflight_path: Path,
    dir_2022: Path,
    dir_2023: Path,
    output_dir: Path,
) -> None:
    require(preflight.get("artifact_name") == "anomaly_clean_cache_preflight",
            "PREFLIGHT_ARTIFACT_MISMATCH")
    require(int(preflight.get("artifact_version", -1)) == 1, "PREFLIGHT_VERSION_MISMATCH")
    require(preflight.get("verification", {}).get("overall_pass") is True,
            "PREFLIGHT_NOT_PASS")

    contract = preflight.get("locked_contract", {})
    require(canonical(contract.get("tensor_dir_2022", "")) == canonical(dir_2022),
            "PREFLIGHT_2022_DIR_MISMATCH")
    require(canonical(contract.get("tensor_dir_2023", "")) == canonical(dir_2023),
            "PREFLIGHT_2023_DIR_MISMATCH")
    require(canonical(contract.get("prospective_cache_dir", "")) == canonical(output_dir),
            "PREFLIGHT_OUTPUT_DIR_MISMATCH")
    require(contract.get("train_months_2022") == TRAIN_MONTHS, "PREFLIGHT_TRAIN_MONTHS_MISMATCH")
    require(contract.get("validation_months_2022") == VAL_MONTHS, "PREFLIGHT_VAL_MONTHS_MISMATCH")
    require(contract.get("blacklist_months_2022") == BLACKLIST_MONTHS,
            "PREFLIGHT_BLACKLIST_MONTHS_MISMATCH")
    require(contract.get("external_months_2023") == EXTERNAL_MONTHS,
            "PREFLIGHT_EXTERNAL_MONTHS_MISMATCH")
    require(contract.get("cache_schema") == "lean_daily_anomaly_with_source_reference",
            "PREFLIGHT_CACHE_SCHEMA_MISMATCH")
    require(contract.get("aef_storage") == "source_file_reference_only",
            "PREFLIGHT_AEF_STORAGE_MISMATCH")
    require(contract.get("emission_storage") == "not_duplicated_in_anomaly_cache",
            "PREFLIGHT_EMISSION_STORAGE_MISMATCH")

    source_inventory = preflight.get("source_inventory", {})
    require(source_inventory.get("split_counts") == EXPECTED_SPLIT_COUNTS,
            "PREFLIGHT_SPLIT_COUNTS_MISMATCH")
    require(preflight_path.is_file(), f"PREFLIGHT_FILE_NOT_FOUND: {preflight_path}")


def final_path(output_dir: Path, *parts: str) -> str:
    return str(output_dir.joinpath(*parts).resolve())


def temp_path(temp_dir: Path, *parts: str) -> Path:
    return temp_dir.joinpath(*parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean lean anomaly cache.")
    parser.add_argument("--tensor-dir-2022", default=DEFAULT_2022_DIR)
    parser.add_argument("--tensor-dir-2023", default=DEFAULT_2023_DIR)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-obs-days-per-pixel", type=int, default=5)
    args = parser.parse_args()

    require(args.min_obs_days_per_pixel == 5, "MIN_OBS_DAYS_LOCKED_TO_5")

    dir_2022 = Path(args.tensor_dir_2022).expanduser().resolve()
    dir_2023 = Path(args.tensor_dir_2023).expanduser().resolve()
    preflight_path = Path(args.preflight_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    temp_dir = output_dir.with_name(output_dir.name + ".building")

    require(dir_2022.is_dir(), f"2022_DIR_NOT_FOUND: {dir_2022}")
    require(dir_2023.is_dir(), f"2023_DIR_NOT_FOUND: {dir_2023}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_EXISTS: {output_dir}")
    require(not temp_dir.exists(), f"STOP_TEMP_BUILD_DIR_EXISTS: {temp_dir}")
    require(output_dir.parent.is_dir(), f"OUTPUT_PARENT_NOT_FOUND: {output_dir.parent}")

    preflight = load_json(preflight_path)
    verify_preflight(preflight, preflight_path, dir_2022, dir_2023, output_dir)
    preflight_sha256 = sha256_file(preflight_path)
    builder_sha256 = sha256_file(Path(__file__).resolve())

    groups, split_counts = build_source_groups(dir_2022, dir_2023)
    disk_before = shutil.disk_usage(output_dir.parent)
    require(disk_before.free >= 12 * 1024 ** 3,
            f"INSUFFICIENT_FREE_DISK: {disk_before.free / 1024**3:.2f} GiB")

    print("=" * 92)
    print("Clean daily-anomaly cache build")
    print("=" * 92)
    print(f"TENSOR_DIR_2022={dir_2022}")
    print(f"TENSOR_DIR_2023={dir_2023}")
    print(f"PREFLIGHT_REPORT={preflight_path}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"TEMP_BUILD_DIR={temp_dir}")
    print(f"MONTHLY_GROUPS={len(groups)}")
    print(f"DAILY_FILES_EXPECTED={EXPECTED_DAILY_FILES}")
    print(f"FREE_DISK_GIB_BEFORE={disk_before.free / 1024**3:.2f}")
    print("OLD_06_PHASE2_CACHE_MODIFIED=False")
    print()

    temp_dir.mkdir(parents=True, exist_ok=False)
    temp_path(temp_dir, "manifests").mkdir(parents=True, exist_ok=True)
    for split in ("2022_train", "2022_val", "2023_all"):
        temp_path(temp_dir, "monthly_background", split).mkdir(parents=True, exist_ok=True)
        temp_path(temp_dir, "daily_anomaly", split).mkdir(parents=True, exist_ok=True)

    build_started = time.time()
    source_records: list[dict[str, Any]] = []
    monthly_entries: list[dict[str, Any]] = []
    daily_entries_by_split: dict[str, list[dict[str, Any]]] = {
        "2022_train": [],
        "2022_val": [],
        "2023_all": [],
    }
    source_hashes: dict[str, str] = {}

    for group_index, ((year, split, month), source_paths) in enumerate(
        sorted(groups.items()),
        start=1,
    ):
        season = month_to_season(month)
        print(
            f"[GROUP {group_index:02d}/{len(groups):02d}] "
            f"year={year} split={split} month={month:02d} n_days={len(source_paths)}"
        )

        target_sum = np.zeros((H, W), dtype=np.float64)
        target_count = np.zeros((H, W), dtype=np.int16)
        meteo_sum = np.zeros((5, H, W), dtype=np.float64)
        fire_sum = np.zeros((1, H, W), dtype=np.float64)

        group_source_records: list[dict[str, Any]] = []

        for source_index, source_path in enumerate(source_paths, start=1):
            doy = parse_doy(source_path)
            data, source_sha = load_pt_from_bytes(source_path)
            target, meteo, fire, obs = validate_and_extract(
                data,
                source_path,
                year,
                doy,
            )

            source_hashes[canonical(source_path)] = source_sha
            obs_np = obs.numpy()
            target_np = target[0].numpy().astype(np.float32, copy=False)
            meteo_np = meteo.numpy().astype(np.float32, copy=False)
            fire_np = fire.numpy().astype(np.float32, copy=False)

            target_sum[obs_np] += target_np[obs_np]
            target_count[obs_np] += 1
            meteo_sum += meteo_np
            fire_sum += fire_np

            source_record = {
                "file": str(source_path),
                "file_name": source_path.name,
                "sha256": source_sha,
                "size_bytes": source_path.stat().st_size,
                "year": year,
                "split": split,
                "month": month,
                "season": season,
                "doy": doy,
            }
            source_records.append(source_record)
            group_source_records.append(source_record)

            if source_index == len(source_paths) or source_index % 10 == 0:
                print(f"  aggregate_source={source_index}/{len(source_paths)}")

            del data, target, meteo, fire, obs

        target_mean = np.zeros((H, W), dtype=np.float32)
        observed = target_count > 0
        target_mean[observed] = (
            target_sum[observed] / target_count[observed].astype(np.float64)
        ).astype(np.float32)

        month_valid_mask = target_count >= args.min_obs_days_per_pixel
        meteo_mean = (meteo_sum / float(len(source_paths))).astype(np.float32)
        fire_mean = (fire_sum / float(len(source_paths))).astype(np.float32)

        require(bool(np.isfinite(target_mean).all()), "NONFINITE_TARGET_MONTH_MEAN")
        require(bool(np.isfinite(meteo_mean).all()), "NONFINITE_METEO_MONTH_MEAN")
        require(bool(np.isfinite(fire_mean).all()), "NONFINITE_FIRE_MONTH_MEAN")
        require(int(month_valid_mask.sum()) > 0, f"NO_VALID_PIXELS_GROUP: {(year, split, month)}")

        monthly_name = f"background_{year}_{month:02d}_{split}.pt"
        monthly_rel = ("monthly_background", split, monthly_name)
        monthly_temp = temp_path(temp_dir, *monthly_rel)
        monthly_final = final_path(output_dir, *monthly_rel)

        monthly_payload = {
            "artifact_name": "clean_anomaly_monthly_background",
            "artifact_version": 1,
            "year": year,
            "split": split,
            "month": month,
            "season": season,
            "n_days": len(source_paths),
            "min_obs_days_per_pixel": args.min_obs_days_per_pixel,
            "target_log_month_mean": torch.from_numpy(target_mean).unsqueeze(0),
            "obs_count_map": torch.from_numpy(target_count.copy()),
            "month_valid_mask": torch.from_numpy(month_valid_mask.copy()),
            "meteo_month_mean": torch.from_numpy(meteo_mean),
            "fire_month_mean": torch.from_numpy(fire_mean),
        }
        torch.save(monthly_payload, monthly_temp)
        monthly_sha = sha256_file(monthly_temp)
        monthly_size = monthly_temp.stat().st_size

        monthly_entries.append(
            {
                "year": year,
                "split": split,
                "month": month,
                "season": season,
                "n_days": len(source_paths),
                "n_valid_pixels": int(month_valid_mask.sum()),
                "valid_pixel_fraction": float(month_valid_mask.mean()),
                "min_obs_days_per_pixel": args.min_obs_days_per_pixel,
                "file": monthly_final,
                "sha256": monthly_sha,
                "size_bytes": monthly_size,
                "source_files": group_source_records,
            }
        )

        target_bg_t = torch.from_numpy(target_mean).unsqueeze(0)
        meteo_bg_t = torch.from_numpy(meteo_mean)
        fire_bg_t = torch.from_numpy(fire_mean)
        month_valid_t = torch.from_numpy(month_valid_mask.copy())

        group_target_sum = np.zeros((H, W), dtype=np.float64)
        group_target_count = np.zeros((H, W), dtype=np.int16)
        group_meteo_anom_sum = np.zeros((5, H, W), dtype=np.float64)
        group_fire_anom_sum = np.zeros((1, H, W), dtype=np.float64)

        for source_index, source_path in enumerate(source_paths, start=1):
            doy = parse_doy(source_path)
            data = load_pt(source_path)
            target, meteo, fire, obs = validate_and_extract(
                data,
                source_path,
                year,
                doy,
            )

            anomaly_valid = obs & month_valid_t
            target_anomaly = (target - target_bg_t) * anomaly_valid.unsqueeze(0)
            meteo_anomaly = meteo - meteo_bg_t
            fire_anomaly = fire - fire_bg_t

            require(bool(torch.isfinite(target_anomaly).all().item()),
                    f"NONFINITE_TARGET_ANOMALY: {source_path}")
            require(bool(torch.isfinite(meteo_anomaly).all().item()),
                    f"NONFINITE_METEO_ANOMALY: {source_path}")
            require(bool(torch.isfinite(fire_anomaly).all().item()),
                    f"NONFINITE_FIRE_ANOMALY: {source_path}")

            daily_rel = ("daily_anomaly", split, source_path.name)
            daily_temp = temp_path(temp_dir, *daily_rel)
            daily_final = final_path(output_dir, *daily_rel)

            daily_payload = {
                "artifact_name": "clean_daily_anomaly",
                "artifact_version": 1,
                "year": year,
                "split": split,
                "month": month,
                "season": season,
                "doy": doy,
                "source_file": str(source_path),
                "source_file_name": source_path.name,
                "source_file_sha256": source_hashes[canonical(source_path)],
                "background_month_file": monthly_final,
                "background_month_sha256": monthly_sha,
                "target_log_anomaly": target_anomaly.float(),
                "anomaly_valid_mask": anomaly_valid.bool(),
                "meteo_anomaly": meteo_anomaly.float(),
                "fire_anomaly": fire_anomaly.float(),
                "aef_storage": "source_file_reference_only",
                "emission_stored": False,
            }
            torch.save(daily_payload, daily_temp)
            daily_sha = sha256_file(daily_temp)

            daily_entry = {
                "file": daily_final,
                "sha256": daily_sha,
                "size_bytes": daily_temp.stat().st_size,
                "source_file": str(source_path),
                "source_file_sha256": source_hashes[canonical(source_path)],
                "background_month_file": monthly_final,
                "background_month_sha256": monthly_sha,
                "year": year,
                "split": split,
                "month": month,
                "season": season,
                "doy": doy,
                "n_valid_pixels": int(anomaly_valid.sum().item()),
                "valid_pixel_fraction": float(anomaly_valid.float().mean().item()),
            }
            daily_entries_by_split[split].append(daily_entry)

            valid_np = anomaly_valid.numpy()
            target_anom_np = target_anomaly[0].numpy()
            group_target_sum[valid_np] += target_anom_np[valid_np]
            group_target_count[valid_np] += 1
            group_meteo_anom_sum += meteo_anomaly.numpy().astype(np.float32, copy=False)
            group_fire_anom_sum += fire_anomaly.numpy().astype(np.float32, copy=False)

            if source_index == len(source_paths) or source_index % 10 == 0:
                print(f"  write_daily={source_index}/{len(source_paths)}")

            del data, target, meteo, fire, obs
            del target_anomaly, meteo_anomaly, fire_anomaly, anomaly_valid

        target_group_mean = np.zeros((H, W), dtype=np.float64)
        valid_group = group_target_count > 0
        target_group_mean[valid_group] = (
            group_target_sum[valid_group] / group_target_count[valid_group]
        )
        max_abs_target_group_mean = (
            float(np.max(np.abs(target_group_mean[valid_group])))
            if np.any(valid_group)
            else 0.0
        )
        meteo_group_mean_absmax = float(
            np.max(np.abs(group_meteo_anom_sum / float(len(source_paths))))
        )
        fire_group_mean_absmax = float(
            np.max(np.abs(group_fire_anom_sum / float(len(source_paths))))
        )

        monthly_entries[-1]["construction_zero_mean_diagnostic"] = {
            "target_max_abs_group_mean": max_abs_target_group_mean,
            "meteo_max_abs_group_mean": meteo_group_mean_absmax,
            "fire_max_abs_group_mean": fire_group_mean_absmax,
        }

        require(max_abs_target_group_mean <= 2e-6,
                f"TARGET_ZERO_MEAN_CONSTRUCTION_FAIL: {max_abs_target_group_mean}")
        require(meteo_group_mean_absmax <= 2e-6,
                f"METEO_ZERO_MEAN_CONSTRUCTION_FAIL: {meteo_group_mean_absmax}")
        require(fire_group_mean_absmax <= 2e-6,
                f"FIRE_ZERO_MEAN_CONSTRUCTION_FAIL: {fire_group_mean_absmax}")

        print(
            "  group_complete "
            f"valid_fraction={float(month_valid_mask.mean()):.6f} "
            f"target_zero_mean_absmax={max_abs_target_group_mean:.3e}"
        )

    for split, expected in (
        ("2022_train", EXPECTED_SPLIT_COUNTS["2022_train"]),
        ("2022_val", EXPECTED_SPLIT_COUNTS["2022_val"]),
        ("2023_all", EXPECTED_SPLIT_COUNTS["2023_all"]),
    ):
        actual = len(daily_entries_by_split[split])
        require(actual == expected, f"DAILY_COUNT_MISMATCH_{split}: {actual}")

    total_daily = sum(len(v) for v in daily_entries_by_split.values())
    require(total_daily == EXPECTED_DAILY_FILES, f"TOTAL_DAILY_FILES_MISMATCH: {total_daily}")

    split_manifest = {
        "artifact_name": "clean_anomaly_split_manifest",
        "artifact_version": 1,
        "contract": {
            "train_months_2022": TRAIN_MONTHS,
            "validation_months_2022": VAL_MONTHS,
            "blacklist_months_2022": BLACKLIST_MONTHS,
            "external_months_2023": EXTERNAL_MONTHS,
        },
        "counts": split_counts,
        "source_records": sorted(
            source_records,
            key=lambda r: (r["year"], r["doy"]),
        ),
    }

    monthly_manifest = {
        "artifact_name": "clean_anomaly_monthly_manifest",
        "artifact_version": 1,
        "n_groups": len(monthly_entries),
        "groups": monthly_entries,
    }

    anomaly_manifest = {
        "artifact_name": "clean_anomaly_daily_manifest",
        "artifact_version": 1,
        "n_files_total": total_daily,
        "groups": [
            {
                "split": split,
                "n_files": len(daily_entries_by_split[split]),
                "files": sorted(daily_entries_by_split[split], key=lambda r: r["doy"]),
            }
            for split in ("2022_train", "2022_val", "2023_all")
        ],
    }

    elapsed = time.time() - build_started
    temp_size_bytes = sum(
        path.stat().st_size
        for path in temp_dir.rglob("*")
        if path.is_file()
    )

    build_report = {
        "artifact_name": "clean_anomaly_cache_build",
        "artifact_version": 1,
        "output_dir": str(output_dir),
        "temporary_build_dir": str(temp_dir),
        "source_dirs": {
            "2022": str(dir_2022),
            "2023": str(dir_2023),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": preflight_sha256,
            "accepted": True,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": builder_sha256,
        },
        "contract": {
            "train_months_2022": TRAIN_MONTHS,
            "validation_months_2022": VAL_MONTHS,
            "blacklist_months_2022": BLACKLIST_MONTHS,
            "external_months_2023": EXTERNAL_MONTHS,
            "min_obs_days_per_pixel": args.min_obs_days_per_pixel,
            "cache_schema": "lean_daily_anomaly_with_source_reference",
            "daily_stored_fields": [
                "target_log_anomaly",
                "anomaly_valid_mask",
                "meteo_anomaly",
                "fire_anomaly",
            ],
            "monthly_stored_fields": [
                "target_log_month_mean",
                "obs_count_map",
                "month_valid_mask",
                "meteo_month_mean",
                "fire_month_mean",
            ],
            "aef_storage": "source_file_reference_only",
            "emission_storage": "not_duplicated_in_anomaly_cache",
        },
        "counts": {
            "monthly_groups": len(monthly_entries),
            "daily_files_total": total_daily,
            "daily_files_by_split": {
                split: len(entries)
                for split, entries in daily_entries_by_split.items()
            },
            "blacklist_files_excluded": split_counts["2022_blacklist"],
        },
        "size_and_runtime": {
            "cache_size_bytes_before_manifests": temp_size_bytes,
            "cache_size_gib_before_manifests": temp_size_bytes / 1024 ** 3,
            "elapsed_seconds_before_manifests": elapsed,
        },
        "verification": {
            "preflight_provenance_pass": True,
            "exact_split_contract_pass": True,
            "all_source_payloads_validated": True,
            "all_source_files_sha256_recorded": True,
            "all_monthly_artifacts_sha256_recorded": True,
            "all_daily_artifacts_sha256_recorded": True,
            "all_group_zero_mean_construction_checks_pass": True,
            "old_06_phase2_cache_modified": False,
            "model_training_started": False,
            "overall_pass": True,
        },
    }

    save_json(split_manifest, temp_path(temp_dir, "manifests", "split_manifest.json"))
    save_json(monthly_manifest, temp_path(temp_dir, "manifests", "monthly_manifest.json"))
    save_json(anomaly_manifest, temp_path(temp_dir, "manifests", "anomaly_manifest.json"))
    save_json(build_report, temp_path(temp_dir, "manifests", "build_report.json"))

    manifest_hashes = {
        name: sha256_file(temp_path(temp_dir, "manifests", name))
        for name in (
            "split_manifest.json",
            "monthly_manifest.json",
            "anomaly_manifest.json",
            "build_report.json",
        )
    }
    seal = {
        "artifact_name": "clean_anomaly_cache_build_seal",
        "artifact_version": 1,
        "output_dir": str(output_dir),
        "manifest_sha256": manifest_hashes,
        "verification": {
            "build_complete": True,
            "ready_for_independent_audit": True,
            "overall_pass": True,
        },
    }
    save_json(seal, temp_path(temp_dir, "manifests", "build_seal.json"))

    # Same-volume rename; final directory appears only after a complete build.
    temp_dir.rename(output_dir)

    disk_after = shutil.disk_usage(output_dir.parent)
    final_size_bytes = sum(
        path.stat().st_size
        for path in output_dir.rglob("*")
        if path.is_file()
    )

    print()
    print("ANOMALY_CLEAN_CACHE_BUILD_PASS")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"MONTHLY_GROUPS={len(monthly_entries)}")
    print(f"DAILY_FILES_TOTAL={total_daily}")
    print(f"2022_TRAIN_FILES={len(daily_entries_by_split['2022_train'])}")
    print(f"2022_VAL_FILES={len(daily_entries_by_split['2022_val'])}")
    print(f"2022_BLACKLIST_FILES_EXCLUDED={split_counts['2022_blacklist']}")
    print(f"2023_EXTERNAL_FILES={len(daily_entries_by_split['2023_all'])}")
    print(f"MIN_OBS_DAYS_PER_PIXEL={args.min_obs_days_per_pixel}")
    print(f"CACHE_SIZE_GIB={final_size_bytes / 1024**3:.3f}")
    print(f"ELAPSED_MINUTES={(time.time() - build_started) / 60.0:.2f}")
    print(f"FREE_DISK_GIB_AFTER={disk_after.free / 1024**3:.2f}")
    print(f"PREFLIGHT_SHA256={preflight_sha256}")
    print(f"BUILDER_SHA256={builder_sha256}")
    print("ALL_SOURCE_FILES_SHA256_RECORDED=True")
    print("ALL_OUTPUT_ARTIFACTS_SHA256_RECORDED=True")
    print("GROUP_ZERO_MEAN_CONSTRUCTION_PASS=True")
    print("OLD_06_PHASE2_CACHE_MODIFIED=False")
    print("MODEL_TRAINING_STARTED=False")
    print(f"BUILD_REPORT={output_dir / 'manifests' / 'build_report.json'}")
    print(f"BUILD_SEAL={output_dir / 'manifests' / 'build_seal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
