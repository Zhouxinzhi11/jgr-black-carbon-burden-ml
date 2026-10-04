# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
39_rebuild_daily_tensors_meteo_trainonly_from_raw.py

Gate 8B:
Build the final 2022 train-only meteorological daily tensors directly from
the original fused NetCDF, without using the previously saved float16
meteorological z-scores.

The script streams the NetCDF once and aggregates all available integer DOYs
and five meteorological variables to the 361 x 576 MERRA-2 grid. It then
loads each original daily tensor and replaces only x[43:48] with:

    z = (daily_grid_mean(raw_meteo) - train_mean) / train_std

cast to the original x dtype.

All other input channels, targets, masks, flags, and metadata are preserved
exactly. The old tensors and Gate-8A tensors are never overwritten.

Temporary aggregation arrays use about 2.2 GB of disk and are deleted after
a successful run unless --keep-work-files is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


H = 361
W = 576
N_GRID = H * W
LAT_MIN = -90.0
LON_MIN = -180.0
LAT_RES = 0.5
LON_RES = 0.625

METEO_VARS = [
    "Temp",
    "RH",
    "BLH",
    "Windspeed_U",
    "Windspeed_V",
]

FEATURE_INDICES = {
    "Temp": 43,
    "RH": 44,
    "BLH": 45,
    "Windspeed_U": 46,
    "Windspeed_V": 47,
}

EXPECTED_X_SHAPE = (112, H, W)
EXPECTED_Y_SHAPE = (10, H, W)
EXPECTED_MASK_SHAPE = (H, W)


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


def day_from_filename(path: Path) -> int:
    match = re.fullmatch(r"day_(\d{3})\.pt", path.name)
    if not match:
        raise ValueError(f"Unexpected tensor filename: {path.name}")
    return int(match.group(1))


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and torch.equal(a, b)
        )
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def verify_unchanged_except_meteo(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> None:
    if set(old_payload) != set(new_payload):
        raise AssertionError("Payload keys changed")

    old_x = old_payload["x"]
    new_x = new_payload["x"]

    if not torch.equal(old_x[:43], new_x[:43]):
        raise AssertionError("x[0:43] changed unexpectedly")
    if not torch.equal(old_x[48:], new_x[48:]):
        raise AssertionError("x[48:112] changed unexpectedly")

    for key in old_payload:
        if key == "x":
            continue
        if not values_equal(old_payload[key], new_payload[key]):
            raise AssertionError(f"Field changed unexpectedly: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Directly rebuild only the five meteorological daily tensor "
            "channels from the original fused NetCDF using train-only stats."
        )
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-stats", required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--expected-files",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--keep-work-files",
        action="store_true",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    old_dir = Path(args.old_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_stats_path = Path(args.train_stats).expanduser().resolve()

    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)
    if not old_dir.is_dir():
        raise FileNotFoundError(old_dir)
    if not train_stats_path.is_file():
        raise FileNotFoundError(train_stats_path)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if old_dir == output_dir:
        raise ValueError("Old and output directories must differ")

    source_files = sorted(old_dir.glob("day_*.pt"))
    if len(source_files) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} source tensors, "
            f"found {len(source_files)}"
        )

    if output_dir.exists():
        existing = list(output_dir.glob("day_*.pt"))
        if existing:
            raise RuntimeError(
                f"Output directory already contains {len(existing)} day tensors. "
                "Use a new empty directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_artifact = load_json(train_stats_path)
    if not train_artifact.get("old_full_year_stats_reproduced", False):
        raise RuntimeError("Train statistics are not approved by Gate 7")
    if train_artifact.get("train_months") != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError(
            f"Unexpected train months: {train_artifact.get('train_months')}"
        )

    train_only_stats = train_artifact["train_only_stats"]
    feature_indices = train_artifact["feature_indices"]
    for var in METEO_VARS:
        if int(feature_indices[var]) != FEATURE_INDICES[var]:
            raise RuntimeError(f"Feature index mismatch for {var}")

    source_days = [day_from_filename(path) for path in source_files]
    if len(set(source_days)) != len(source_days):
        raise RuntimeError("Duplicate daily tensor DOYs found")

    day_to_slot = np.full(367, -1, dtype=np.int16)
    for slot, day in enumerate(source_days):
        if day < 1 or day > 366:
            raise ValueError(f"Invalid tensor DOY: {day}")
        day_to_slot[day] = slot

    n_slots = len(source_days) * N_GRID
    work_dir = output_dir / "_gate8b_direct_raw_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    counts_path = work_dir / "counts.int32.dat"
    sums_path = work_dir / "meteo_sums.float64.dat"

    counts = np.memmap(
        counts_path,
        mode="w+",
        dtype=np.int32,
        shape=(n_slots,),
    )
    sums = np.memmap(
        sums_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(METEO_VARS), n_slots),
    )
    counts[:] = 0
    sums[:] = 0.0
    counts.flush()
    sums.flush()

    started = time.time()
    processed_rows = 0
    matched_rows = 0
    invalid_coordinate_rows = 0

    try:
        with xr.open_dataset(
            nc_path,
            decode_times=False,
            mask_and_scale=False,
        ) as ds:
            required = [
                "DOY",
                "Latitude",
                "Longitude",
                *METEO_VARS,
            ]
            missing = [name for name in required if name not in ds.variables]
            if missing:
                raise KeyError(f"Missing NetCDF variables: {missing}")
            if "num_samples" not in ds.sizes:
                raise KeyError("Dimension num_samples not found")

            n_samples = int(ds.sizes["num_samples"])
            print("STAGE=STREAM_RAW_NETCDF", flush=True)

            for start in range(0, n_samples, args.chunk_size):
                stop = min(start + args.chunk_size, n_samples)
                selector = {"num_samples": slice(start, stop)}

                doy = np.asarray(
                    ds["DOY"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                lat = np.asarray(
                    ds["Latitude"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                lon = np.asarray(
                    ds["Longitude"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)

                finite = (
                    np.isfinite(doy)
                    & np.isfinite(lat)
                    & np.isfinite(lon)
                )
                integer_doy = np.zeros(doy.shape, dtype=np.int16)
                integer_doy[finite] = doy[finite].astype(
                    np.int16,
                    copy=False,
                )

                valid_day = (
                    finite
                    & (integer_doy >= 1)
                    & (integer_doy <= 366)
                    & (day_to_slot[integer_doy.clip(0, 366)] >= 0)
                )

                lat_idx = np.rint(
                    (lat - LAT_MIN) / LAT_RES
                ).astype(np.int32, copy=False)
                lon_idx = np.rint(
                    (lon - LON_MIN) / LON_RES
                ).astype(np.int32, copy=False)

                valid_coordinate = (
                    valid_day
                    & (lat_idx >= 0)
                    & (lat_idx < H)
                    & (lon_idx >= 0)
                    & (lon_idx < W)
                )
                invalid_coordinate_rows += int(
                    np.sum(valid_day & ~valid_coordinate)
                )

                valid_indices = np.flatnonzero(valid_coordinate)
                if valid_indices.size == 0:
                    processed_rows += stop - start
                    print(f"PROGRESS={stop}/{n_samples}", flush=True)
                    continue

                slots = day_to_slot[
                    integer_doy[valid_indices]
                ].astype(np.int64, copy=False)
                grid_ids = (
                    lat_idx[valid_indices].astype(np.int64) * W
                    + lon_idx[valid_indices].astype(np.int64)
                )
                keys = slots * N_GRID + grid_ids

                order = np.argsort(keys, kind="mergesort")
                sorted_keys = keys[order]
                unique_keys, first_indices, local_counts = np.unique(
                    sorted_keys,
                    return_index=True,
                    return_counts=True,
                )

                counts[unique_keys] += local_counts.astype(
                    np.int32,
                    copy=False,
                )

                for var_index, var in enumerate(METEO_VARS):
                    values_all = np.asarray(
                        ds[var].isel(selector).values,
                        dtype=np.float32,
                    ).reshape(-1)
                    values = np.nan_to_num(
                        values_all[valid_indices],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    sorted_values = values[order].astype(
                        np.float64,
                        copy=False,
                    )
                    local_sums = np.add.reduceat(
                        sorted_values,
                        first_indices,
                    )
                    sums[var_index, unique_keys] += local_sums

                matched_rows += int(valid_indices.size)
                processed_rows += stop - start
                print(
                    f"PROGRESS={stop}/{n_samples} "
                    f"MATCHED_ROWS={matched_rows}",
                    flush=True,
                )

        counts.flush()
        sums.flush()

        print("STAGE=WRITE_DIRECT_DAILY_TENSORS", flush=True)

        file_records: list[dict[str, Any]] = []
        total_output_bytes = 0

        for position, source_path in enumerate(source_files, start=1):
            day = day_from_filename(source_path)
            slot = int(day_to_slot[day])
            left = slot * N_GRID
            right = left + N_GRID

            count_segment = np.asarray(
                counts[left:right],
                dtype=np.float64,
            )
            safe_counts = count_segment.copy()
            safe_counts[safe_counts == 0] = 1.0

            old_payload = load_pt(source_path)
            if tuple(old_payload["x"].shape) != EXPECTED_X_SHAPE:
                raise ValueError(
                    f"{source_path.name}: invalid x shape "
                    f"{tuple(old_payload['x'].shape)}"
                )
            if tuple(old_payload["y"].shape) != EXPECTED_Y_SHAPE:
                raise ValueError(
                    f"{source_path.name}: invalid y shape"
                )
            if tuple(old_payload["obs_mask"].shape) != EXPECTED_MASK_SHAPE:
                raise ValueError(
                    f"{source_path.name}: invalid obs_mask shape"
                )

            x_new = old_payload["x"].clone()
            per_variable_stats: dict[str, Any] = {}

            for var_index, var in enumerate(METEO_VARS):
                raw_grid = (
                    np.asarray(
                        sums[var_index, left:right],
                        dtype=np.float64,
                    )
                    / safe_counts
                ).reshape(H, W)

                mean = float(train_only_stats[var]["mean"])
                std = float(
                    train_only_stats[var]["std_population"]
                )
                normalized = (
                    (raw_grid - mean) / std
                ).astype(np.float32, copy=False)

                tensor_channel = torch.from_numpy(normalized).to(
                    dtype=old_payload["x"].dtype
                )
                feature_index = FEATURE_INDICES[var]
                x_new[feature_index] = tensor_channel

                occupied = count_segment.reshape(H, W) > 0
                occupied_values = tensor_channel.float()[occupied]
                all_values = tensor_channel.float()

                per_variable_stats[var] = {
                    "feature_index": feature_index,
                    "occupied_grid_count": int(occupied.sum()),
                    "all_grid_min": float(torch.min(all_values).item()),
                    "all_grid_max": float(torch.max(all_values).item()),
                    "occupied_mean": (
                        float(torch.mean(occupied_values).item())
                        if occupied_values.numel()
                        else None
                    ),
                    "occupied_std_population": (
                        float(
                            torch.std(
                                occupied_values,
                                unbiased=False,
                            ).item()
                        )
                        if occupied_values.numel()
                        else None
                    ),
                }

            new_payload = dict(old_payload)
            new_payload["x"] = x_new

            destination = output_dir / source_path.name
            temp_path = destination.with_suffix(".pt.tmp")
            if temp_path.exists():
                temp_path.unlink()
            torch.save(new_payload, temp_path)
            os.replace(temp_path, destination)

            reloaded = load_pt(destination)
            verify_unchanged_except_meteo(
                old_payload,
                reloaded,
            )
            if not torch.equal(
                reloaded["x"][43:48],
                x_new[43:48],
            ):
                raise AssertionError(
                    f"Saved meteo channels changed after reload: "
                    f"{source_path.name}"
                )
            if not torch.isfinite(
                reloaded["x"][43:48].float()
            ).all():
                raise AssertionError(
                    f"Non-finite meteo values in {source_path.name}"
                )

            file_size = destination.stat().st_size
            total_output_bytes += file_size
            file_records.append(
                {
                    "file": source_path.name,
                    "day": day,
                    "raw_row_count": int(
                        np.sum(count_segment)
                    ),
                    "occupied_grid_count": int(
                        np.sum(count_segment > 0)
                    ),
                    "output_bytes": int(file_size),
                    "variables": per_variable_stats,
                }
            )

            print(
                f"WRITE_PROGRESS={position}/{len(source_files)} "
                f"FILE={source_path.name}",
                flush=True,
            )

            del old_payload, new_payload, reloaded, x_new

        output_count = len(list(output_dir.glob("day_*.pt")))
        if output_count != len(source_files):
            raise RuntimeError(
                f"Output count mismatch: "
                f"expected={len(source_files)}, found={output_count}"
            )

        manifest = {
            "artifact_name": (
                "daily_tensor_meteo_trainonly_direct_raw_gate8b"
            ),
            "version": 1,
            "source_nc": str(nc_path),
            "source_tensor_dir": str(old_dir),
            "output_dir": str(output_dir),
            "train_stats": str(train_stats_path),
            "train_stats_sha256": sha256_file(train_stats_path),
            "source_file_count": len(source_files),
            "output_file_count": output_count,
            "processed_netcdf_rows": int(processed_rows),
            "matched_netcdf_rows": int(matched_rows),
            "invalid_coordinate_rows": int(invalid_coordinate_rows),
            "changed_channels": FEATURE_INDICES,
            "construction": (
                "Direct raw-NetCDF same-day same-grid arithmetic mean, "
                "followed by train-only z-score and cast to original x dtype."
            ),
            "unchanged_content_contract": (
                "x[0:43], x[48:112], y, obs_mask, flags, and "
                "all metadata are preserved exactly."
            ),
            "total_output_bytes": int(total_output_bytes),
            "file_records": file_records,
            "elapsed_seconds": time.time() - started,
            "next_required_gate": (
                "Run a revised Gate 9 against direct raw-NetCDF "
                "normalization before constructing phase2 caches."
            ),
        }

        manifest_path = (
            output_dir
            / "meteo_trainonly_direct_raw_manifest_gate8b.json"
        )
        txt_path = (
            output_dir
            / "meteo_trainonly_direct_raw_manifest_gate8b.txt"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        txt_lines = [
            "Daily tensor direct raw meteorological rebuild - Gate 8B",
            f"Source NetCDF: {nc_path}",
            f"Source tensors: {old_dir}",
            f"Output: {output_dir}",
            f"Train stats: {train_stats_path}",
            f"Source files: {len(source_files)}",
            f"Output files: {output_count}",
            f"Processed NetCDF rows: {processed_rows}",
            f"Matched NetCDF rows: {matched_rows}",
            f"Invalid coordinate rows: {invalid_coordinate_rows}",
            f"Total output bytes: {total_output_bytes}",
            "Changed channels: [43, 44, 45, 46, 47]",
            "All other content exactly preserved: True",
        ]
        txt_path.write_text(
            "\n".join(txt_lines),
            encoding="utf-8",
        )

        print("GATE8B_DIRECT_RAW_DAILY_TENSORS_PASS")
        print(f"MANIFEST={manifest_path}")
        print(f"TXT_REPORT={txt_path}")
        print(f"SOURCE_FILES={len(source_files)}")
        print(f"OUTPUT_FILES={output_count}")
        print(f"PROCESSED_NETCDF_ROWS={processed_rows}")
        print(f"MATCHED_NETCDF_ROWS={matched_rows}")
        print(
            "NON_METEOROLOGICAL_CONTENT_EXACTLY_UNCHANGED=True"
        )

    finally:
        try:
            del counts
            del sums
        except Exception:
            pass
        if work_dir.exists() and not args.keep_work_files:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
