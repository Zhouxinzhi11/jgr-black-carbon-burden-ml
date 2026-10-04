# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
43_rebuild_2023_tensors_union_trainonly.py

Gate 10B:
Build the final 2023 clean daily tensors from the row-wise union of the two
source NetCDF files.

Why a union is required:
- The two source files contain overlapping integer DOYs.
- Gate 10A2 showed that their exact (DOY, Latitude, Longitude) keys are
  disjoint on the overlap days, with no conflicting common keys.
- The original sequential checkpoint logic skipped the second file for
  already completed days and therefore omitted complementary samples.

For every source day this script reconstructs directly from raw NetCDF rows:
- x[0:43] : POSP and angular features, using the original 2022 statistics.
- x[43:48]: meteorological features, using approved 2022 train-month-only
            statistics.
- y[0]    : BCCMASS processed target.
- obs_mask: validity from I_380.

The following are copied exactly from the existing 2023 tensor:
- x[48:112] (already-audited AEF channels)
- y[1:9]    (2023 emission placeholders)
- y[9]      (external fire grid)
- day, has_emission, normalized, and any other metadata.

The old 2023 tensors are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
GRID_SIZE = H * W
LAT_MIN = -90.0
LON_MIN = -180.0
LAT_RES = 0.5
LON_RES = 0.625
CONC_SCALE = 1e9

WAVELENGTHS = [
    "380", "410", "443", "490",
    "670", "865", "1610", "2250",
]
POSP_VARS = [
    f"{kind}_{wavelength}"
    for wavelength in WAVELENGTHS
    for kind in ["I", "Q", "U", "Ip", "DOLP"]
]
ANGLES = ["SZA", "VZA", "RAA"]
METEO_VARS = [
    "Temp",
    "RH",
    "BLH",
    "Windspeed_U",
    "Windspeed_V",
]
FEATURE_VARS = POSP_VARS + ANGLES + METEO_VARS

if len(FEATURE_VARS) != 48:
    raise RuntimeError("Internal feature definition is not 48 channels")

METEO_INDICES = {
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


def prepare_day_index(
    integer_doy: np.ndarray,
) -> dict[int, np.ndarray]:
    valid = (integer_doy >= 1) & (integer_doy <= 365)
    valid_indices = np.flatnonzero(valid)
    valid_days = integer_doy[valid_indices]

    order = np.argsort(valid_days, kind="stable")
    sorted_indices = valid_indices[order]
    sorted_days = valid_days[order]

    unique_days, first, counts = np.unique(
        sorted_days,
        return_index=True,
        return_counts=True,
    )

    result: dict[int, np.ndarray] = {}
    for day, start, count in zip(unique_days, first, counts):
        result[int(day)] = sorted_indices[start:start + count]
    return result


def read_rows(
    ds: xr.Dataset,
    indices: np.ndarray,
    variable: str,
    dtype: np.dtype,
) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0,), dtype=dtype)

    selector = xr.DataArray(
        indices.astype(np.int64, copy=False),
        dims="selected_rows",
    )
    return np.asarray(
        ds[variable].isel(num_samples=selector).values,
        dtype=dtype,
    ).reshape(-1)


def combine_rows(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    idx1: np.ndarray,
    idx2: np.ndarray,
    variable: str,
    dtype: np.dtype,
) -> np.ndarray:
    part1 = read_rows(ds1, idx1, variable, dtype)
    part2 = read_rows(ds2, idx2, variable, dtype)

    if part1.size == 0:
        return part2
    if part2.size == 0:
        return part1
    return np.concatenate([part1, part2], axis=0)


def tensor_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float | int]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = torch.abs(af - bf)
    n = int(diff.numel())
    return {
        "count": n,
        "unequal_count": int(
            torch.count_nonzero(diff != 0).item()
        ),
        "exact_fraction": (
            int(torch.count_nonzero(diff == 0).item()) / n
        ),
        "mean_abs": float(torch.mean(diff).item()),
        "max_abs": float(torch.max(diff).item()),
        "rmse": float(
            torch.sqrt(torch.mean((af - bf) ** 2)).item()
        ),
    }


def mask_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float | int | bool]:
    unequal = a.bool().reshape(-1) != b.bool().reshape(-1)
    n = int(unequal.numel())
    count = int(torch.count_nonzero(unequal).item())
    return {
        "count": n,
        "unequal_count": count,
        "unequal_fraction": count / n,
        "exact_match": count == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild 2023 tensors from the union of two complementary "
            "NetCDF sources using 2022 train-only meteorological statistics."
        )
    )
    parser.add_argument("--nc1", required=True)
    parser.add_argument("--nc2", required=True)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-stats", required=True)
    parser.add_argument("--train-stats", required=True)
    parser.add_argument("--gate10a2-report", required=True)
    parser.add_argument(
        "--expected-extra-nc2-keys",
        type=int,
        default=546,
    )
    args = parser.parse_args()

    nc1_path = Path(args.nc1).expanduser().resolve()
    nc2_path = Path(args.nc2).expanduser().resolve()
    old_dir = Path(args.old_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    old_stats_path = Path(args.old_stats).expanduser().resolve()
    train_stats_path = Path(args.train_stats).expanduser().resolve()
    overlap_report_path = Path(
        args.gate10a2_report
    ).expanduser().resolve()

    for path in [
        nc1_path,
        nc2_path,
        old_stats_path,
        train_stats_path,
        overlap_report_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not old_dir.is_dir():
        raise FileNotFoundError(old_dir)
    if old_dir == output_dir:
        raise ValueError("Old and output directories must differ")

    old_files = sorted(old_dir.glob("day_*.pt"))
    if not old_files:
        raise RuntimeError(f"No old tensors in {old_dir}")

    if output_dir.exists():
        existing = list(output_dir.glob("day_*.pt"))
        if existing:
            raise RuntimeError(
                f"Output directory already contains {len(existing)} "
                "day tensors. Use a new empty directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    old_stats = load_json(old_stats_path)
    train_stats = load_json(train_stats_path)
    overlap_report = load_json(overlap_report_path)

    if int(old_stats.get("year", -1)) != 2022:
        raise RuntimeError("Old statistics are not anchored to 2022")
    if len(old_stats.get("mean", [])) < 48:
        raise RuntimeError("Old statistics mean list is too short")
    if len(old_stats.get("std", [])) < 48:
        raise RuntimeError("Old statistics std list is too short")

    if train_stats.get("train_months") != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError("Unexpected approved training months")
    if not train_stats.get(
        "old_full_year_stats_reproduced",
        False,
    ):
        raise RuntimeError("Train-only stats are not Gate-7 approved")

    if overlap_report.get("scientific_classification") != (
        "OVERLAP_CONTAINS_COMPLEMENTARY_NC2_SAMPLES"
    ):
        raise RuntimeError(
            "Gate-10A2 report does not authorize complementary-source union"
        )

    overlap_summary = overlap_report["overlap_summary"]
    if int(
        overlap_summary.get(
            "total_conflicting_common_keys",
            -1,
        )
    ) != 0:
        raise RuntimeError("Gate-10A2 reported conflicting common keys")
    if int(
        overlap_summary.get(
            "total_common_unique_keys",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Cross-file common keys exist; this script expects a disjoint union"
        )
    if int(
        overlap_summary.get(
            "total_nc2_only_unique_keys",
            -1,
        )
    ) != args.expected_extra_nc2_keys:
        raise RuntimeError(
            "Unexpected complementary NC2 key count: "
            f"{overlap_summary.get('total_nc2_only_unique_keys')}"
        )

    old_means = np.asarray(
        old_stats["mean"],
        dtype=np.float32,
    )
    old_stds = np.asarray(
        old_stats["std"],
        dtype=np.float32,
    )
    meteo_train_stats = train_stats["train_only_stats"]

    old_tensor_days = {
        day_from_filename(path): path
        for path in old_files
    }

    started = time.time()

    with xr.open_dataset(
        nc1_path,
        decode_times=False,
        mask_and_scale=False,
    ) as ds1, xr.open_dataset(
        nc2_path,
        decode_times=False,
        mask_and_scale=False,
    ) as ds2:

        required = [
            "DOY",
            "Latitude",
            "Longitude",
            "BCCMASS",
            *FEATURE_VARS,
        ]
        for label, ds in [("nc1", ds1), ("nc2", ds2)]:
            missing = [
                variable
                for variable in required
                if variable not in ds.variables
            ]
            if missing:
                raise KeyError(
                    f"{label} missing variables: {missing}"
                )
            if "num_samples" not in ds.sizes:
                raise KeyError(
                    f"{label} lacks dimension num_samples"
                )

        doy1 = np.asarray(
            ds1["DOY"].values,
            dtype=np.float64,
        ).astype(np.int16, copy=False)
        doy2 = np.asarray(
            ds2["DOY"].values,
            dtype=np.float64,
        ).astype(np.int16, copy=False)

        day_index1 = prepare_day_index(doy1)
        day_index2 = prepare_day_index(doy2)
        source_days = sorted(
            set(day_index1) | set(day_index2)
        )

        if set(source_days) != set(old_tensor_days):
            raise RuntimeError(
                "Source-day union and existing tensor-day inventory differ. "
                f"source_without_tensor="
                f"{sorted(set(source_days) - set(old_tensor_days))}; "
                f"tensor_without_source="
                f"{sorted(set(old_tensor_days) - set(source_days))}"
            )

        overlap_days = sorted(
            set(day_index1) & set(day_index2)
        )

        file_records: list[dict[str, Any]] = []
        total_output_bytes = 0
        total_rows_nc1 = 0
        total_rows_nc2 = 0
        overlap_changed_days = 0
        nonoverlap_old_source_content_pass = True

        for position, day in enumerate(
            source_days,
            start=1,
        ):
            idx1 = day_index1.get(
                day,
                np.empty((0,), dtype=np.int64),
            )
            idx2 = day_index2.get(
                day,
                np.empty((0,), dtype=np.int64),
            )
            total_rows_nc1 += int(idx1.size)
            total_rows_nc2 += int(idx2.size)

            old_path = old_tensor_days[day]
            old_payload = load_pt(old_path)

            if tuple(old_payload["x"].shape) != EXPECTED_X_SHAPE:
                raise RuntimeError(
                    f"{old_path.name}: invalid x shape"
                )
            if tuple(old_payload["y"].shape) != EXPECTED_Y_SHAPE:
                raise RuntimeError(
                    f"{old_path.name}: invalid y shape"
                )
            if tuple(
                old_payload["obs_mask"].shape
            ) != EXPECTED_MASK_SHAPE:
                raise RuntimeError(
                    f"{old_path.name}: invalid mask shape"
                )
            if int(old_payload["day"]) != day:
                raise RuntimeError(
                    f"{old_path.name}: internal day mismatch"
                )

            lats = combine_rows(
                ds1, ds2, idx1, idx2,
                "Latitude", np.float64,
            )
            lons = combine_rows(
                ds1, ds2, idx1, idx2,
                "Longitude", np.float64,
            )

            if lats.size != idx1.size + idx2.size:
                raise RuntimeError(
                    f"Day {day}: combined row count mismatch"
                )

            lat_idx = np.rint(
                (lats - LAT_MIN) / LAT_RES
            ).astype(np.int32)
            lon_idx = np.rint(
                (lons - LON_MIN) / LON_RES
            ).astype(np.int32)

            if (
                np.any(lat_idx < 0)
                or np.any(lat_idx >= H)
                or np.any(lon_idx < 0)
                or np.any(lon_idx >= W)
            ):
                raise RuntimeError(
                    f"Day {day}: out-of-range coordinates"
                )

            flat_idx = lat_idx * W + lon_idx
            counts = np.bincount(
                flat_idx,
                minlength=GRID_SIZE,
            ).astype(np.float32)
            safe_counts = counts.copy()
            safe_counts[safe_counts == 0] = 1.0

            x_new = old_payload["x"].clone()
            y_new = old_payload["y"].clone()

            for feature_index, variable in enumerate(
                FEATURE_VARS
            ):
                values = combine_rows(
                    ds1, ds2, idx1, idx2,
                    variable, np.float32,
                )
                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                sums = np.bincount(
                    flat_idx,
                    weights=values,
                    minlength=GRID_SIZE,
                )
                raw_grid = (
                    sums / safe_counts
                ).reshape(H, W)

                if feature_index < 43:
                    normalized = (
                        raw_grid
                        - float(old_means[feature_index])
                    ) / float(old_stds[feature_index])
                else:
                    item = meteo_train_stats[variable]
                    normalized = (
                        raw_grid - float(item["mean"])
                    ) / float(item["std_population"])

                x_new[feature_index] = torch.from_numpy(
                    normalized.astype(
                        np.float32,
                        copy=False,
                    )
                ).to(dtype=old_payload["x"].dtype)

            bccmass = combine_rows(
                ds1, ds2, idx1, idx2,
                "BCCMASS", np.float32,
            )
            bccmass = np.nan_to_num(
                bccmass,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            bcc_sums = np.bincount(
                flat_idx,
                weights=bccmass,
                minlength=GRID_SIZE,
            )
            bcc_grid = (
                bcc_sums / safe_counts
            ).reshape(H, W)
            y_new[0] = torch.from_numpy(
                np.log1p(
                    bcc_grid * CONC_SCALE
                ).astype(np.float32, copy=False)
            )

            i380 = combine_rows(
                ds1, ds2, idx1, idx2,
                "I_380", np.float32,
            )
            valid_obs = (~np.isnan(i380)).astype(
                np.float32
            )
            obs_counts = np.bincount(
                flat_idx,
                weights=valid_obs,
                minlength=GRID_SIZE,
            )
            obs_mask_new = torch.from_numpy(
                (obs_counts > 0).reshape(H, W)
            ).bool()

            new_payload = dict(old_payload)
            new_payload["x"] = x_new
            new_payload["y"] = y_new
            new_payload["obs_mask"] = obs_mask_new

            # Fields independent of the fused source-row union must be exact.
            if not torch.equal(
                old_payload["x"][48:],
                new_payload["x"][48:],
            ):
                raise AssertionError(
                    f"Day {day}: AEF channels changed"
                )
            if not torch.equal(
                old_payload["y"][1:],
                new_payload["y"][1:],
            ):
                raise AssertionError(
                    f"Day {day}: emission/fire channels changed"
                )
            for key in old_payload:
                if key in {"x", "y", "obs_mask"}:
                    continue
                if not values_equal(
                    old_payload[key],
                    new_payload[key],
                ):
                    raise AssertionError(
                        f"Day {day}: metadata changed: {key}"
                    )

            destination = output_dir / old_path.name
            temp_path = destination.with_suffix(".pt.tmp")
            if temp_path.exists():
                temp_path.unlink()
            torch.save(new_payload, temp_path)
            os.replace(temp_path, destination)

            reloaded = load_pt(destination)
            if not torch.equal(
                reloaded["x"],
                x_new,
            ):
                raise AssertionError(
                    f"Day {day}: saved x changed after reload"
                )
            if not torch.equal(
                reloaded["y"],
                y_new,
            ):
                raise AssertionError(
                    f"Day {day}: saved y changed after reload"
                )
            if not torch.equal(
                reloaded["obs_mask"],
                obs_mask_new,
            ):
                raise AssertionError(
                    f"Day {day}: saved mask changed after reload"
                )

            is_overlap = day in overlap_days

            posp_angle_change = tensor_diff(
                old_payload["x"][:43],
                x_new[:43],
            )
            meteo_change = tensor_diff(
                old_payload["x"][43:48],
                x_new[43:48],
            )
            target_change = tensor_diff(
                old_payload["y"][0],
                y_new[0],
            )
            mask_change = mask_diff(
                old_payload["obs_mask"],
                obs_mask_new,
            )

            if is_overlap and (
                posp_angle_change["unequal_count"] > 0
                or target_change["unequal_count"] > 0
                or mask_change["unequal_count"] > 0
            ):
                overlap_changed_days += 1

            if not is_overlap:
                # Outside overlap days, only meteo normalization should change.
                if posp_angle_change["max_abs"] > 0.00390625:
                    nonoverlap_old_source_content_pass = False
                if target_change["max_abs"] > 1e-5:
                    nonoverlap_old_source_content_pass = False
                if not mask_change["exact_match"]:
                    nonoverlap_old_source_content_pass = False

            file_size = destination.stat().st_size
            total_output_bytes += file_size

            file_records.append(
                {
                    "file": old_path.name,
                    "integer_doy": day,
                    "is_cross_file_overlap_day": is_overlap,
                    "nc1_row_count": int(idx1.size),
                    "nc2_row_count": int(idx2.size),
                    "combined_row_count": int(
                        idx1.size + idx2.size
                    ),
                    "occupied_grid_count": int(
                        np.sum(counts > 0)
                    ),
                    "old_to_new_change": {
                        "posp_and_angles_x_0_43": (
                            posp_angle_change
                        ),
                        "meteorology_x_43_48": (
                            meteo_change
                        ),
                        "bccmass_y0": target_change,
                        "obs_mask": mask_change,
                    },
                    "output_bytes": int(file_size),
                }
            )

            print(
                f"PROGRESS={position}/{len(source_days)} "
                f"DOY={day} "
                f"NC1_ROWS={idx1.size} "
                f"NC2_ROWS={idx2.size} "
                f"OVERLAP={is_overlap}",
                flush=True,
            )

            del (
                old_payload,
                new_payload,
                reloaded,
                x_new,
                y_new,
                obs_mask_new,
            )

    output_files = sorted(output_dir.glob("day_*.pt"))
    if len(output_files) != len(old_files):
        raise RuntimeError(
            f"Output count mismatch: expected={len(old_files)}, "
            f"found={len(output_files)}"
        )
    if [p.name for p in output_files] != [
        p.name for p in old_files
    ]:
        raise RuntimeError(
            "Output filename set differs from old tensor set"
        )
    if not nonoverlap_old_source_content_pass:
        raise RuntimeError(
            "At least one non-overlap day changed in POSP/target/mask "
            "beyond the declared numerical tolerance"
        )

    manifest = {
        "artifact_name": (
            "daily_tensor_2023_union_trainonly_gate10b"
        ),
        "version": 1,
        "inputs": {
            "nc1": str(nc1_path),
            "nc2": str(nc2_path),
            "old_tensor_dir": str(old_dir),
            "old_stats": str(old_stats_path),
            "train_stats": str(train_stats_path),
            "gate10a2_report": str(overlap_report_path),
        },
        "hashes": {
            "old_stats_sha256": sha256_file(
                old_stats_path
            ),
            "train_stats_sha256": sha256_file(
                train_stats_path
            ),
            "gate10a2_report_sha256": sha256_file(
                overlap_report_path
            ),
        },
        "output_dir": str(output_dir),
        "source_day_count": len(source_days),
        "output_file_count": len(output_files),
        "cross_file_overlap_day_count": len(
            overlap_days
        ),
        "cross_file_overlap_days": overlap_days,
        "total_rows_from_nc1": int(total_rows_nc1),
        "total_rows_from_nc2": int(total_rows_nc2),
        "gate10a2_complementary_nc2_unique_keys": int(
            overlap_summary[
                "total_nc2_only_unique_keys"
            ]
        ),
        "overlap_days_with_source_content_change": int(
            overlap_changed_days
        ),
        "nonoverlap_old_source_content_reproduction_pass": bool(
            nonoverlap_old_source_content_pass
        ),
        "reconstructed_fields": {
            "x_0_43": (
                "POSP and angular channels from raw source-row union "
                "using original 2022 normalization statistics"
            ),
            "x_43_48": (
                "Meteorological channels from raw source-row union "
                "using 2022 train-month-only statistics"
            ),
            "y_0": (
                "BCCMASS same-day same-grid arithmetic mean, "
                "scaled by 1e9 and log1p transformed"
            ),
            "obs_mask": (
                "At least one finite I_380 source sample in the grid cell"
            ),
        },
        "exactly_preserved_fields": [
            "x[48:112]",
            "y[1:10]",
            "day",
            "has_emission",
            "normalized",
            "all other metadata",
        ],
        "total_output_bytes": int(
            total_output_bytes
        ),
        "file_records": file_records,
        "elapsed_seconds": time.time() - started,
        "next_required_gate": (
            "Run Gate 10C independent raw-source audit before "
            "building any phase2 cache."
        ),
    }

    manifest_path = (
        output_dir
        / "tensor_2023_union_trainonly_manifest_gate10b.json"
    )
    txt_path = (
        output_dir
        / "tensor_2023_union_trainonly_manifest_gate10b.txt"
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

    lines = [
        "2023 tensor union and train-only rebuild - Gate 10B",
        f"NC1: {nc1_path}",
        f"NC2: {nc2_path}",
        f"Old tensors: {old_dir}",
        f"Output: {output_dir}",
        f"Source days: {len(source_days)}",
        f"Output files: {len(output_files)}",
        f"Overlap days: {len(overlap_days)}",
        f"Total rows from NC1: {total_rows_nc1}",
        f"Total rows from NC2: {total_rows_nc2}",
        (
            "Complementary NC2 unique keys on overlap days: "
            f"{overlap_summary['total_nc2_only_unique_keys']}"
        ),
        (
            "Overlap days with POSP/target/mask change: "
            f"{overlap_changed_days}"
        ),
        (
            "Non-overlap old source-content reproduction pass: "
            f"{nonoverlap_old_source_content_pass}"
        ),
        f"Total output bytes: {total_output_bytes}",
    ]
    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("GATE10B_2023_UNION_TENSORS_PASS")
    print(f"MANIFEST={manifest_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"SOURCE_DAYS={len(source_days)}")
    print(f"OUTPUT_FILES={len(output_files)}")
    print(f"OVERLAP_DAYS={len(overlap_days)}")
    print(
        "COMPLEMENTARY_NC2_UNIQUE_KEYS="
        f"{overlap_summary['total_nc2_only_unique_keys']}"
    )
    print(
        "OVERLAP_DAYS_WITH_SOURCE_CONTENT_CHANGE="
        f"{overlap_changed_days}"
    )
    print(
        "NONOVERLAP_SOURCE_CONTENT_REPRODUCTION_PASS=True"
    )
    print(
        "PRESERVED_AEF_FIRE_EMISSION_METADATA_EXACTLY=True"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
