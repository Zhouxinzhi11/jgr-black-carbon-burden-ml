# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
44_audit_2023_union_trainonly_tensors_gate10c.py

Gate 10C: independent, read-only audit of the final 2023 clean daily tensors.

The Gate-10B builder used np.bincount aggregation. This audit uses an
independent sorted reduce-at implementation to reconstruct the source fields
from the two raw NetCDF files.

Audit coverage:
- all cross-file overlap days declared by Gate 10B;
- four automatically selected NC1-only days;
- four automatically selected NC2-only days.

For every audited day, reconstruct and verify:
- x[0:43]  POSP and angular channels using the original 2022 statistics;
- x[43:48] meteorological channels using approved 2022 train-month-only stats;
- y[0]     BCCMASS grid mean, x1e9, log1p;
- obs_mask finite I_380 presence.

Also verify:
- x[48:112], y[1:10], and all metadata are exactly preserved from old tensors;
- cross-file sample keys remain disjoint on all overlap days;
- the total number of complementary NC2 unique keys is exactly 546;
- non-overlap POSP/angle, BCCMASS, and mask content reproduces the old tensors;
- every overlap day has a genuine source-content change after adding NC2 rows.

No file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def sample_positions(days: list[int], count: int = 4) -> list[int]:
    if not days:
        return []
    if len(days) <= count:
        return list(days)

    positions = np.linspace(
        0,
        len(days) - 1,
        num=count,
        dtype=np.int64,
    )
    return [days[int(position)] for position in positions]


def make_exact_keys(
    doy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("doy", np.float64),
            ("lat", np.float64),
            ("lon", np.float64),
        ]
    )
    keys = np.empty(doy.size, dtype=dtype)
    keys["doy"] = np.asarray(doy, dtype=np.float64)
    keys["lat"] = np.asarray(lat, dtype=np.float64)
    keys["lon"] = np.asarray(lon, dtype=np.float64)
    return keys


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


def prepare_sorted_groups(
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict[str, np.ndarray]:
    lat_idx = np.rint(
        (np.asarray(lats, dtype=np.float64) - LAT_MIN) / LAT_RES
    ).astype(np.int32)
    lon_idx = np.rint(
        (np.asarray(lons, dtype=np.float64) - LON_MIN) / LON_RES
    ).astype(np.int32)

    if (
        np.any(lat_idx < 0)
        or np.any(lat_idx >= H)
        or np.any(lon_idx < 0)
        or np.any(lon_idx >= W)
    ):
        raise RuntimeError("Out-of-range coordinates in audited source rows")

    flat_idx = lat_idx * W + lon_idx
    order = np.argsort(flat_idx, kind="stable")
    sorted_flat = flat_idx[order]

    unique_flat, starts, counts = np.unique(
        sorted_flat,
        return_index=True,
        return_counts=True,
    )

    return {
        "order": order,
        "unique_flat": unique_flat,
        "starts": starts,
        "counts": counts.astype(np.float64),
    }


def aggregate_numeric(
    values: np.ndarray,
    grouping: dict[str, np.ndarray],
) -> np.ndarray:
    clean = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    sorted_values = clean[grouping["order"]]
    sums = np.add.reduceat(
        sorted_values,
        grouping["starts"],
    )
    means = sums / grouping["counts"]

    grid = np.zeros(GRID_SIZE, dtype=np.float64)
    grid[grouping["unique_flat"]] = means
    return grid.reshape(H, W)


def aggregate_valid_mask(
    i380: np.ndarray,
    grouping: dict[str, np.ndarray],
) -> np.ndarray:
    valid = (~np.isnan(
        np.asarray(i380, dtype=np.float64)
    )).astype(np.float64)
    sorted_valid = valid[grouping["order"]]
    valid_sums = np.add.reduceat(
        sorted_valid,
        grouping["starts"],
    )

    flat = np.zeros(GRID_SIZE, dtype=bool)
    flat[grouping["unique_flat"]] = valid_sums > 0
    return flat.reshape(H, W)


def reconstruct_union_day(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    idx1: np.ndarray,
    idx2: np.ndarray,
    old_means: np.ndarray,
    old_stds: np.ndarray,
    train_stats: dict[str, Any],
    output_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    lats = combine_rows(
        ds1, ds2, idx1, idx2,
        "Latitude", np.float64,
    )
    lons = combine_rows(
        ds1, ds2, idx1, idx2,
        "Longitude", np.float64,
    )
    grouping = prepare_sorted_groups(lats, lons)

    x = torch.empty(
        (48, H, W),
        dtype=output_dtype,
    )

    for feature_index, variable in enumerate(FEATURE_VARS):
        values = combine_rows(
            ds1, ds2, idx1, idx2,
            variable, np.float32,
        )
        raw_grid = aggregate_numeric(values, grouping)

        if feature_index < 43:
            normalized = (
                raw_grid - float(old_means[feature_index])
            ) / float(old_stds[feature_index])
        else:
            item = train_stats[variable]
            normalized = (
                raw_grid - float(item["mean"])
            ) / float(item["std_population"])

        x[feature_index] = torch.from_numpy(
            normalized.astype(np.float32, copy=False)
        ).to(dtype=output_dtype)

    bccmass = combine_rows(
        ds1, ds2, idx1, idx2,
        "BCCMASS", np.float32,
    )
    bcc_grid = aggregate_numeric(
        bccmass,
        grouping,
    )
    y0 = torch.from_numpy(
        np.log1p(
            bcc_grid * CONC_SCALE
        ).astype(np.float32, copy=False)
    )

    i380 = combine_rows(
        ds1, ds2, idx1, idx2,
        "I_380", np.float32,
    )
    obs_mask = torch.from_numpy(
        aggregate_valid_mask(
            i380,
            grouping,
        )
    ).bool()

    return {
        "x_0_48": x,
        "y0": y0,
        "obs_mask": obs_mask,
    }


def diff_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    diff = torch.abs(a - b)
    n = int(diff.numel())

    sorted_diff = torch.sort(diff).values
    q99_index = min(
        max(int(math.ceil(0.99 * n)) - 1, 0),
        n - 1,
    )

    return {
        "count": n,
        "unequal_count": int(
            torch.count_nonzero(diff != 0).item()
        ),
        "exact_fraction": (
            int(torch.count_nonzero(diff == 0).item()) / n
        ),
        "mean_abs": float(torch.mean(diff).item()),
        "q99_abs": float(sorted_diff[q99_index].item()),
        "max_abs": float(torch.max(diff).item()),
        "rmse": float(
            torch.sqrt(torch.mean((a - b) ** 2)).item()
        ),
    }


def mask_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int | bool]:
    unequal = (
        actual.bool().reshape(-1)
        != expected.bool().reshape(-1)
    )
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
            "Independent sorted-reduction audit of Gate-10B 2023 "
            "union and train-only daily tensors."
        )
    )
    parser.add_argument("--nc1", required=True)
    parser.add_argument("--nc2", required=True)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--old-stats", required=True)
    parser.add_argument("--train-stats", required=True)
    parser.add_argument("--gate10a2-report", required=True)
    parser.add_argument("--gate10b-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--x-mean-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--x-q99-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--x-max-threshold",
        type=float,
        default=0.0078125,
    )
    parser.add_argument(
        "--y-mean-threshold",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--y-q99-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--y-max-threshold",
        type=float,
        default=1e-5,
    )
    args = parser.parse_args()

    nc1_path = Path(args.nc1).expanduser().resolve()
    nc2_path = Path(args.nc2).expanduser().resolve()
    old_dir = Path(args.old_dir).expanduser().resolve()
    new_dir = Path(args.new_dir).expanduser().resolve()
    old_stats_path = Path(args.old_stats).expanduser().resolve()
    train_stats_path = Path(args.train_stats).expanduser().resolve()
    gate10a2_path = Path(
        args.gate10a2_report
    ).expanduser().resolve()
    gate10b_path = Path(
        args.gate10b_manifest
    ).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in [
        nc1_path,
        nc2_path,
        old_stats_path,
        train_stats_path,
        gate10a2_path,
        gate10b_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in [old_dir, new_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / "tensor_2023_union_trainonly_audit_gate10c.json"
    )
    txt_path = (
        output_dir
        / "tensor_2023_union_trainonly_audit_gate10c.txt"
    )
    if json_path.exists() or txt_path.exists():
        raise RuntimeError(
            "Gate-10C report already exists. Use a new output directory "
            "or archive the previous report."
        )

    old_stats_artifact = load_json(old_stats_path)
    train_stats_artifact = load_json(train_stats_path)
    gate10a2 = load_json(gate10a2_path)
    gate10b = load_json(gate10b_path)

    if int(old_stats_artifact.get("year", -1)) != 2022:
        raise RuntimeError("Old statistics are not anchored to 2022")
    if train_stats_artifact.get(
        "train_months"
    ) != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError("Unexpected train-month definition")
    if not train_stats_artifact.get(
        "old_full_year_stats_reproduced",
        False,
    ):
        raise RuntimeError("Gate-7 train-only statistics are not approved")

    if gate10a2.get(
        "scientific_classification"
    ) != "OVERLAP_CONTAINS_COMPLEMENTARY_NC2_SAMPLES":
        raise RuntimeError(
            "Gate-10A2 classification does not authorize a source union"
        )

    if gate10b.get(
        "artifact_name"
    ) != "daily_tensor_2023_union_trainonly_gate10b":
        raise RuntimeError("Unexpected Gate-10B artifact")
    if int(gate10b.get("source_day_count", -1)) != 361:
        raise RuntimeError("Gate-10B source_day_count is not 361")
    if int(gate10b.get("output_file_count", -1)) != 361:
        raise RuntimeError("Gate-10B output_file_count is not 361")
    if int(
        gate10b.get(
            "cross_file_overlap_day_count",
            -1,
        )
    ) != 44:
        raise RuntimeError("Gate-10B overlap day count is not 44")
    if int(
        gate10b.get(
            "gate10a2_complementary_nc2_unique_keys",
            -1,
        )
    ) != 546:
        raise RuntimeError(
            "Gate-10B complementary NC2 key count is not 546"
        )
    if int(
        gate10b.get(
            "overlap_days_with_source_content_change",
            -1,
        )
    ) != 44:
        raise RuntimeError(
            "Gate-10B did not report source-content change on all overlap days"
        )
    if not gate10b.get(
        "nonoverlap_old_source_content_reproduction_pass",
        False,
    ):
        raise RuntimeError(
            "Gate-10B non-overlap reproduction was not approved"
        )

    declared_hashes = gate10b.get("hashes", {})
    checks = {
        "old_stats_sha256": (
            sha256_file(old_stats_path)
        ),
        "train_stats_sha256": (
            sha256_file(train_stats_path)
        ),
        "gate10a2_report_sha256": (
            sha256_file(gate10a2_path)
        ),
    }
    for key, actual_hash in checks.items():
        if declared_hashes.get(key) != actual_hash:
            raise RuntimeError(
                f"Input artifact hash mismatch: {key}"
            )

    old_files = sorted(old_dir.glob("day_*.pt"))
    new_files = sorted(new_dir.glob("day_*.pt"))
    if len(old_files) != 361 or len(new_files) != 361:
        raise RuntimeError(
            f"Expected 361 old and 361 new tensors; "
            f"found old={len(old_files)}, new={len(new_files)}"
        )
    if [p.name for p in old_files] != [p.name for p in new_files]:
        raise RuntimeError("Old and new tensor filename sets differ")

    old_means = np.asarray(
        old_stats_artifact["mean"],
        dtype=np.float64,
    )
    old_stds = np.asarray(
        old_stats_artifact["std"],
        dtype=np.float64,
    )
    train_stats = train_stats_artifact["train_only_stats"]

    overlap_days = sorted(
        int(day)
        for day in gate10b["cross_file_overlap_days"]
    )

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

        doy1_raw = np.asarray(
            ds1["DOY"].values,
            dtype=np.float64,
        ).reshape(-1)
        doy2_raw = np.asarray(
            ds2["DOY"].values,
            dtype=np.float64,
        ).reshape(-1)
        int_doy1 = doy1_raw.astype(np.int16, copy=False)
        int_doy2 = doy2_raw.astype(np.int16, copy=False)

        days1 = set(
            int(value)
            for value in np.unique(int_doy1)
            if 1 <= int(value) <= 365
        )
        days2 = set(
            int(value)
            for value in np.unique(int_doy2)
            if 1 <= int(value) <= 365
        )

        observed_overlap = sorted(days1 & days2)
        if observed_overlap != overlap_days:
            raise RuntimeError(
                "Raw-source overlap days differ from Gate-10B manifest"
            )

        nc1_only_days = sorted(days1 - days2)
        nc2_only_days = sorted(days2 - days1)
        selected_nc1_only = sample_positions(
            nc1_only_days,
            count=4,
        )
        selected_nc2_only = sample_positions(
            nc2_only_days,
            count=4,
        )
        audited_days = sorted(
            set(
                overlap_days
                + selected_nc1_only
                + selected_nc2_only
            )
        )

        print(
            f"AUDITED_OVERLAP_DAYS={overlap_days}",
            flush=True,
        )
        print(
            f"SELECTED_NC1_ONLY_DAYS={selected_nc1_only}",
            flush=True,
        )
        print(
            f"SELECTED_NC2_ONLY_DAYS={selected_nc2_only}",
            flush=True,
        )

        day_reports: list[dict[str, Any]] = []
        total_nc2_unique_keys = 0
        total_common_cross_file_keys = 0

        raw_union_pass = True
        preserved_fields_pass = True
        nonoverlap_reproduction_pass = True
        overlap_change_pass = True

        for position, day in enumerate(
            audited_days,
            start=1,
        ):
            idx1 = np.flatnonzero(int_doy1 == day)
            idx2 = np.flatnonzero(int_doy2 == day)

            if idx1.size == 0 and idx2.size == 0:
                raise RuntimeError(
                    f"No raw rows for audited DOY {day}"
                )

            is_overlap = day in overlap_days

            if is_overlap:
                lat1 = read_rows(
                    ds1, idx1, "Latitude", np.float64
                )
                lon1 = read_rows(
                    ds1, idx1, "Longitude", np.float64
                )
                lat2 = read_rows(
                    ds2, idx2, "Latitude", np.float64
                )
                lon2 = read_rows(
                    ds2, idx2, "Longitude", np.float64
                )

                keys1 = np.unique(
                    make_exact_keys(
                        doy1_raw[idx1],
                        lat1,
                        lon1,
                    )
                )
                keys2 = np.unique(
                    make_exact_keys(
                        doy2_raw[idx2],
                        lat2,
                        lon2,
                    )
                )
                common_keys = np.intersect1d(
                    keys1,
                    keys2,
                    assume_unique=True,
                )

                total_nc2_unique_keys += int(keys2.size)
                total_common_cross_file_keys += int(
                    common_keys.size
                )

                if common_keys.size != 0:
                    raise RuntimeError(
                        f"DOY {day}: cross-file exact keys are not disjoint"
                    )

            old_path = old_dir / f"day_{day:03d}.pt"
            new_path = new_dir / f"day_{day:03d}.pt"
            if not old_path.is_file():
                raise FileNotFoundError(old_path)
            if not new_path.is_file():
                raise FileNotFoundError(new_path)

            old_payload = load_pt(old_path)
            new_payload = load_pt(new_path)

            if tuple(new_payload["x"].shape) != EXPECTED_X_SHAPE:
                raise RuntimeError(
                    f"{new_path.name}: invalid x shape"
                )
            if tuple(new_payload["y"].shape) != EXPECTED_Y_SHAPE:
                raise RuntimeError(
                    f"{new_path.name}: invalid y shape"
                )
            if tuple(
                new_payload["obs_mask"].shape
            ) != EXPECTED_MASK_SHAPE:
                raise RuntimeError(
                    f"{new_path.name}: invalid mask shape"
                )
            if int(new_payload["day"]) != day:
                raise RuntimeError(
                    f"{new_path.name}: internal day mismatch"
                )

            reconstructed = reconstruct_union_day(
                ds1=ds1,
                ds2=ds2,
                idx1=idx1,
                idx2=idx2,
                old_means=old_means,
                old_stds=old_stds,
                train_stats=train_stats,
                output_dtype=new_payload["x"].dtype,
            )

            x_posp_angle = diff_metrics(
                new_payload["x"][:43],
                reconstructed["x_0_48"][:43],
            )
            x_meteo = diff_metrics(
                new_payload["x"][43:48],
                reconstructed["x_0_48"][43:48],
            )
            y0 = diff_metrics(
                new_payload["y"][0],
                reconstructed["y0"],
            )
            obs_mask = mask_metrics(
                new_payload["obs_mask"],
                reconstructed["obs_mask"],
            )

            x_posp_pass = (
                x_posp_angle["mean_abs"]
                <= args.x_mean_threshold
                and x_posp_angle["q99_abs"]
                <= args.x_q99_threshold
                and x_posp_angle["max_abs"]
                <= args.x_max_threshold
            )
            x_meteo_pass = (
                x_meteo["mean_abs"]
                <= args.x_mean_threshold
                and x_meteo["q99_abs"]
                <= args.x_q99_threshold
                and x_meteo["max_abs"]
                <= args.x_max_threshold
            )
            y_pass = (
                y0["mean_abs"]
                <= args.y_mean_threshold
                and y0["q99_abs"]
                <= args.y_q99_threshold
                and y0["max_abs"]
                <= args.y_max_threshold
            )
            mask_pass = bool(obs_mask["exact_match"])

            day_raw_union_pass = (
                x_posp_pass
                and x_meteo_pass
                and y_pass
                and mask_pass
            )
            raw_union_pass = (
                raw_union_pass
                and day_raw_union_pass
            )

            day_preserved_fields_pass = True
            if not torch.equal(
                old_payload["x"][48:],
                new_payload["x"][48:],
            ):
                day_preserved_fields_pass = False
            if not torch.equal(
                old_payload["y"][1:],
                new_payload["y"][1:],
            ):
                day_preserved_fields_pass = False

            for key in old_payload:
                if key in {"x", "y", "obs_mask"}:
                    continue
                if not values_equal(
                    old_payload[key],
                    new_payload[key],
                ):
                    day_preserved_fields_pass = False

            preserved_fields_pass = (
                preserved_fields_pass
                and day_preserved_fields_pass
            )

            old_posp = diff_metrics(
                old_payload["x"][:43],
                new_payload["x"][:43],
            )
            old_target = diff_metrics(
                old_payload["y"][0],
                new_payload["y"][0],
            )
            old_mask = mask_metrics(
                old_payload["obs_mask"],
                new_payload["obs_mask"],
            )

            if is_overlap:
                day_overlap_change = (
                    old_posp["unequal_count"] > 0
                    or old_target["unequal_count"] > 0
                    or old_mask["unequal_count"] > 0
                )
                overlap_change_pass = (
                    overlap_change_pass
                    and day_overlap_change
                )
            else:
                day_nonoverlap_reproduction = (
                    old_posp["mean_abs"]
                    <= args.x_mean_threshold
                    and old_posp["q99_abs"]
                    <= args.x_q99_threshold
                    and old_posp["max_abs"]
                    <= args.x_max_threshold
                    and old_target["mean_abs"]
                    <= args.y_mean_threshold
                    and old_target["q99_abs"]
                    <= args.y_q99_threshold
                    and old_target["max_abs"]
                    <= args.y_max_threshold
                    and old_mask["exact_match"]
                )
                nonoverlap_reproduction_pass = (
                    nonoverlap_reproduction_pass
                    and day_nonoverlap_reproduction
                )
                day_overlap_change = None

            day_reports.append(
                {
                    "integer_doy": day,
                    "source_class": (
                        "overlap"
                        if is_overlap
                        else (
                            "nc1_only"
                            if idx1.size > 0
                            else "nc2_only"
                        )
                    ),
                    "nc1_row_count": int(idx1.size),
                    "nc2_row_count": int(idx2.size),
                    "direct_raw_union_vs_gate10b": {
                        "x_0_43": x_posp_angle,
                        "x_43_48": x_meteo,
                        "y_0": y0,
                        "obs_mask": obs_mask,
                        "pass": bool(day_raw_union_pass),
                    },
                    "preserved_fields_exact": bool(
                        day_preserved_fields_pass
                    ),
                    "old_to_new_source_content": {
                        "x_0_43": old_posp,
                        "y_0": old_target,
                        "obs_mask": old_mask,
                        "overlap_change_verified": (
                            day_overlap_change
                        ),
                    },
                }
            )

            print(
                f"AUDIT_PROGRESS={position}/{len(audited_days)} "
                f"DOY={day} "
                f"SOURCE_CLASS="
                f"{'overlap' if is_overlap else ('nc1_only' if idx1.size else 'nc2_only')} "
                f"RAW_UNION_PASS={day_raw_union_pass} "
                f"PRESERVED_PASS={day_preserved_fields_pass}",
                flush=True,
            )

    cross_file_key_pass = (
        total_common_cross_file_keys == 0
        and total_nc2_unique_keys == 546
    )

    overall_pass = (
        raw_union_pass
        and preserved_fields_pass
        and nonoverlap_reproduction_pass
        and overlap_change_pass
        and cross_file_key_pass
    )

    report = {
        "audit_name": (
            "tensor_2023_union_trainonly_gate10c"
        ),
        "read_only": True,
        "inputs": {
            "nc1": str(nc1_path),
            "nc2": str(nc2_path),
            "old_dir": str(old_dir),
            "new_dir": str(new_dir),
            "old_stats": str(old_stats_path),
            "train_stats": str(train_stats_path),
            "gate10a2_report": str(gate10a2_path),
            "gate10b_manifest": str(gate10b_path),
        },
        "independent_aggregation": (
            "stable sort by grid index followed by numpy.add.reduceat; "
            "Gate 10B used numpy.bincount"
        ),
        "audit_coverage": {
            "all_overlap_days": overlap_days,
            "overlap_day_count": len(overlap_days),
            "selected_nc1_only_days": selected_nc1_only,
            "selected_nc2_only_days": selected_nc2_only,
            "total_audited_day_count": len(audited_days),
        },
        "thresholds": {
            "x_mean_abs": args.x_mean_threshold,
            "x_q99_abs": args.x_q99_threshold,
            "x_max_abs": args.x_max_threshold,
            "y_mean_abs": args.y_mean_threshold,
            "y_q99_abs": args.y_q99_threshold,
            "y_max_abs": args.y_max_threshold,
            "mask_requires_exact_match": True,
        },
        "cross_file_key_diagnostic": {
            "total_nc2_unique_keys_on_overlap_days": int(
                total_nc2_unique_keys
            ),
            "total_common_exact_keys": int(
                total_common_cross_file_keys
            ),
            "pass": bool(cross_file_key_pass),
        },
        "verification": {
            "raw_union_reconstruction_pass": bool(
                raw_union_pass
            ),
            "preserved_fields_exactly_unchanged": bool(
                preserved_fields_pass
            ),
            "nonoverlap_source_content_reproduction_pass": bool(
                nonoverlap_reproduction_pass
            ),
            "overlap_source_content_change_verified": bool(
                overlap_change_pass
            ),
            "cross_file_key_diagnostic_pass": bool(
                cross_file_key_pass
            ),
            "overall_pass": bool(overall_pass),
        },
        "day_reports": day_reports,
        "next_gate_boundary": (
            "Only after overall_pass is true may the clean phase2 "
            "mean-state and anomaly caches be rebuilt."
        ),
    }

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "2023 union and train-only tensor audit - Gate 10C",
        f"Overlap days audited: {len(overlap_days)}",
        f"NC1-only days audited: {selected_nc1_only}",
        f"NC2-only days audited: {selected_nc2_only}",
        f"Total audited days: {len(audited_days)}",
        (
            "Raw union reconstruction pass: "
            f"{raw_union_pass}"
        ),
        (
            "Preserved fields exactly unchanged: "
            f"{preserved_fields_pass}"
        ),
        (
            "Non-overlap source-content reproduction pass: "
            f"{nonoverlap_reproduction_pass}"
        ),
        (
            "Overlap source-content change verified: "
            f"{overlap_change_pass}"
        ),
        (
            "Cross-file key diagnostic pass: "
            f"{cross_file_key_pass}"
        ),
        (
            "Total NC2 unique keys on overlap days: "
            f"{total_nc2_unique_keys}"
        ),
        (
            "Total common exact cross-file keys: "
            f"{total_common_cross_file_keys}"
        ),
        f"Overall pass: {overall_pass}",
    ]
    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    if not overall_pass:
        print("GATE10C_2023_UNION_TENSOR_AUDIT_FAILED")
        print(f"JSON_REPORT={json_path}")
        print(f"TXT_REPORT={txt_path}")
        print(
            f"RAW_UNION_RECONSTRUCTION_PASS={raw_union_pass}"
        )
        print(
            "PRESERVED_FIELDS_EXACTLY_UNCHANGED="
            f"{preserved_fields_pass}"
        )
        print(
            "NONOVERLAP_SOURCE_CONTENT_REPRODUCTION_PASS="
            f"{nonoverlap_reproduction_pass}"
        )
        print(
            "OVERLAP_SOURCE_CONTENT_CHANGE_VERIFIED="
            f"{overlap_change_pass}"
        )
        print(
            f"CROSS_FILE_KEY_DIAGNOSTIC_PASS={cross_file_key_pass}"
        )
        raise RuntimeError(
            "Gate 10C failed. Do not rebuild phase2 caches."
        )

    print("GATE10C_2023_UNION_TENSOR_AUDIT_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"AUDITED_OVERLAP_DAYS={len(overlap_days)}")
    print(
        "AUDITED_NONOVERLAP_DAYS="
        f"{len(selected_nc1_only) + len(selected_nc2_only)}"
    )
    print("RAW_UNION_RECONSTRUCTION_PASS=True")
    print("PRESERVED_FIELDS_EXACTLY_UNCHANGED=True")
    print(
        "NONOVERLAP_SOURCE_CONTENT_REPRODUCTION_PASS=True"
    )
    print(
        "OVERLAP_SOURCE_CONTENT_CHANGE_VERIFIED=True"
    )
    print("CROSS_FILE_KEY_DIAGNOSTIC_PASS=True")
    print("TOTAL_NC2_UNIQUE_KEYS_ON_OVERLAP_DAYS=546")
    print("TOTAL_COMMON_EXACT_CROSS_FILE_KEYS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
