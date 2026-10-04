# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
82_compute_spatial_train_only_meteo_stats_gateS9B.py

Gate S9B: remove the remaining preprocessing leakage before spatial holdout training.

Why this gate exists
--------------------
The clean mean-state cache uses meteorological z-scores estimated from the six
2022 training MONTHS, but those statistics were computed from rows across the
whole globe. For a strict East-Asia spatial holdout, the held-out geography
should not contribute even to fitted preprocessing statistics.

This script therefore recomputes the five meteorological normalization
statistics using only:
  1) the approved 2022 training months [02,03,05,06,08,09], and
  2) raw rows mapped OUTSIDE the Gate-S9A East-Asia training exclusion
     (core holdout + 32-grid-cell guard).

It is read-only with respect to source NetCDF, caches, checkpoints and models.

Important implementation detail
-------------------------------
The existing monthly cache does NOT need to be rebuilt. Because z-score
normalization is affine and the monthly mean is arithmetic,

    z_new = (std_old / std_new) * z_old
            + (mean_old - mean_new) / std_new

is exactly the corresponding transformation of the cached monthly meteo_mean
apart from ordinary floating-point roundoff.

This gate only computes/audits the new statistics and affine coefficients.
It does not train any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


H = 361
W = 576
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
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_S9A_ARTIFACT = "s2_meanstate_spatial_holdout_preflight_gateS9A"
EXPECTED_PRIMARY_REGION = "East_Asia"
EXPECTED_GUARD_CELLS = 32
EXPECTED_GRID_SHAPE = [361, 576]
EXPECTED_MONTHLY_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def doy_to_month_lookup() -> np.ndarray:
    lookup = np.zeros(367, dtype=np.int16)
    start = 1
    for month, n_days in enumerate(MONTH_DAYS, start=1):
        stop = start + n_days
        lookup[start:stop] = month
        start = stop
    return lookup


class RunningStats:
    """Numerically stable batch-combined population statistics."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return

        batch_count = int(arr.size)
        batch_mean = float(np.mean(arr, dtype=np.float64))
        centered = arr - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            self.m2 = (
                self.m2
                + batch_m2
                + delta * delta * self.count * batch_count / total
            )
            self.count = total

        self.minimum = min(self.minimum, float(np.min(arr)))
        self.maximum = max(self.maximum, float(np.max(arr)))

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std_population": None,
                "min": None,
                "max": None,
            }
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        require(math.isfinite(std) and std > 1e-8, "DEGENERATE_STANDARD_DEVIATION")
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std_population": float(std),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


def stats_from_gate7(obj: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    artifact_name = obj.get("artifact_name")
    if artifact_name == "train_only_meteorological_normalization_statistics":
        block = obj.get("train_only_stats")
    elif obj.get("audit_name") == "train_only_meteo_normalization_gate7":
        block = obj.get("train_only_stats")
    else:
        raise RuntimeError("GATE7_STATS_ARTIFACT_NAME_UNRECOGNIZED")

    require(isinstance(block, dict), "GATE7_TRAIN_ONLY_STATS_MISSING")
    result = {}
    for var in METEO_VARS:
        item = block.get(var)
        require(isinstance(item, dict), f"GATE7_VARIABLE_MISSING: {var}")
        result[var] = {
            "count": int(item["count"]),
            "mean": float(item["mean"]),
            "std_population": float(item["std_population"]),
        }
    return result


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S9B: spatial-train-only meteorological normalization statistics."
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--gate7-train-stats", required=True)
    parser.add_argument("--s9a-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--reproduction-atol", type=float, default=1e-8)
    parser.add_argument("--reproduction-rtol", type=float, default=1e-10)
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    gate7_path = Path(args.gate7_train_stats).expanduser().resolve()
    s9a_path = Path(args.s9a_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(nc_path.is_file(), f"NETCDF_NOT_FOUND: {nc_path}")
    require(gate7_path.is_file(), f"GATE7_STATS_NOT_FOUND: {gate7_path}")
    require(s9a_path.is_file(), f"S9A_REPORT_NOT_FOUND: {s9a_path}")
    require(args.chunk_size > 0, "CHUNK_SIZE_MUST_BE_POSITIVE")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    gate7 = load_json(gate7_path)
    s9a = load_json(s9a_path)
    gate7_stats = stats_from_gate7(gate7)

    # Lock the exact S9A design that was already approved.
    require(
        s9a.get("artifact_name") == EXPECTED_S9A_ARTIFACT,
        "S9A_ARTIFACT_NAME_MISMATCH",
    )
    require(
        s9a.get("status") == "DESIGNABLE_PRIMARY_REGION",
        "S9A_STATUS_NOT_DESIGNABLE",
    )
    require(
        s9a.get("verification", {}).get("overall_pass") is True,
        "S9A_OVERALL_PASS_NOT_TRUE",
    )
    contract = s9a.get("design_contract", {})
    require(
        contract.get("primary_region") == EXPECTED_PRIMARY_REGION,
        "S9A_PRIMARY_REGION_MISMATCH",
    )
    require(
        list(contract.get("grid_shape", [])) == EXPECTED_GRID_SHAPE,
        "S9A_GRID_SHAPE_MISMATCH",
    )
    require(
        int(contract.get("guard_cells", -1)) == EXPECTED_GUARD_CELLS,
        "S9A_GUARD_CELLS_MISMATCH",
    )
    require(
        s9a.get("inputs", {}).get("monthly_manifest_sha256")
        == EXPECTED_MONTHLY_MANIFEST_SHA256,
        "S9A_MONTHLY_MANIFEST_HASH_MISMATCH",
    )

    east = s9a["region_results"][EXPECTED_PRIMARY_REGION]
    exclusion_bbox = east["training_exclusion_bbox_indices"]
    y0 = int(exclusion_bbox["y0"])
    y1 = int(exclusion_bbox["y1"])
    x0 = int(exclusion_bbox["x0"])
    x1 = int(exclusion_bbox["x1"])
    require((y0, y1, x0, x1) == (188, 322, 416, 552), "S9A_EXCLUSION_BBOX_UNEXPECTED")

    exclusion_mask = np.zeros((H, W), dtype=bool)
    exclusion_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    require(
        int(exclusion_mask.sum()) == int(east["training_exclusion_grid_cells"]),
        "S9A_EXCLUSION_CELL_COUNT_MISMATCH",
    )

    lookup = doy_to_month_lookup()
    time_only_stats = {var: RunningStats() for var in METEO_VARS}
    spatial_train_stats = {var: RunningStats() for var in METEO_VARS}

    processed_rows = 0
    train_month_rows = 0
    train_month_rows_with_valid_grid = 0
    excluded_spatial_rows = 0
    retained_spatial_rows = 0
    row_counts_by_train_month = {str(m): 0 for m in EXPECTED_TRAIN_MONTHS}
    retained_counts_by_train_month = {str(m): 0 for m in EXPECTED_TRAIN_MONTHS}

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
        require(not missing, f"NETCDF_MISSING_VARIABLES: {missing}")
        require("num_samples" in ds.sizes, "NETCDF_NUM_SAMPLES_DIMENSION_MISSING")
        n_samples = int(ds.sizes["num_samples"])

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

            finite_doy = np.isfinite(doy)
            integer_doy = np.zeros(doy.shape, dtype=np.int16)
            integer_doy[finite_doy] = doy[finite_doy].astype(np.int16, copy=False)
            valid_doy = (
                finite_doy
                & (integer_doy >= 1)
                & (integer_doy <= 366)
            )
            months = np.zeros(integer_doy.shape, dtype=np.int16)
            months[valid_doy] = lookup[integer_doy[valid_doy]]
            train_time = valid_doy & np.isin(months, EXPECTED_TRAIN_MONTHS)

            # Exactly match the raw-row -> analysis-grid index mapping used by
            # 39_rebuild_daily_tensors_meteo_trainonly_from_raw.py.
            finite_coord = np.isfinite(lat) & np.isfinite(lon)
            lat_idx = np.zeros(lat.shape, dtype=np.int64)
            lon_idx = np.zeros(lon.shape, dtype=np.int64)
            lat_idx[finite_coord] = np.rint(
                (lat[finite_coord] - LAT_MIN) / LAT_RES
            ).astype(np.int64)
            lon_idx[finite_coord] = np.rint(
                (lon[finite_coord] - LON_MIN) / LON_RES
            ).astype(np.int64)
            valid_grid = (
                finite_coord
                & (lat_idx >= 0)
                & (lat_idx < H)
                & (lon_idx >= 0)
                & (lon_idx < W)
            )

            mapped_train = train_time & valid_grid
            inside_exclusion = np.zeros(train_time.shape, dtype=bool)
            mapped_indices = np.flatnonzero(mapped_train)
            if mapped_indices.size:
                inside_exclusion[mapped_indices] = exclusion_mask[
                    lat_idx[mapped_indices],
                    lon_idx[mapped_indices],
                ]
            spatial_train = mapped_train & (~inside_exclusion)

            train_month_rows += int(train_time.sum())
            train_month_rows_with_valid_grid += int(mapped_train.sum())
            excluded_spatial_rows += int((mapped_train & inside_exclusion).sum())
            retained_spatial_rows += int(spatial_train.sum())

            for month in EXPECTED_TRAIN_MONTHS:
                mm = train_time & (months == month)
                sm = spatial_train & (months == month)
                row_counts_by_train_month[str(month)] += int(mm.sum())
                retained_counts_by_train_month[str(month)] += int(sm.sum())

            for var in METEO_VARS:
                values = np.asarray(
                    ds[var].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                # Reproduce Gate 7 on the same training-month raw rows.
                time_only_stats[var].update(values[train_time])
                # Strict spatial training domain.
                spatial_train_stats[var].update(values[spatial_train])

            processed_rows += stop - start
            print(f"PROGRESS={stop}/{n_samples}", flush=True)

    time_only_results = {var: time_only_stats[var].result() for var in METEO_VARS}
    spatial_results = {var: spatial_train_stats[var].result() for var in METEO_VARS}

    reproduction = {}
    all_reproduced = True
    affine = {}
    shifts = {}

    for var in METEO_VARS:
        expected = gate7_stats[var]
        got = time_only_results[var]

        mean_abs = abs(float(got["mean"]) - float(expected["mean"]))
        std_abs = abs(
            float(got["std_population"])
            - float(expected["std_population"])
        )
        mean_rel = relative_difference(float(got["mean"]), float(expected["mean"]))
        std_rel = relative_difference(
            float(got["std_population"]),
            float(expected["std_population"]),
        )
        count_exact = int(got["count"]) == int(expected["count"])
        reproduced = (
            count_exact
            and (
                mean_abs <= args.reproduction_atol
                or mean_rel <= args.reproduction_rtol
            )
            and (
                std_abs <= args.reproduction_atol
                or std_rel <= args.reproduction_rtol
            )
        )
        all_reproduced = all_reproduced and reproduced
        reproduction[var] = {
            "count_exact": count_exact,
            "expected_count": int(expected["count"]),
            "recomputed_count": int(got["count"]),
            "mean_abs_diff": mean_abs,
            "mean_rel_diff": mean_rel,
            "std_abs_diff": std_abs,
            "std_rel_diff": std_rel,
            "reproduced_within_tolerance": bool(reproduced),
        }

        old_mean = float(expected["mean"])
        old_std = float(expected["std_population"])
        new_mean = float(spatial_results[var]["mean"])
        new_std = float(spatial_results[var]["std_population"])
        require(math.isfinite(new_mean), f"SPATIAL_MEAN_NONFINITE: {var}")
        require(math.isfinite(new_std) and new_std > 1e-8, f"SPATIAL_STD_INVALID: {var}")

        a = old_std / new_std
        b = (old_mean - new_mean) / new_std
        affine[var] = {
            "old_z_to_spatial_z_scale": a,
            "old_z_to_spatial_z_offset": b,
            "formula": "z_spatial = scale * z_existing + offset",
        }
        shifts[var] = {
            "old_time_train_mean": old_mean,
            "new_spatial_train_mean": new_mean,
            "mean_difference_new_minus_old": new_mean - old_mean,
            "mean_relative_difference": relative_difference(new_mean, old_mean),
            "old_time_train_std": old_std,
            "new_spatial_train_std": new_std,
            "std_difference_new_minus_old": new_std - old_std,
            "std_relative_difference": relative_difference(new_std, old_std),
        }

    require(all_reproduced, "GATE7_TIME_ONLY_STATS_REPRODUCTION_FAILED")
    require(retained_spatial_rows > 0, "NO_SPATIAL_TRAIN_ROWS_RETAINED")
    require(excluded_spatial_rows > 0, "NO_ROWS_EXCLUDED_BY_SPATIAL_HOLDOUT")
    require(
        retained_spatial_rows + excluded_spatial_rows
        == train_month_rows_with_valid_grid,
        "SPATIAL_ROW_ACCOUNTING_MISMATCH",
    )
    require(
        all(v > 0 for v in retained_counts_by_train_month.values()),
        "A_TRAIN_MONTH_HAS_ZERO_RETAINED_SPATIAL_ROWS",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    stats_path = (
        output_dir
        / "meteo_normalization_stats_2022_train_months_spatial_EastAsia_guard32.json"
    )
    report_path = output_dir / "spatial_train_only_meteo_stats_gateS9B.json"
    txt_path = output_dir / "spatial_train_only_meteo_stats_gateS9B.txt"

    stats_artifact = {
        "artifact_name": "spatial_train_only_meteorological_normalization_statistics_gateS9B",
        "artifact_version": 1,
        "source_nc": str(nc_path),
        "source_nc_sha256": sha256_file(nc_path),
        "source_gate7_stats": str(gate7_path),
        "source_gate7_stats_sha256": sha256_file(gate7_path),
        "source_s9a_report": str(s9a_path),
        "source_s9a_report_sha256": sha256_file(s9a_path),
        "year": 2022,
        "train_months": EXPECTED_TRAIN_MONTHS,
        "validation_months": EXPECTED_VAL_MONTHS,
        "primary_spatial_holdout": EXPECTED_PRIMARY_REGION,
        "training_exclusion_bbox_indices": {
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
        },
        "guard_cells": EXPECTED_GUARD_CELLS,
        "grid_mapping": {
            "lat_index": "rint((Latitude - (-90.0)) / 0.5)",
            "lon_index": "rint((Longitude - (-180.0)) / 0.625)",
            "matches_script_39": True,
        },
        "statistics_definition": {
            "sample_domain": (
                "Finite meteorological values on raw fused-NetCDF rows from the "
                "approved 2022 training months, mapped to valid analysis-grid cells "
                "outside the East-Asia core plus 32-cell training guard."
            ),
            "std_definition": "Population standard deviation: sqrt(M2 / count)",
            "nonfinite_handling": "Excluded per meteorological variable",
        },
        "spatial_train_only_stats": spatial_results,
        "existing_time_train_only_stats": gate7_stats,
        "old_z_to_spatial_z_affine": affine,
    }

    report = {
        "artifact_name": "s2_meanstate_spatial_train_only_meteo_stats_gateS9B",
        "artifact_version": 1,
        "status": "PASS_SPATIAL_TRAIN_ONLY_PREPROCESSING",
        "inputs": {
            "nc": str(nc_path),
            "nc_sha256": sha256_file(nc_path),
            "gate7_train_stats": str(gate7_path),
            "gate7_train_stats_sha256": sha256_file(gate7_path),
            "s9a_report": str(s9a_path),
            "s9a_report_sha256": sha256_file(s9a_path),
        },
        "row_accounting": {
            "processed_rows": int(processed_rows),
            "train_month_rows": int(train_month_rows),
            "train_month_rows_with_valid_grid": int(train_month_rows_with_valid_grid),
            "excluded_spatial_rows": int(excluded_spatial_rows),
            "retained_spatial_rows": int(retained_spatial_rows),
            "excluded_fraction_of_mapped_training_rows": (
                excluded_spatial_rows / train_month_rows_with_valid_grid
            ),
            "row_counts_by_train_month": row_counts_by_train_month,
            "retained_counts_by_train_month": retained_counts_by_train_month,
        },
        "gate7_reproduction": reproduction,
        "existing_time_train_only_stats_reproduced": bool(all_reproduced),
        "spatial_train_only_stats": spatial_results,
        "normalization_shifts": shifts,
        "old_z_to_spatial_z_affine": affine,
        "outputs": {
            "stats_artifact": str(stats_path),
            "json_report": str(report_path),
            "txt_report": str(txt_path),
        },
        "verification": {
            "s9a_contract_exact": True,
            "grid_mapping_matches_clean_rebuild": True,
            "gate7_time_train_stats_reproduced": True,
            "spatial_exclusion_removed_rows": True,
            "all_train_months_retained_rows": True,
            "spatial_stats_finite_and_nondegenerate": True,
            "monthly_cache_rebuild_required": False,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This gate removes held-out-geography participation from learned "
            "meteorological normalization statistics. It does not train or evaluate "
            "a model. The affine transform is used only to convert the existing "
            "time-train-normalized monthly meteorology to the stricter spatial-train-"
            "normalized scale for Gate S9."
        ),
    }

    stats_path.write_text(
        json.dumps(stats_artifact, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "GATE_S9B_SPATIAL_TRAIN_ONLY_METEO_STATS_PASS",
        f"PRIMARY_REGION={EXPECTED_PRIMARY_REGION}",
        f"TRAINING_EXCLUSION_BBOX_INDICES={(y0, y1, x0, x1)}",
        f"PROCESSED_ROWS={processed_rows}",
        f"TRAIN_MONTH_ROWS={train_month_rows}",
        f"TRAIN_MONTH_ROWS_WITH_VALID_GRID={train_month_rows_with_valid_grid}",
        f"EXCLUDED_SPATIAL_ROWS={excluded_spatial_rows}",
        f"RETAINED_SPATIAL_ROWS={retained_spatial_rows}",
        "GATE7_TIME_TRAIN_STATS_REPRODUCED=True",
        "MONTHLY_CACHE_REBUILD_REQUIRED=False",
    ]
    for var in METEO_VARS:
        lines.extend(
            [
                f"{var}_OLD_MEAN={gate7_stats[var]['mean']:.12g}",
                f"{var}_NEW_MEAN={spatial_results[var]['mean']:.12g}",
                f"{var}_OLD_STD={gate7_stats[var]['std_population']:.12g}",
                f"{var}_NEW_STD={spatial_results[var]['std_population']:.12g}",
                f"{var}_AFFINE_SCALE={affine[var]['old_z_to_spatial_z_scale']:.12g}",
                f"{var}_AFFINE_OFFSET={affine[var]['old_z_to_spatial_z_offset']:.12g}",
            ]
        )
    lines.extend(
        [
            "SCIENTIFIC_TRAINING_STARTED=False",
            f"STATS_ARTIFACT={stats_path}",
            f"OUTPUT_JSON={report_path}",
            f"OUTPUT_TXT={txt_path}",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
