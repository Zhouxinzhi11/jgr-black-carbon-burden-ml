#!/usr/bin/env python
# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
87_audit_posp_bccmass_sampling_support.py

Read-only audit of the POSP-derived sampling support used by the clean
mean-state experiments.

The audit answers four separate questions without modifying source data,
caches, checkpoints, or evaluation reports:

1. What monthly and daily sampling coverage is inherited from I_380?
2. Are the historical code-faithful ``~isnan(I_380)`` rule and the stricter
   ``isfinite(I_380)`` wording equivalent in the actual fused files?
3. How do the gridded I_380 and BCCMASS availability masks overlap at daily,
   monthly-any-day, and monthly >=5-day levels?
4. Can the raw-row I_380 rule reproduce every stored monthly
   ``obs_count_map`` and ``valid_mask`` exactly, and did every formal
   mean-state model use those same cache masks?

Scientific boundary
-------------------
This is a support/provenance audit, not a model-skill experiment. A difference
between the I_380 and BCCMASS masks is characterized but is not automatically
treated as an error: the historical pipeline deliberately used I_380 to define
source availability. Gate failure is reserved for a broken lineage, ambiguous
missing-value semantics, invalid source coordinates/dates, or inconsistent
formal evaluation records.

Expected local inputs
---------------------
- 2022 fused NetCDF: BCCMASS_train.nc
- 2023 fused NetCDF union: BCCMASS_train_1.nc and BCCMASS_train_2.nc
- Clean monthly cache with manifests/monthly_manifest.json
- The code root containing the 18 clean/support/binary formal evaluation
  reports.

The script is CPU-only and streams NetCDF rows in chunks. It refuses to
overwrite an existing output directory. Core tabular/JSON inputs for plotting
are written before plotting starts, and figures are rendered in a separate
Python process that does not import PyTorch. This isolates Windows OpenMP
runtimes used by PyTorch and Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


H = 361
W = 576
N_GRID = H * W
PACKED_BYTES = (N_GRID + 7) // 8
LAT_MIN = -90.0
LON_MIN = -180.0
LAT_RES = 0.5
LON_RES = 0.625
MIN_DOY = 1
MAX_DOY = 365

MASK_NAMES = (
    "source_present",
    "i380_non_nan",
    "i380_finite",
    "bccmass_finite",
    "same_row_both_finite",
)

DEFAULT_BASE_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED"
DEFAULT_NC_2022 = os.path.join(
    DEFAULT_BASE_DIR,
    r"02_source_nc\2022\BCCMASS_train.nc",
)
DEFAULT_NC_2023_1 = os.path.join(
    DEFAULT_BASE_DIR,
    r"02_source_nc\2023\BCCMASS_train_1.nc",
)
DEFAULT_NC_2023_2 = os.path.join(
    DEFAULT_BASE_DIR,
    r"02_source_nc\2023\BCCMASS_train_2.nc",
)
DEFAULT_GRID_REF = os.path.join(
    DEFAULT_BASE_DIR,
    r"00_grid_ref\merra2_grid_coords.npz",
)
DEFAULT_CACHE_DIR = os.path.join(
    DEFAULT_BASE_DIR,
    "07_phase2_meanstate_fullmonth_trainonly_union",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_BASE_DIR,
    r"sanity_reports\posp_bccmass_sampling_support_gate87_v01",
)

EXPECTED_CONFIGS = {
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
    "mean_emis_meteo_clim_support_preserving",
    "mean_binary_support_meteo_clim",
}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_REPORT_COUNT = len(EXPECTED_CONFIGS) * len(EXPECTED_SEEDS)
EXPECTED_TRAIN_MONTHS_2022 = [2, 3, 5, 6, 8, 9]
EXPECTED_VALIDATION_MONTHS_2022 = [11, 12]
EXPECTED_BLACKLIST_MONTHS_2022 = [7]
EXPECTED_MONTHS_2022 = sorted(
    EXPECTED_TRAIN_MONTHS_2022
    + EXPECTED_VALIDATION_MONTHS_2022
    + EXPECTED_BLACKLIST_MONTHS_2022
)
EXPECTED_MONTHS_2023 = list(range(1, 13))

FORMAL_REPORT_PATTERNS = (
    "runs_s2_meanstate_clean/formal/seed*/mean_*_seed*/"
    "eval_clean_generic/eval_report.json",
    "runs_s2_meanstate_support_preserving/formal/seed*/mean_*_seed*/"
    "eval_support_preserving_seed*/eval_report.json",
    "runs_s2_meanstate_binary_support_gates8/formal/seed*/"
    "mean_*_seed*/eval_gateS8E_v02/eval_report.json",
)

CSV_DAILY_COLUMNS = [
    "year",
    "doy",
    "date",
    "month",
    "source_present_cells",
    "source_present_fraction",
    "i380_non_nan_cells",
    "i380_non_nan_fraction",
    "i380_finite_cells",
    "i380_finite_fraction",
    "bccmass_finite_cells",
    "bccmass_finite_fraction",
    "same_row_both_finite_cells",
    "same_row_both_finite_fraction",
    "grid_intersection_i380_bccmass_cells",
    "grid_union_i380_bccmass_cells",
    "i380_only_cells",
    "bccmass_only_cells",
    "jaccard_i380_bccmass",
    "jaccard_i380_non_nan_vs_finite",
    "i380_non_nan_mask_equals_finite",
    "same_row_pair_equals_grid_intersection",
]

CSV_MONTHLY_COLUMNS = [
    "analysis_role",
    "year",
    "month",
    "split",
    "n_source_days",
    "source_doys",
    "missing_calendar_doys",
    "cached_obs_any_cells",
    "cached_obs_any_fraction",
    "cached_eval_cells",
    "cached_eval_fraction",
    "i380_non_nan_any_cells",
    "i380_non_nan_any_fraction",
    "i380_non_nan_ge_min_cells",
    "i380_non_nan_ge_min_fraction",
    "i380_finite_any_cells",
    "i380_finite_any_fraction",
    "i380_finite_ge_min_cells",
    "i380_finite_ge_min_fraction",
    "bccmass_finite_any_cells",
    "bccmass_finite_any_fraction",
    "bccmass_finite_ge_min_cells",
    "bccmass_finite_ge_min_fraction",
    "jaccard_i380_bccmass_any",
    "jaccard_i380_bccmass_ge_min",
    "i380_only_any_cells",
    "bccmass_only_any_cells",
    "i380_only_ge_min_cells",
    "bccmass_only_ge_min_cells",
    "jaccard_i380_non_nan_vs_finite_any",
    "jaccard_i380_non_nan_vs_finite_ge_min",
    "i380_non_nan_equals_finite_any",
    "i380_non_nan_equals_finite_ge_min",
    "code_count_map_exact_cache",
    "code_valid_mask_exact_cache",
    "cache_internal_valid_mask_exact",
    "cache_file_sha256",
    "daily_i380_finite_fraction_min",
    "daily_i380_finite_fraction_median",
    "daily_i380_finite_fraction_mean",
    "daily_i380_finite_fraction_max",
]

CSV_SOURCE_COLUMNS = [
    "year",
    "file",
    "file_size_bytes",
    "num_rows",
    "mapped_rows",
    "invalid_doy_rows",
    "invalid_coordinate_rows",
    "i380_nan_rows",
    "i380_posinf_rows",
    "i380_neginf_rows",
    "i380_finite_rows",
    "i380_non_nan_declared_sentinel_rows",
    "bccmass_nan_rows",
    "bccmass_posinf_rows",
    "bccmass_neginf_rows",
    "bccmass_finite_rows",
    "bccmass_non_nan_declared_sentinel_rows",
    "mapped_both_finite_rows",
    "mapped_i380_only_finite_rows",
    "mapped_bccmass_only_finite_rows",
    "mapped_neither_finite_rows",
    "mapped_row_jaccard_i380_bccmass",
    "first_doy",
    "last_doy",
    "n_unique_doys",
]

CSV_FORMAL_COLUMNS = [
    "experiment",
    "seed",
    "artifact_name",
    "report",
    "report_sha256",
    "overall_pass",
    "monthly_manifest_sha256",
    "manifest_sha256_matches_current",
    "2022_validation_total_valid_pixels",
    "2023_external_total_valid_pixels",
    "per_group_keys_exact",
    "per_group_valid_counts_exact",
    "per_group_valid_fractions_exact",
    "per_group_valid_fraction_fields_complete",
    "per_group_cache_hash_fields_complete",
    "per_group_reported_cache_hashes_exact",
    "report_common_mask_pass",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def strict_json_dump(obj: Any, path: Path) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            clean(obj),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    require(isinstance(obj, dict), "JSON root is not an object: %s" % path)
    return obj


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if row.get(key) is None
                        else row.get(key)
                    )
                    for key in columns
                }
            )


def file_inventory(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def doy_to_date(year: int, doy: int) -> date:
    require(1 <= doy <= 365, "DOY outside 1..365: %s" % doy)
    return date(year, 1, 1) + timedelta(days=doy - 1)


def doy_to_month(year: int, doy: int) -> int:
    return int(doy_to_date(year, doy).month)


def role(year: int, month: int) -> str:
    if year == 2022 and month in EXPECTED_TRAIN_MONTHS_2022:
        return "2022_train"
    if year == 2022 and month in EXPECTED_VALIDATION_MONTHS_2022:
        return "2022_model_selection_validation"
    if year == 2022 and month in EXPECTED_BLACKLIST_MONTHS_2022:
        return "2022_blacklisted_weak_group"
    if year == 2023 and month in EXPECTED_MONTHS_2023:
        return "2023_external_evaluation_grid"
    return "other"


def pack_indices(indices: np.ndarray) -> np.ndarray:
    packed = np.zeros(PACKED_BYTES, dtype=np.uint8)
    if indices.size == 0:
        return packed
    unique = np.unique(indices.astype(np.int64, copy=False))
    require(
        bool(np.all((unique >= 0) & (unique < N_GRID))),
        "Grid index outside valid range",
    )
    byte_index = unique >> 3
    bit_value = (1 << (unique & 7)).astype(np.uint8, copy=False)
    np.bitwise_or.at(packed, byte_index, bit_value)
    return packed


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    require(
        packed.dtype == np.uint8 and packed.size == PACKED_BYTES,
        "Invalid packed-mask representation",
    )
    return np.unpackbits(packed, bitorder="little")[:N_GRID].astype(bool, copy=False)


def merge_packed(
    day_masks: MutableMapping[int, Dict[int, Dict[str, np.ndarray]]],
    year: int,
    doy: int,
    additions: Mapping[str, np.ndarray],
) -> None:
    year_map = day_masks.setdefault(year, {})
    current = year_map.get(doy)
    if current is None:
        current = {
            name: np.zeros(PACKED_BYTES, dtype=np.uint8)
            for name in MASK_NAMES
        }
        year_map[doy] = current
    for name in MASK_NAMES:
        np.bitwise_or(current[name], additions[name], out=current[name])


def count_true(packed: np.ndarray) -> int:
    return int(np.count_nonzero(unpack_mask(packed)))


def overlap_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    require(a.dtype == bool and b.dtype == bool, "Overlap inputs must be bool")
    require(a.shape == b.shape, "Overlap inputs have different shapes")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    a_only = int(np.count_nonzero(a & ~b))
    b_only = int(np.count_nonzero(b & ~a))
    return {
        "a_cells": int(np.count_nonzero(a)),
        "b_cells": int(np.count_nonzero(b)),
        "intersection_cells": intersection,
        "union_cells": union,
        "a_only_cells": a_only,
        "b_only_cells": b_only,
        "jaccard": ratio(intersection, union),
        "exact_equal": bool(np.array_equal(a, b)),
    }


def numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "q95": None,
            "max": None,
        }
    array = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def count_map_summary(count_map: np.ndarray, n_days: int) -> Dict[str, Any]:
    require(count_map.shape == (H, W), "Count map has unexpected shape")
    flat = count_map.reshape(-1).astype(np.int64, copy=False)
    require(bool(np.all(flat >= 0)), "Count map contains negative values")
    require(bool(np.all(flat <= n_days)), "Count map exceeds source-day count")
    histogram = np.bincount(flat, minlength=n_days + 1)
    positive = flat[flat > 0]
    return {
        "n_days": int(n_days),
        "histogram_cell_count_by_valid_days": {
            str(day_count): int(histogram[day_count])
            for day_count in range(histogram.size)
        },
        "all_cells_quantiles": numeric_summary(flat.astype(np.float64)),
        "positive_cells_quantiles": numeric_summary(positive.astype(np.float64)),
    }


def variable_sentinels(variable: Any) -> List[float]:
    sentinels: List[float] = []
    for attr in ("_FillValue", "missing_value"):
        if attr not in variable.ncattrs():
            continue
        values = np.atleast_1d(variable.getncattr(attr)).reshape(-1)
        for value in values:
            try:
                sentinels.append(float(value))
            except (TypeError, ValueError):
                continue
    unique: List[float] = []
    for value in sentinels:
        if math.isnan(value):
            if not any(math.isnan(v) for v in unique):
                unique.append(value)
        elif value not in unique:
            unique.append(value)
    return unique


def non_nan_sentinel_mask(values: np.ndarray, sentinels: Sequence[float]) -> np.ndarray:
    """Return hits for finite/infinite declared sentinels.

    A declared NaN fill value is not ambiguous for the historical ``isnan``
    rule and is therefore intentionally excluded here. It remains recorded in
    variable metadata.
    """
    result = np.zeros(values.shape, dtype=bool)
    for sentinel in sentinels:
        if not math.isnan(float(sentinel)):
            result |= values == sentinel
    return result


def variable_metadata(variable: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for name in variable.ncattrs():
        value = variable.getncattr(name)
        if isinstance(value, np.ndarray):
            attrs[name] = value.tolist()
        elif isinstance(value, np.generic):
            attrs[name] = value.item()
        else:
            attrs[name] = value
    try:
        chunking = variable.chunking()
    except Exception:
        chunking = None
    return {
        "dtype": str(variable.dtype),
        "dimensions": list(variable.dimensions),
        "shape": [int(v) for v in variable.shape],
        "chunking": chunking,
        "attributes": attrs,
        "declared_missing_sentinels": variable_sentinels(variable),
    }


def empty_source_stats(path: Path, year: int, n_rows: int) -> Dict[str, Any]:
    return {
        "year": int(year),
        "file": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "num_rows": int(n_rows),
        "mapped_rows": 0,
        "invalid_doy_rows": 0,
        "invalid_coordinate_rows": 0,
        "i380_nan_rows": 0,
        "i380_posinf_rows": 0,
        "i380_neginf_rows": 0,
        "i380_finite_rows": 0,
        "i380_non_nan_declared_sentinel_rows": 0,
        "bccmass_nan_rows": 0,
        "bccmass_posinf_rows": 0,
        "bccmass_neginf_rows": 0,
        "bccmass_finite_rows": 0,
        "bccmass_non_nan_declared_sentinel_rows": 0,
        "mapped_both_finite_rows": 0,
        "mapped_i380_only_finite_rows": 0,
        "mapped_bccmass_only_finite_rows": 0,
        "mapped_neither_finite_rows": 0,
        "first_doy": None,
        "last_doy": None,
        "n_unique_doys": 0,
        "unique_doys": [],
        "mapped_row_jaccard_i380_bccmass": None,
        "variable_metadata": {},
    }


def read_numeric(variable: Any, start: int, stop: int) -> np.ndarray:
    values = variable[start:stop]
    if np.ma.isMaskedArray(values):
        values = np.ma.getdata(values)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def scan_source_file(
    *,
    netcdf4_module: Any,
    path: Path,
    year: int,
    chunk_rows: int,
    day_masks: MutableMapping[int, Dict[int, Dict[str, np.ndarray]]],
) -> Dict[str, Any]:
    required_variables = ["DOY", "Latitude", "Longitude", "I_380", "BCCMASS"]
    print("SOURCE_SCAN_START year=%d file=%s" % (year, path), flush=True)

    with netcdf4_module.Dataset(str(path), mode="r") as dataset:
        dataset.set_auto_maskandscale(False)
        missing = [name for name in required_variables if name not in dataset.variables]
        require(not missing, "%s missing variables: %s" % (path, missing))

        variables = {name: dataset.variables[name] for name in required_variables}
        lengths = {name: int(np.prod(variables[name].shape)) for name in required_variables}
        require(len(set(lengths.values())) == 1, "Required variable lengths differ: %s" % lengths)
        n_rows = next(iter(lengths.values()))
        for name, variable in variables.items():
            require(len(variable.shape) == 1, "%s must be one-dimensional" % name)

        stats = empty_source_stats(path, year, n_rows)
        stats["variable_metadata"] = {
            name: variable_metadata(variable)
            for name, variable in variables.items()
        }
        i_sentinels = variable_sentinels(variables["I_380"])
        b_sentinels = variable_sentinels(variables["BCCMASS"])
        unique_days: set[int] = set()

        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            doy = read_numeric(variables["DOY"], start, stop)
            lat = read_numeric(variables["Latitude"], start, stop)
            lon = read_numeric(variables["Longitude"], start, stop)
            i380 = read_numeric(variables["I_380"], start, stop)
            bccmass = read_numeric(variables["BCCMASS"], start, stop)

            n_chunk = stop - start
            for name, array in (
                ("DOY", doy),
                ("Latitude", lat),
                ("Longitude", lon),
                ("I_380", i380),
                ("BCCMASS", bccmass),
            ):
                require(array.size == n_chunk, "%s chunk length mismatch" % name)

            doy_finite = np.isfinite(doy)
            integer_doy = np.zeros(n_chunk, dtype=np.int16)
            integer_doy[doy_finite] = doy[doy_finite].astype(np.int16, copy=False)
            valid_doy = doy_finite & (integer_doy >= MIN_DOY) & (integer_doy <= MAX_DOY)

            coordinate_finite = np.isfinite(lat) & np.isfinite(lon)
            lat_index = np.zeros(n_chunk, dtype=np.int32)
            lon_index = np.zeros(n_chunk, dtype=np.int32)
            lat_index[coordinate_finite] = np.rint(
                (lat[coordinate_finite] - LAT_MIN) / LAT_RES
            ).astype(np.int32, copy=False)
            lon_index[coordinate_finite] = np.rint(
                (lon[coordinate_finite] - LON_MIN) / LON_RES
            ).astype(np.int32, copy=False)

            coordinate_in_range = (
                coordinate_finite
                & (lat_index >= 0)
                & (lat_index < H)
                & (lon_index >= 0)
                & (lon_index < W)
            )
            mapped = valid_doy & coordinate_in_range

            i_nan = np.isnan(i380)
            i_posinf = np.isposinf(i380)
            i_neginf = np.isneginf(i380)
            i_finite = np.isfinite(i380)
            b_nan = np.isnan(bccmass)
            b_posinf = np.isposinf(bccmass)
            b_neginf = np.isneginf(bccmass)
            b_finite = np.isfinite(bccmass)

            stats["mapped_rows"] += int(np.count_nonzero(mapped))
            stats["invalid_doy_rows"] += int(np.count_nonzero(~valid_doy))
            stats["invalid_coordinate_rows"] += int(
                np.count_nonzero(valid_doy & ~coordinate_in_range)
            )
            stats["i380_nan_rows"] += int(np.count_nonzero(i_nan))
            stats["i380_posinf_rows"] += int(np.count_nonzero(i_posinf))
            stats["i380_neginf_rows"] += int(np.count_nonzero(i_neginf))
            stats["i380_finite_rows"] += int(np.count_nonzero(i_finite))
            stats["i380_non_nan_declared_sentinel_rows"] += int(
                np.count_nonzero(non_nan_sentinel_mask(i380, i_sentinels))
            )
            stats["bccmass_nan_rows"] += int(np.count_nonzero(b_nan))
            stats["bccmass_posinf_rows"] += int(np.count_nonzero(b_posinf))
            stats["bccmass_neginf_rows"] += int(np.count_nonzero(b_neginf))
            stats["bccmass_finite_rows"] += int(np.count_nonzero(b_finite))
            stats["bccmass_non_nan_declared_sentinel_rows"] += int(
                np.count_nonzero(non_nan_sentinel_mask(bccmass, b_sentinels))
            )

            mapped_both = mapped & i_finite & b_finite
            mapped_i_only = mapped & i_finite & ~b_finite
            mapped_b_only = mapped & ~i_finite & b_finite
            mapped_neither = mapped & ~i_finite & ~b_finite
            stats["mapped_both_finite_rows"] += int(np.count_nonzero(mapped_both))
            stats["mapped_i380_only_finite_rows"] += int(np.count_nonzero(mapped_i_only))
            stats["mapped_bccmass_only_finite_rows"] += int(np.count_nonzero(mapped_b_only))
            stats["mapped_neither_finite_rows"] += int(np.count_nonzero(mapped_neither))

            for current_doy in np.unique(integer_doy[mapped]):
                day = int(current_doy)
                unique_days.add(day)
                day_rows = mapped & (integer_doy == day)
                grid_ids = (
                    lat_index[day_rows].astype(np.int64, copy=False) * W
                    + lon_index[day_rows].astype(np.int64, copy=False)
                )
                day_i380 = i380[day_rows]
                day_bccmass = bccmass[day_rows]
                day_i_non_nan = ~np.isnan(day_i380)
                day_i_finite = np.isfinite(day_i380)
                day_b_finite = np.isfinite(day_bccmass)

                additions = {
                    "source_present": pack_indices(grid_ids),
                    "i380_non_nan": pack_indices(grid_ids[day_i_non_nan]),
                    "i380_finite": pack_indices(grid_ids[day_i_finite]),
                    "bccmass_finite": pack_indices(grid_ids[day_b_finite]),
                    "same_row_both_finite": pack_indices(
                        grid_ids[day_i_finite & day_b_finite]
                    ),
                }
                merge_packed(day_masks, year, day, additions)

            print(
                "SOURCE_SCAN_PROGRESS year=%d file=%s rows=%d/%d"
                % (year, path.name, stop, n_rows),
                flush=True,
            )

        stats["unique_doys"] = sorted(unique_days)
        stats["n_unique_doys"] = len(unique_days)
        stats["first_doy"] = min(unique_days) if unique_days else None
        stats["last_doy"] = max(unique_days) if unique_days else None
        row_union = (
            int(stats["mapped_both_finite_rows"])
            + int(stats["mapped_i380_only_finite_rows"])
            + int(stats["mapped_bccmass_only_finite_rows"])
        )
        stats["mapped_row_jaccard_i380_bccmass"] = ratio(
            int(stats["mapped_both_finite_rows"]),
            row_union,
        )

    print("SOURCE_SCAN_DONE year=%d file=%s" % (year, path), flush=True)
    return stats


def audit_grid_reference(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        lats = np.asarray(payload["lats"], dtype=np.float64).reshape(-1)
        lons = np.asarray(payload["lons"], dtype=np.float64).reshape(-1)
    expected_lats = LAT_MIN + np.arange(H, dtype=np.float64) * LAT_RES
    expected_lons = LON_MIN + np.arange(W, dtype=np.float64) * LON_RES
    lat_exact = bool(np.array_equal(lats, expected_lats))
    lon_exact = bool(np.array_equal(lons, expected_lons))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "keys": keys,
        "lat_shape": list(lats.shape),
        "lon_shape": list(lons.shape),
        "lat_exact_expected": lat_exact,
        "lon_exact_expected": lon_exact,
        "overall_pass": lat_exact and lon_exact,
    }


def parse_day_from_source_name(value: Any) -> int:
    match = re.search(r"day_(\d{3})\.pt$", str(value), flags=re.IGNORECASE)
    require(match is not None, "Cannot parse day tensor name: %s" % value)
    doy = int(match.group(1))
    require(MIN_DOY <= doy <= MAX_DOY, "Invalid tensor DOY: %s" % value)
    return doy


def resolve_cache_file(cache_dir: Path, entry: Mapping[str, Any]) -> Path:
    manifest_value = str(entry.get("file", ""))
    manifest_path = Path(manifest_value)
    if manifest_path.is_file():
        return manifest_path.resolve()
    filename_match = re.search(r"([^\\/]+\.pt)$", manifest_value)
    require(filename_match is not None, "Cannot resolve manifest cache filename")
    candidate = (
        cache_dir
        / "monthly_mean_full"
        / str(entry["split"])
        / filename_match.group(1)
    )
    require(candidate.is_file(), "Cache file not found: %s" % candidate)
    return candidate.resolve()


def load_pt(torch_module: Any, path: Path) -> Dict[str, Any]:
    try:
        payload = torch_module.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch_module.load(str(path), map_location="cpu")
    except Exception:
        payload = torch_module.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    require(isinstance(payload, dict), "PT payload is not a dict: %s" % path)
    return payload


def tensor_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def build_count_map(
    day_masks: Mapping[int, Dict[int, Dict[str, np.ndarray]]],
    year: int,
    days: Sequence[int],
    mask_name: str,
) -> np.ndarray:
    count = np.zeros(N_GRID, dtype=np.uint16)
    for doy in days:
        require(year in day_masks and doy in day_masks[year], "No raw mask for %d DOY %03d" % (year, doy))
        count += unpack_mask(day_masks[year][doy][mask_name]).astype(np.uint16)
    return count.reshape(H, W)


def expected_calendar_doys(year: int, month: int) -> List[int]:
    first = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return [
        int((first + timedelta(days=offset) - date(year, 1, 1)).days + 1)
        for offset in range((next_month - first).days)
    ]


def audit_monthly_cache(
    *,
    torch_module: Any,
    cache_dir: Path,
    day_masks: Mapping[int, Dict[int, Dict[str, np.ndarray]]],
    min_obs_days: int,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[Tuple[int, int], Dict[str, Any]],
    Dict[str, np.ndarray],
]:
    manifest_path = cache_dir / "manifests" / "monthly_manifest.json"
    require(manifest_path.is_file(), "Monthly manifest not found: %s" % manifest_path)
    manifest = load_json(manifest_path)
    entries = list(manifest.get("groups", []))
    require(len(entries) == 21, "Expected 21 monthly groups, found %d" % len(entries))

    keys = [(int(e["year"]), int(e["month"])) for e in entries]
    require(len(keys) == len(set(keys)), "Duplicate year-month keys in manifest")
    months_2022 = sorted(month for year, month in keys if year == 2022)
    months_2023 = sorted(month for year, month in keys if year == 2023)
    require(months_2022 == EXPECTED_MONTHS_2022, "Unexpected 2022 months: %s" % months_2022)
    require(months_2023 == EXPECTED_MONTHS_2023, "Unexpected 2023 months: %s" % months_2023)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    reference: Dict[Tuple[int, int], Dict[str, Any]] = {}
    arrays: Dict[str, np.ndarray] = {}
    seen_source_days: Dict[int, List[int]] = defaultdict(list)

    for position, entry in enumerate(
        sorted(entries, key=lambda item: (int(item["year"]), int(item["month"]))),
        start=1,
    ):
        year = int(entry["year"])
        month = int(entry["month"])
        split = str(entry["split"])
        source_files = list(entry.get("source_files", []))
        days = [parse_day_from_source_name(item) for item in source_files]
        require(len(days) == len(set(days)), "Duplicate source days for %d-%02d" % (year, month))
        require(int(entry["n_days"]) == len(days), "Manifest n_days mismatch for %d-%02d" % (year, month))
        require(
            all(doy_to_month(year, doy) == month for doy in days),
            "Source day assigned to wrong month for %d-%02d" % (year, month),
        )
        seen_source_days[year].extend(days)

        count_maps = {
            name: build_count_map(day_masks, year, days, name)
            for name in MASK_NAMES
        }

        cache_path = resolve_cache_file(cache_dir, entry)
        cache_sha256 = sha256_file(cache_path)
        payload = load_pt(torch_module, cache_path)
        for required_key in (
            "obs_count_map",
            "valid_mask",
            "n_days",
            "source_files",
            "year",
            "month",
            "split",
        ):
            require(required_key in payload, "Cache missing %s: %s" % (required_key, cache_path))

        stored_count = tensor_numpy(payload["obs_count_map"]).astype(np.int32, copy=False)
        stored_valid = tensor_numpy(payload["valid_mask"]).astype(np.float32, copy=False)
        require(stored_count.shape == (H, W), "Stored obs_count_map shape mismatch")
        require(stored_valid.shape == (H, W), "Stored valid_mask shape mismatch")
        require(int(payload["n_days"]) == len(days), "Cache n_days mismatch")
        require(int(payload["year"]) == year, "Cache year mismatch")
        require(int(payload["month"]) == month, "Cache month mismatch")
        require(str(payload["split"]) == split, "Cache split mismatch")
        require(
            [parse_day_from_source_name(v) for v in payload["source_files"]] == days,
            "Cache source_files differ from manifest",
        )

        stored_eval = stored_valid > 0.5
        internal_expected_eval = stored_count >= min_obs_days
        cache_internal_exact = bool(np.array_equal(stored_eval, internal_expected_eval))
        code_count_exact = bool(
            np.array_equal(
                count_maps["i380_non_nan"].astype(np.int32, copy=False),
                stored_count,
            )
        )
        code_valid_exact = bool(
            np.array_equal(
                count_maps["i380_non_nan"] >= min_obs_days,
                stored_eval,
            )
        )

        masks_any = {
            name: value > 0
            for name, value in count_maps.items()
        }
        masks_min = {
            name: value >= min_obs_days
            for name, value in count_maps.items()
        }
        overlap_any = overlap_stats(
            masks_any["i380_finite"],
            masks_any["bccmass_finite"],
        )
        overlap_min = overlap_stats(
            masks_min["i380_finite"],
            masks_min["bccmass_finite"],
        )
        i_semantics_any = overlap_stats(
            masks_any["i380_non_nan"],
            masks_any["i380_finite"],
        )
        i_semantics_min = overlap_stats(
            masks_min["i380_non_nan"],
            masks_min["i380_finite"],
        )

        daily_i_fractions = [
            ratio(count_true(day_masks[year][doy]["i380_finite"]), N_GRID)
            for doy in days
        ]
        daily_i_summary = numeric_summary(
            value for value in daily_i_fractions if value is not None
        )
        calendar_days = expected_calendar_doys(year, month)
        missing_calendar_days = sorted(set(calendar_days) - set(days))

        row = {
            "analysis_role": role(year, month),
            "year": year,
            "month": month,
            "split": split,
            "n_source_days": len(days),
            "source_doys": ";".join("%03d" % value for value in days),
            "missing_calendar_doys": ";".join("%03d" % value for value in missing_calendar_days),
            "cached_obs_any_cells": int(np.count_nonzero(stored_count > 0)),
            "cached_obs_any_fraction": ratio(int(np.count_nonzero(stored_count > 0)), N_GRID),
            "cached_eval_cells": int(np.count_nonzero(stored_eval)),
            "cached_eval_fraction": ratio(int(np.count_nonzero(stored_eval)), N_GRID),
            "i380_non_nan_any_cells": int(np.count_nonzero(masks_any["i380_non_nan"])),
            "i380_non_nan_any_fraction": ratio(int(np.count_nonzero(masks_any["i380_non_nan"])), N_GRID),
            "i380_non_nan_ge_min_cells": int(np.count_nonzero(masks_min["i380_non_nan"])),
            "i380_non_nan_ge_min_fraction": ratio(int(np.count_nonzero(masks_min["i380_non_nan"])), N_GRID),
            "i380_finite_any_cells": int(np.count_nonzero(masks_any["i380_finite"])),
            "i380_finite_any_fraction": ratio(int(np.count_nonzero(masks_any["i380_finite"])), N_GRID),
            "i380_finite_ge_min_cells": int(np.count_nonzero(masks_min["i380_finite"])),
            "i380_finite_ge_min_fraction": ratio(int(np.count_nonzero(masks_min["i380_finite"])), N_GRID),
            "bccmass_finite_any_cells": int(np.count_nonzero(masks_any["bccmass_finite"])),
            "bccmass_finite_any_fraction": ratio(int(np.count_nonzero(masks_any["bccmass_finite"])), N_GRID),
            "bccmass_finite_ge_min_cells": int(np.count_nonzero(masks_min["bccmass_finite"])),
            "bccmass_finite_ge_min_fraction": ratio(int(np.count_nonzero(masks_min["bccmass_finite"])), N_GRID),
            "jaccard_i380_bccmass_any": overlap_any["jaccard"],
            "jaccard_i380_bccmass_ge_min": overlap_min["jaccard"],
            "i380_only_any_cells": overlap_any["a_only_cells"],
            "bccmass_only_any_cells": overlap_any["b_only_cells"],
            "i380_only_ge_min_cells": overlap_min["a_only_cells"],
            "bccmass_only_ge_min_cells": overlap_min["b_only_cells"],
            "jaccard_i380_non_nan_vs_finite_any": i_semantics_any["jaccard"],
            "jaccard_i380_non_nan_vs_finite_ge_min": i_semantics_min["jaccard"],
            "i380_non_nan_equals_finite_any": i_semantics_any["exact_equal"],
            "i380_non_nan_equals_finite_ge_min": i_semantics_min["exact_equal"],
            "code_count_map_exact_cache": code_count_exact,
            "code_valid_mask_exact_cache": code_valid_exact,
            "cache_internal_valid_mask_exact": cache_internal_exact,
            "cache_file_sha256": cache_sha256,
            "daily_i380_finite_fraction_min": daily_i_summary["min"],
            "daily_i380_finite_fraction_median": daily_i_summary["median"],
            "daily_i380_finite_fraction_mean": daily_i_summary["mean"],
            "daily_i380_finite_fraction_max": daily_i_summary["max"],
        }
        rows.append(row)

        key = "%d-%02d" % (year, month)
        details[key] = {
            "analysis_role": role(year, month),
            "year": year,
            "month": month,
            "split": split,
            "cache_file": str(cache_path),
            "cache_file_sha256": cache_sha256,
            "source_doys": days,
            "source_dates": [doy_to_date(year, day).isoformat() for day in days],
            "missing_calendar_doys": missing_calendar_days,
            "missing_calendar_dates": [
                doy_to_date(year, day).isoformat()
                for day in missing_calendar_days
            ],
            "daily_i380_finite_fraction_summary": daily_i_summary,
            "count_distributions": {
                name: count_map_summary(value, len(days))
                for name, value in count_maps.items()
            },
            "overlap": {
                "i380_finite_vs_bccmass_finite_any_day": overlap_any,
                "i380_finite_vs_bccmass_finite_ge_min_days": overlap_min,
                "i380_non_nan_vs_i380_finite_any_day": i_semantics_any,
                "i380_non_nan_vs_i380_finite_ge_min_days": i_semantics_min,
            },
            "cache_lineage": {
                "code_faithful_i380_non_nan_count_map_exact": code_count_exact,
                "code_faithful_i380_non_nan_valid_mask_exact": code_valid_exact,
                "stored_valid_mask_equals_stored_count_ge_min": cache_internal_exact,
                "stored_count_unequal_cells": int(
                    np.count_nonzero(
                        stored_count
                        != count_maps["i380_non_nan"].astype(np.int32, copy=False)
                    )
                ),
                "stored_valid_unequal_cells": int(
                    np.count_nonzero(
                        stored_eval
                        != (count_maps["i380_non_nan"] >= min_obs_days)
                    )
                ),
            },
        }

        reference[(year, month)] = {
            "valid_cells": int(np.count_nonzero(stored_eval)),
            "valid_fraction": float(np.mean(stored_eval)),
            "cache_file": str(cache_path),
            "cache_file_sha256": cache_sha256,
        }
        arrays["%s_i380_non_nan_count" % key] = count_maps["i380_non_nan"].astype(np.uint16)
        arrays["%s_i380_finite_count" % key] = count_maps["i380_finite"].astype(np.uint16)
        arrays["%s_bccmass_finite_count" % key] = count_maps["bccmass_finite"].astype(np.uint16)
        arrays["%s_cached_obs_count" % key] = stored_count.astype(np.uint16)
        arrays["%s_eval_difference_class" % key] = (
            (masks_min["i380_finite"].astype(np.uint8) * 2)
            + masks_min["bccmass_finite"].astype(np.uint8)
        )

        del payload
        print(
            "CACHE_AUDIT_PROGRESS=%d/%d year=%d month=%02d count_exact=%s valid_exact=%s"
            % (position, len(entries), year, month, code_count_exact, code_valid_exact),
            flush=True,
        )

    for year in (2022, 2023):
        manifest_days = seen_source_days[year]
        require(
            len(manifest_days) == len(set(manifest_days)),
            "A source day appears in more than one monthly group for %d" % year,
        )
        raw_days = sorted(day_masks.get(year, {}).keys())
        require(
            sorted(manifest_days) == raw_days,
            "Raw-source day inventory differs from monthly manifest for %d; raw_only=%s manifest_only=%s"
            % (
                year,
                sorted(set(raw_days) - set(manifest_days)),
                sorted(set(manifest_days) - set(raw_days)),
            ),
        )

    manifest_info = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "group_count": len(entries),
        "months_2022": months_2022,
        "months_2023": months_2023,
        "source_days_2022": sorted(seen_source_days[2022]),
        "source_days_2023": sorted(seen_source_days[2023]),
    }
    return rows, {"manifest": manifest_info, "months": details}, reference, arrays


def build_daily_rows(
    day_masks: Mapping[int, Dict[int, Dict[str, np.ndarray]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for year in sorted(day_masks):
        for doy in sorted(day_masks[year]):
            masks = {
                name: unpack_mask(day_masks[year][doy][name])
                for name in MASK_NAMES
            }
            i_b = overlap_stats(masks["i380_finite"], masks["bccmass_finite"])
            i_semantics = overlap_stats(masks["i380_non_nan"], masks["i380_finite"])
            same_row_pair_equal = bool(
                np.array_equal(
                    masks["same_row_both_finite"],
                    masks["i380_finite"] & masks["bccmass_finite"],
                )
            )
            row = {
                "year": year,
                "doy": doy,
                "date": doy_to_date(year, doy).isoformat(),
                "month": doy_to_month(year, doy),
                "source_present_cells": int(np.count_nonzero(masks["source_present"])),
                "source_present_fraction": float(np.mean(masks["source_present"])),
                "i380_non_nan_cells": int(np.count_nonzero(masks["i380_non_nan"])),
                "i380_non_nan_fraction": float(np.mean(masks["i380_non_nan"])),
                "i380_finite_cells": int(np.count_nonzero(masks["i380_finite"])),
                "i380_finite_fraction": float(np.mean(masks["i380_finite"])),
                "bccmass_finite_cells": int(np.count_nonzero(masks["bccmass_finite"])),
                "bccmass_finite_fraction": float(np.mean(masks["bccmass_finite"])),
                "same_row_both_finite_cells": int(np.count_nonzero(masks["same_row_both_finite"])),
                "same_row_both_finite_fraction": float(np.mean(masks["same_row_both_finite"])),
                "grid_intersection_i380_bccmass_cells": i_b["intersection_cells"],
                "grid_union_i380_bccmass_cells": i_b["union_cells"],
                "i380_only_cells": i_b["a_only_cells"],
                "bccmass_only_cells": i_b["b_only_cells"],
                "jaccard_i380_bccmass": i_b["jaccard"],
                "jaccard_i380_non_nan_vs_finite": i_semantics["jaccard"],
                "i380_non_nan_mask_equals_finite": i_semantics["exact_equal"],
                "same_row_pair_equals_grid_intersection": same_row_pair_equal,
            }
            rows.append(row)
    return rows


def discover_formal_reports(code_root: Path) -> List[Path]:
    paths: set[Path] = set()
    for pattern in FORMAL_REPORT_PATTERNS:
        paths.update(path.resolve() for path in code_root.glob(pattern))
    return sorted(paths)


def audit_formal_reports(
    *,
    code_root: Path,
    manifest_sha256: str,
    monthly_reference: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = discover_formal_reports(code_root)
    require(
        len(paths) == EXPECTED_REPORT_COUNT,
        "Expected %d formal evaluation reports, found %d under %s"
        % (EXPECTED_REPORT_COUNT, len(paths), code_root),
    )
    rows: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, int]] = set()

    expected_eval_keys = (
        [(2022, month) for month in EXPECTED_VALIDATION_MONTHS_2022]
        + [(2023, month) for month in EXPECTED_MONTHS_2023]
    )

    for path in paths:
        report = load_json(path)
        experiment = str(report.get("experiment", {}).get("name", ""))
        seed = int(report.get("seed", -1))
        require(experiment in EXPECTED_CONFIGS, "Unexpected formal experiment: %s" % experiment)
        require(seed in EXPECTED_SEEDS, "Unexpected formal seed: %s" % seed)
        require((experiment, seed) not in seen_pairs, "Duplicate formal report: %s seed %d" % (experiment, seed))
        seen_pairs.add((experiment, seed))

        report_manifest_sha = str(
            report.get("provenance", {}).get("monthly_manifest_sha256", "")
        )
        manifest_match = report_manifest_sha == manifest_sha256
        overall_pass = report.get("verification", {}).get("overall_pass") is True

        observed_group_keys: List[Tuple[int, int]] = []
        valid_counts_exact = True
        valid_fractions_exact = True
        valid_fraction_fields_complete = True
        cache_hash_fields_complete = True
        reported_cache_hashes_exact = True
        section_totals: Dict[int, int] = {}

        for section_name, expected_year in (
            ("eval_2022_validation", 2022),
            ("eval_2023_external", 2023),
        ):
            section = report.get(section_name, {})
            group_rows = list(section.get("per_group", []))
            recomputed_total = 0
            for group in group_rows:
                year = int(group["year"])
                month = int(group["month"])
                require(year == expected_year, "Report section year mismatch: %s" % path)
                key = (year, month)
                observed_group_keys.append(key)
                require(key in monthly_reference, "Report group absent from cache: %s" % (key,))
                reference = monthly_reference[key]
                reported_n = int(group["n_valid_pixels"])
                if "valid_pixel_fraction" in group:
                    reported_fraction = float(group["valid_pixel_fraction"])
                else:
                    valid_fraction_fields_complete = False
                    reported_fraction = reported_n / float(N_GRID)
                recomputed_total += reported_n
                valid_counts_exact = valid_counts_exact and (
                    reported_n == int(reference["valid_cells"])
                )
                valid_fractions_exact = valid_fractions_exact and math.isclose(
                    reported_fraction,
                    float(reference["valid_fraction"]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                if group.get("file_sha256"):
                    reported_cache_hashes_exact = reported_cache_hashes_exact and (
                        str(group["file_sha256"])
                        == str(reference["cache_file_sha256"])
                    )
                else:
                    cache_hash_fields_complete = False
            reported_total = int(section.get("n_total_valid_pixels", -1))
            require(
                recomputed_total == reported_total,
                "Per-group total differs from section total: %s" % path,
            )
            section_totals[expected_year] = reported_total

        per_group_keys_exact = sorted(observed_group_keys) == sorted(expected_eval_keys)
        report_pass = bool(
            overall_pass
            and manifest_match
            and per_group_keys_exact
            and valid_counts_exact
            and valid_fractions_exact
            and reported_cache_hashes_exact
        )
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "artifact_name": str(report.get("artifact_name", "")),
                "report": str(path),
                "report_sha256": sha256_file(path),
                "overall_pass": overall_pass,
                "monthly_manifest_sha256": report_manifest_sha,
                "manifest_sha256_matches_current": manifest_match,
                "2022_validation_total_valid_pixels": section_totals.get(2022),
                "2023_external_total_valid_pixels": section_totals.get(2023),
                "per_group_keys_exact": per_group_keys_exact,
                "per_group_valid_counts_exact": valid_counts_exact,
                "per_group_valid_fractions_exact": valid_fractions_exact,
                "per_group_valid_fraction_fields_complete": valid_fraction_fields_complete,
                "per_group_cache_hash_fields_complete": cache_hash_fields_complete,
                "per_group_reported_cache_hashes_exact": reported_cache_hashes_exact,
                "report_common_mask_pass": report_pass,
            }
        )

    expected_pairs = {
        (experiment, seed)
        for experiment in EXPECTED_CONFIGS
        for seed in EXPECTED_SEEDS
    }
    require(seen_pairs == expected_pairs, "Formal experiment/seed matrix is incomplete")
    manifest_hashes = sorted(set(row["monthly_manifest_sha256"] for row in rows))
    totals_2022 = sorted(set(int(row["2022_validation_total_valid_pixels"]) for row in rows))
    totals_2023 = sorted(set(int(row["2023_external_total_valid_pixels"]) for row in rows))
    summary = {
        "report_count": len(rows),
        "expected_report_count": EXPECTED_REPORT_COUNT,
        "experiments": sorted(set(row["experiment"] for row in rows)),
        "seeds": sorted(set(int(row["seed"]) for row in rows)),
        "unique_monthly_manifest_sha256_values": manifest_hashes,
        "unique_2022_validation_total_valid_pixels": totals_2022,
        "unique_2023_external_total_valid_pixels": totals_2023,
        "all_reports_common_mask_pass": all(
            bool(row["report_common_mask_pass"])
            for row in rows
        ),
    }
    return rows, summary


def aggregate_source_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_keys = [
        "num_rows",
        "mapped_rows",
        "invalid_doy_rows",
        "invalid_coordinate_rows",
        "i380_nan_rows",
        "i380_posinf_rows",
        "i380_neginf_rows",
        "i380_finite_rows",
        "i380_non_nan_declared_sentinel_rows",
        "bccmass_nan_rows",
        "bccmass_posinf_rows",
        "bccmass_neginf_rows",
        "bccmass_finite_rows",
        "bccmass_non_nan_declared_sentinel_rows",
        "mapped_both_finite_rows",
        "mapped_i380_only_finite_rows",
        "mapped_bccmass_only_finite_rows",
        "mapped_neither_finite_rows",
    ]
    by_year: Dict[int, Dict[str, Any]] = {}
    for year in (2022, 2023):
        selected = [row for row in rows if int(row["year"]) == year]
        values = {
            key: sum(int(row[key]) for row in selected)
            for key in total_keys
        }
        row_union = (
            values["mapped_both_finite_rows"]
            + values["mapped_i380_only_finite_rows"]
            + values["mapped_bccmass_only_finite_rows"]
        )
        values["mapped_row_jaccard_i380_bccmass"] = ratio(
            values["mapped_both_finite_rows"], row_union
        )
        values["source_files"] = len(selected)
        by_year[year] = values
    overall = {
        key: sum(int(row[key]) for row in rows)
        for key in total_keys
    }
    row_union = (
        overall["mapped_both_finite_rows"]
        + overall["mapped_i380_only_finite_rows"]
        + overall["mapped_bccmass_only_finite_rows"]
    )
    overall["mapped_row_jaccard_i380_bccmass"] = ratio(
        overall["mapped_both_finite_rows"], row_union
    )
    return {"by_year": by_year, "overall": overall}


def monthly_aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "i380_bccmass_any_day_jaccard": numeric_summary(
            float(row["jaccard_i380_bccmass_any"])
            for row in rows
            if row["jaccard_i380_bccmass_any"] is not None
        ),
        "i380_bccmass_ge_min_days_jaccard": numeric_summary(
            float(row["jaccard_i380_bccmass_ge_min"])
            for row in rows
            if row["jaccard_i380_bccmass_ge_min"] is not None
        ),
        "i380_finite_eval_fraction": numeric_summary(
            float(row["i380_finite_ge_min_fraction"])
            for row in rows
        ),
        "bccmass_finite_eval_fraction": numeric_summary(
            float(row["bccmass_finite_ge_min_fraction"])
            for row in rows
        ),
        "all_code_count_maps_exact_cache": all(
            bool(row["code_count_map_exact_cache"]) for row in rows
        ),
        "all_code_valid_masks_exact_cache": all(
            bool(row["code_valid_mask_exact_cache"]) for row in rows
        ),
        "all_cache_internal_valid_masks_exact": all(
            bool(row["cache_internal_valid_mask_exact"]) for row in rows
        ),
        "all_i380_non_nan_and_finite_monthly_any_masks_equal": all(
            bool(row["i380_non_nan_equals_finite_any"])
            for row in rows
        ),
        "all_i380_non_nan_and_finite_monthly_eval_masks_equal": all(
            bool(row["i380_non_nan_equals_finite_ge_min"])
            for row in rows
        ),
    }


def daily_aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for year in (2022, 2023):
        selected = [row for row in rows if int(row["year"]) == year]
        actual_days = sorted(int(row["doy"]) for row in selected)
        result[str(year)] = {
            "n_source_days": len(actual_days),
            "source_doys": actual_days,
            "missing_calendar_doys": sorted(set(range(1, 366)) - set(actual_days)),
            "missing_calendar_dates": [
                doy_to_date(year, value).isoformat()
                for value in sorted(set(range(1, 366)) - set(actual_days))
            ],
            "i380_finite_fraction": numeric_summary(
                float(row["i380_finite_fraction"]) for row in selected
            ),
            "bccmass_finite_fraction": numeric_summary(
                float(row["bccmass_finite_fraction"]) for row in selected
            ),
            "i380_bccmass_daily_jaccard": numeric_summary(
                float(row["jaccard_i380_bccmass"])
                for row in selected
                if row["jaccard_i380_bccmass"] is not None
            ),
            "all_i380_non_nan_and_finite_daily_masks_equal": all(
                bool(row["i380_non_nan_mask_equals_finite"])
                for row in selected
            ),
            "all_same_row_pair_masks_equal_grid_intersections": all(
                bool(row["same_row_pair_equals_grid_intersection"])
                for row in selected
            ),
        }
    return result


def make_plots(
    *,
    output_dir: Path,
    monthly_rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    min_obs_days: int,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    created: List[str] = []
    ordered = sorted(monthly_rows, key=lambda row: (int(row["year"]), int(row["month"])))
    labels = ["%d-%02d" % (int(row["year"]), int(row["month"])) for row in ordered]
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.8), constrained_layout=True)
    axes[0].plot(
        x,
        [float(row["i380_finite_ge_min_fraction"]) for row in ordered],
        marker="o",
        linewidth=1.5,
        label=r"finite $I_{380}$, count $\geq$ %d" % min_obs_days,
    )
    axes[0].plot(
        x,
        [float(row["bccmass_finite_ge_min_fraction"]) for row in ordered],
        marker="s",
        linewidth=1.5,
        label=r"finite BCCMASS, count $\geq$ %d" % min_obs_days,
    )
    axes[0].plot(
        x,
        [float(row["cached_eval_fraction"]) for row in ordered],
        linestyle="--",
        linewidth=1.2,
        label="stored evaluation mask",
    )
    axes[0].set_ylabel("Fraction of global grid cells")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)

    jaccard_values = [
        np.nan if row["jaccard_i380_bccmass_ge_min"] is None else float(row["jaccard_i380_bccmass_ge_min"])
        for row in ordered
    ]
    axes[1].plot(x, jaccard_values, color="#6a3d9a", marker="o", linewidth=1.5)
    axes[1].set_ylabel("Jaccard index")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=55, ha="right")
    axes[1].set_xlabel("Year-month")
    figure_path = output_dir / "monthly_coverage_and_jaccard.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    created.append(str(figure_path))

    max_days = max(int(row["n_source_days"]) for row in ordered)
    matrices: Dict[str, np.ndarray] = {}
    for variable in ("i380_finite", "bccmass_finite"):
        matrix = np.zeros((len(ordered), max_days + 1), dtype=np.float64)
        for row_index, row in enumerate(ordered):
            key = "%d-%02d" % (int(row["year"]), int(row["month"]))
            count_map = arrays["%s_%s_count" % (key, variable)].reshape(-1).astype(np.int64)
            hist = np.bincount(count_map, minlength=max_days + 1).astype(np.float64)
            matrix[row_index, : hist.size] = hist / float(N_GRID)
        matrices[variable] = matrix

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.4), constrained_layout=True, sharey=True)
    for axis, variable, title in (
        (axes[0], "i380_finite", r"finite $I_{380}$"),
        (axes[1], "bccmass_finite", "finite BCCMASS"),
    ):
        image = axis.imshow(
            np.log10(matrices[variable] + 1e-8),
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap="viridis",
            vmin=-8,
            vmax=0,
        )
        axis.set_title(title)
        axis.set_xlabel("Valid days per grid cell")
        axis.set_xticks(np.arange(0, max_days + 1, 5))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
    axes[0].set_ylabel("Year-month")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82)
    colorbar.set_label(r"$\log_{10}$(fraction of global grid cells + $10^{-8}$)")
    figure_path = output_dir / "valid_day_count_distributions.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    created.append(str(figure_path))

    for year in (2022, 2023):
        year_rows = [row for row in ordered if int(row["year"]) == year]
        n = len(year_rows)
        ncols = 4 if n > 9 else 3
        nrows = int(math.ceil(n / ncols))

        differences = []
        for row in year_rows:
            key = "%d-%02d" % (year, int(row["month"]))
            differences.append(
                arrays["%s_bccmass_finite_count" % key].astype(np.int16)
                - arrays["%s_i380_finite_count" % key].astype(np.int16)
            )
        limit = max(1, max(int(np.max(np.abs(value))) for value in differences))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.6 * nrows), constrained_layout=True)
        flat_axes = np.atleast_1d(axes).reshape(-1)
        image = None
        for axis, row, difference in zip(flat_axes, year_rows, differences):
            image = axis.imshow(
                difference,
                origin="lower",
                extent=(-180, 180, -90, 90),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="auto",
            )
            axis.set_title("%d-%02d" % (year, int(row["month"])))
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        for axis in flat_axes[n:]:
            axis.axis("off")
        if image is not None:
            colorbar = fig.colorbar(image, ax=flat_axes[:n].tolist(), shrink=0.78)
            colorbar.set_label("BCCMASS finite-day count minus I_380 finite-day count")
        figure_path = output_dir / ("valid_day_count_difference_%d.png" % year)
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        created.append(str(figure_path))

        cmap = ListedColormap(["#f2f2f2", "#6baed6", "#e34a33", "#31a354"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.6 * nrows), constrained_layout=True)
        flat_axes = np.atleast_1d(axes).reshape(-1)
        image = None
        for axis, row in zip(flat_axes, year_rows):
            key = "%d-%02d" % (year, int(row["month"]))
            class_map = arrays["%s_eval_difference_class" % key]
            image = axis.imshow(
                class_map,
                origin="lower",
                extent=(-180, 180, -90, 90),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            axis.set_title("%d-%02d" % (year, int(row["month"])))
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        for axis in flat_axes[n:]:
            axis.axis("off")
        if image is not None:
            colorbar = fig.colorbar(image, ax=flat_axes[:n].tolist(), shrink=0.78, ticks=[0, 1, 2, 3])
            colorbar.ax.set_yticklabels(["neither", "BCCMASS only", "I_380 only", "both"])
            colorbar.set_label("Evaluation-mask class (count >= %d)" % min_obs_days)
        figure_path = output_dir / ("evaluation_mask_difference_%d.png" % year)
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        created.append(str(figure_path))

    return created


def plot_only_from_snapshot(output_dir: Path) -> int:
    """Render figures in a fresh process that never imports PyTorch."""
    snapshot_path = output_dir / "gate87_preplot_core_snapshot.json"
    arrays_path = output_dir / "posp_bccmass_sampling_maps.npz"
    status_path = output_dir / "plot_generation_status.json"
    require(snapshot_path.is_file(), "Pre-plot snapshot not found: %s" % snapshot_path)
    require(arrays_path.is_file(), "Plot-array archive not found: %s" % arrays_path)
    require(not status_path.exists(), "Plot status already exists: %s" % status_path)

    snapshot = load_json(snapshot_path)
    monthly_rows = list(snapshot.get("monthly_rows", []))
    min_obs_days = int(snapshot.get("min_obs_days", -1))
    require(len(monthly_rows) == 21, "Pre-plot snapshot does not contain 21 months")
    require(min_obs_days >= 1, "Invalid min_obs_days in pre-plot snapshot")

    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    created = make_plots(
        output_dir=output_dir,
        monthly_rows=monthly_rows,
        arrays=arrays,
        min_obs_days=min_obs_days,
    )
    all_exist = len(created) == 6 and all(Path(value).is_file() for value in created)
    plot_report = {
        "plot_process_isolated_from_torch": True,
        "plot_count": len(created),
        "expected_plot_count": 6,
        "files": created,
        "all_six_required_plots_generated": all_exist,
    }
    strict_json_dump(plot_report, status_path)
    print("ISOLATED_PLOT_PROCESS_STATUS=%s" % ("PASS" if all_exist else "FAIL"), flush=True)
    print("PLOT_STATUS_REPORT=%s" % status_path, flush=True)
    return 0 if all_exist else 2


def run_isolated_plot_process(output_dir: Path) -> Tuple[List[str], Optional[str]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--plot-only-dir",
        str(output_dir),
    ]
    print("ISOLATED_PLOT_PROCESS_START", flush=True)
    completed = subprocess.run(command, check=False)
    status_path = output_dir / "plot_generation_status.json"
    if completed.returncode != 0:
        return [], "Isolated plotting process exited with code %d" % completed.returncode
    if not status_path.is_file():
        return [], "Isolated plotting process did not write its status report"
    status = load_json(status_path)
    files = [str(value) for value in status.get("files", [])]
    all_exist = (
        status.get("all_six_required_plots_generated") is True
        and len(files) == 6
        and all(Path(value).is_file() for value in files)
    )
    if not all_exist:
        return files, "Isolated plotting process returned an incomplete figure set"
    print("ISOLATED_PLOT_PROCESS_PASS", flush=True)
    return files, None


def write_summary(
    *,
    path: Path,
    report: Mapping[str, Any],
    monthly_rows: Sequence[Mapping[str, Any]],
) -> None:
    verification = report["verification"]
    source_overall = report["source_row_summary"]["overall"]
    formal = report["formal_common_mask_audit"]
    formal_summary = formal["summary"]
    lines = [
        "POSP / BCCMASS SAMPLING-SUPPORT AUDIT",
        "=" * 56,
        "",
        "STATUS: %s" % report["status"],
        "READ_ONLY: True",
        "",
        "Core lineage checks",
        "-------------------",
        "All code-faithful I_380 count maps exactly reproduce cache: %s"
        % verification["all_code_count_maps_exact_cache"],
        "All code-faithful I_380 valid masks exactly reproduce cache: %s"
        % verification["all_code_valid_masks_exact_cache"],
        "I_380 non-NaN and finite semantics are equivalent: %s"
        % verification["i380_non_nan_equals_finite_everywhere"],
        "No declared missing-value sentinel occurred in I_380/BCCMASS: %s"
        % verification["no_declared_sentinel_hits"],
        "Formal reports use the same cache masks: %s"
        % verification["formal_reports_common_mask_pass"],
        "",
        "Source-row totals",
        "-----------------",
        "Rows: %s" % source_overall["num_rows"],
        "Mapped rows: %s" % source_overall["mapped_rows"],
        "I_380 finite rows: %s" % source_overall["i380_finite_rows"],
        "BCCMASS finite rows: %s" % source_overall["bccmass_finite_rows"],
        "I_380/BCCMASS mapped-row Jaccard: %s"
        % source_overall["mapped_row_jaccard_i380_bccmass"],
        "",
        "Formal common-mask evidence",
        "---------------------------",
        "Formal reports: %s" % formal_summary.get("report_count", "SKIPPED"),
        "Experiments: %s" % ", ".join(formal_summary.get("experiments", [])),
        "Seeds: %s" % formal_summary.get("seeds", []),
        "2022 validation total valid pixels: %s"
        % formal_summary.get("unique_2022_validation_total_valid_pixels", "SKIPPED"),
        "2023 external total valid pixels: %s"
        % formal_summary.get("unique_2023_external_total_valid_pixels", "SKIPPED"),
        "",
        "Monthly results",
        "---------------",
        "year-month | n_days | I_380 eval fraction | BCCMASS eval fraction | Jaccard | cache exact",
    ]
    for row in monthly_rows:
        lines.append(
            "%d-%02d | %d | %.8f | %.8f | %s | %s"
            % (
                int(row["year"]),
                int(row["month"]),
                int(row["n_source_days"]),
                float(row["i380_finite_ge_min_fraction"]),
                float(row["bccmass_finite_ge_min_fraction"]),
                "NA" if row["jaccard_i380_bccmass_ge_min"] is None else "%.8f" % float(row["jaccard_i380_bccmass_ge_min"]),
                bool(row["code_count_map_exact_cache"] and row["code_valid_mask_exact_cache"]),
            )
        )
    lines += [
        "",
        "Interpretation boundary",
        "-----------------------",
        "I_380/BCCMASS mask equality is not required for Gate success. The historical",
        "pipeline deliberately used I_380 to define source availability. This audit",
        "tests whether that declared rule is traceable and consistently applied; it",
        "also quantifies, rather than assumes away, any sampling-support difference.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> int:
    indices_a = np.asarray([0, 1, 7, 8, 100, N_GRID - 1], dtype=np.int64)
    indices_b = np.asarray([1, 8, 101, N_GRID - 1], dtype=np.int64)
    mask_a = unpack_mask(pack_indices(indices_a))
    mask_b = unpack_mask(pack_indices(indices_b))
    require(set(np.flatnonzero(mask_a).tolist()) == set(indices_a.tolist()), "pack/unpack A failed")
    require(set(np.flatnonzero(mask_b).tolist()) == set(indices_b.tolist()), "pack/unpack B failed")
    overlap = overlap_stats(mask_a, mask_b)
    require(overlap["intersection_cells"] == 3, "Overlap intersection test failed")
    require(overlap["union_cells"] == 7, "Overlap union test failed")
    require(math.isclose(float(overlap["jaccard"]), 3.0 / 7.0), "Jaccard test failed")
    count = np.zeros((H, W), dtype=np.uint16)
    count.reshape(-1)[:6] = np.asarray([0, 1, 1, 2, 3, 3], dtype=np.uint16)
    summary = count_map_summary(count, 3)
    require(summary["histogram_cell_count_by_valid_days"]["1"] == 2, "Histogram test failed")
    require(doy_to_date(2023, 32).isoformat() == "2023-02-01", "Calendar test failed")
    print("SELF_TEST_PASS")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of I_380/BCCMASS sampling support, monthly cache "
            "lineage, and formal common evaluation masks."
        )
    )
    parser.add_argument("--nc-2022", default=DEFAULT_NC_2022)
    parser.add_argument("--nc-2023", nargs=2, default=[DEFAULT_NC_2023_1, DEFAULT_NC_2023_2])
    parser.add_argument("--grid-ref", default=DEFAULT_GRID_REF)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--code-root",
        default=str(Path(__file__).resolve().parent),
        help="Repository root containing the formal run directories.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-obs-days", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=500000)
    parser.add_argument(
        "--skip-formal-report-audit",
        action="store_true",
        help="Diagnostic-only escape hatch; not acceptable for closing the manuscript comment.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Diagnostic-only escape hatch; CSV/JSON/NPZ are still written.",
    )
    parser.add_argument(
        "--plot-only-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def load_dependencies() -> Tuple[Any, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency netCDF4. Install it in the active experiment environment."
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency torch. Run this script in the experiment environment."
        ) from exc
    return netCDF4, torch


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.plot_only_dir is not None:
        return plot_only_from_snapshot(Path(args.plot_only_dir).expanduser().resolve())

    require(args.min_obs_days >= 1, "--min-obs-days must be >= 1")
    require(args.chunk_rows >= 1000, "--chunk-rows must be >= 1000")

    netcdf4_module, torch_module = load_dependencies()
    try:
        torch_module.set_num_threads(1)
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass

    nc_2022 = Path(args.nc_2022).expanduser().resolve()
    nc_2023 = [Path(value).expanduser().resolve() for value in args.nc_2023]
    grid_ref = Path(args.grid_ref).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    code_root = Path(args.code_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in [nc_2022, *nc_2023, grid_ref]:
        require(path.is_file(), "Required input file not found: %s" % path)
    require(cache_dir.is_dir(), "Cache directory not found: %s" % cache_dir)
    require(code_root.is_dir(), "Code root not found: %s" % code_root)
    require(not output_dir.exists(), "Output directory exists; refusing overwrite: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    print("GATE87_READ_ONLY_AUDIT_START", flush=True)
    print("OUTPUT_DIR=%s" % output_dir, flush=True)

    grid_report = audit_grid_reference(grid_ref)
    require(grid_report["overall_pass"], "Grid reference differs from the formal 361x576 grid")

    day_masks: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    source_rows: List[Dict[str, Any]] = []
    source_rows.append(
        scan_source_file(
            netcdf4_module=netcdf4_module,
            path=nc_2022,
            year=2022,
            chunk_rows=args.chunk_rows,
            day_masks=day_masks,
        )
    )
    for path in nc_2023:
        source_rows.append(
            scan_source_file(
                netcdf4_module=netcdf4_module,
                path=path,
                year=2023,
                chunk_rows=args.chunk_rows,
                day_masks=day_masks,
            )
        )

    daily_rows = build_daily_rows(day_masks)
    monthly_rows, monthly_detail, monthly_reference, arrays = audit_monthly_cache(
        torch_module=torch_module,
        cache_dir=cache_dir,
        day_masks=day_masks,
        min_obs_days=args.min_obs_days,
    )

    manifest_sha256 = monthly_detail["manifest"]["sha256"]
    if args.skip_formal_report_audit:
        formal_rows: List[Dict[str, Any]] = []
        formal_summary = {
            "skipped": True,
            "all_reports_common_mask_pass": False,
            "reason": "--skip-formal-report-audit was supplied",
        }
    else:
        formal_rows, formal_summary = audit_formal_reports(
            code_root=code_root,
            manifest_sha256=manifest_sha256,
            monthly_reference=monthly_reference,
        )

    source_summary = aggregate_source_stats(source_rows)
    monthly_summary = monthly_aggregate_summary(monthly_rows)
    daily_summary = daily_aggregate_summary(daily_rows)

    no_invalid_source_rows = all(
        int(row["invalid_doy_rows"]) == 0
        and int(row["invalid_coordinate_rows"]) == 0
        for row in source_rows
    )
    no_declared_sentinel_hits = all(
        int(row["i380_non_nan_declared_sentinel_rows"]) == 0
        and int(row["bccmass_non_nan_declared_sentinel_rows"]) == 0
        for row in source_rows
    )
    no_i380_inf_rows = all(
        int(row["i380_posinf_rows"]) == 0
        and int(row["i380_neginf_rows"]) == 0
        for row in source_rows
    )
    bccmass_all_rows_finite = all(
        int(row["bccmass_nan_rows"]) == 0
        and int(row["bccmass_posinf_rows"]) == 0
        and int(row["bccmass_neginf_rows"]) == 0
        and int(row["bccmass_finite_rows"]) == int(row["num_rows"])
        for row in source_rows
    )
    i_semantics_daily = all(
        bool(item["all_i380_non_nan_and_finite_daily_masks_equal"])
        for item in daily_summary.values()
    )
    i_semantics_monthly = bool(
        monthly_summary["all_i380_non_nan_and_finite_monthly_any_masks_equal"]
        and monthly_summary["all_i380_non_nan_and_finite_monthly_eval_masks_equal"]
    )
    i_semantics_equivalent = bool(
        no_i380_inf_rows and i_semantics_daily and i_semantics_monthly
    )
    no_finite_i380_rows_with_nonfinite_bccmass = all(
        int(row["mapped_i380_only_finite_rows"]) == 0
        for row in source_rows
    )
    formal_pass = bool(formal_summary.get("all_reports_common_mask_pass", False))

    verification: Dict[str, Any] = {
        "grid_reference_exact": bool(grid_report["overall_pass"]),
        "no_invalid_doy_or_coordinate_rows": no_invalid_source_rows,
        "no_declared_sentinel_hits": no_declared_sentinel_hits,
        "i380_has_no_infinite_rows": no_i380_inf_rows,
        "i380_non_nan_equals_finite_everywhere": i_semantics_equivalent,
        "bccmass_all_source_rows_finite": bccmass_all_rows_finite,
        "no_finite_i380_rows_with_nonfinite_bccmass": (
            no_finite_i380_rows_with_nonfinite_bccmass
        ),
        "all_code_count_maps_exact_cache": bool(
            monthly_summary["all_code_count_maps_exact_cache"]
        ),
        "all_code_valid_masks_exact_cache": bool(
            monthly_summary["all_code_valid_masks_exact_cache"]
        ),
        "all_cache_internal_valid_masks_exact": bool(
            monthly_summary["all_cache_internal_valid_masks_exact"]
        ),
        "formal_reports_common_mask_pass": formal_pass,
        "i380_bccmass_equality_is_not_required_for_gate_pass": True,
    }

    arrays_path = output_dir / "posp_bccmass_sampling_maps.npz"
    np.savez_compressed(arrays_path, **arrays)

    # Persist every non-figure result before launching the plotting process.
    # If a platform-specific plotting failure occurs, the expensive source
    # scan and cache audit remain recoverable from these files.
    write_csv(daily_rows, CSV_DAILY_COLUMNS, output_dir / "posp_bccmass_daily_coverage.csv")
    write_csv(monthly_rows, CSV_MONTHLY_COLUMNS, output_dir / "posp_bccmass_monthly_coverage_overlap.csv")
    write_csv(source_rows, CSV_SOURCE_COLUMNS, output_dir / "posp_bccmass_source_row_audit.csv")
    write_csv(formal_rows, CSV_FORMAL_COLUMNS, output_dir / "formal_meanstate_common_mask_audit.csv")
    preplot_snapshot_path = output_dir / "gate87_preplot_core_snapshot.json"
    strict_json_dump(
        {
            "snapshot_name": "gate87_preplot_core_snapshot",
            "read_only": True,
            "min_obs_days": int(args.min_obs_days),
            "source_files": source_rows,
            "source_row_summary": source_summary,
            "daily_rows": daily_rows,
            "daily_summary": daily_summary,
            "monthly_rows": monthly_rows,
            "monthly_summary": monthly_summary,
            "monthly_detail": monthly_detail,
            "formal_rows": formal_rows,
            "formal_summary": formal_summary,
            "verification_before_plots": verification,
            "maps_npz": str(arrays_path),
        },
        preplot_snapshot_path,
    )
    print("PREPLOT_CORE_SNAPSHOT_WRITTEN=%s" % preplot_snapshot_path, flush=True)

    plot_files: List[str] = []
    plot_error: Optional[str] = None
    if not args.skip_plots:
        plot_files, plot_error = run_isolated_plot_process(output_dir)
        if plot_error is not None:
            print("PLOT_GENERATION_FAILED=%s" % plot_error, file=sys.stderr, flush=True)
    verification["all_six_required_plots_generated"] = len(plot_files) == 6

    overall_pass = all(
        bool(value)
        for key, value in verification.items()
        if key != "i380_bccmass_equality_is_not_required_for_gate_pass"
    )
    status = (
        "PASS_EXACT_LINEAGE_AND_COMMON_MASK_CHARACTERIZED"
        if overall_pass
        else "FAIL_REQUIRES_DIAGNOSIS_BEFORE_MANUSCRIPT_CLOSURE"
    )

    report = {
        "audit_name": "posp_bccmass_sampling_support_gate87",
        "audit_version": 1,
        "read_only": True,
        "status": status,
        "started_unix": started,
        "elapsed_seconds": float(time.time() - started),
        "definitions": {
            "historical_code_mask": "At least one mapped source row with ~isnan(I_380) in a day-grid cell.",
            "strict_i380_mask": "At least one mapped source row with isfinite(I_380) in a day-grid cell.",
            "strict_bccmass_mask": "At least one mapped source row with isfinite(BCCMASS) in a day-grid cell.",
            "monthly_count": "Sum of the corresponding binary daily grid masks over manifest-listed source days.",
            "monthly_evaluation_threshold": int(args.min_obs_days),
            "jaccard": "intersection / union; null only when both masks are empty.",
            "doy_mapping": "Positive finite DOY values are cast to int16, reproducing the historical truncation toward zero.",
            "grid_mapping": "rint((latitude+90)/0.5), rint((longitude+180)/0.625) on the 361x576 grid.",
        },
        "inputs": {
            "netcdf_files": [file_inventory(nc_2022)] + [file_inventory(path) for path in nc_2023],
            "grid_reference": file_inventory(grid_ref),
            "cache_dir": str(cache_dir),
            "code_root": str(code_root),
            "chunk_rows": int(args.chunk_rows),
            "min_obs_days": int(args.min_obs_days),
            "skip_formal_report_audit": bool(args.skip_formal_report_audit),
            "skip_plots": bool(args.skip_plots),
        },
        "grid_reference_audit": grid_report,
        "source_files": source_rows,
        "source_row_summary": source_summary,
        "daily_summary": daily_summary,
        "monthly_summary": monthly_summary,
        "monthly_detail": monthly_detail,
        "formal_common_mask_audit": {
            "summary": formal_summary,
            "reports": formal_rows,
        },
        "verification": verification,
        "outputs": {
            "daily_csv": str(output_dir / "posp_bccmass_daily_coverage.csv"),
            "monthly_csv": str(output_dir / "posp_bccmass_monthly_coverage_overlap.csv"),
            "source_csv": str(output_dir / "posp_bccmass_source_row_audit.csv"),
            "formal_csv": str(output_dir / "formal_meanstate_common_mask_audit.csv"),
            "maps_npz": str(arrays_path),
            "preplot_core_snapshot": str(preplot_snapshot_path),
            "plot_status_report": str(output_dir / "plot_generation_status.json"),
            "plots": plot_files,
            "plot_error": plot_error,
        },
        "interpretation_boundary": (
            "This audit characterizes processed collocation support. It does not "
            "treat MERRA-2 BCCMASS as independent observational truth and does not "
            "infer a causal emission-to-burden relationship. I_380/BCCMASS mask "
            "inequality is descriptive and is not by itself a Gate failure."
        ),
    }

    strict_json_dump(report, output_dir / "posp_bccmass_sampling_support_audit.json")
    write_summary(
        path=output_dir / "posp_bccmass_sampling_support_summary.txt",
        report=report,
        monthly_rows=monthly_rows,
    )

    print("JSON_REPORT=%s" % (output_dir / "posp_bccmass_sampling_support_audit.json"), flush=True)
    print("TEXT_REPORT=%s" % (output_dir / "posp_bccmass_sampling_support_summary.txt"), flush=True)
    print("STATUS=%s" % status, flush=True)
    print("GATE87_POSP_BCCMASS_SAMPLING_SUPPORT_COMPLETED", flush=True)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED_BY_USER", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(
            "GATE87_FAILED=%s: %s" % (type(exc).__name__, exc),
            file=sys.stderr,
            flush=True,
        )
        raise
