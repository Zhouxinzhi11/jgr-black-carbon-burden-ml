#!/usr/bin/env python3
# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""
Read-only fingerprint audit for inherited fused BCCMASS values.

The script compares randomly sampled rows from one or more POSP-centred fused
NetCDF files with the corresponding daily MERRA-2 M2T1NXAER BCCMASS fields.
It does not modify any source file and it does not retrain a model.

It tests these hypotheses:
  1. The fused latitude and longitude address the same native MERRA-2 cell.
  2. Latitude was accidentally mirrored north/south.
  3. A fixed one-native-cell latitude/longitude offset was used.
  4. Time was matched by proper interpolation using the actual MERRA-2 times.
  5. Time was matched to the nearest/lower/upper MERRA-2 record.
  6. Time followed the weighting expression found in the supplied 2024 script.

Outputs:
  merra2_bccmass_fingerprint_report.json
  merra2_bccmass_fingerprint_summary.txt
  merra2_bccmass_fingerprint_samples.csv

This is an evidence gate. A close match can identify a likely operation. A
weak or ambiguous match does not prove that the fused data are wrong because
the final file may have been built from a different historical script or an
orbit-level mean time that is not preserved in the fused rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from netCDF4 import Dataset, num2date
except ImportError as exc:  # pragma: no cover - exercised on the user's machine
    raise SystemExit(
        "Missing dependency 'netCDF4'. Install it in the active environment with:\n"
        "  python -m pip install netCDF4 numpy\n"
        f"Original import error: {exc}"
    )


SCRIPT_VERSION = "1.1.0"
DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")
RAW_SUFFIXES = {".nc", ".nc4", ".cdf"}
FUSED_NAMES = {
    "doy": ("DOY", "doy", "day_of_year"),
    "lat": ("Latitude", "latitude", "lat"),
    "lon": ("Longitude", "longitude", "lon"),
    "bcc": ("BCCMASS", "bccmass"),
}
RAW_NAMES = {
    "time": ("time", "Time", "valid_time"),
    "lat": ("lat", "latitude", "Latitude"),
    "lon": ("lon", "longitude", "Longitude"),
    "bcc": ("BCCMASS", "bccmass"),
}

CORE_CANDIDATE_NAMES = [
    "aligned_linear_actual",
    "aligned_nearest_actual",
    "aligned_lower_actual",
    "aligned_upper_actual",
    "aligned_supplied_2024_code",
    "mirrored_linear_actual",
    "mirrored_nearest_actual",
    "mirrored_supplied_2024_code",
]


def signed_grid_tag(value: int) -> str:
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "p0"


def neighbor_candidate_name(delta_lat: int, delta_lon: int, time_method: str) -> str:
    return (
        f"neighbor_dlat_{signed_grid_tag(delta_lat)}_"
        f"dlon_{signed_grid_tag(delta_lon)}_{time_method}"
    )


def all_candidate_names() -> List[str]:
    names = list(CORE_CANDIDATE_NAMES)
    for delta_lat in (-1, 0, 1):
        for delta_lon in (-1, 0, 1):
            if delta_lat == 0 and delta_lon == 0:
                continue
            for method in (
                "linear_actual",
                "nearest_actual",
                "supplied_2024_code",
            ):
                names.append(neighbor_candidate_name(delta_lat, delta_lon, method))
    return names


@dataclass(frozen=True)
class FusedSpec:
    path: Path
    year: int


@dataclass(frozen=True)
class RowCandidate:
    priority: float
    fused_index: int
    row_index: int
    date_key: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only raw-MERRA-2 versus fused-BCCMASS fingerprint audit."
    )
    parser.add_argument(
        "--raw-root",
        action="append",
        required=True,
        help=(
            "Directory or individual file containing MERRA-2 M2T1NXAER files. "
            "Repeat this option for multiple roots. Directories are scanned recursively."
        ),
    )
    parser.add_argument(
        "--fused",
        action="append",
        nargs=2,
        required=True,
        metavar=("FILE", "YEAR"),
        help=(
            "Fused NetCDF file followed by its calendar year. Repeat for each file, "
            "for example: --fused BCCMASS_train_1.nc 2023"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the JSON, TXT, and CSV reports.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Approximate total number of fused rows to compare (default: 100).",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=10,
        help="Maximum number of overlapping dates represented in the sample (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260910,
        help="Random seed used only for row sampling (default: 20260910).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows read at once while scanning fused DOY (default: 1000000).",
    )
    parser.add_argument(
        "--raw-name-contains",
        default="tavg1_2d_aer_Nx",
        help=(
            "Case-sensitive filename text used to restrict raw files "
            "(default: tavg1_2d_aer_Nx). Use an empty string to disable."
        ),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.max_dates < 1:
        parser.error("--max-dates must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    return args


def resolve_specs(items: Sequence[Sequence[str]]) -> List[FusedSpec]:
    specs: List[FusedSpec] = []
    for file_text, year_text in items:
        path = Path(file_text)
        if not path.is_file():
            raise FileNotFoundError(f"Fused file not found: {path}")
        year = int(year_text)
        if year < 1900 or year > 2100:
            raise ValueError(f"Implausible fused year {year} for {path}")
        specs.append(FusedSpec(path=path, year=year))
    return specs


def extract_filename_date(path: Path) -> Optional[datetime]:
    matches = DATE_RE.findall(path.name)
    for text in reversed(matches):
        try:
            return datetime.strptime(text, "%Y%m%d")
        except ValueError:
            continue
    return None


def scan_raw_files(
    roots: Sequence[str], name_contains: str
) -> Tuple[Dict[str, Path], List[str], List[str]]:
    candidates: List[Path] = []
    warnings: List[str] = []
    for root_text in roots:
        root = Path(root_text)
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in RAW_SUFFIXES
            )
        else:
            warnings.append(f"Raw root does not exist and was skipped: {root}")

    candidates = sorted(set(candidates), key=lambda p: str(p).lower())
    if name_contains:
        candidates = [p for p in candidates if name_contains in p.name]

    by_date_all: MutableMapping[str, List[Path]] = defaultdict(list)
    undated: List[str] = []
    for path in candidates:
        date = extract_filename_date(path)
        if date is None:
            undated.append(str(path))
            continue
        by_date_all[date.strftime("%Y-%m-%d")].append(path)

    chosen: Dict[str, Path] = {}
    for key, paths in sorted(by_date_all.items()):
        # Prefer the canonical aerosol diagnostic collection if duplicates occur.
        ranked = sorted(
            paths,
            key=lambda p: (
                "tavg1_2d_aer_Nx" not in p.name,
                len(str(p)),
                str(p).lower(),
            ),
        )
        chosen[key] = ranked[0]
        if len(ranked) > 1:
            warnings.append(
                f"Multiple raw files found for {key}; using {ranked[0]} and ignoring "
                + ", ".join(str(p) for p in ranked[1:])
            )
    return chosen, warnings, undated


def find_variable(ds: Dataset, aliases: Sequence[str], label: str):
    exact = {name: name for name in ds.variables}
    lower = {name.lower(): name for name in ds.variables}
    for alias in aliases:
        if alias in exact:
            return ds.variables[exact[alias]]
        if alias.lower() in lower:
            return ds.variables[lower[alias.lower()]]
    raise KeyError(
        f"Could not find {label}. Tried {list(aliases)}; available variables include "
        f"{list(ds.variables)[:80]}"
    )


def as_float_array(value) -> np.ndarray:
    arr = np.ma.asarray(value)
    return np.asarray(np.ma.filled(arr, np.nan), dtype=np.float64)


def as_float_scalar(value) -> float:
    arr = as_float_array(value).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def doy_to_date_key(year: int, day: int) -> Optional[str]:
    max_day = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    if day < 1 or day > max_day:
        return None
    return (datetime(year, 1, 1) + timedelta(days=day - 1)).strftime("%Y-%m-%d")


def keep_lowest_priority(
    current: List[RowCandidate], incoming: List[RowCandidate], capacity: int
) -> List[RowCandidate]:
    if not incoming:
        return current
    combined = current + incoming
    combined.sort(key=lambda item: item.priority)
    return combined[:capacity]


def scan_fused_candidates(
    specs: Sequence[FusedSpec],
    raw_by_date: Mapping[str, Path],
    samples: int,
    max_dates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[
    Dict[str, List[RowCandidate]],
    Dict[str, int],
    Dict[str, List[str]],
    List[dict],
]:
    rng = np.random.default_rng(seed)
    capacity_per_date = max(samples, int(math.ceil(samples / max_dates)) * 4)
    reservoirs: Dict[str, List[RowCandidate]] = defaultdict(list)
    row_counts: Dict[str, int] = defaultdict(int)
    present_dates_by_file: Dict[str, List[str]] = {}
    file_scan_info: List[dict] = []

    for fused_index, spec in enumerate(specs):
        print(f"Scanning fused DOY read-only: {spec.path}", flush=True)
        present_days = set()
        overlap_days = set()
        with Dataset(spec.path, mode="r") as ds:
            doy_var = find_variable(ds, FUSED_NAMES["doy"], "fused DOY")
            if doy_var.ndim != 1:
                raise ValueError(f"Fused DOY must be 1-D in {spec.path}; got {doy_var.shape}")
            n_rows = int(doy_var.shape[0])
            for start in range(0, n_rows, chunk_size):
                stop = min(n_rows, start + chunk_size)
                doy = as_float_array(doy_var[start:stop])
                finite = np.isfinite(doy)
                integer_day = np.full(doy.shape, -1, dtype=np.int32)
                # DOY is 1-based; its fractional component is the within-day time.
                integer_day[finite] = np.floor(doy[finite] + 1.0e-7).astype(np.int32)
                unique_days = np.unique(integer_day[integer_day > 0])
                present_days.update(int(day) for day in unique_days)

                for day_value in unique_days:
                    day = int(day_value)
                    date_key = doy_to_date_key(spec.year, day)
                    if date_key is None or date_key not in raw_by_date:
                        continue
                    overlap_days.add(day)
                    local = np.flatnonzero(integer_day == day)
                    if local.size == 0:
                        continue
                    row_counts[date_key] += int(local.size)
                    priorities = rng.random(local.size)
                    if local.size > capacity_per_date:
                        selected_local = np.argpartition(
                            priorities, capacity_per_date - 1
                        )[:capacity_per_date]
                    else:
                        selected_local = np.arange(local.size)
                    incoming = [
                        RowCandidate(
                            priority=float(priorities[pos]),
                            fused_index=fused_index,
                            row_index=int(start + local[pos]),
                            date_key=date_key,
                        )
                        for pos in selected_local
                    ]
                    reservoirs[date_key] = keep_lowest_priority(
                        reservoirs[date_key], incoming, capacity_per_date
                    )

        present_dates = [
            key
            for day in sorted(present_days)
            if (key := doy_to_date_key(spec.year, day)) is not None
        ]
        overlap_dates = [
            key
            for day in sorted(overlap_days)
            if (key := doy_to_date_key(spec.year, day)) is not None
        ]
        present_dates_by_file[str(spec.path)] = present_dates
        file_scan_info.append(
            {
                "path": str(spec.path),
                "year": spec.year,
                "rows": n_rows,
                "present_date_count": len(present_dates),
                "present_dates": present_dates,
                "overlap_date_count": len(overlap_dates),
                "overlap_dates": overlap_dates,
            }
        )

    return dict(reservoirs), dict(row_counts), present_dates_by_file, file_scan_info


def select_spread_dates(keys: Sequence[str], max_dates: int) -> List[str]:
    ordered = sorted(keys)
    if len(ordered) <= max_dates:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, max_dates)
    indices: List[int] = []
    for value in positions:
        index = int(round(float(value)))
        if index not in indices:
            indices.append(index)
    # Rounding can theoretically collapse positions; fill deterministically.
    for index in range(len(ordered)):
        if len(indices) >= max_dates:
            break
        if index not in indices:
            indices.append(index)
    return [ordered[index] for index in sorted(indices[:max_dates])]


def choose_rows(
    reservoirs: Mapping[str, List[RowCandidate]], samples: int, max_dates: int
) -> Tuple[List[RowCandidate], List[str]]:
    selected_dates = select_spread_dates(list(reservoirs), max_dates)
    if not selected_dates:
        return [], []
    base, extra = divmod(samples, len(selected_dates))
    chosen: List[RowCandidate] = []
    unused: List[RowCandidate] = []
    for index, key in enumerate(selected_dates):
        need = base + (1 if index < extra else 0)
        ordered = sorted(reservoirs[key], key=lambda item: item.priority)
        chosen.extend(ordered[:need])
        unused.extend(ordered[need:])
    if len(chosen) < samples:
        unused.sort(key=lambda item: item.priority)
        chosen.extend(unused[: samples - len(chosen)])
    chosen.sort(key=lambda item: (item.date_key, item.fused_index, item.row_index))
    return chosen[:samples], selected_dates


def read_fused_rows(
    specs: Sequence[FusedSpec], candidates: Sequence[RowCandidate]
) -> List[dict]:
    by_file: MutableMapping[int, List[RowCandidate]] = defaultdict(list)
    for item in candidates:
        by_file[item.fused_index].append(item)

    rows: List[dict] = []
    for fused_index, items in sorted(by_file.items()):
        spec = specs[fused_index]
        print(f"Reading {len(items)} sampled fused rows: {spec.path}", flush=True)
        with Dataset(spec.path, mode="r") as ds:
            variables = {
                key: find_variable(ds, aliases, f"fused {key}")
                for key, aliases in FUSED_NAMES.items()
            }
            for item in items:
                row_index = item.row_index
                doy = as_float_scalar(variables["doy"][row_index])
                day = int(math.floor(doy + 1.0e-7))
                rows.append(
                    {
                        "fused_index": fused_index,
                        "fused_file": str(spec.path),
                        "fused_year": spec.year,
                        "row_index": row_index,
                        "date": item.date_key,
                        "doy": doy,
                        "day_integer": day,
                        "hour_from_doy": (doy - day) * 24.0,
                        "latitude": as_float_scalar(variables["lat"][row_index]),
                        "longitude": as_float_scalar(variables["lon"][row_index]),
                        "fused_bccmass": as_float_scalar(variables["bcc"][row_index]),
                    }
                )
    rows.sort(key=lambda row: (row["date"], row["fused_file"], row["row_index"]))
    for sample_id, row in enumerate(rows, start=1):
        row["sample_id"] = sample_id
    return rows


def raw_hours_from_time(time_var, raw_date: datetime) -> np.ndarray:
    units = getattr(time_var, "units", None)
    if not units:
        raise ValueError("Raw MERRA-2 time variable has no 'units' attribute")
    calendar = getattr(time_var, "calendar", "standard")
    decoded = num2date(
        time_var[:],
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    base = datetime(raw_date.year, raw_date.month, raw_date.day)
    hours = []
    for item in np.asarray(decoded).reshape(-1):
        current = datetime(
            int(item.year),
            int(item.month),
            int(item.day),
            int(getattr(item, "hour", 0)),
            int(getattr(item, "minute", 0)),
            int(getattr(item, "second", 0)),
            int(getattr(item, "microsecond", 0)),
        )
        hours.append((current - base).total_seconds() / 3600.0)
    return np.asarray(hours, dtype=np.float64)


def load_raw_day(path: Path, date_key: str) -> dict:
    raw_date = datetime.strptime(date_key, "%Y-%m-%d")
    with Dataset(path, mode="r") as ds:
        time_var = find_variable(ds, RAW_NAMES["time"], "raw time")
        lat_var = find_variable(ds, RAW_NAMES["lat"], "raw latitude")
        lon_var = find_variable(ds, RAW_NAMES["lon"], "raw longitude")
        bcc_var = find_variable(ds, RAW_NAMES["bcc"], "raw BCCMASS")

        hours = raw_hours_from_time(time_var, raw_date)
        lats = as_float_array(lat_var[:]).reshape(-1)
        lons = as_float_array(lon_var[:]).reshape(-1)
        bcc = as_float_array(bcc_var[:])

        time_dim = time_var.dimensions[0]
        lat_dim = lat_var.dimensions[0]
        lon_dim = lon_var.dimensions[0]
        dims = list(bcc_var.dimensions)
        missing_dims = [dim for dim in (time_dim, lat_dim, lon_dim) if dim not in dims]
        if missing_dims:
            raise ValueError(
                f"Raw BCCMASS dimensions {dims} do not contain expected dimensions {missing_dims}"
            )
        if bcc.ndim != 3:
            raise ValueError(
                f"Expected 3-D raw BCCMASS (time, lat, lon); got shape {bcc.shape}"
            )
        axes = (dims.index(time_dim), dims.index(lat_dim), dims.index(lon_dim))
        bcc = np.transpose(bcc, axes=axes)

        order = np.argsort(hours)
        hours = hours[order]
        bcc = bcc[order, :, :]
        return {
            "path": str(path),
            "hours": hours,
            "lats": lats,
            "lons": lons,
            "bcc": bcc,
            "bcc_units": getattr(bcc_var, "units", None),
            "bcc_long_name": getattr(bcc_var, "long_name", None),
            "time_units": getattr(time_var, "units", None),
            "time_calendar": getattr(time_var, "calendar", "standard"),
            "bcc_dimensions_original": dims,
            "bcc_shape_original": list(bcc_var.shape),
            "lat_ascending": bool(lats[-1] > lats[0]) if lats.size > 1 else None,
            "lon_ascending": bool(lons[-1] > lons[0]) if lons.size > 1 else None,
        }


def nearest_lat_index(lats: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(lats - target)))


def nearest_lon_index(lons: np.ndarray, target: float) -> int:
    difference = np.abs(((lons - target + 180.0) % 360.0) - 180.0)
    return int(np.nanargmin(difference))


def circular_lon_delta(value: float, reference: float) -> float:
    return float(((value - reference + 180.0) % 360.0) - 180.0)


def nearest_time_value(hours: np.ndarray, values: np.ndarray, hour: float) -> float:
    index = int(np.nanargmin(np.abs(hours - hour)))
    return float(values[index])


def lower_time_value(hours: np.ndarray, values: np.ndarray, hour: float) -> float:
    eligible = np.flatnonzero(hours <= hour)
    index = int(eligible[-1]) if eligible.size else 0
    return float(values[index])


def upper_time_value(hours: np.ndarray, values: np.ndarray, hour: float) -> float:
    eligible = np.flatnonzero(hours >= hour)
    index = int(eligible[0]) if eligible.size else len(hours) - 1
    return float(values[index])


def linear_time_value(hours: np.ndarray, values: np.ndarray, hour: float) -> float:
    if hour <= hours[0]:
        return float(values[0])
    if hour >= hours[-1]:
        return float(values[-1])
    upper = int(np.searchsorted(hours, hour, side="right"))
    lower = upper - 1
    span = float(hours[upper] - hours[lower])
    if span == 0.0:
        return float(values[lower])
    weight_upper = float((hour - hours[lower]) / span)
    return float(values[lower] * (1.0 - weight_upper) + values[upper] * weight_upper)


def supplied_2024_code_value(
    hours: np.ndarray, values: np.ndarray, hour: float
) -> float:
    """Reproduce the literal MERRA-2 weighting in the supplied 2024 script."""
    if hour > hours[-1]:
        return float(values[-1])
    if hour < hours[0]:
        return float(values[0])
    lower_candidates = np.flatnonzero((hour - hours) > 0)
    upper_candidates = np.flatnonzero((hour - hours) < 0)
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        return nearest_time_value(hours, values, hour)
    lower = int(lower_candidates[-1])
    upper = int(upper_candidates[0])
    fraction = hour - math.floor(hour)
    return float(values[lower] * fraction + values[upper] * (1.0 - fraction))


def attach_raw_candidates(
    rows: List[dict], raw_by_date: Mapping[str, Path]
) -> Tuple[List[dict], List[dict]]:
    by_date: MutableMapping[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)

    raw_metadata: List[dict] = []
    for date_key, date_rows in sorted(by_date.items()):
        raw_path = raw_by_date[date_key]
        print(f"Comparing {len(date_rows)} rows with raw day: {raw_path}", flush=True)
        raw = load_raw_day(raw_path, date_key)
        metadata_item = {
            key: value
            for key, value in raw.items()
            if key not in {"bcc", "hours", "lats", "lons"}
        }
        metadata_item.update(
            {
                "date": date_key,
                "time_hours": raw["hours"].tolist(),
                "lat_min": float(np.nanmin(raw["lats"])),
                "lat_max": float(np.nanmax(raw["lats"])),
                "lat_count": int(raw["lats"].size),
                "lon_min": float(np.nanmin(raw["lons"])),
                "lon_max": float(np.nanmax(raw["lons"])),
                "lon_count": int(raw["lons"].size),
            }
        )
        raw_metadata.append(metadata_item)

        lat_spacing = float(np.median(np.abs(np.diff(np.sort(np.unique(raw["lats"]))))))
        lon_spacing = float(np.median(np.abs(np.diff(np.sort(np.unique(raw["lons"]))))))
        metadata_item["median_lat_spacing_deg"] = lat_spacing
        metadata_item["median_lon_spacing_deg"] = lon_spacing

        for row in date_rows:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            hour = float(row["hour_from_doy"])
            aligned_lat_index = nearest_lat_index(raw["lats"], lat)
            mirrored_lat_index = nearest_lat_index(raw["lats"], -lat)
            lon_index = nearest_lon_index(raw["lons"], lon)
            aligned_series = raw["bcc"][:, aligned_lat_index, lon_index]
            mirrored_series = raw["bcc"][:, mirrored_lat_index, lon_index]

            row.update(
                {
                    "raw_file": str(raw_path),
                    "raw_lat_aligned": float(raw["lats"][aligned_lat_index]),
                    "raw_lat_mirrored": float(raw["lats"][mirrored_lat_index]),
                    "raw_lon": float(raw["lons"][lon_index]),
                    "aligned_lat_delta_deg": float(
                        raw["lats"][aligned_lat_index] - lat
                    ),
                    "mirrored_lat_delta_from_negated_deg": float(
                        raw["lats"][mirrored_lat_index] + lat
                    ),
                    "lon_delta_deg": circular_lon_delta(
                        float(raw["lons"][lon_index]), lon
                    ),
                    "aligned_linear_actual": linear_time_value(
                        raw["hours"], aligned_series, hour
                    ),
                    "aligned_nearest_actual": nearest_time_value(
                        raw["hours"], aligned_series, hour
                    ),
                    "aligned_lower_actual": lower_time_value(
                        raw["hours"], aligned_series, hour
                    ),
                    "aligned_upper_actual": upper_time_value(
                        raw["hours"], aligned_series, hour
                    ),
                    "aligned_supplied_2024_code": supplied_2024_code_value(
                        raw["hours"], aligned_series, hour
                    ),
                    "mirrored_linear_actual": linear_time_value(
                        raw["hours"], mirrored_series, hour
                    ),
                    "mirrored_nearest_actual": nearest_time_value(
                        raw["hours"], mirrored_series, hour
                    ),
                    "mirrored_supplied_2024_code": supplied_2024_code_value(
                        raw["hours"], mirrored_series, hour
                    ),
                }
            )
            # Test every fixed one-cell coordinate offset around the stored
            # coordinate. These candidates detect a consistent off-by-one
            # spatial index without loading the complete fused variables.
            for delta_lat in (-1, 0, 1):
                for delta_lon in (-1, 0, 1):
                    if delta_lat == 0 and delta_lon == 0:
                        continue
                    candidate_lat = lat + delta_lat * lat_spacing
                    candidate_lon = lon + delta_lon * lon_spacing
                    neighbor_lat_index = nearest_lat_index(raw["lats"], candidate_lat)
                    neighbor_lon_index = nearest_lon_index(raw["lons"], candidate_lon)
                    neighbor_series = raw["bcc"][
                        :, neighbor_lat_index, neighbor_lon_index
                    ]
                    row[
                        neighbor_candidate_name(
                            delta_lat, delta_lon, "linear_actual"
                        )
                    ] = linear_time_value(raw["hours"], neighbor_series, hour)
                    row[
                        neighbor_candidate_name(
                            delta_lat, delta_lon, "nearest_actual"
                        )
                    ] = nearest_time_value(raw["hours"], neighbor_series, hour)
                    row[
                        neighbor_candidate_name(
                            delta_lat, delta_lon, "supplied_2024_code"
                        )
                    ] = supplied_2024_code_value(
                        raw["hours"], neighbor_series, hour
                    )
    return rows, raw_metadata


def metric_block(y: np.ndarray, prediction: np.ndarray) -> dict:
    valid = np.isfinite(y) & np.isfinite(prediction)
    yv = y[valid]
    pv = prediction[valid]
    if yv.size == 0:
        return {"n": 0, "status": "NO_FINITE_PAIRS"}
    error = pv - yv
    abs_error = np.abs(error)
    rms_y = float(np.sqrt(np.mean(yv**2)))
    scale_floor = max(float(np.median(np.abs(yv))) * 1.0e-6, 1.0e-30)
    denominator = np.maximum(np.maximum(np.abs(yv), np.abs(pv)), scale_floor)
    relative = abs_error / denominator
    sse = float(np.sum(error**2))
    centered = yv - float(np.mean(yv))
    sst = float(np.sum(centered**2))
    corr = (
        float(np.corrcoef(yv, pv)[0, 1])
        if yv.size >= 3 and np.std(yv) > 0 and np.std(pv) > 0
        else None
    )
    nonzero_prediction = np.abs(pv) > scale_floor
    median_ratio = (
        float(np.median(yv[nonzero_prediction] / pv[nonzero_prediction]))
        if np.any(nonzero_prediction)
        else None
    )
    through_origin_scale = (
        float(np.sum(yv * pv) / np.sum(pv**2)) if np.sum(pv**2) > 0 else None
    )
    return {
        "n": int(yv.size),
        "status": "OK",
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse_by_rms_fused": (
            float(np.sqrt(np.mean(error**2)) / rms_y) if rms_y > 0 else None
        ),
        "bias_prediction_minus_fused": float(np.mean(error)),
        "median_absolute_error": float(np.median(abs_error)),
        "p95_absolute_error": float(np.quantile(abs_error, 0.95)),
        "median_symmetric_relative_error": float(np.median(relative)),
        "p95_symmetric_relative_error": float(np.quantile(relative, 0.95)),
        "max_symmetric_relative_error": float(np.max(relative)),
        "correlation": corr,
        "r2_against_identity": float(1.0 - sse / sst) if sst > 0 else None,
        "median_fused_to_candidate_ratio": median_ratio,
        "through_origin_scale_candidate_to_fused": through_origin_scale,
        "exactish_count_rtol_1e-5": int(
            np.count_nonzero(np.isclose(yv, pv, rtol=1.0e-5, atol=1.0e-14))
        ),
    }


def analyse_candidates(rows: Sequence[dict]) -> dict:
    candidate_names = all_candidate_names()
    y = np.asarray([row["fused_bccmass"] for row in rows], dtype=np.float64)
    metrics = {
        name: metric_block(
            y, np.asarray([row[name] for row in rows], dtype=np.float64)
        )
        for name in candidate_names
    }
    rankable = [
        (name, block.get("normalized_rmse_by_rms_fused"))
        for name, block in metrics.items()
        if block.get("normalized_rmse_by_rms_fused") is not None
    ]
    rankable.sort(key=lambda item: item[1])
    ranking = [{"candidate": name, "score": score} for name, score in rankable]
    best_name = rankable[0][0] if rankable else None
    best_score = rankable[0][1] if rankable else None

    aligned_rank = [item for item in rankable if item[0].startswith("aligned_")]
    mirrored_rank = [item for item in rankable if item[0].startswith("mirrored_")]
    neighbor_rank = [item for item in rankable if item[0].startswith("neighbor_")]
    best_aligned = aligned_rank[0] if aligned_rank else (None, None)
    best_mirrored = mirrored_rank[0] if mirrored_rank else (None, None)
    best_neighbor = neighbor_rank[0] if neighbor_rank else (None, None)

    spatial_scores = [
        score
        for score in (best_aligned[1], best_mirrored[1], best_neighbor[1])
        if score is not None
    ]
    if len(spatial_scores) < 3:
        spatial_verdict = "INCONCLUSIVE"
    elif best_aligned[1] < 0.5 * min(best_mirrored[1], best_neighbor[1]):
        spatial_verdict = "ALIGNED_LATITUDE_STRONGLY_PREFERRED"
    elif best_mirrored[1] < 0.5 * min(best_aligned[1], best_neighbor[1]):
        spatial_verdict = "MIRRORED_LATITUDE_STRONGLY_PREFERRED_REVIEW_REQUIRED"
    elif best_neighbor[1] < 0.5 * min(best_aligned[1], best_mirrored[1]):
        spatial_verdict = "FIXED_ONE_CELL_OFFSET_STRONGLY_PREFERRED_REVIEW_REQUIRED"
    else:
        spatial_verdict = "SPATIAL_INDEX_HYPOTHESES_NOT_SEPARATED"

    time_rank = aligned_rank
    if not time_rank:
        time_verdict = "INCONCLUSIVE"
    elif len(time_rank) == 1:
        time_verdict = f"ONLY_CANDIDATE_{time_rank[0][0]}"
    else:
        first, second = time_rank[0], time_rank[1]
        if first[1] == 0.0:
            clear = second[1] > 1.0e-8
        else:
            clear = second[1] / first[1] >= 1.25
        time_verdict = (
            f"{first[0]}_CLEARLY_PREFERRED"
            if clear
            else f"AMBIGUOUS_BEST_{first[0]}_SECOND_{second[0]}"
        )

    if best_name is None:
        status = "INCONCLUSIVE_NO_FINITE_COMPARISONS"
    elif best_name.startswith("mirrored_") and spatial_verdict.startswith("MIRRORED"):
        status = "REVIEW_POSSIBLE_LATITUDE_MIRROR"
    elif best_name.startswith("neighbor_") and spatial_verdict.startswith("FIXED"):
        status = "REVIEW_POSSIBLE_FIXED_ONE_CELL_OFFSET"
    else:
        best_block = metrics[best_name]
        median_rel = best_block["median_symmetric_relative_error"]
        p95_rel = best_block["p95_symmetric_relative_error"]
        if median_rel <= 1.0e-4 and p95_rel <= 1.0e-3:
            status = "PASS_NEAR_EXACT_FINGERPRINT"
        elif median_rel <= 0.01 and p95_rel <= 0.05:
            status = "PASS_STRONG_APPROXIMATE_FINGERPRINT"
        elif median_rel <= 0.05 and p95_rel <= 0.20:
            status = "PASS_BROADLY_CONSISTENT_FINGERPRINT"
        else:
            status = "INCONCLUSIVE_NO_CLOSE_HYPOTHESIS"

    coordinate_summary = {
        "max_abs_aligned_lat_delta_deg": float(
            np.max(np.abs([row["aligned_lat_delta_deg"] for row in rows]))
        ),
        "max_abs_lon_delta_deg": float(
            np.max(np.abs([row["lon_delta_deg"] for row in rows]))
        ),
    }
    return {
        "status": status,
        "best_candidate": best_name,
        "best_score_normalized_rmse": best_score,
        "spatial_orientation_verdict": spatial_verdict,
        "best_aligned_candidate": best_aligned[0],
        "best_aligned_score": best_aligned[1],
        "best_mirrored_candidate": best_mirrored[0],
        "best_mirrored_score": best_mirrored[1],
        "best_neighbor_candidate": best_neighbor[0],
        "best_neighbor_score": best_neighbor[1],
        "time_matching_verdict": time_verdict,
        "coordinate_summary": coordinate_summary,
        "ranking": ranking,
        "metrics": metrics,
        "interpretation_limits": [
            "A close candidate supports, but does not prove, the complete historical pipeline.",
            "Similar candidate scores mean that slowly varying BCCMASS cannot separate the time methods.",
            "A weak match may reflect an unpreserved orbit-mean time, another historical script, or another raw-file revision.",
            "No manuscript claim or formal experiment should be changed from this gate alone unless a material mismatch is reproduced and explained.",
        ],
    }


def serializable_row(row: Mapping[str, object]) -> dict:
    result = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    candidate_names = all_candidate_names()
    csv_rows: List[dict] = []
    for row in rows:
        item = serializable_row(row)
        fused = float(row["fused_bccmass"])
        for name in candidate_names:
            prediction = float(row[name])
            item[f"{name}_error"] = prediction - fused
        csv_rows.append(item)
    fieldnames: List[str] = []
    for row in csv_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def build_summary(report: dict) -> str:
    lines = [
        "MERRA-2 BCCMASS FINGERPRINT AUDIT",
        f"Script version: {report['script_version']}",
        f"Generated UTC: {report['generated_utc']}",
        "Read only: true",
        f"Requested samples: {report['sampling']['requested_samples']}",
        f"Compared samples: {report['sampling']['compared_samples']}",
        f"Selected dates: {', '.join(report['sampling']['selected_dates'])}",
        "",
    ]
    analysis = report.get("analysis")
    if not analysis:
        lines.extend(
            [
                f"Status: {report['status']}",
                "No raw/fused date overlap was available for comparison.",
                "Raw dates found: " + ", ".join(report["raw_inventory"]["dates"]),
                "Review fused_present_dates_by_file in the JSON report and supply at least one matching raw daily file.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"Status: {analysis['status']}",
            f"Best candidate: {analysis['best_candidate']}",
            f"Best normalized RMSE: {analysis['best_score_normalized_rmse']}",
            f"Spatial verdict: {analysis['spatial_orientation_verdict']}",
            f"Time verdict: {analysis['time_matching_verdict']}",
            "",
            "Candidate ranking (lower normalized RMSE is better):",
        ]
    )
    displayed_ranking = analysis["ranking"][:15]
    for item in displayed_ranking:
        block = analysis["metrics"][item["candidate"]]
        lines.append(
            f"  {item['candidate']}: score={item['score']:.8g}, "
            f"median_rel={block['median_symmetric_relative_error']:.8g}, "
            f"p95_rel={block['p95_symmetric_relative_error']:.8g}, "
            f"scale={block['through_origin_scale_candidate_to_fused']}"
        )
    if len(analysis["ranking"]) > len(displayed_ranking):
        lines.append(
            f"  ... {len(analysis['ranking']) - len(displayed_ranking)} additional candidates are recorded in the JSON report."
        )
    lines.extend(
        [
            "",
            "Candidate definitions:",
            "  aligned_linear_actual: correct linear interpolation using decoded raw times at the aligned cell.",
            "  aligned_nearest_actual: nearest decoded raw time at the aligned cell.",
            "  aligned_lower_actual / aligned_upper_actual: step selection at the aligned cell.",
            "  aligned_supplied_2024_code: literal weighting in the supplied 2024 matching script.",
            "  mirrored_*: the same time hypotheses after north/south latitude mirroring.",
            "  neighbor_*: fixed one-native-cell coordinate offsets around the stored latitude/longitude.",
            "",
            "Limits:",
        ]
    )
    lines.extend(f"  - {line}" for line in analysis["interpretation_limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "merra2_bccmass_fingerprint_report.json"
    txt_path = output_dir / "merra2_bccmass_fingerprint_summary.txt"
    csv_path = output_dir / "merra2_bccmass_fingerprint_samples.csv"

    specs = resolve_specs(args.fused)
    raw_by_date, raw_warnings, undated_raw = scan_raw_files(
        args.raw_root, args.raw_name_contains
    )
    if not raw_by_date:
        raise RuntimeError(
            "No dated raw MERRA-2 files were found. Check --raw-root and "
            "--raw-name-contains. Expected names similar to "
            "MERRA2_400.tavg1_2d_aer_Nx.20230109.nc4."
        )

    print(f"Dated raw MERRA-2 files found: {len(raw_by_date)}", flush=True)
    reservoirs, row_counts, present_by_file, file_scan_info = scan_fused_candidates(
        specs=specs,
        raw_by_date=raw_by_date,
        samples=args.samples,
        max_dates=args.max_dates,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    candidates, selected_dates = choose_rows(
        reservoirs=reservoirs,
        samples=args.samples,
        max_dates=args.max_dates,
    )

    base_report = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now_iso(),
        "read_only": True,
        "command": " ".join(sys.argv),
        "raw_inventory": {
            "roots": args.raw_root,
            "name_contains": args.raw_name_contains,
            "dated_file_count": len(raw_by_date),
            "dates": sorted(raw_by_date),
            "chosen_files": {key: str(value) for key, value in raw_by_date.items()},
            "warnings": raw_warnings,
            "undated_files": undated_raw,
        },
        "fused_inventory": file_scan_info,
        "fused_present_dates_by_file": present_by_file,
        "overlap_row_counts_by_date": row_counts,
        "sampling": {
            "strategy": "temporally_spread_dates_plus_seeded_uniform_row_priorities",
            "seed": args.seed,
            "requested_samples": args.samples,
            "selected_dates": selected_dates,
            "compared_samples": 0,
        },
    }

    if not candidates:
        report = dict(base_report)
        report.update(
            {
                "status": "NO_RAW_FUSED_DATE_OVERLAP",
                "analysis": None,
            }
        )
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_path.write_text(build_summary(report), encoding="utf-8")
        print(f"JSON report: {json_path}")
        print(f"Text report: {txt_path}")
        print("Status: NO_RAW_FUSED_DATE_OVERLAP")
        return 2

    rows = read_fused_rows(specs, candidates)
    rows, raw_metadata = attach_raw_candidates(rows, raw_by_date)
    analysis = analyse_candidates(rows)
    base_report["sampling"]["compared_samples"] = len(rows)
    report = dict(base_report)
    report.update(
        {
            "status": analysis["status"],
            "raw_file_metadata_for_selected_dates": raw_metadata,
            "analysis": analysis,
        }
    )

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(build_summary(report), encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")
    print(f"CSV samples: {csv_path}")
    print(f"Status: {analysis['status']}")
    print("MERRA2_BCCMASS_FINGERPRINT_COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
