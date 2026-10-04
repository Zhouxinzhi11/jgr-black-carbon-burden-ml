# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
36_compute_train_only_meteo_stats.py

Gate 7: read-only computation and audit of meteorological normalization
statistics using only the 2022 mean-state training months.

Training months: 02, 03, 05, 06, 08, 09
Validation months: 11, 12
Blacklisted month: 07

The script:
1. Streams the original 2022 fused NetCDF.
2. Recomputes full-2022 statistics for the five meteorological variables.
3. Compares them with the existing normalization_stats.json.
4. Computes train-month-only statistics.
5. Writes a new versioned JSON file without overwriting the old statistics.

No tensor, cache, model, or source NetCDF is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


METEO_VARS = [
    "Temp",
    "RH",
    "BLH",
    "Windspeed_U",
    "Windspeed_V",
]

# Positions of the five meteorological variables in FEATURE_VARS
OLD_STATS_INDICES = [43, 44, 45, 46, 47]

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


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
        if std < 1e-8:
            std = 1.0
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std_population": float(std),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


def safe_rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-month-only meteorological normalization statistics "
            "and reproduce the existing full-2022 statistics."
        )
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--old-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-months",
        default="2,3,5,6,8,9",
    )
    parser.add_argument(
        "--val-months",
        default="11,12",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--reproduction-atol",
        type=float,
        default=1e-5,
        help="Maximum absolute difference allowed when reproducing old means/stds.",
    )
    parser.add_argument(
        "--reproduction-rtol",
        type=float,
        default=1e-7,
        help="Maximum relative difference allowed when reproducing old means/stds.",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    old_stats_path = Path(args.old_stats).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")
    if not old_stats_path.is_file():
        raise FileNotFoundError(f"Old statistics file not found: {old_stats_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    train_months = sorted(set(parse_int_list(args.train_months)))
    val_months = sorted(set(parse_int_list(args.val_months)))
    overlap = sorted(set(train_months) & set(val_months))
    if overlap:
        raise ValueError(f"Train and validation months overlap: {overlap}")

    with old_stats_path.open("r", encoding="utf-8") as f:
        old_stats = json.load(f)

    old_mean_all = old_stats.get("mean")
    old_std_all = old_stats.get("std")
    if not isinstance(old_mean_all, list) or not isinstance(old_std_all, list):
        raise KeyError("Old normalization_stats.json lacks mean/std lists")
    if len(old_mean_all) <= max(OLD_STATS_INDICES):
        raise ValueError("Old mean list is shorter than expected")
    if len(old_std_all) <= max(OLD_STATS_INDICES):
        raise ValueError("Old std list is shorter than expected")

    old_meteo = {}
    for var, index in zip(METEO_VARS, OLD_STATS_INDICES):
        old_meteo[var] = {
            "feature_index": index,
            "mean": float(old_mean_all[index]),
            "std_population": float(old_std_all[index]),
        }

    lookup = doy_to_month_lookup()
    train_set = set(train_months)
    val_set = set(val_months)

    full_stats = {var: RunningStats() for var in METEO_VARS}
    train_stats = {var: RunningStats() for var in METEO_VARS}
    val_stats = {var: RunningStats() for var in METEO_VARS}

    row_counts_by_month = {str(month): 0 for month in range(1, 13)}
    processed_rows = 0

    with xr.open_dataset(
        nc_path,
        decode_times=False,
        mask_and_scale=False,
    ) as ds:
        required = ["DOY", *METEO_VARS]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables: {missing}")
        if "num_samples" not in ds.sizes:
            raise KeyError("Dimension 'num_samples' not found")

        n_samples = int(ds.sizes["num_samples"])

        for start in range(0, n_samples, args.chunk_size):
            stop = min(start + args.chunk_size, n_samples)
            selector = {"num_samples": slice(start, stop)}

            doy = np.asarray(
                ds["DOY"].isel(selector).values,
                dtype=np.float64,
            ).reshape(-1)

            finite_doy = np.isfinite(doy)
            integer_doy = np.zeros(doy.shape, dtype=np.int16)
            integer_doy[finite_doy] = doy[finite_doy].astype(
                np.int16,
                copy=False,
            )

            valid_doy = (
                finite_doy
                & (integer_doy >= 1)
                & (integer_doy <= 366)
            )
            months = np.zeros(integer_doy.shape, dtype=np.int16)
            months[valid_doy] = lookup[integer_doy[valid_doy]]

            for month in range(1, 13):
                row_counts_by_month[str(month)] += int(
                    np.sum(valid_doy & (months == month))
                )

            train_mask = valid_doy & np.isin(months, train_months)
            val_mask = valid_doy & np.isin(months, val_months)

            for var in METEO_VARS:
                values = np.asarray(
                    ds[var].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)

                full_stats[var].update(values[valid_doy])
                train_stats[var].update(values[train_mask])
                val_stats[var].update(values[val_mask])

            processed_rows += stop - start
            print(
                f"PROGRESS={stop}/{n_samples}",
                flush=True,
            )

    full_results = {
        var: full_stats[var].result()
        for var in METEO_VARS
    }
    train_results = {
        var: train_stats[var].result()
        for var in METEO_VARS
    }
    val_results = {
        var: val_stats[var].result()
        for var in METEO_VARS
    }

    reproduction = {}
    all_reproduced = True

    for var in METEO_VARS:
        old_mean = old_meteo[var]["mean"]
        old_std = old_meteo[var]["std_population"]
        new_mean = float(full_results[var]["mean"])
        new_std = float(full_results[var]["std_population"])

        mean_abs = abs(new_mean - old_mean)
        std_abs = abs(new_std - old_std)
        mean_rel = safe_rel_diff(new_mean, old_mean)
        std_rel = safe_rel_diff(new_std, old_std)

        reproduced = (
            (mean_abs <= args.reproduction_atol or mean_rel <= args.reproduction_rtol)
            and
            (std_abs <= args.reproduction_atol or std_rel <= args.reproduction_rtol)
        )
        all_reproduced = all_reproduced and reproduced

        reproduction[var] = {
            "old_mean": old_mean,
            "recomputed_mean": new_mean,
            "mean_abs_diff": mean_abs,
            "mean_rel_diff": mean_rel,
            "old_std": old_std,
            "recomputed_std": new_std,
            "std_abs_diff": std_abs,
            "std_rel_diff": std_rel,
            "reproduced_within_tolerance": bool(reproduced),
        }

    train_stats_filename = (
        "meteo_normalization_stats_2022_"
        "train_months_02_03_05_06_08_09.json"
    )
    train_stats_path = output_dir / train_stats_filename

    train_stats_artifact = {
        "artifact_name": "train_only_meteorological_normalization_statistics",
        "version": 1,
        "source_nc": str(nc_path),
        "source_old_stats": str(old_stats_path),
        "year": 2022,
        "train_months": train_months,
        "validation_months": val_months,
        "excluded_or_unavailable_months": [
            month
            for month in range(1, 13)
            if month not in train_set and month not in val_set
        ],
        "variables": METEO_VARS,
        "feature_indices": dict(zip(METEO_VARS, OLD_STATS_INDICES)),
        "statistics_definition": {
            "sample_domain": (
                "Raw fused-NetCDF rows whose integer DOY falls within "
                "the specified 2022 training months"
            ),
            "nan_handling": "Non-finite meteorological values excluded per variable",
            "std_definition": "Population standard deviation: sqrt(M2 / count)",
        },
        "train_only_stats": train_results,
        "old_full_year_stats_reproduced": bool(all_reproduced),
    }

    report = {
        "audit_name": "train_only_meteo_normalization_gate7",
        "read_only": True,
        "inputs": {
            "nc": str(nc_path),
            "old_stats": str(old_stats_path),
        },
        "processed_rows": int(processed_rows),
        "train_months": train_months,
        "validation_months": val_months,
        "row_counts_by_month": row_counts_by_month,
        "old_meteorological_stats": old_meteo,
        "recomputed_full_2022_stats": full_results,
        "train_only_stats": train_results,
        "validation_month_stats_for_diagnosis_only": val_results,
        "old_stats_reproduction": reproduction,
        "old_full_year_stats_reproduced": bool(all_reproduced),
        "output_train_stats": str(train_stats_path),
        "interpretation": {
            "purpose": (
                "Remove preprocessing leakage by defining meteorological "
                "normalization using only the 2022 mean-state training months."
            ),
            "boundary": (
                "This gate computes statistics only. It does not rebuild tensors, "
                "caches, models, or manuscript result tables."
            ),
        },
    }

    report_path = output_dir / "train_only_meteo_normalization_gate7.json"
    txt_path = output_dir / "train_only_meteo_normalization_gate7.txt"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    train_stats_path.write_text(
        json.dumps(
            train_stats_artifact,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "Train-only meteorological normalization audit - Gate 7",
        f"NetCDF: {nc_path}",
        f"Old stats: {old_stats_path}",
        f"Processed rows: {processed_rows}",
        f"Train months: {train_months}",
        f"Validation months: {val_months}",
        f"Old full-year stats reproduced: {all_reproduced}",
        "",
        "Variables:",
    ]
    for var in METEO_VARS:
        item = train_results[var]
        rep = reproduction[var]
        lines.extend(
            [
                f"- {var}",
                f"  train_count={item['count']}",
                f"  train_mean={item['mean']}",
                f"  train_std={item['std_population']}",
                f"  old_mean={rep['old_mean']}",
                f"  old_std={rep['old_std']}",
                f"  reproduced={rep['reproduced_within_tolerance']}",
            ]
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if not all_reproduced:
        print("GATE7_OLD_STATS_REPRODUCTION_FAILED")
        print(f"JSON_REPORT={report_path}")
        print(f"TXT_REPORT={txt_path}")
        raise RuntimeError(
            "Recomputed full-2022 meteorological statistics do not reproduce "
            "the existing normalization_stats.json within tolerance. "
            "Do not use the train-only statistics until this discrepancy is resolved."
        )

    print("GATE7_TRAIN_ONLY_METEO_STATS_PASS")
    print(f"JSON_REPORT={report_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"TRAIN_STATS={train_stats_path}")
    print(f"PROCESSED_ROWS={processed_rows}")
    print(f"TRAIN_MONTHS={train_months}")
    print(f"VALIDATION_MONTHS={val_months}")
    print("OLD_FULL_YEAR_STATS_REPRODUCED=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
