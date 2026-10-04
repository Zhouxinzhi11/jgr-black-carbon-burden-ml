# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
40_audit_meteo_trainonly_direct_tensors_gate9b.py

Gate 9B:
Independently audit the Gate-8B directly rebuilt 2022 daily tensors against
one-shot per-day aggregation from the original fused NetCDF.

For one representative day from each core month
(02, 03, 05, 06, 08, 09, 11, 12), this script:

1. Reconstructs the five raw daily meteorological grids using one-shot
   np.bincount aggregation.
2. Applies the train-month-only normalization statistics.
3. Casts the expected result to the saved tensor dtype.
4. Compares it with the Gate-8B tensor over the complete 361 x 576 grid.
5. Verifies that every non-meteorological channel, target, mask, flag, and
   metadata field is exactly unchanged relative to the original tensor.

The Gate-8B construction used chunked float64 accumulation, whereas this
audit uses one-shot per-day accumulation. At most a very small number of
half-precision boundary differences are permitted.

No file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

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

FEATURE_INDICES = {
    "Temp": 43,
    "RH": 44,
    "BLH": 45,
    "Windspeed_U": 46,
    "Windspeed_V": 47,
}

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_months(text: str) -> list[int]:
    return sorted(
        set(
            int(part.strip())
            for part in text.split(",")
            if part.strip()
        )
    )


def build_month_lookup() -> np.ndarray:
    lookup = np.zeros(367, dtype=np.int16)
    first = 1
    for month, n_days in enumerate(MONTH_DAYS, start=1):
        lookup[first:first + n_days] = month
        first += n_days
    return lookup


def day_from_filename(path: Path) -> int:
    match = re.fullmatch(r"day_(\d{3})\.pt", path.name)
    if not match:
        raise ValueError(f"Unexpected tensor filename: {path.name}")
    return int(match.group(1))


def select_representative_days(
    tensor_dir: Path,
    months: list[int],
) -> dict[int, int]:
    lookup = build_month_lookup()
    grouped: dict[int, list[int]] = {month: [] for month in months}

    for path in sorted(tensor_dir.glob("day_*.pt")):
        day = day_from_filename(path)
        if 1 <= day <= 366:
            month = int(lookup[day])
            if month in grouped:
                grouped[month].append(day)

    selected: dict[int, int] = {}
    for month in months:
        days = sorted(grouped[month])
        if not days:
            raise RuntimeError(
                f"No daily tensor found for required month {month:02d}"
            )
        selected[month] = days[len(days) // 2]
    return selected


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


def verify_non_meteo_exact(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> None:
    if set(old_payload) != set(new_payload):
        raise AssertionError("Payload keys differ")

    if not torch.equal(old_payload["x"][:43], new_payload["x"][:43]):
        raise AssertionError("x[0:43] changed")
    if not torch.equal(old_payload["x"][48:], new_payload["x"][48:]):
        raise AssertionError("x[48:112] changed")

    for key in old_payload:
        if key == "x":
            continue
        if not values_equal(old_payload[key], new_payload[key]):
            raise AssertionError(f"Field changed: {key}")


def diff_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    diff = torch.abs(a - b)
    n = int(diff.numel())

    sorted_diff = torch.sort(diff).values
    q99_index = min(max(int(math.ceil(0.99 * n)) - 1, 0), n - 1)

    return {
        "count": n,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independent one-shot raw-NetCDF audit of Gate-8B "
            "train-only meteorological daily tensors."
        )
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--train-stats", required=True)
    parser.add_argument("--gate8b-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--months",
        default="2,3,5,6,8,9,11,12",
    )
    parser.add_argument(
        "--mean-abs-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--q99-abs-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-abs-threshold",
        type=float,
        default=0.001953125,
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    old_dir = Path(args.old_dir).expanduser().resolve()
    new_dir = Path(args.new_dir).expanduser().resolve()
    train_stats_path = Path(args.train_stats).expanduser().resolve()
    manifest_path = Path(args.gate8b_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [nc_path, train_stats_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in [old_dir, new_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    old_files = sorted(old_dir.glob("day_*.pt"))
    new_files = sorted(new_dir.glob("day_*.pt"))
    if len(old_files) != 240 or len(new_files) != 240:
        raise RuntimeError(
            f"Expected 240 old and 240 new tensors; "
            f"found old={len(old_files)}, new={len(new_files)}"
        )
    if [p.name for p in old_files] != [p.name for p in new_files]:
        raise RuntimeError("Old and new tensor filename sets differ")

    train_artifact = load_json(train_stats_path)
    manifest = load_json(manifest_path)

    if train_artifact.get("train_months") != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError("Unexpected train-month definition")
    if not train_artifact.get("old_full_year_stats_reproduced", False):
        raise RuntimeError("Gate-7 statistics are not approved")
    if manifest.get("source_file_count") != 240:
        raise RuntimeError("Gate-8B source_file_count is not 240")
    if manifest.get("output_file_count") != 240:
        raise RuntimeError("Gate-8B output_file_count is not 240")
    if manifest.get("processed_netcdf_rows") != 26075892:
        raise RuntimeError("Unexpected processed_netcdf_rows")
    if manifest.get("matched_netcdf_rows") != 26075892:
        raise RuntimeError("Not all NetCDF rows were matched")
    if manifest.get("invalid_coordinate_rows") != 0:
        raise RuntimeError("Gate-8B reported invalid coordinates")
    if manifest.get("changed_channels") != FEATURE_INDICES:
        raise RuntimeError("Gate-8B changed-channel declaration differs")

    train_stats = train_artifact["train_only_stats"]
    months = parse_months(args.months)
    selected = select_representative_days(old_dir, months)
    print(f"SELECTED_DAYS={selected}", flush=True)

    day_reports: list[dict[str, Any]] = []
    all_non_meteo_exact = True
    all_variable_metrics: dict[str, list[dict[str, float | int]]] = {
        var: [] for var in METEO_VARS
    }

    with xr.open_dataset(
        nc_path,
        decode_times=False,
        mask_and_scale=False,
    ) as ds:
        required = ["DOY", "Latitude", "Longitude", *METEO_VARS]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        doy_all = np.asarray(ds["DOY"].values, dtype=np.float64)
        integer_doy = doy_all.astype(np.int16, copy=False)

        for month in months:
            day = selected[month]
            filename = f"day_{day:03d}.pt"

            old_payload = load_pt(old_dir / filename)
            new_payload = load_pt(new_dir / filename)

            if int(old_payload["day"]) != day:
                raise RuntimeError(f"Old internal day mismatch: {filename}")
            if int(new_payload["day"]) != day:
                raise RuntimeError(f"New internal day mismatch: {filename}")

            try:
                verify_non_meteo_exact(old_payload, new_payload)
            except Exception:
                all_non_meteo_exact = False
                raise

            day_indices = np.flatnonzero(integer_doy == day)
            if day_indices.size == 0:
                raise RuntimeError(f"No raw rows for DOY {day}")

            lats = np.asarray(
                ds["Latitude"].values[day_indices],
                dtype=np.float64,
            )
            lons = np.asarray(
                ds["Longitude"].values[day_indices],
                dtype=np.float64,
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
                    f"Out-of-range grid index for DOY {day}"
                )

            flat_idx = lat_idx * W + lon_idx
            counts = np.bincount(
                flat_idx,
                minlength=H * W,
            ).astype(np.float64)
            safe_counts = counts.copy()
            safe_counts[safe_counts == 0] = 1.0

            variable_reports: dict[str, Any] = {}

            for var in METEO_VARS:
                raw_values = np.asarray(
                    ds[var].values[day_indices],
                    dtype=np.float32,
                )
                raw_values = np.nan_to_num(
                    raw_values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                sums = np.bincount(
                    flat_idx,
                    weights=raw_values,
                    minlength=H * W,
                )
                raw_grid = (sums / safe_counts).reshape(H, W)

                mean = float(train_stats[var]["mean"])
                std = float(train_stats[var]["std_population"])
                expected = (
                    (raw_grid - mean) / std
                ).astype(np.float32, copy=False)

                expected_tensor = torch.from_numpy(expected).to(
                    dtype=new_payload["x"].dtype
                )
                feature_index = FEATURE_INDICES[var]

                metrics = diff_metrics(
                    new_payload["x"][feature_index],
                    expected_tensor,
                )
                all_variable_metrics[var].append(metrics)

                variable_reports[var] = {
                    "feature_index": feature_index,
                    "direct_raw_vs_gate8b": metrics,
                }

            day_reports.append(
                {
                    "month": month,
                    "integer_doy": day,
                    "file": filename,
                    "raw_sample_count": int(day_indices.size),
                    "non_meteorological_content_exact": True,
                    "variables": variable_reports,
                }
            )

            print(
                f"AUDITED_MONTH={month:02d} "
                f"DOY={day} FILE={filename}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    direct_pass = True

    for var, records in all_variable_metrics.items():
        item = {
            "days": len(records),
            "worst_mean_abs": max(
                float(record["mean_abs"]) for record in records
            ),
            "worst_q99_abs": max(
                float(record["q99_abs"]) for record in records
            ),
            "worst_max_abs": max(
                float(record["max_abs"]) for record in records
            ),
            "minimum_exact_fraction": min(
                float(record["exact_fraction"]) for record in records
            ),
        }
        item_pass = (
            item["worst_mean_abs"] <= args.mean_abs_threshold
            and item["worst_q99_abs"] <= args.q99_abs_threshold
            and item["worst_max_abs"] <= args.max_abs_threshold
        )
        item["pass"] = bool(item_pass)
        direct_pass = direct_pass and item_pass
        summary[var] = item

    overall_pass = direct_pass and all_non_meteo_exact

    report = {
        "audit_name": "meteo_trainonly_direct_tensor_gate9b",
        "read_only": True,
        "inputs": {
            "nc": str(nc_path),
            "old_dir": str(old_dir),
            "new_dir": str(new_dir),
            "train_stats": str(train_stats_path),
            "gate8b_manifest": str(manifest_path),
        },
        "selected_representative_days": {
            str(month): day for month, day in selected.items()
        },
        "thresholds": {
            "mean_abs": args.mean_abs_threshold,
            "q99_abs": args.q99_abs_threshold,
            "max_abs": args.max_abs_threshold,
        },
        "summary": summary,
        "day_reports": day_reports,
        "verification": {
            "direct_raw_reconstruction_pass": bool(direct_pass),
            "all_non_meteorological_content_exactly_unchanged": bool(
                all_non_meteo_exact
            ),
            "overall_pass": bool(overall_pass),
        },
        "interpretation": {
            "construction_independence": (
                "Gate-8B used chunked accumulation over the full NetCDF; "
                "Gate-9B uses one-shot per-day np.bincount aggregation."
            ),
            "allowed_difference": (
                "Only rare half-precision boundary differences up to one "
                "float16 step are permitted."
            ),
        },
    }

    json_path = output_dir / "meteo_trainonly_direct_tensor_gate9b.json"
    txt_path = output_dir / "meteo_trainonly_direct_tensor_gate9b.txt"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "Train-only direct daily tensor audit - Gate 9B",
        f"Selected days: {selected}",
        f"Direct raw reconstruction pass: {direct_pass}",
        (
            "All non-meteorological content exactly unchanged: "
            f"{all_non_meteo_exact}"
        ),
        f"Overall pass: {overall_pass}",
        "",
        "Variable summaries:",
    ]
    for var, item in summary.items():
        lines.extend(
            [
                f"- {var}",
                f"  worst_mean_abs={item['worst_mean_abs']}",
                f"  worst_q99_abs={item['worst_q99_abs']}",
                f"  worst_max_abs={item['worst_max_abs']}",
                f"  minimum_exact_fraction={item['minimum_exact_fraction']}",
                f"  pass={item['pass']}",
            ]
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if not overall_pass:
        print("GATE9B_DIRECT_TENSOR_AUDIT_FAILED")
        print(f"JSON_REPORT={json_path}")
        print(f"TXT_REPORT={txt_path}")
        print(f"DIRECT_RAW_RECONSTRUCTION_PASS={direct_pass}")
        print(
            "NON_METEOROLOGICAL_CONTENT_EXACTLY_UNCHANGED="
            f"{all_non_meteo_exact}"
        )
        raise RuntimeError(
            "Gate 9B failed. Do not build phase2 caches."
        )

    print("GATE9B_DIRECT_TENSOR_AUDIT_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"SELECTED_DAYS={selected}")
    print("DIRECT_RAW_RECONSTRUCTION_PASS=True")
    print("NON_METEOROLOGICAL_CONTENT_EXACTLY_UNCHANGED=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
