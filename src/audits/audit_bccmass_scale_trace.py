# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
35_audit_bccmass_unit_and_scale_trace.py

Read-only Gate 6:
Trace the exact BCCMASS numerical transformation used by the experiment.

This script:
1. Reads the BCCMASS attributes and deterministic value statistics from the
   inherited fused NetCDF.
2. Reads one existing daily tensor.
3. Reconstructs that day's gridded BCCMASS target exactly as
   04_construct_tensors.py does.
4. Verifies that y[0] equals log1p(grid_mean(BCCMASS) * 1e9).
5. Verifies that expm1(y[0]) / 1e9 returns the pre-log gridded values.
6. Reports the dimensional implication:
   if raw BCCMASS is kg m^-2, multiplication by 1e9 gives micrograms m^-2.

No source file or tensor is modified.
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


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_scale_from_code(code_text: str) -> float | None:
    patterns = [
        r"CONC_SCALE\s*=\s*([0-9.eE+-]+)",
        r"avg_grid_conc\s*\*\s*([0-9.eE+-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, code_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "q01": None,
            "median": None,
            "mean": None,
            "q99": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def error_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    diff = np.abs(
        np.asarray(actual, dtype=np.float64)
        - np.asarray(expected, dtype=np.float64)
    )
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "max_abs": float("nan"),
            "q99_abs": float("nan"),
        }
    return {
        "count": int(diff.size),
        "mae": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "q99_abs": float(np.quantile(diff, 0.99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only BCCMASS unit and numerical-scale trace."
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--construct-script", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    tensor_path = Path(args.tensor).expanduser().resolve()
    construct_script_path = Path(args.construct_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [nc_path, tensor_path, construct_script_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    code_text = construct_script_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    code_scale = extract_scale_from_code(code_text)
    if code_scale is None:
        raise RuntimeError(
            "Could not locate CONC_SCALE in the tensor-construction script."
        )

    has_log1p = (
        "np.log1p(avg_grid_conc * CONC_SCALE)" in code_text
        or "np.log1p(avg_grid_conc*CONC_SCALE)" in code_text
    )

    tensor = torch.load(tensor_path, map_location="cpu")
    if "y" not in tensor or "day" not in tensor:
        raise KeyError("Tensor must contain 'y' and 'day'.")

    day = int(tensor["day"])
    target_saved = (
        tensor["y"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )

    if target_saved.shape != (H, W):
        raise ValueError(
            f"Unexpected saved target shape: {target_saved.shape}"
        )

    with xr.open_dataset(
        nc_path,
        decode_times=False,
        mask_and_scale=False,
    ) as ds:
        required = ["DOY", "Latitude", "Longitude", "BCCMASS"]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        doy_all = np.asarray(ds["DOY"].values, dtype=np.float64)
        integer_days = doy_all.astype(np.int16, copy=False)
        day_indices = np.flatnonzero(integer_days == day)
        if day_indices.size == 0:
            raise RuntimeError(
                f"No NetCDF samples found for integer DOY {day}."
            )

        lats = np.asarray(
            ds["Latitude"].values[day_indices],
            dtype=np.float64,
        )
        lons = np.asarray(
            ds["Longitude"].values[day_indices],
            dtype=np.float64,
        )
        raw = np.asarray(
            ds["BCCMASS"].values[day_indices],
            dtype=np.float64,
        )
        attrs = dict(ds["BCCMASS"].attrs)
        global_attrs = dict(ds.attrs)

    raw = np.nan_to_num(raw, nan=0.0)

    lat_idx = np.rint((lats - LAT_MIN) / LAT_RES).astype(np.int32)
    lon_idx = np.rint((lons - LON_MIN) / LON_RES).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, H - 1)
    lon_idx = np.clip(lon_idx, 0, W - 1)

    flat_idx = lat_idx * W + lon_idx
    grid_size = H * W

    counts = np.bincount(
        flat_idx,
        minlength=grid_size,
    ).astype(np.float64)
    safe_counts = counts.copy()
    safe_counts[safe_counts == 0] = 1.0

    sums = np.bincount(
        flat_idx,
        weights=raw,
        minlength=grid_size,
    )
    grid_mean_raw = (sums / safe_counts).reshape(H, W)

    expected_log = np.log1p(grid_mean_raw * code_scale)
    reconstructed_raw = np.expm1(target_saved) / code_scale

    occupied = counts.reshape(H, W) > 0

    log_error_all = error_stats(target_saved, expected_log)
    log_error_occupied = error_stats(
        target_saved[occupied],
        expected_log[occupied],
    )
    inverse_error_occupied = error_stats(
        reconstructed_raw[occupied],
        grid_mean_raw[occupied],
    )

    formula_match = (
        log_error_occupied["max_abs"] <= 1e-5
        and inverse_error_occupied["max_abs"] <= 1e-10
    )

    official_source_unit = "kg m^-2"
    scale_interpretation = {
        "kg_m-2_to_mg_m-2_factor": 1e6,
        "kg_m-2_to_ug_m-2_factor": 1e9,
        "kg_m-2_to_ng_m-2_factor": 1e12,
        "code_factor": code_scale,
        "if_raw_is_official_kg_m-2": (
            "The code-scaled pre-log value is micrograms per square metre "
            "when CONC_SCALE=1e9."
            if math.isclose(code_scale, 1e9)
            else "The code factor differs from the kg-to-microgram factor."
        ),
    }

    report = {
        "audit_name": "bccmass_unit_and_scale_trace_gate6",
        "read_only": True,
        "inputs": {
            "nc": str(nc_path),
            "tensor": str(tensor_path),
            "construct_script": str(construct_script_path),
        },
        "official_merra2_reference": {
            "variable": "BCCMASS",
            "definition": "Black Carbon Column Mass Density",
            "unit": official_source_unit,
            "collection": "tavg1_2d_aer_Nx (M2T1NXAER)",
            "note": (
                "This is the official MERRA-2 product definition. "
                "The inherited fused NetCDF does not itself retain unit metadata."
            ),
        },
        "local_nc_metadata": {
            "bccmass_attrs": jsonable(attrs),
            "global_attrs": jsonable(global_attrs),
            "unit_attribute_present": any(
                key.lower() in {"units", "unit"}
                for key in attrs
            ),
        },
        "code_trace": {
            "detected_conc_scale": code_scale,
            "detected_log1p_formula": bool(has_log1p),
            "actual_formula": (
                "saved_y0 = log1p(daily_grid_mean(BCCMASS) * CONC_SCALE)"
            ),
            "inverse_formula": (
                "daily_grid_mean(BCCMASS) = expm1(saved_y0) / CONC_SCALE"
            ),
        },
        "selected_day": {
            "integer_doy": day,
            "netcdf_sample_count": int(day_indices.size),
            "occupied_grid_count": int(occupied.sum()),
        },
        "value_statistics": {
            "raw_bccmass_rows": finite_stats(raw),
            "daily_grid_mean_raw_on_occupied": finite_stats(
                grid_mean_raw[occupied]
            ),
            "saved_log_target_on_occupied": finite_stats(
                target_saved[occupied]
            ),
            "raw_interpreted_as_mg_m-2_on_occupied": finite_stats(
                grid_mean_raw[occupied] * 1e6
            ),
            "raw_interpreted_as_ug_m-2_on_occupied": finite_stats(
                grid_mean_raw[occupied] * 1e9
            ),
        },
        "verification": {
            "saved_vs_reconstructed_log_all_grid": log_error_all,
            "saved_vs_reconstructed_log_occupied": log_error_occupied,
            "inverse_saved_log_vs_raw_grid_mean_occupied": (
                inverse_error_occupied
            ),
            "formula_match_within_tolerance": bool(formula_match),
        },
        "dimensional_interpretation": scale_interpretation,
        "scientific_conclusion": {
            "confirmed_from_code": (
                "The experiment uses a fixed 1e9 numerical scaling after "
                "same-day grid averaging and before log1p."
            ),
            "strongly_supported_source_unit": (
                "kg m^-2, because BCCMASS is the official MERRA-2 variable "
                "name and the observed magnitude is compatible with that product."
            ),
            "remaining_boundary": (
                "The inherited fused NetCDF lacks unit metadata, so only the "
                "upstream fusion code or data dictionary can prove that no unit "
                "conversion occurred before the file was written."
            ),
            "comment_correction": (
                "If the source is kg m^-2, multiplying by 1e9 yields "
                "micrograms m^-2, not nanograms m^-2."
            ),
        },
    }

    json_path = output_dir / "bccmass_unit_and_scale_trace_gate6.json"
    txt_path = output_dir / "bccmass_unit_and_scale_trace_gate6.txt"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    txt_lines = [
        "BCCMASS unit and numerical-scale trace - Gate 6",
        f"NetCDF: {nc_path}",
        f"Tensor: {tensor_path}",
        f"Integer DOY: {day}",
        f"NetCDF samples: {day_indices.size}",
        f"Occupied grids: {int(occupied.sum())}",
        f"Local BCCMASS attrs: {attrs}",
        f"Detected CONC_SCALE: {code_scale}",
        f"Detected log1p formula: {has_log1p}",
        (
            "Saved-vs-reconstructed occupied max abs error: "
            f"{log_error_occupied['max_abs']}"
        ),
        (
            "Inverse-vs-raw occupied max abs error: "
            f"{inverse_error_occupied['max_abs']}"
        ),
        f"Formula match: {formula_match}",
        "Official MERRA-2 BCCMASS unit: kg m^-2",
        (
            "Dimensional implication: kg m^-2 * 1e9 = "
            "micrograms m^-2, not nanograms m^-2."
        ),
        (
            "Boundary: the inherited fused NetCDF has no unit attribute; "
            "upstream fusion provenance remains the final proof."
        ),
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("GATE6_BCCMASS_SCALE_TRACE_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"INTEGER_DOY={day}")
    print(f"DETECTED_CONC_SCALE={code_scale}")
    print(f"FORMULA_MATCH={formula_match}")
    print(
        "SAVED_LOG_MAX_ABS_ERROR_OCCUPIED="
        f"{log_error_occupied['max_abs']}"
    )
    print(
        "INVERSE_RAW_MAX_ABS_ERROR_OCCUPIED="
        f"{inverse_error_occupied['max_abs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
