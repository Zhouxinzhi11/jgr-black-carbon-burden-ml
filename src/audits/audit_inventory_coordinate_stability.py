# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


EMISSION_VARS = [
    "AGRICULTURE",
    "TRANSPORT",
    "IND_COMBUSTION",
    "POWER_INDUSTRYE",
    "BUILDINGS",
    "FUEL_EXPLOITATION",
    "IND_PROCESSES",
    "WASTE_arr",
]

N_LAT = 361
N_LON = 576
N_GRID = N_LAT * N_LON
LAT_MIN = -90.0
LON_MIN = -180.0
LAT_STEP = 0.5
LON_STEP = 0.625


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"q50": None, "q90": None, "q99": None, "max": None}
    return {
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def grid_examples(
    score: np.ndarray,
    eligible: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    limit: int = 10,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(eligible & np.isfinite(score))
    if indices.size == 0:
        return []
    count = min(limit, indices.size)
    local = np.argpartition(score[indices], -count)[-count:]
    selected = indices[local]
    selected = selected[np.argsort(score[selected])[::-1]]

    result: list[dict[str, Any]] = []
    for grid_id in selected:
        lat_idx = int(grid_id // N_LON)
        lon_idx = int(grid_id % N_LON)
        lat = LAT_MIN + lat_idx * LAT_STEP
        lon = LON_MIN + lon_idx * LON_STEP
        result.append(
            {
                "grid_id": int(grid_id),
                "lat": float(lat),
                "lon": float(lon),
                "min": float(min_values[grid_id]),
                "max": float(max_values[grid_id]),
                "range": float(max_values[grid_id] - min_values[grid_id]),
                "relative_range": float(score[grid_id]),
            }
        )
    return result


def classify_sector(
    cross_day_count: int,
    tolerant_constant_fraction: float | None,
    nonzero_cross_day_count: int,
    nonzero_tolerant_constant_fraction: float | None,
) -> str:
    # The non-zero cross-day comparison is the strongest diagnostic because it is
    # less sensitive to missing/fill zeros.
    if nonzero_cross_day_count >= 100 and nonzero_tolerant_constant_fraction is not None:
        if nonzero_tolerant_constant_fraction >= 0.999:
            return "consistent_with_fixed_spatial_lookup"
        if nonzero_tolerant_constant_fraction <= 0.95:
            return "not_consistent_with_fixed_spatial_lookup"
        return "mixed_or_ambiguous"

    if cross_day_count >= 100 and tolerant_constant_fraction is not None:
        if tolerant_constant_fraction >= 0.999:
            return "consistent_with_fixed_spatial_lookup_but_nonzero_evidence_limited"
        if tolerant_constant_fraction <= 0.95:
            return "not_consistent_with_fixed_spatial_lookup"
    return "insufficient_repeated_coordinate_evidence"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Gate 2 audit: test whether embedded sector values are stable "
            "at the same MERRA-2 grid coordinate across integer DOYs."
        )
    )
    parser.add_argument("--nc", required=True, help="Path to BCCMASS_train.nc")
    parser.add_argument("--output-dir", required=True, help="Output report directory")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows per streaming chunk (default: 250000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row limit for a smoke run; 0 means scan the full file",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for stability (default: 1e-7)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for stability (default: 1e-6)",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")

    start_time = time.time()

    grid_sample_count = np.zeros(N_GRID, dtype=np.int64)
    grid_day_min = np.full(N_GRID, np.iinfo(np.int16).max, dtype=np.int16)
    grid_day_max = np.full(N_GRID, np.iinfo(np.int16).min, dtype=np.int16)

    sector_state: dict[str, dict[str, np.ndarray]] = {}
    for name in EMISSION_VARS:
        sector_state[name] = {
            "min": np.full(N_GRID, np.inf, dtype=np.float64),
            "max": np.full(N_GRID, -np.inf, dtype=np.float64),
            "finite_count": np.zeros(N_GRID, dtype=np.int64),
            "nz_min": np.full(N_GRID, np.inf, dtype=np.float64),
            "nz_max": np.full(N_GRID, -np.inf, dtype=np.float64),
            "nz_count": np.zeros(N_GRID, dtype=np.int64),
            "nz_day_min": np.full(N_GRID, np.iinfo(np.int16).max, dtype=np.int16),
            "nz_day_max": np.full(N_GRID, np.iinfo(np.int16).min, dtype=np.int16),
        }

    integer_day_counts = np.zeros(367, dtype=np.int64)
    coordinate_invalid_count = 0
    coordinate_nonfinite_count = 0
    max_lat_reconstruction_error = 0.0
    max_lon_reconstruction_error = 0.0
    processed_samples = 0

    with xr.open_dataset(nc_path, decode_times=False, mask_and_scale=False) as ds:
        missing = [
            name
            for name in ["Latitude", "Longitude", "DOY", *EMISSION_VARS]
            if name not in ds.variables
        ]
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

        if "num_samples" not in ds.sizes:
            raise KeyError("Expected dimension 'num_samples' was not found")

        total_samples = int(ds.sizes["num_samples"])
        scan_samples = total_samples
        if args.max_samples:
            scan_samples = min(scan_samples, int(args.max_samples))

        for start in range(0, scan_samples, args.chunk_size):
            stop = min(start + args.chunk_size, scan_samples)
            selector = {"num_samples": slice(start, stop)}

            lat = np.asarray(ds["Latitude"].isel(selector).values, dtype=np.float64).reshape(-1)
            lon = np.asarray(ds["Longitude"].isel(selector).values, dtype=np.float64).reshape(-1)
            doy_raw = np.asarray(ds["DOY"].isel(selector).values, dtype=np.float64).reshape(-1)

            coordinate_finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(doy_raw)
            coordinate_nonfinite_count += int((~coordinate_finite).sum())

            lat_idx = np.rint((lat - LAT_MIN) / LAT_STEP).astype(np.int32, copy=False)
            lon_idx = np.rint((lon - LON_MIN) / LON_STEP).astype(np.int32, copy=False)

            reconstructed_lat = LAT_MIN + lat_idx.astype(np.float64) * LAT_STEP
            reconstructed_lon = LON_MIN + lon_idx.astype(np.float64) * LON_STEP
            if coordinate_finite.any():
                max_lat_reconstruction_error = max(
                    max_lat_reconstruction_error,
                    float(np.max(np.abs(lat[coordinate_finite] - reconstructed_lat[coordinate_finite]))),
                )
                max_lon_reconstruction_error = max(
                    max_lon_reconstruction_error,
                    float(np.max(np.abs(lon[coordinate_finite] - reconstructed_lon[coordinate_finite]))),
                )

            coordinate_valid = (
                coordinate_finite
                & (lat_idx >= 0)
                & (lat_idx < N_LAT)
                & (lon_idx >= 0)
                & (lon_idx < N_LON)
                & (np.abs(lat - reconstructed_lat) <= 1e-4)
                & (np.abs(lon - reconstructed_lon) <= 1e-4)
            )
            coordinate_invalid_count += int((~coordinate_valid & coordinate_finite).sum())

            if not coordinate_valid.any():
                processed_samples += stop - start
                print(f"PROGRESS={stop}/{scan_samples} valid_rows=0", flush=True)
                continue

            ids = lat_idx[coordinate_valid] * N_LON + lon_idx[coordinate_valid]

            # Match the historical tensor code: positive fractional DOY is truncated
            # by astype(int16), yielding the integer calendar day.
            days = doy_raw[coordinate_valid].astype(np.int16, copy=False)
            valid_day_range = (days >= 1) & (days <= 366)
            ids = ids[valid_day_range]
            days = days[valid_day_range]

            np.add.at(grid_sample_count, ids, 1)
            np.minimum.at(grid_day_min, ids, days)
            np.maximum.at(grid_day_max, ids, days)
            np.add.at(integer_day_counts, days, 1)

            original_valid_indices = np.flatnonzero(coordinate_valid)[valid_day_range]

            for name in EMISSION_VARS:
                values_all = np.asarray(
                    ds[name].isel(selector).values, dtype=np.float64
                ).reshape(-1)
                values = values_all[original_valid_indices]
                finite = np.isfinite(values)
                if not finite.any():
                    continue

                finite_ids = ids[finite]
                finite_days = days[finite]
                finite_values = values[finite]
                state = sector_state[name]

                np.minimum.at(state["min"], finite_ids, finite_values)
                np.maximum.at(state["max"], finite_ids, finite_values)
                np.add.at(state["finite_count"], finite_ids, 1)

                nonzero = finite_values != 0.0
                if nonzero.any():
                    nz_ids = finite_ids[nonzero]
                    nz_days = finite_days[nonzero]
                    nz_values = finite_values[nonzero]
                    np.minimum.at(state["nz_min"], nz_ids, nz_values)
                    np.maximum.at(state["nz_max"], nz_ids, nz_values)
                    np.add.at(state["nz_count"], nz_ids, 1)
                    np.minimum.at(state["nz_day_min"], nz_ids, nz_days)
                    np.maximum.at(state["nz_day_max"], nz_ids, nz_days)

            processed_samples += stop - start
            print(
                f"PROGRESS={stop}/{scan_samples} "
                f"observed_grids={int(np.count_nonzero(grid_sample_count))}",
                flush=True,
            )

    observed_grid = grid_sample_count > 0
    cross_day_grid = observed_grid & (grid_day_max > grid_day_min)

    day_records = [
        {"integer_doy": int(day), "sample_count": int(integer_day_counts[day])}
        for day in np.flatnonzero(integer_day_counts)
    ]

    sector_results: dict[str, Any] = {}
    sector_decisions: list[str] = []

    for name in EMISSION_VARS:
        state = sector_state[name]
        finite_grid = state["finite_count"] > 0
        eligible = cross_day_grid & finite_grid

        value_range = state["max"] - state["min"]
        scale = np.maximum(np.abs(state["min"]), np.abs(state["max"]))
        tolerance = args.atol + args.rtol * scale
        exact_constant = eligible & (value_range == 0.0)
        tolerant_constant = eligible & (value_range <= tolerance)
        varying = eligible & ~tolerant_constant

        nz_cross_day = (
            (state["nz_count"] > 0)
            & (state["nz_day_max"] > state["nz_day_min"])
        )
        nz_range = state["nz_max"] - state["nz_min"]
        nz_scale = np.maximum(np.abs(state["nz_min"]), np.abs(state["nz_max"]))
        nz_tolerance = args.atol + args.rtol * nz_scale
        nz_exact_constant = nz_cross_day & (nz_range == 0.0)
        nz_tolerant_constant = nz_cross_day & (nz_range <= nz_tolerance)
        nz_varying = nz_cross_day & ~nz_tolerant_constant

        zero_and_nonzero = eligible & (state["min"] == 0.0) & (state["max"] > 0.0)

        safe_denom = np.maximum(
            np.maximum(np.abs(state["min"]), np.abs(state["max"])),
            1e-30,
        )
        relative_range = value_range / safe_denom
        nz_safe_denom = np.maximum(
            np.maximum(np.abs(state["nz_min"]), np.abs(state["nz_max"])),
            1e-30,
        )
        nz_relative_range = nz_range / nz_safe_denom

        eligible_count = int(eligible.sum())
        nz_cross_day_count = int(nz_cross_day.sum())
        tolerant_fraction = (
            float(tolerant_constant.sum() / eligible_count)
            if eligible_count
            else None
        )
        nz_tolerant_fraction = (
            float(nz_tolerant_constant.sum() / nz_cross_day_count)
            if nz_cross_day_count
            else None
        )

        decision = classify_sector(
            eligible_count,
            tolerant_fraction,
            nz_cross_day_count,
            nz_tolerant_fraction,
        )
        sector_decisions.append(decision)

        sector_results[name] = {
            "finite_grid_count": int(finite_grid.sum()),
            "cross_integer_day_grid_count": eligible_count,
            "exact_constant_cross_day_grid_count": int(exact_constant.sum()),
            "tolerant_constant_cross_day_grid_count": int(tolerant_constant.sum()),
            "tolerant_constant_cross_day_fraction": tolerant_fraction,
            "varying_cross_day_grid_count": int(varying.sum()),
            "zero_and_nonzero_cross_day_grid_count": int(zero_and_nonzero.sum()),
            "nonzero_cross_day_grid_count": nz_cross_day_count,
            "nonzero_exact_constant_cross_day_grid_count": int(nz_exact_constant.sum()),
            "nonzero_tolerant_constant_cross_day_grid_count": int(nz_tolerant_constant.sum()),
            "nonzero_tolerant_constant_cross_day_fraction": nz_tolerant_fraction,
            "nonzero_varying_cross_day_grid_count": int(nz_varying.sum()),
            "range_quantiles_on_varying_grids": finite_quantiles(value_range[varying]),
            "relative_range_quantiles_on_varying_grids": finite_quantiles(
                relative_range[varying]
            ),
            "nonzero_range_quantiles_on_varying_grids": finite_quantiles(
                nz_range[nz_varying]
            ),
            "nonzero_relative_range_quantiles_on_varying_grids": finite_quantiles(
                nz_relative_range[nz_varying]
            ),
            "largest_nonzero_cross_day_variation_examples": grid_examples(
                nz_relative_range,
                nz_varying,
                state["nz_min"],
                state["nz_max"],
                limit=10,
            ),
            "decision": decision,
        }

    if all(
        d.startswith("consistent_with_fixed_spatial_lookup")
        for d in sector_decisions
    ):
        overall_decision = "CONSISTENT_WITH_FIXED_ANNUAL_SPATIAL_LOOKUP"
    elif any(d == "not_consistent_with_fixed_spatial_lookup" for d in sector_decisions):
        overall_decision = "NOT_CONSISTENT_WITH_FIXED_ANNUAL_SPATIAL_LOOKUP"
    else:
        overall_decision = "MIXED_OR_INSUFFICIENT_EVIDENCE"

    report = {
        "audit_name": "emission_coordinate_stability_gate2",
        "read_only": True,
        "file": str(nc_path),
        "processed_samples": int(processed_samples),
        "full_file_scan": bool(args.max_samples == 0),
        "chunk_size": int(args.chunk_size),
        "tolerances": {"atol": float(args.atol), "rtol": float(args.rtol)},
        "historical_day_conversion": "integer_doy = DOY.astype(np.int16)",
        "coordinate_grid": {
            "shape": [N_LAT, N_LON],
            "lat_min": LAT_MIN,
            "lon_min": LON_MIN,
            "lat_step": LAT_STEP,
            "lon_step": LON_STEP,
            "coordinate_nonfinite_count": int(coordinate_nonfinite_count),
            "coordinate_invalid_count": int(coordinate_invalid_count),
            "max_lat_reconstruction_error": float(max_lat_reconstruction_error),
            "max_lon_reconstruction_error": float(max_lon_reconstruction_error),
            "observed_grid_count": int(observed_grid.sum()),
            "cross_integer_day_grid_count": int(cross_day_grid.sum()),
        },
        "integer_doy_summary": {
            "n_unique_integer_days": int(np.count_nonzero(integer_day_counts)),
            "min_integer_day": (
                int(np.flatnonzero(integer_day_counts)[0])
                if np.count_nonzero(integer_day_counts)
                else None
            ),
            "max_integer_day": (
                int(np.flatnonzero(integer_day_counts)[-1])
                if np.count_nonzero(integer_day_counts)
                else None
            ),
            "day_records": day_records,
        },
        "sector_results": sector_results,
        "overall_decision": overall_decision,
        "interpretation": {
            "fixed_lookup_signature": (
                "For a fixed annual gridded inventory sampled by coordinate, the same "
                "coordinate should retain the same sector value across integer DOYs, "
                "apart from negligible floating-point differences."
            ),
            "strongest_test": (
                "nonzero_tolerant_constant_cross_day_fraction; this excludes zero/fill "
                "changes and requires nonzero observations on more than one integer DOY."
            ),
            "boundary": (
                "This audit tests coordinate-level temporal stability of the embedded "
                "values. It cannot recover the EDGAR release, inventory year, units, or "
                "upstream join procedure because those metadata are absent."
            ),
        },
        "elapsed_seconds": float(time.time() - start_time),
    }

    json_path = output_dir / "emission_coordinate_stability_gate2.json"
    txt_path = output_dir / "emission_coordinate_stability_gate2.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    lines = [
        "Emission coordinate stability audit - Gate 2",
        f"File: {nc_path}",
        f"Processed samples: {processed_samples}",
        f"Full file scan: {report['full_file_scan']}",
        f"Integer DOYs: {report['integer_doy_summary']['n_unique_integer_days']}",
        f"Observed grids: {report['coordinate_grid']['observed_grid_count']}",
        f"Cross-day grids: {report['coordinate_grid']['cross_integer_day_grid_count']}",
        f"Overall decision: {overall_decision}",
        "",
        "Sector results:",
    ]
    for name, item in sector_results.items():
        lines.extend(
            [
                f"- {name}",
                f"  decision={item['decision']}",
                f"  cross_day_grids={item['cross_integer_day_grid_count']}",
                f"  tolerant_constant_fraction={item['tolerant_constant_cross_day_fraction']}",
                f"  nonzero_cross_day_grids={item['nonzero_cross_day_grid_count']}",
                f"  nonzero_tolerant_constant_fraction="
                f"{item['nonzero_tolerant_constant_cross_day_fraction']}",
                f"  zero_and_nonzero_cross_day_grids="
                f"{item['zero_and_nonzero_cross_day_grid_count']}",
            ]
        )

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("GATE2_COORDINATE_STABILITY_AUDIT_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(
        "N_UNIQUE_INTEGER_DAYS="
        f"{report['integer_doy_summary']['n_unique_integer_days']}"
    )
    print(f"OVERALL_DECISION={overall_decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
