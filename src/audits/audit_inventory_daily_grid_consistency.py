# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


N_LAT = 361
N_LON = 576
N_GRID = N_LAT * N_LON
LAT_MIN = -90.0
LON_MIN = -180.0
LAT_STEP = 0.5
LON_STEP = 0.625

ALLOWED_SECTORS = {
    "AGRICULTURE",
    "TRANSPORT",
    "IND_COMBUSTION",
    "POWER_INDUSTRYE",
    "BUILDINGS",
    "FUEL_EXPLOITATION",
    "IND_PROCESSES",
    "WASTE_arr",
}


def collect_integer_days(
    ds: xr.Dataset,
    chunk_size: int,
) -> list[int]:
    days_seen: set[int] = set()
    n_samples = int(ds.sizes["num_samples"])

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        doy = np.asarray(
            ds["DOY"].isel(num_samples=slice(start, stop)).values,
            dtype=np.float64,
        ).reshape(-1)

        finite = np.isfinite(doy)
        if finite.any():
            days = doy[finite].astype(np.int16, copy=False)
            days = days[(days >= 1) & (days <= 366)]
            days_seen.update(int(day) for day in np.unique(days))

    return sorted(days_seen)


def top_variation_examples(
    relative_range: np.ndarray,
    repeated_mask: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    count_state: np.ndarray,
    integer_days: list[int],
    limit: int = 20,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(repeated_mask & np.isfinite(relative_range))
    if indices.size == 0:
        return []

    take = min(limit, indices.size)
    local = np.argpartition(relative_range[indices], -take)[-take:]
    chosen = indices[local]
    chosen = chosen[np.argsort(relative_range[chosen])[::-1]]

    result: list[dict[str, Any]] = []
    for composite_index in chosen:
        day_slot = int(composite_index // N_GRID)
        grid_id = int(composite_index % N_GRID)
        lat_idx = grid_id // N_LON
        lon_idx = grid_id % N_LON

        result.append(
            {
                "integer_doy": int(integer_days[day_slot]),
                "grid_id": grid_id,
                "lat": float(LAT_MIN + lat_idx * LAT_STEP),
                "lon": float(LON_MIN + lon_idx * LON_STEP),
                "count_state": int(count_state[composite_index]),
                "min": float(min_values[composite_index]),
                "max": float(max_values[composite_index]),
                "range": float(
                    max_values[composite_index] - min_values[composite_index]
                ),
                "relative_range": float(relative_range[composite_index]),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Gate 3A audit. It does not assume that NetCDF rows are "
            "ordered by DOY. It aggregates one selected sector by the composite "
            "key (integer DOY, MERRA-2 grid cell)."
        )
    )
    parser.add_argument("--nc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sector",
        default="TRANSPORT",
        choices=sorted(ALLOWED_SECTORS),
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--keep-work-files",
        action="store_true",
        help="Keep temporary memmap arrays after a successful run.",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    started = time.time()
    work_dir = output_dir / f"_gate3a_work_{args.sector.lower()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    min_path = work_dir / "min_values.float32.dat"
    max_path = work_dir / "max_values.float32.dat"
    count_path = work_dir / "count_state.uint8.dat"

    try:
        with xr.open_dataset(
            nc_path,
            decode_times=False,
            mask_and_scale=False,
        ) as ds:
            required = ["Latitude", "Longitude", "DOY", args.sector]
            missing = [name for name in required if name not in ds.variables]
            if missing:
                raise KeyError(f"Missing required variables: {missing}")
            if "num_samples" not in ds.sizes:
                raise KeyError("Expected dimension 'num_samples' was not found")

            n_samples = int(ds.sizes["num_samples"])

            print("STAGE=COLLECT_INTEGER_DAYS", flush=True)
            integer_days = collect_integer_days(ds, args.chunk_size)
            if not integer_days:
                raise RuntimeError("No valid integer DOYs were found")

            day_to_slot = np.full(367, -1, dtype=np.int16)
            for slot, day in enumerate(integer_days):
                day_to_slot[day] = slot

            n_composite = len(integer_days) * N_GRID
            print(f"N_UNIQUE_INTEGER_DAYS={len(integer_days)}", flush=True)
            print(f"N_DAY_GRID_SLOTS={n_composite}", flush=True)

            min_values = np.memmap(
                min_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_composite,),
            )
            max_values = np.memmap(
                max_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_composite,),
            )
            count_state = np.memmap(
                count_path,
                mode="w+",
                dtype=np.uint8,
                shape=(n_composite,),
            )

            min_values[:] = np.inf
            max_values[:] = -np.inf
            count_state[:] = 0
            min_values.flush()
            max_values.flush()
            count_state.flush()

            coordinate_invalid_count = 0
            value_nonfinite_count = 0

            print("STAGE=AGGREGATE_DAY_GRID_KEYS", flush=True)

            for start in range(0, n_samples, args.chunk_size):
                stop = min(start + args.chunk_size, n_samples)
                selector = {"num_samples": slice(start, stop)}

                lat = np.asarray(
                    ds["Latitude"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                lon = np.asarray(
                    ds["Longitude"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                doy = np.asarray(
                    ds["DOY"].isel(selector).values,
                    dtype=np.float64,
                ).reshape(-1)
                values = np.asarray(
                    ds[args.sector].isel(selector).values,
                    dtype=np.float32,
                ).reshape(-1)

                lat_idx = np.rint((lat - LAT_MIN) / LAT_STEP).astype(
                    np.int32,
                    copy=False,
                )
                lon_idx = np.rint((lon - LON_MIN) / LON_STEP).astype(
                    np.int32,
                    copy=False,
                )
                reconstructed_lat = (
                    LAT_MIN + lat_idx.astype(np.float64) * LAT_STEP
                )
                reconstructed_lon = (
                    LON_MIN + lon_idx.astype(np.float64) * LON_STEP
                )

                finite_coordinates = (
                    np.isfinite(lat) & np.isfinite(lon) & np.isfinite(doy)
                )
                days = np.zeros(doy.shape, dtype=np.int16)
                days[finite_coordinates] = doy[finite_coordinates].astype(
                    np.int16,
                    copy=False,
                )

                valid_coordinates = (
                    finite_coordinates
                    & (days >= 1)
                    & (days <= 366)
                    & (lat_idx >= 0)
                    & (lat_idx < N_LAT)
                    & (lon_idx >= 0)
                    & (lon_idx < N_LON)
                    & (np.abs(lat - reconstructed_lat) <= 1e-4)
                    & (np.abs(lon - reconstructed_lon) <= 1e-4)
                    & (day_to_slot[days.clip(0, 366)] >= 0)
                )
                coordinate_invalid_count += int(
                    (~valid_coordinates).sum()
                )

                finite_values = np.isfinite(values)
                value_nonfinite_count += int(
                    (valid_coordinates & ~finite_values).sum()
                )
                valid = valid_coordinates & finite_values

                if not valid.any():
                    print(f"PROGRESS={stop}/{n_samples}", flush=True)
                    continue

                valid_days = days[valid]
                day_slots = day_to_slot[valid_days].astype(
                    np.int64,
                    copy=False,
                )
                grid_ids = (
                    lat_idx[valid].astype(np.int64) * N_LON
                    + lon_idx[valid].astype(np.int64)
                )
                keys = day_slots * N_GRID + grid_ids
                vals = values[valid]

                order = np.argsort(keys, kind="mergesort")
                sorted_keys = keys[order]
                sorted_vals = vals[order]

                unique_keys, first_indices, local_counts = np.unique(
                    sorted_keys,
                    return_index=True,
                    return_counts=True,
                )
                local_min = np.minimum.reduceat(
                    sorted_vals,
                    first_indices,
                )
                local_max = np.maximum.reduceat(
                    sorted_vals,
                    first_indices,
                )

                old_state = count_state[unique_keys].astype(
                    np.uint16,
                    copy=False,
                )
                local_state = np.minimum(
                    local_counts,
                    2,
                ).astype(np.uint16)
                new_state = np.minimum(
                    old_state + local_state,
                    2,
                ).astype(np.uint8)

                min_values[unique_keys] = np.minimum(
                    min_values[unique_keys],
                    local_min,
                )
                max_values[unique_keys] = np.maximum(
                    max_values[unique_keys],
                    local_max,
                )
                count_state[unique_keys] = new_state

                print(f"PROGRESS={stop}/{n_samples}", flush=True)

            min_values.flush()
            max_values.flush()
            count_state.flush()

            observed = count_state > 0
            repeated = count_state >= 2

            value_range = np.full(
                n_composite,
                np.nan,
                dtype=np.float32,
            )
            value_range[repeated] = (
                max_values[repeated] - min_values[repeated]
            )

            scale = np.full(
                n_composite,
                np.nan,
                dtype=np.float32,
            )
            scale[repeated] = np.maximum(
                np.abs(min_values[repeated]),
                np.abs(max_values[repeated]),
            )
            tolerance = args.atol + args.rtol * scale

            exact_constant = repeated & (value_range == 0.0)
            tolerant_constant = repeated & (value_range <= tolerance)
            varying = repeated & ~tolerant_constant

            nonzero_repeated = repeated & (max_values > 0.0)
            nonzero_tolerant_constant = (
                nonzero_repeated & tolerant_constant
            )
            nonzero_varying = nonzero_repeated & ~tolerant_constant
            zero_and_nonzero = (
                repeated
                & (min_values == 0.0)
                & (max_values > 0.0)
            )

            relative_range = np.full(
                n_composite,
                np.nan,
                dtype=np.float32,
            )
            varying_scale = np.maximum(
                np.maximum(
                    np.abs(min_values[varying]),
                    np.abs(max_values[varying]),
                ),
                np.float32(1e-30),
            )
            relative_range[varying] = (
                value_range[varying] / varying_scale
            )

            repeated_count = int(repeated.sum())
            tolerant_count = int(tolerant_constant.sum())
            nonzero_repeated_count = int(nonzero_repeated.sum())
            nonzero_tolerant_count = int(
                nonzero_tolerant_constant.sum()
            )

            tolerant_fraction = (
                float(tolerant_count / repeated_count)
                if repeated_count
                else None
            )
            nonzero_tolerant_fraction = (
                float(
                    nonzero_tolerant_count
                    / nonzero_repeated_count
                )
                if nonzero_repeated_count
                else None
            )

            if (
                nonzero_repeated_count >= 100
                and nonzero_tolerant_fraction is not None
            ):
                if nonzero_tolerant_fraction >= 0.999:
                    decision = (
                        "CONSISTENT_WITH_DETERMINISTIC_DAILY_GRID_FIELD"
                    )
                elif nonzero_tolerant_fraction <= 0.95:
                    decision = (
                        "NOT_CONSISTENT_WITH_DETERMINISTIC_DAILY_GRID_FIELD"
                    )
                else:
                    decision = "MIXED_OR_AMBIGUOUS"
            else:
                decision = "INSUFFICIENT_REPEATED_DAY_GRID_EVIDENCE"

            examples = top_variation_examples(
                relative_range=relative_range,
                repeated_mask=nonzero_varying,
                min_values=min_values,
                max_values=max_values,
                count_state=count_state,
                integer_days=integer_days,
                limit=20,
            )

            report = {
                "audit_name": (
                    "emission_same_day_grid_consistency_gate3a"
                ),
                "read_only": True,
                "file": str(nc_path),
                "sector": args.sector,
                "processed_samples": n_samples,
                "row_order_assumption": "none",
                "historical_day_conversion": (
                    "integer_doy = DOY.astype(np.int16)"
                ),
                "integer_days": integer_days,
                "n_unique_integer_days": len(integer_days),
                "day_grid_slot_count": n_composite,
                "coordinate_invalid_count": (
                    coordinate_invalid_count
                ),
                "value_nonfinite_count": value_nonfinite_count,
                "tolerances": {
                    "atol": args.atol,
                    "rtol": args.rtol,
                },
                "results": {
                    "observed_day_grid_groups": int(observed.sum()),
                    "repeated_day_grid_groups": repeated_count,
                    "exact_constant_repeated_groups": int(
                        exact_constant.sum()
                    ),
                    "tolerant_constant_repeated_groups": (
                        tolerant_count
                    ),
                    "tolerant_constant_fraction": (
                        tolerant_fraction
                    ),
                    "varying_repeated_groups": int(varying.sum()),
                    "nonzero_repeated_day_grid_groups": (
                        nonzero_repeated_count
                    ),
                    "nonzero_tolerant_constant_groups": (
                        nonzero_tolerant_count
                    ),
                    "nonzero_tolerant_constant_fraction": (
                        nonzero_tolerant_fraction
                    ),
                    "nonzero_varying_groups": int(
                        nonzero_varying.sum()
                    ),
                    "zero_and_nonzero_repeated_groups": int(
                        zero_and_nonzero.sum()
                    ),
                    "largest_nonzero_variation_examples": (
                        examples
                    ),
                },
                "overall_decision": decision,
                "interpretation": {
                    "question": (
                        "At one integer DOY and one stored MERRA-2 "
                        "coordinate, do repeated rows carry the same "
                        "sector value?"
                    ),
                    "if_consistent": (
                        "Cross-day variation may still be compatible "
                        "with a temporally configured daily or monthly "
                        "sector field."
                    ),
                    "if_not_consistent": (
                        "The embedded variable is not a deterministic "
                        "daily grid lookup at the stored coordinate. "
                        "Possible causes include sub-grid information "
                        "collapsed to a coarse coordinate, row-level "
                        "fusion, or another undocumented transformation."
                    ),
                    "boundary": (
                        "This audit does not identify the EDGAR release, "
                        "inventory year, units, or upstream fusion method."
                    ),
                },
                "elapsed_seconds": float(time.time() - started),
            }

            json_path = (
                output_dir
                / f"emission_same_day_grid_gate3a_{args.sector.lower()}.json"
            )
            txt_path = (
                output_dir
                / f"emission_same_day_grid_gate3a_{args.sector.lower()}.txt"
            )

            json_path.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            txt_lines = [
                "Emission same-day grid consistency audit - Gate 3A",
                f"File: {nc_path}",
                f"Sector: {args.sector}",
                f"Processed samples: {n_samples}",
                "Row-order assumption: none",
                f"Integer DOYs: {len(integer_days)}",
                f"Observed day-grid groups: {int(observed.sum())}",
                f"Repeated day-grid groups: {repeated_count}",
                (
                    "Tolerant constant fraction: "
                    f"{tolerant_fraction}"
                ),
                (
                    "Nonzero repeated day-grid groups: "
                    f"{nonzero_repeated_count}"
                ),
                (
                    "Nonzero tolerant constant fraction: "
                    f"{nonzero_tolerant_fraction}"
                ),
                (
                    "Zero-and-nonzero repeated groups: "
                    f"{int(zero_and_nonzero.sum())}"
                ),
                f"Overall decision: {decision}",
            ]
            txt_path.write_text(
                "\n".join(txt_lines),
                encoding="utf-8",
            )

            print("GATE3A_SAME_DAY_GRID_AUDIT_PASS")
            print(f"JSON_REPORT={json_path}")
            print(f"TXT_REPORT={txt_path}")
            print(f"SECTOR={args.sector}")
            print(
                "NONZERO_REPEATED_DAY_GRID_GROUPS="
                f"{nonzero_repeated_count}"
            )
            print(
                "NONZERO_TOLERANT_CONSTANT_FRACTION="
                f"{nonzero_tolerant_fraction}"
            )
            print(f"OVERALL_DECISION={decision}")

            # Close memory maps before optionally deleting temporary files.
            del value_range
            del scale
            del tolerance
            del relative_range
            del min_values
            del max_values
            del count_state

    finally:
        if work_dir.exists() and not args.keep_work_files:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
