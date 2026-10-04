# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
from __future__ import annotations

import argparse
import json
import os
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
FOCUS_VARS = ["DOY", "Latitude", "Longitude", "BCCMASS", *EMISSION_VARS]
KEYWORDS = ("edgar", "emission", "inventory", "version", "source", "history", "unit")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def finite_sample(data_array: xr.DataArray, max_points: int = 12000) -> dict[str, Any]:
    if not data_array.dims:
        raw = np.asarray(data_array.values).reshape(-1)
    else:
        sample_dim = data_array.dims[0]
        n = int(data_array.sizes[sample_dim])
        if n == 0:
            raw = np.asarray([], dtype=np.float64)
        else:
            count = min(max_points, n)
            indices = np.unique(np.linspace(0, n - 1, num=count, dtype=np.int64))
            raw = np.asarray(data_array.isel({sample_dim: indices}).values).reshape(-1)

    numeric = np.issubdtype(raw.dtype, np.number)
    result: dict[str, Any] = {
        "sampled_count": int(raw.size),
        "sample_dtype": str(raw.dtype),
    }
    if raw.size == 0:
        return result

    if numeric:
        values = raw.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        finite_values = values[finite]
        result.update(
            {
                "sampled_finite_count": int(finite.sum()),
                "sampled_nonfinite_count": int((~finite).sum()),
                "sampled_min": float(np.min(finite_values)) if finite_values.size else None,
                "sampled_max": float(np.max(finite_values)) if finite_values.size else None,
                "sampled_mean": float(np.mean(finite_values)) if finite_values.size else None,
                "sampled_first_values": [float(v) for v in finite_values[:10]],
            }
        )
    else:
        result["sampled_first_values"] = [str(v) for v in raw[:10]]
    return result


def keyword_hits(attrs: dict[str, Any]) -> dict[str, Any]:
    hits: dict[str, Any] = {}
    for key, value in attrs.items():
        text = f"{key} {value}".lower()
        if any(word in text for word in KEYWORDS):
            hits[str(key)] = to_jsonable(value)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only metadata audit for BCCMASS_train.nc and embedded emission variables."
    )
    parser.add_argument("--nc", required=True, help="Path to BCCMASS_train.nc")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for JSON and TXT reports (default: current directory)",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    report: dict[str, Any] = {
        "audit_name": "bccmass_nc_metadata_gate1",
        "read_only": True,
        "file": {
            "path": str(nc_path),
            "size_bytes": int(nc_path.stat().st_size),
            "size_gib": round(nc_path.stat().st_size / (1024**3), 4),
            "modified_time": nc_path.stat().st_mtime,
        },
    }

    with xr.open_dataset(nc_path, decode_times=False, mask_and_scale=False) as ds:
        report["dataset"] = {
            "dimensions": {str(k): int(v) for k, v in ds.sizes.items()},
            "coordinate_names": list(ds.coords),
            "data_variable_names": list(ds.data_vars),
            "global_attributes": to_jsonable(dict(ds.attrs)),
            "global_provenance_keyword_hits": keyword_hits(dict(ds.attrs)),
        }

        variables: dict[str, Any] = {}
        missing: list[str] = []
        for name in FOCUS_VARS:
            if name not in ds.variables:
                missing.append(name)
                continue
            da = ds[name]
            variables[name] = {
                "dims": list(da.dims),
                "shape": [int(x) for x in da.shape],
                "dtype": str(da.dtype),
                "attributes": to_jsonable(dict(da.attrs)),
                "provenance_keyword_hits": keyword_hits(dict(da.attrs)),
                "encoding_selected": {
                    key: to_jsonable(da.encoding.get(key))
                    for key in ("source", "original_shape", "dtype", "_FillValue", "scale_factor", "add_offset")
                    if key in da.encoding
                },
                "deterministic_sample": finite_sample(da),
            }

        report["focus_variables"] = variables
        report["missing_focus_variables"] = missing
        report["all_emission_variables_present"] = all(v in ds.variables for v in EMISSION_VARS)

        if "DOY" in ds.variables:
            doy = np.asarray(ds["DOY"].values).reshape(-1)
            finite_doy = doy[np.isfinite(doy)] if np.issubdtype(doy.dtype, np.number) else doy
            unique, counts = np.unique(finite_doy, return_counts=True)
            report["doy_summary"] = {
                "n_samples": int(doy.size),
                "n_unique_days": int(unique.size),
                "min_day": to_jsonable(unique.min()) if unique.size else None,
                "max_day": to_jsonable(unique.max()) if unique.size else None,
                "first_20_day_counts": [
                    {"doy": to_jsonable(day), "count": int(count)}
                    for day, count in zip(unique[:20], counts[:20])
                ],
                "last_20_day_counts": [
                    {"doy": to_jsonable(day), "count": int(count)}
                    for day, count in zip(unique[-20:], counts[-20:])
                ],
            }

        common_dims = {}
        if all(v in ds.variables for v in EMISSION_VARS):
            common_dims = {v: list(ds[v].dims) for v in EMISSION_VARS}
        report["emission_dimension_map"] = common_dims

    json_path = output_dir / "bccmass_nc_metadata_gate1.json"
    txt_path = output_dir / "bccmass_nc_metadata_gate1.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    lines = [
        "BCCMASS NetCDF metadata audit - Gate 1",
        f"File: {report['file']['path']}",
        f"Size GiB: {report['file']['size_gib']}",
        f"Dimensions: {report['dataset']['dimensions']}",
        f"Coordinates: {report['dataset']['coordinate_names']}",
        f"All 8 emission variables present: {report['all_emission_variables_present']}",
        f"Missing focus variables: {report['missing_focus_variables']}",
        f"Global provenance hits: {report['dataset']['global_provenance_keyword_hits']}",
    ]
    if "doy_summary" in report:
        lines.append(f"DOY summary: {report['doy_summary']}")
    lines.append("")
    lines.append("Focus variables:")
    for name, info in report["focus_variables"].items():
        lines.append(
            f"- {name}: dims={info['dims']}, shape={info['shape']}, dtype={info['dtype']}, "
            f"attrs={info['attributes']}, sample={info['deterministic_sample']}"
        )

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("GATE1_METADATA_AUDIT_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"ALL_8_EMISSION_VARS_PRESENT={report['all_emission_variables_present']}")
    print(f"MISSING_FOCUS_VARS={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
