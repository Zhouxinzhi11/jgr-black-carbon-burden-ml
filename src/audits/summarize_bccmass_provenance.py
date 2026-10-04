#!/usr/bin/env python3
# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""Create reviewer-auditable BCCMASS provenance summaries from an existing audit.

This script is read-only with respect to the fused and MERRA-2 source data.  It
uses the JSON and sample-level CSV produced by audit_merra2_bccmass_fingerprint.py
and writes a compact JSON/CSV/TXT summary plus manuscript-ready Markdown text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = (0.01, 0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--samples-csv",
        help=(
            "Sample CSV from the fingerprint audit. If omitted, the script "
            "looks beside the JSON for merra2_bccmass_fingerprint_samples.csv."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise TypeError("The report JSON must contain an object at its root.")
    return obj


def read_pairs(path: Path, candidate: str) -> tuple[np.ndarray, np.ndarray]:
    fused: list[float] = []
    predicted: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"fused_bccmass", candidate}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Sample CSV lacks columns: {sorted(missing)}")
        for row in reader:
            y = float(row["fused_bccmass"])
            p = float(row[candidate])
            if math.isfinite(y) and math.isfinite(p):
                fused.append(y)
                predicted.append(p)
    if not fused:
        raise RuntimeError("No finite fused/candidate pairs were found.")
    return np.asarray(fused, dtype=np.float64), np.asarray(predicted, dtype=np.float64)


def recompute(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    error = p - y
    rms_y = float(np.sqrt(np.mean(y**2)))
    floor = max(float(np.median(np.abs(y))) * 1.0e-6, 1.0e-30)
    symmetric_relative = np.abs(error) / np.maximum(
        np.maximum(np.abs(y), np.abs(p)), floor
    )
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    threshold_rows = []
    for threshold in THRESHOLDS:
        count = int(np.count_nonzero(symmetric_relative <= threshold))
        threshold_rows.append(
            {
                "threshold_fraction": threshold,
                "threshold_percent": 100.0 * threshold,
                "count_within": count,
                "n": int(y.size),
                "consistency_rate": count / int(y.size),
                "consistency_percent": 100.0 * count / int(y.size),
            }
        )
    return {
        "n": int(y.size),
        "normalized_rmse_by_rms_fused": float(np.sqrt(np.mean(error**2)) / rms_y),
        "median_symmetric_relative_error": float(np.median(symmetric_relative)),
        "p95_symmetric_relative_error": float(np.quantile(symmetric_relative, 0.95)),
        "correlation": float(np.corrcoef(y, p)[0, 1]),
        "r2_against_identity": float(1.0 - sse / sst),
        "through_origin_scale_candidate_to_fused": float(np.sum(y * p) / np.sum(p**2)),
        "threshold_consistency": threshold_rows,
    }


def require_close(name: str, expected: float, actual: float) -> None:
    if not math.isclose(expected, actual, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(
            f"Metric reproduction failed for {name}: report={expected}, recomputed={actual}"
        )


def classify(metrics: dict[str, Any]) -> dict[str, Any]:
    median = float(metrics["median_symmetric_relative_error"])
    p95 = float(metrics["p95_symmetric_relative_error"])
    if median <= 0.0001 and p95 <= 0.001:
        label = "near_exact"
    elif median <= 0.01 and p95 <= 0.05:
        label = "strong_approximate"
    elif median <= 0.05 and p95 <= 0.20:
        label = "broadly_consistent"
    else:
        label = "not_close_under_prespecified_rules"
    return {
        "classification": label,
        "prespecified_rules": {
            "near_exact": "median symmetric relative error <=0.01% and p95 <=0.1%",
            "strong_approximate": "median <=1% and p95 <=5%",
            "broadly_consistent": "median <=5% and p95 <=20%",
        },
    }


def write_outputs(
    output_dir: Path,
    report_path: Path,
    samples_path: Path,
    report: dict[str, Any],
    candidate: str,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    sampling = report["sampling"]
    analysis = report["analysis"]
    classification = classify(metrics)
    dates = list(sampling["selected_dates"])
    rate5 = next(
        row for row in metrics["threshold_consistency"]
        if math.isclose(row["threshold_fraction"], 0.05)
    )
    summary = {
        "artifact_name": "bccmass_provenance_si_summary",
        "read_only": True,
        "inputs": {
            "report_json": str(report_path.resolve()),
            "samples_csv": str(samples_path.resolve()),
        },
        "sampling": {
            "strategy": sampling["strategy"],
            "seed": sampling["seed"],
            "requested_samples": sampling["requested_samples"],
            "compared_samples": sampling["compared_samples"],
            "selected_dates": dates,
            "rationale": (
                "Ten dates were selected at approximately even positions across the "
                "365 overlapping 2023 dates; seeded uniform row priorities selected "
                "10 fused records within each date."
            ),
        },
        "matching": {
            "best_candidate": candidate,
            "spatial_method": (
                "Stored fused coordinates were matched to the nearest native MERRA-2 "
                "grid coordinate; aligned, north/south-mirrored, and all fixed +/-1-cell "
                "offsets were compared."
            ),
            "observed_max_coordinate_difference_degrees": analysis["coordinate_summary"],
            "temporal_method": (
                "Fractional DOY supplied within-day time; the primary candidate used "
                "linear interpolation between decoded hourly MERRA-2 record centres."
            ),
            "temporal_bracketing_interval_hours": 1.0,
            "hard_acceptance_window_used": False,
        },
        "metrics": metrics,
        "classification": classification,
        "interpretation_boundary": (
            "The fingerprint supports the source-variable identity, numerical scale, "
            "and tested spatial alignment. It does not reconstruct every historical "
            "collocation or aggregation operation and is not independent observational validation."
        ),
    }
    with (output_dir / "bccmass_provenance_si_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    with (output_dir / "bccmass_provenance_si_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "candidate", "n", "dates", "normalized_rmse_percent",
            "median_symmetric_relative_error_percent",
            "p95_symmetric_relative_error_percent", "correlation",
            "identity_line_r2", "through_origin_scale", "within_5_percent_n",
            "within_5_percent_rate_percent", "classification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "candidate": candidate,
                "n": metrics["n"],
                "dates": len(dates),
                "normalized_rmse_percent": 100 * metrics["normalized_rmse_by_rms_fused"],
                "median_symmetric_relative_error_percent": 100 * metrics["median_symmetric_relative_error"],
                "p95_symmetric_relative_error_percent": 100 * metrics["p95_symmetric_relative_error"],
                "correlation": metrics["correlation"],
                "identity_line_r2": metrics["r2_against_identity"],
                "through_origin_scale": metrics["through_origin_scale_candidate_to_fused"],
                "within_5_percent_n": rate5["count_within"],
                "within_5_percent_rate_percent": rate5["consistency_percent"],
                "classification": classification["classification"],
            }
        )

    main_text = (
        f"A read-only provenance audit compared {metrics['n']} fused records from "
        f"{len(dates)} temporally distributed 2023 dates with independently downloaded "
        "MERRA-2 M2T1NXAER BCCMASS fields. Fractional day of year defined within-day "
        "time, and the best tested hypothesis used the stored native grid cell with "
        "linear interpolation between adjacent hourly records. The comparison yielded "
        f"r = {metrics['correlation']:.5f}, identity-line R^2 = "
        f"{metrics['r2_against_identity']:.5f}, a median symmetric relative error of "
        f"{100 * metrics['median_symmetric_relative_error']:.2f}%, and a 95th-percentile "
        f"error of {100 * metrics['p95_symmetric_relative_error']:.2f}% (Supporting "
        "Information)."
    )
    si_text = (
        "The dates were selected at approximately even positions across the 365 dates "
        "with raw/fused overlap, and 10 records per date were selected using seeded "
        f"uniform row priorities (seed {sampling['seed']}). Stored coordinates matched "
        "the native grid exactly in the sample (maximum latitude and longitude "
        "differences 0 degrees). The audit compared aligned, north/south-mirrored, and "
        "all fixed one-cell-offset spatial candidates together with linear, nearest, "
        "lower, upper, and historical-code temporal candidates. Under the prespecified "
        "rules, broad consistency required median symmetric relative error <=5% and "
        "95th-percentile error <=20%; strong approximate consistency required <=1% "
        "and <=5%, respectively. The aligned linear candidate had normalized RMSE "
        f"{100 * metrics['normalized_rmse_by_rms_fused']:.2f}%, median error "
        f"{100 * metrics['median_symmetric_relative_error']:.2f}%, 95th-percentile "
        f"error {100 * metrics['p95_symmetric_relative_error']:.2f}%, r = "
        f"{metrics['correlation']:.5f}, identity-line R^2 = "
        f"{metrics['r2_against_identity']:.5f}, and a through-origin scale of "
        f"{metrics['through_origin_scale_candidate_to_fused']:.5f}; "
        f"{rate5['count_within']}/{rate5['n']} records "
        f"({rate5['consistency_percent']:.1f}%) were within 5% symmetric relative error. "
        "This supports the inherited field's MERRA-2 BCCMASS fingerprint and numerical "
        "scale but does not reconstruct every upstream fusion step or provide independent "
        "observational validation."
    )
    (output_dir / "bccmass_provenance_manuscript_text.md").write_text(
        "## Main-text replacement\n\n" + main_text + "\n\n"
        "## Supporting Information note\n\n" + si_text + "\n",
        encoding="utf-8",
    )
    (output_dir / "bccmass_provenance_si_summary.txt").write_text(
        f"candidate={candidate}\n"
        f"n={metrics['n']}\n"
        f"dates={len(dates)}\n"
        f"normalized_rmse={metrics['normalized_rmse_by_rms_fused']:.9f}\n"
        f"median_symmetric_relative_error={metrics['median_symmetric_relative_error']:.9f}\n"
        f"p95_symmetric_relative_error={metrics['p95_symmetric_relative_error']:.9f}\n"
        f"correlation={metrics['correlation']:.9f}\n"
        f"identity_line_r2={metrics['r2_against_identity']:.9f}\n"
        f"within_5_percent={rate5['count_within']}/{rate5['n']} "
        f"({rate5['consistency_percent']:.1f}%)\n"
        f"classification={classification['classification']}\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_json).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    samples_path = (
        Path(args.samples_csv).expanduser().resolve()
        if args.samples_csv
        else report_path.parent / "merra2_bccmass_fingerprint_samples.csv"
    )
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"Sample CSV not found: {samples_path}. Use --samples-csv to provide it."
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    report = read_json(report_path)
    for key in ("sampling", "analysis"):
        if key not in report:
            raise KeyError(f"Report JSON lacks required key: {key}")
    candidate = str(report["analysis"]["best_candidate"])
    expected = report["analysis"]["metrics"][candidate]
    y, p = read_pairs(samples_path, candidate)
    metrics = recompute(y, p)
    if int(report["sampling"]["compared_samples"]) != metrics["n"]:
        raise RuntimeError("Sample count differs between report JSON and sample CSV.")
    for key in (
        "normalized_rmse_by_rms_fused",
        "median_symmetric_relative_error",
        "p95_symmetric_relative_error",
        "correlation",
        "r2_against_identity",
        "through_origin_scale_candidate_to_fused",
    ):
        require_close(key, float(expected[key]), float(metrics[key]))

    write_outputs(output_dir, report_path, samples_path, report, candidate, metrics)
    print("BCCMASS_PROVENANCE_SI_SUMMARY_PASS")
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
