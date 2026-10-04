# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

# Load torch first and restrict its CPU thread pools. The audit below avoids
# NumPy BLAS/LAPACK calls to prevent libomp/libiomp runtime conflicts.
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import numpy as np


SECTORS = [
    "AGRICULTURE",
    "TRANSPORT",
    "IND_COMBUSTION",
    "POWER_INDUSTRYE",
    "BUILDINGS",
    "FUEL_EXPLOITATION",
    "IND_PROCESSES",
    "WASTE_arr",
]

H = 361
W = 576


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def manual_l2(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(math.sqrt(float(np.sum(x * x, dtype=np.float64))))


def safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    mask = mask & np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return None

    xx = x[mask].astype(np.float64, copy=False)
    yy = y[mask].astype(np.float64, copy=False)
    xx = xx - float(np.mean(xx))
    yy = yy - float(np.mean(yy))

    numerator = float(np.sum(xx * yy, dtype=np.float64))
    denom_x = float(np.sum(xx * xx, dtype=np.float64))
    denom_y = float(np.sum(yy * yy, dtype=np.float64))
    denominator = math.sqrt(max(denom_x * denom_y, 0.0))

    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    union = a | b
    n_union = int(union.sum())
    if n_union == 0:
        return None
    return float((a & b).sum() / n_union)


def cv(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-30:
        return None
    return float(np.std(arr, ddof=1) / abs(mean))


def summarize_values(values: list[float | None]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def month_from_name(path: Path) -> int:
    match = re.search(r"mean_\d{4}_(\d{2})_", path.name)
    if not match:
        raise ValueError(f"Cannot parse month from filename: {path.name}")
    return int(match.group(1))


def provisional_pattern_description(
    median_corr: float | None,
    min_corr: float | None,
    median_common_corr: float | None,
    median_jaccard: float | None,
) -> str:
    # Transparent descriptive heuristic only; not a statistical test.
    if None in (
        median_corr,
        min_corr,
        median_common_corr,
        median_jaccard,
    ):
        return "insufficient_for_pattern_description"

    if (
        median_corr >= 0.95
        and min_corr >= 0.85
        and median_common_corr >= 0.95
    ):
        if median_jaccard >= 0.90:
            return "highly_stable_spatial_pattern"
        return "stable_pattern_with_changing_support"

    if median_corr >= 0.80 and median_common_corr >= 0.85:
        return "moderately_stable_pattern"

    return "substantial_month_to_month_pattern_variation"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, BLAS-free audit of the monthly emission_log_mean "
            "tensors used by the phase-2 mean-state models."
        )
    )
    parser.add_argument("--monthly-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exclude-months",
        default="7",
        help="Comma-separated months excluded from core comparison; default 7",
    )
    parser.add_argument(
        "--nonzero-threshold",
        type=float,
        default=1e-8,
    )
    args = parser.parse_args()

    monthly_dir = Path(args.monthly_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not monthly_dir.is_dir():
        raise FileNotFoundError(f"Monthly directory not found: {monthly_dir}")

    exclude_months = {
        int(part.strip())
        for part in args.exclude_months.split(",")
        if part.strip()
    }

    files = sorted(monthly_dir.glob("mean_2022_*_2022_full.pt"))
    if not files:
        raise FileNotFoundError(
            f"No mean_2022_MM_2022_full.pt files found in {monthly_dir}"
        )

    month_records: list[dict[str, Any]] = []
    emission_by_month: dict[int, np.ndarray] = {}
    core_months: list[int] = []

    for path in files:
        print(f"LOADING={path.name}", flush=True)
        month = month_from_name(path)
        payload = load_pt(path)

        required = [
            "emission_log_mean",
            "obs_count_map",
            "valid_mask",
            "n_days",
            "n_days_with_emission",
            "emission_available",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{path.name} missing keys: {missing}")

        emission = tensor_np(payload["emission_log_mean"])
        obs_count = tensor_np(payload["obs_count_map"])
        valid_mask = tensor_np(payload["valid_mask"]) > 0.5

        if emission.shape != (8, H, W):
            raise ValueError(
                f"{path.name}: emission_log_mean shape={emission.shape}"
            )
        if obs_count.shape != (H, W):
            raise ValueError(
                f"{path.name}: obs_count_map shape={obs_count.shape}"
            )
        if valid_mask.shape != (H, W):
            raise ValueError(
                f"{path.name}: valid_mask shape={valid_mask.shape}"
            )
        if not np.isfinite(emission).all():
            raise ValueError(
                f"{path.name}: emission_log_mean contains NaN/Inf"
            )

        emission_by_month[month] = emission

        total_emission = np.sum(emission, axis=0, dtype=np.float64)
        emission_support = total_emission > args.nonzero_threshold
        obs_support = obs_count > 0

        month_records.append(
            {
                "file": str(path),
                "month": month,
                "excluded_from_core": month in exclude_months,
                "n_days": int(payload["n_days"]),
                "n_days_with_emission": int(
                    payload["n_days_with_emission"]
                ),
                "emission_available": bool(
                    payload["emission_available"]
                ),
                "valid_pixel_fraction": float(np.mean(valid_mask)),
                "obs_support_fraction": float(np.mean(obs_support)),
                "emission_support_fraction": float(
                    np.mean(emission_support)
                ),
                "emission_obs_support_jaccard": jaccard(
                    emission_support,
                    obs_support,
                ),
                "emission_valid_mask_jaccard": jaccard(
                    emission_support,
                    valid_mask,
                ),
                "total_emission_obs_count_corr_all_grid": safe_corr(
                    total_emission.ravel(),
                    obs_count.ravel(),
                    np.ones(H * W, dtype=bool),
                ),
                "total_emission_sum": float(
                    np.sum(total_emission, dtype=np.float64)
                ),
                "total_emission_l2": manual_l2(total_emission),
            }
        )

        if month not in exclude_months:
            core_months.append(month)

        del payload
        del obs_count
        del valid_mask
        del total_emission
        del emission_support
        del obs_support

    core_months = sorted(core_months)
    if len(core_months) < 2:
        raise RuntimeError(
            f"Need at least two core months, got {core_months}"
        )

    pairwise_rows: list[dict[str, Any]] = []
    sector_summaries: dict[str, Any] = {}

    for sector_index, sector_name in enumerate(SECTORS):
        print(f"ANALYZING_SECTOR={sector_name}", flush=True)
        fields = {
            month: emission_by_month[month][sector_index]
            for month in core_months
        }

        pair_union_corrs: list[float | None] = []
        pair_common_corrs: list[float | None] = []
        pair_jaccards: list[float | None] = []
        pair_nrmse: list[float | None] = []

        monthly_sums: list[float] = []
        monthly_l2s: list[float] = []

        for month in core_months:
            field = fields[month]
            monthly_sums.append(
                float(np.sum(field, dtype=np.float64))
            )
            monthly_l2s.append(manual_l2(field))

        for i, month_a in enumerate(core_months):
            field_a = fields[month_a]
            support_a = field_a > args.nonzero_threshold

            for month_b in core_months[i + 1 :]:
                field_b = fields[month_b]
                support_b = field_b > args.nonzero_threshold

                union = support_a | support_b
                common = support_a & support_b

                corr_union = safe_corr(
                    field_a.ravel(),
                    field_b.ravel(),
                    union.ravel(),
                )
                corr_common = safe_corr(
                    field_a.ravel(),
                    field_b.ravel(),
                    common.ravel(),
                )
                jac = jaccard(support_a, support_b)

                n_union = int(union.sum())
                if n_union > 0:
                    diff = field_a[union] - field_b[union]
                    rmse = math.sqrt(
                        float(
                            np.mean(
                                diff * diff,
                                dtype=np.float64,
                            )
                        )
                    )
                    rms_a = math.sqrt(
                        float(
                            np.mean(
                                field_a[union] * field_a[union],
                                dtype=np.float64,
                            )
                        )
                    )
                    rms_b = math.sqrt(
                        float(
                            np.mean(
                                field_b[union] * field_b[union],
                                dtype=np.float64,
                            )
                        )
                    )
                    scale = 0.5 * (rms_a + rms_b)
                    nrmse = float(rmse / scale) if scale > 0 else None
                else:
                    nrmse = None

                pair_union_corrs.append(corr_union)
                pair_common_corrs.append(corr_common)
                pair_jaccards.append(jac)
                pair_nrmse.append(nrmse)

                pairwise_rows.append(
                    {
                        "sector": sector_name,
                        "month_a": month_a,
                        "month_b": month_b,
                        "union_support_pixels": n_union,
                        "common_support_pixels": int(common.sum()),
                        "support_jaccard": jac,
                        "pattern_corr_union_support": corr_union,
                        "pattern_corr_common_support": corr_common,
                        "normalized_rmse_union_support": nrmse,
                    }
                )

        union_summary = summarize_values(pair_union_corrs)
        common_summary = summarize_values(pair_common_corrs)
        jaccard_summary = summarize_values(pair_jaccards)
        nrmse_summary = summarize_values(pair_nrmse)

        sector_summaries[sector_name] = {
            "core_months": core_months,
            "pairwise_pattern_corr_union": union_summary,
            "pairwise_pattern_corr_common": common_summary,
            "pairwise_support_jaccard": jaccard_summary,
            "pairwise_normalized_rmse": nrmse_summary,
            "monthly_total_sum": {
                str(month): monthly_sums[idx]
                for idx, month in enumerate(core_months)
            },
            "monthly_l2_norm": {
                str(month): monthly_l2s[idx]
                for idx, month in enumerate(core_months)
            },
            "monthly_total_sum_cv": cv(monthly_sums),
            "monthly_l2_norm_cv": cv(monthly_l2s),
            "provisional_pattern_description": (
                provisional_pattern_description(
                    union_summary["median"],
                    union_summary["min"],
                    common_summary["median"],
                    jaccard_summary["median"],
                )
            ),
        }

    core_month_records = [
        record
        for record in month_records
        if not record["excluded_from_core"]
    ]

    report = {
        "audit_name": (
            "meanstate_monthly_emission_input_audit_gate4a_blas_free"
        ),
        "read_only": True,
        "blas_free": True,
        "monthly_dir": str(monthly_dir),
        "files_found": len(files),
        "all_months_found": sorted(emission_by_month),
        "core_months": core_months,
        "excluded_months": sorted(exclude_months),
        "nonzero_threshold": args.nonzero_threshold,
        "month_records": month_records,
        "sector_summaries": sector_summaries,
        "cross_month_sampling_imprint_summary": {
            "emission_obs_support_jaccard": summarize_values(
                [
                    record["emission_obs_support_jaccard"]
                    for record in core_month_records
                ]
            ),
            "total_emission_obs_count_corr_all_grid": (
                summarize_values(
                    [
                        record[
                            "total_emission_obs_count_corr_all_grid"
                        ]
                        for record in core_month_records
                    ]
                )
            ),
        },
        "interpretation_boundaries": {
            "purpose": (
                "Characterize the actual monthly emission_log_mean "
                "tensors entering the mean-state models."
            ),
            "does_not_prove": [
                "the EDGAR release or inventory year",
                "physical emission units",
                "the upstream fusion algorithm",
                "causal emission-to-concentration attribution",
            ],
            "decision_rule": (
                "Pattern descriptions are descriptive heuristics, "
                "not formal pass/fail statistical tests."
            ),
        },
    }

    json_path = (
        output_dir
        / "meanstate_monthly_emission_input_gate4a.json"
    )
    txt_path = (
        output_dir
        / "meanstate_monthly_emission_input_gate4a.txt"
    )
    csv_path = (
        output_dir
        / "meanstate_monthly_emission_pairwise_gate4a.csv"
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

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sector",
                "month_a",
                "month_b",
                "union_support_pixels",
                "common_support_pixels",
                "support_jaccard",
                "pattern_corr_union_support",
                "pattern_corr_common_support",
                "normalized_rmse_union_support",
            ],
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)

    lines = [
        "Mean-state monthly emission input audit - Gate 4A",
        "Implementation: BLAS-free",
        f"Directory: {monthly_dir}",
        f"Files found: {len(files)}",
        f"All months: {sorted(emission_by_month)}",
        f"Core months: {core_months}",
        f"Excluded months: {sorted(exclude_months)}",
        "",
        "Month-level diagnostics:",
    ]

    for record in month_records:
        lines.append(
            f"- month={record['month']:02d}, "
            f"n_days={record['n_days']}, "
            f"n_days_with_emission="
            f"{record['n_days_with_emission']}, "
            f"valid_fraction="
            f"{record['valid_pixel_fraction']:.6f}, "
            f"emission_support_fraction="
            f"{record['emission_support_fraction']:.6f}, "
            f"emission_obs_jaccard="
            f"{record['emission_obs_support_jaccard']}, "
            f"emission_obs_count_corr="
            f"{record['total_emission_obs_count_corr_all_grid']}"
        )

    lines.extend(
        ["", "Sector cross-month pattern diagnostics:"]
    )
    for sector, item in sector_summaries.items():
        lines.extend(
            [
                f"- {sector}",
                f"  description="
                f"{item['provisional_pattern_description']}",
                f"  union_corr_min="
                f"{item['pairwise_pattern_corr_union']['min']}",
                f"  union_corr_median="
                f"{item['pairwise_pattern_corr_union']['median']}",
                f"  common_corr_median="
                f"{item['pairwise_pattern_corr_common']['median']}",
                f"  support_jaccard_median="
                f"{item['pairwise_support_jaccard']['median']}",
                f"  total_sum_cv="
                f"{item['monthly_total_sum_cv']}",
                f"  l2_norm_cv="
                f"{item['monthly_l2_norm_cv']}",
            ]
        )

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("GATE4A_MONTHLY_EMISSION_INPUT_AUDIT_PASS")
    print(f"JSON_REPORT={json_path}")
    print(f"TXT_REPORT={txt_path}")
    print(f"CSV_REPORT={csv_path}")
    print(f"ALL_MONTHS={sorted(emission_by_month)}")
    print(f"CORE_MONTHS={core_months}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
