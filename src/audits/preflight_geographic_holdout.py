# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
81_preflight_meanstate_spatial_holdout_gateS9A.py

Gate S9A read-only feasibility audit for one minimal spatially independent
mean-state validation design.

Purpose
-------
Do NOT train. Do NOT evaluate predictive performance.

This audit asks whether a predefined geographic holdout can be implemented
without contaminating training patches or checkpoint-selection targets.

Primary predefined holdout:
    East Asia: 20-55 N, 100-145 E
This is the same fixed region already used in the existing formal evaluator
(52/62), so the region is not chosen after seeing Gate-S9 model performance.

Geometry contract
-----------------
Grid:
    361 x 576
    latitude  -90 to 90 at 0.5 deg
    longitude -180 to 179.375 at 0.625 deg

Core holdout:
    region used only for final spatial test metrics.

Guard band:
    32 grid cells around the core holdout.
    The current U-Net has a finite local inference receptive field substantially
    smaller than this guard; 32 is deliberately conservative.

Training exclusion:
    core + 32-cell guard.
    A 128x128 training crop is eligible only if its full footprint has ZERO
    overlap with the training-exclusion mask. This is especially important
    because BatchNorm during training couples spatial positions within a crop.

Checkpoint-selection safe mask:
    original validation mask AND outside core + 64-cell exclusion.
    This is a stricter double-guard rule so model-selection targets are well
    separated from the final holdout.

Candidate regions are audited only as diagnostics:
    East_Asia, Europe, North_America, South_Asia, Africa.
The script does NOT automatically switch the primary region based on coverage.

Scientific boundary
-------------------
Passing S9A establishes designability only. It does not establish spatial
generalization and produces no manuscript performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


H = 361
W = 576
CROP_SIZE = 128
GUARD_CELLS = 32
SELECTION_GUARD_CELLS = 64

EXPECTED_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_BLACKLIST_MONTHS = [7]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
EXPECTED_SUPPORT_DEFINITION = "any(abs(emission_log_mean) > 0.0, dim=0)"

REGIONS = {
    "East_Asia": {"lat": (20.0, 55.0), "lon": (100.0, 145.0)},
    "Europe": {"lat": (35.0, 72.0), "lon": (-10.0, 40.0)},
    "North_America": {"lat": (15.0, 75.0), "lon": (-170.0, -50.0)},
    "South_Asia": {"lat": (5.0, 35.0), "lon": (65.0, 100.0)},
    "Africa": {"lat": (-35.0, 37.0), "lon": (-20.0, 55.0)},
}
PRIMARY_REGION = "East_Asia"

# Predeclared engineering feasibility thresholds.
MIN_ELIGIBLE_PATCH_FRACTION = 0.20
MIN_CORE_VALID_PIXELS_PER_VAL_MONTH = 500
MIN_SELECTION_VALID_PIXELS_PER_VAL_MONTH = 10000
MIN_TRAIN_VALID_PIXELS_OUTSIDE_EXCLUSION_PER_MONTH = 10000

LAT = np.linspace(-90.0, 90.0, H, dtype=np.float32)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float32)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    require(isinstance(obj, dict), f"PT_ROOT_NOT_DICT: {path}")
    return obj


def region_mask(region_name: str) -> np.ndarray:
    cfg = REGIONS[region_name]
    lat0, lat1 = cfg["lat"]
    lon0, lon1 = cfg["lon"]
    return (
        (LAT_GRID >= lat0)
        & (LAT_GRID <= lat1)
        & (LON_GRID >= lon0)
        & (LON_GRID <= lon1)
    )


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    require(len(ys) > 0, "EMPTY_REGION_MASK")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def dilate_rect_mask(mask: np.ndarray, cells: int) -> np.ndarray:
    y0, y1, x0, x1 = bbox_from_mask(mask)
    y0 = max(0, y0 - cells)
    y1 = min(H - 1, y1 + cells)
    x0 = max(0, x0 - cells)
    x1 = min(W - 1, x1 + cells)
    out = np.zeros((H, W), dtype=bool)
    out[y0 : y1 + 1, x0 : x1 + 1] = True
    return out


def integral_image(mask: np.ndarray) -> np.ndarray:
    # One-cell padded integral image for O(1) rectangular sums.
    arr = mask.astype(np.int64, copy=False)
    return np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def rect_sum(ii: np.ndarray, top: int, left: int, height: int, width: int) -> int:
    bottom = top + height
    right = left + width
    return int(ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left])


def eligible_patch_geometry(exclusion_mask: np.ndarray) -> dict[str, Any]:
    ii = integral_image(exclusion_mask)
    max_top = H - CROP_SIZE
    max_left = W - CROP_SIZE
    total = (max_top + 1) * (max_left + 1)
    eligible = 0
    row_counts: list[int] = []
    for top in range(max_top + 1):
        count = 0
        for left in range(max_left + 1):
            if rect_sum(ii, top, left, CROP_SIZE, CROP_SIZE) == 0:
                eligible += 1
                count += 1
        row_counts.append(count)
    fraction = eligible / total
    return {
        "crop_size": CROP_SIZE,
        "total_top_left_positions": total,
        "eligible_top_left_positions": eligible,
        "eligible_fraction": fraction,
        "rows_with_at_least_one_eligible_position": int(sum(v > 0 for v in row_counts)),
    }


def find_month_entry(
    manifest: dict[str, Any],
    *,
    year: int,
    month: int,
    split: str,
) -> dict[str, Any]:
    # Support the known monthly_manifest layouts without silently guessing.
    candidates: list[dict[str, Any]] = []
    for key in ["monthly_entries", "entries", "rows", "months"]:
        value = manifest.get(key)
        if isinstance(value, list):
            candidates.extend(x for x in value if isinstance(x, dict))
    if not candidates:
        # Fall back to recursive search over JSON dictionaries/lists.
        stack: list[Any] = [manifest]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if all(k in node for k in ["year", "month"]) and "file" in node:
                    candidates.append(node)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    matches = []
    for row in candidates:
        try:
            same = (
                int(row.get("year")) == year
                and int(row.get("month")) == month
            )
        except Exception:
            continue
        if not same:
            continue
        row_split = row.get("split")
        if row_split is not None and str(row_split) != split:
            continue
        matches.append(row)

    # Deduplicate by file.
    unique = {}
    for row in matches:
        unique[str(row.get("file"))] = row
    matches = list(unique.values())
    require(
        len(matches) == 1,
        f"MONTH_ENTRY_RESOLUTION_FAILED year={year} month={month} split={split} n={len(matches)}",
    )
    return matches[0]


def resolve_payload_path(cache_dir: Path, row: dict[str, Any]) -> Path:
    raw = Path(str(row["file"]))
    if raw.is_file():
        return raw.resolve()
    if not raw.is_absolute():
        candidate = (cache_dir / raw).resolve()
        if candidate.is_file():
            return candidate
    # Known cache layout recovery using basename.
    basename = raw.name
    for candidate in cache_dir.rglob(basename):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"MONTHLY_PAYLOAD_NOT_FOUND: {raw}")


def payload_diagnostics(
    payload: dict[str, Any],
    *,
    core: np.ndarray,
    train_exclusion: np.ndarray,
    selection_exclusion: np.ndarray,
) -> dict[str, Any]:
    valid_t = payload.get("valid_mask")
    emission_t = payload.get("emission_log_mean")
    require(isinstance(valid_t, torch.Tensor), "VALID_MASK_MISSING")
    require(tuple(valid_t.shape) == (H, W), f"VALID_MASK_SHAPE_INVALID: {tuple(valid_t.shape)}")
    valid = valid_t.detach().cpu().float().numpy() > 0.5

    support = None
    if isinstance(emission_t, torch.Tensor):
        require(tuple(emission_t.shape) == (8, H, W), "EMISSION_SHAPE_INVALID")
        emission = emission_t.detach().cpu().float()
        require(bool(torch.isfinite(emission).all().item()), "EMISSION_NONFINITE")
        support = torch.any(torch.abs(emission) > 0.0, dim=0).numpy()

    core_valid = valid & core
    train_valid = valid & (~train_exclusion)
    selection_valid = valid & (~selection_exclusion)

    out = {
        "total_valid_pixels": int(valid.sum()),
        "core_valid_pixels": int(core_valid.sum()),
        "train_valid_pixels_outside_exclusion": int(train_valid.sum()),
        "selection_valid_pixels_outside_double_guard": int(selection_valid.sum()),
    }
    if support is not None:
        n = int(core_valid.sum())
        inside = int((core_valid & support).sum())
        out["core_support_inside_valid_pixels"] = inside
        out["core_support_fraction_within_valid"] = (inside / n) if n > 0 else None
    return out


def audit_region(
    *,
    region_name: str,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    core = region_mask(region_name)
    train_exclusion = dilate_rect_mask(core, GUARD_CELLS)
    selection_exclusion = dilate_rect_mask(core, SELECTION_GUARD_CELLS)

    patch_diag = eligible_patch_geometry(train_exclusion)
    y0, y1, x0, x1 = bbox_from_mask(core)
    gy0, gy1, gx0, gx1 = bbox_from_mask(train_exclusion)
    sy0, sy1, sx0, sx1 = bbox_from_mask(selection_exclusion)

    rows: list[dict[str, Any]] = []

    for month in EXPECTED_TRAIN_MONTHS:
        entry = find_month_entry(manifest, year=2022, month=month, split="2022_full")
        path = resolve_payload_path(cache_dir, entry)
        payload = load_pt(path)
        diag = payload_diagnostics(
            payload,
            core=core,
            train_exclusion=train_exclusion,
            selection_exclusion=selection_exclusion,
        )
        rows.append(
            {
                "role": "train",
                "year": 2022,
                "month": month,
                "file": str(path),
                **diag,
            }
        )

    for month in EXPECTED_VAL_MONTHS:
        entry = find_month_entry(manifest, year=2022, month=month, split="2022_full")
        path = resolve_payload_path(cache_dir, entry)
        payload = load_pt(path)
        diag = payload_diagnostics(
            payload,
            core=core,
            train_exclusion=train_exclusion,
            selection_exclusion=selection_exclusion,
        )
        rows.append(
            {
                "role": "model_selection_and_spatial_test",
                "year": 2022,
                "month": month,
                "file": str(path),
                **diag,
            }
        )

    checks = {
        "eligible_patch_fraction_pass": (
            patch_diag["eligible_fraction"] >= MIN_ELIGIBLE_PATCH_FRACTION
        ),
        "all_train_months_have_sufficient_valid_pixels_outside_exclusion": all(
            row["train_valid_pixels_outside_exclusion"]
            >= MIN_TRAIN_VALID_PIXELS_OUTSIDE_EXCLUSION_PER_MONTH
            for row in rows
            if row["role"] == "train"
        ),
        "both_validation_months_have_sufficient_core_test_pixels": all(
            row["core_valid_pixels"] >= MIN_CORE_VALID_PIXELS_PER_VAL_MONTH
            for row in rows
            if row["role"] == "model_selection_and_spatial_test"
        ),
        "both_validation_months_have_sufficient_safe_selection_pixels": all(
            row["selection_valid_pixels_outside_double_guard"]
            >= MIN_SELECTION_VALID_PIXELS_PER_VAL_MONTH
            for row in rows
            if row["role"] == "model_selection_and_spatial_test"
        ),
    }
    checks["overall_designable"] = all(checks.values())

    return {
        "region": region_name,
        "is_primary_predeclared_region": region_name == PRIMARY_REGION,
        "region_definition": REGIONS[region_name],
        "core_bbox_indices": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "training_exclusion_bbox_indices": {"y0": gy0, "y1": gy1, "x0": gx0, "x1": gx1},
        "selection_exclusion_bbox_indices": {"y0": sy0, "y1": sy1, "x0": sx0, "x1": sx1},
        "core_grid_cells": int(core.sum()),
        "training_exclusion_grid_cells": int(train_exclusion.sum()),
        "selection_exclusion_grid_cells": int(selection_exclusion.sum()),
        "patch_geometry": patch_diag,
        "monthly_rows": rows,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S9A read-only spatial-holdout feasibility audit."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--monthly-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    manifest_path = Path(args.monthly_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(manifest_path.is_file(), f"MONTHLY_MANIFEST_NOT_FOUND: {manifest_path}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    manifest_hash = sha256_file(manifest_path)
    require(
        manifest_hash == EXPECTED_MANIFEST_SHA256,
        f"MONTHLY_MANIFEST_SHA256_MISMATCH: {manifest_hash}",
    )
    manifest = load_json(manifest_path)

    results = {
        name: audit_region(
            region_name=name,
            cache_dir=cache_dir,
            manifest=manifest,
        )
        for name in REGIONS
    }

    primary = results[PRIMARY_REGION]
    status = (
        "DESIGNABLE_PRIMARY_REGION"
        if primary["checks"]["overall_designable"]
        else "PRIMARY_REGION_NOT_DESIGNABLE_STOP_BEFORE_TRAINING"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "meanstate_spatial_holdout_preflight_gateS9A.json"
    txt_path = output_dir / "meanstate_spatial_holdout_preflight_gateS9A.txt"

    report = {
        "artifact_name": "s2_meanstate_spatial_holdout_preflight_gateS9A",
        "artifact_version": 1,
        "status": status,
        "inputs": {
            "cache_dir": str(cache_dir),
            "monthly_manifest": str(manifest_path),
            "monthly_manifest_sha256": manifest_hash,
        },
        "design_contract": {
            "primary_region": PRIMARY_REGION,
            "primary_region_definition": REGIONS[PRIMARY_REGION],
            "candidate_regions_diagnostic_only": [
                name for name in REGIONS if name != PRIMARY_REGION
            ],
            "grid_shape": [H, W],
            "crop_size": CROP_SIZE,
            "guard_cells": GUARD_CELLS,
            "selection_guard_cells": SELECTION_GUARD_CELLS,
            "training_crop_rule": (
                "A training crop is eligible only if its complete 128x128 footprint "
                "does not intersect the core holdout plus 32-cell guard."
            ),
            "checkpoint_selection_rule": (
                "November-December 2022 checkpoint-selection metrics must use only "
                "valid pixels outside the core holdout plus 64-cell double guard."
            ),
            "final_spatial_test_rule": (
                "Final spatial test metrics are computed only inside the predefined "
                "core holdout, which is never used for training loss or checkpoint selection."
            ),
            "planned_core_models": [
                "mean_meteo_clim_only",
                "mean_binary_support_meteo_clim",
                "mean_emis_meteo_clim",
            ],
            "planned_seeds": [42, 43, 44],
            "support_definition": EXPECTED_SUPPORT_DEFINITION,
        },
        "feasibility_thresholds": {
            "min_eligible_patch_fraction": MIN_ELIGIBLE_PATCH_FRACTION,
            "min_core_valid_pixels_per_validation_month": MIN_CORE_VALID_PIXELS_PER_VAL_MONTH,
            "min_selection_valid_pixels_per_validation_month": MIN_SELECTION_VALID_PIXELS_PER_VAL_MONTH,
            "min_train_valid_pixels_outside_exclusion_per_month": MIN_TRAIN_VALID_PIXELS_OUTSIDE_EXCLUSION_PER_MONTH,
        },
        "region_results": results,
        "verification": {
            "monthly_manifest_hash_exact": True,
            "primary_region_predeclared_before_spatial_training": True,
            "no_predictive_performance_used_for_region_selection": True,
            "no_training_performed": True,
            "primary_region_designable": primary["checks"]["overall_designable"],
            "overall_pass": primary["checks"]["overall_designable"],
        },
        "scientific_boundary": (
            "This audit evaluates only geometry, target-mask coverage, processed-prior "
            "support coverage and training-patch feasibility. It does not evaluate model "
            "skill and does not justify changing the primary holdout based on future "
            "performance. Passing this gate is not evidence of spatial generalization."
        ),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    lines = [
        "GATE_S9A_MEANSTATE_SPATIAL_HOLDOUT_PREFLIGHT",
        f"STATUS={status}",
        f"PRIMARY_REGION={PRIMARY_REGION}",
        f"PRIMARY_REGION_LAT={REGIONS[PRIMARY_REGION]['lat']}",
        f"PRIMARY_REGION_LON={REGIONS[PRIMARY_REGION]['lon']}",
        f"CROP_SIZE={CROP_SIZE}",
        f"GUARD_CELLS={GUARD_CELLS}",
        f"SELECTION_GUARD_CELLS={SELECTION_GUARD_CELLS}",
        f"PRIMARY_ELIGIBLE_PATCH_FRACTION={primary['patch_geometry']['eligible_fraction']:.8f}",
        f"PRIMARY_CORE_GRID_CELLS={primary['core_grid_cells']}",
        f"PRIMARY_TRAINING_EXCLUSION_GRID_CELLS={primary['training_exclusion_grid_cells']}",
        f"PRIMARY_SELECTION_EXCLUSION_GRID_CELLS={primary['selection_exclusion_grid_cells']}",
    ]

    for row in primary["monthly_rows"]:
        if row["role"] == "train":
            lines.append(
                f"TRAIN_2022_{row['month']:02d}_VALID_OUTSIDE_EXCLUSION="
                f"{row['train_valid_pixels_outside_exclusion']}"
            )
        else:
            lines.append(
                f"VAL_2022_{row['month']:02d}_CORE_TEST_VALID={row['core_valid_pixels']}"
            )
            lines.append(
                f"VAL_2022_{row['month']:02d}_SAFE_SELECTION_VALID="
                f"{row['selection_valid_pixels_outside_double_guard']}"
            )
            sf = row.get("core_support_fraction_within_valid")
            if sf is not None:
                lines.append(
                    f"VAL_2022_{row['month']:02d}_CORE_SUPPORT_FRACTION={sf:.8f}"
                )

    for key, value in primary["checks"].items():
        lines.append(f"{key.upper()}={value}")
    lines.extend(
        [
            "PREDICTIVE_PERFORMANCE_USED_FOR_REGION_SELECTION=False",
            "SCIENTIFIC_TRAINING_STARTED=False",
            f"OUTPUT_JSON={json_path}",
            f"OUTPUT_TXT={txt_path}",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not primary["checks"]["overall_designable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
