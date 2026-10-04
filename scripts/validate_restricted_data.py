#!/usr/bin/env python3
"""Validate the separate restricted-review data archive without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate byte-preserved restricted inputs and their public-code interface."
    )
    parser.add_argument("--restricted-root", required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check paths, counts and sizes but skip the full approximately 36 GiB SHA pass.",
    )
    args = parser.parse_args()
    root = Path(args.restricted_root).expanduser().resolve()
    require(root.is_dir(), f"RESTRICTED_ROOT_NOT_FOUND: {root}")

    manifest_path = root / "manifests" / "restricted_file_manifest.csv"
    require(manifest_path.is_file(), f"MANIFEST_NOT_FOUND: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1268, f"RESTRICTED_DATA_ROW_COUNT_MISMATCH: {len(rows)}")

    groups: Counter[str] = Counter()
    checked_bytes = 0
    for row in rows:
        path = root / row["archive_path"]
        require(path.is_file(), f"MISSING_RESTRICTED_FILE: {row['archive_path']}")
        size = path.stat().st_size
        require(size == int(row["size_bytes"]), f"SIZE_MISMATCH: {row['archive_path']}")
        if not args.quick:
            require(sha256_file(path) == row["sha256"], f"SHA256_MISMATCH: {row['archive_path']}")
        checked_bytes += size
        groups[row["object_group"]] += 1

    require(groups["formal_meanstate_analysis_ready_cache"] == 32, "MEANSTATE_FILE_COUNT_MISMATCH")
    require(groups["formal_daily_anomaly_analysis_ready_cache"] == 625, "ANOMALY_FILE_COUNT_MISMATCH")
    require(groups["public_AEF_source_cache_with_restricted_composite_tensor_content"] == 600, "AEF_SOURCE_COUNT_MISMATCH")
    require(groups["Figure_3_pointwise_target_prediction_error_grids"] == 1, "FIGURE3_RESTRICTED_SOURCE_MISSING")
    require(groups["Figure_S3_cellwise_monthly_support_grids"] == 1, "FIGURES3_RESTRICTED_SOURCE_MISSING")

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        sample_mean = root / "inputs" / "meanstate_cache" / "07_phase2_meanstate_fullmonth_trainonly_union" / "monthly_mean_full" / "2022_full" / "mean_2022_02_2022_full.pt"
        sample_anomaly = root / "inputs" / "anomaly_cache" / "08_phase2_anomaly_clean_trainonly_union" / "daily_anomaly" / "2022_train" / "day_032.pt"
        sample_source = root / "inputs" / "anomaly_aef_source_tensors" / "2022" / "day_032.pt"
        for path in (sample_mean, sample_anomaly, sample_source):
            require(path.is_file(), f"SCHEMA_SAMPLE_MISSING: {path}")
        mean = torch.load(sample_mean, map_location="cpu", weights_only=False)
        anomaly = torch.load(sample_anomaly, map_location="cpu", weights_only=False)
        source = torch.load(sample_source, map_location="cpu", weights_only=False)
        require(tuple(mean["target_log_mean_conc"].shape) == (361, 576), "MEANSTATE_TARGET_SHAPE_MISMATCH")
        require(tuple(mean["meteo_mean"].shape) == (5, 361, 576), "MEANSTATE_METEO_SHAPE_MISMATCH")
        require(tuple(mean["aef_mean"].shape) == (64, 361, 576), "MEANSTATE_AEF_SHAPE_MISMATCH")
        require(tuple(anomaly["target_log_anomaly"].shape) == (1, 361, 576), "ANOMALY_TARGET_SHAPE_MISMATCH")
        require(tuple(source["x"].shape) == (112, 361, 576), "SOURCE_X_SHAPE_MISMATCH")
        require(tuple(source["y"].shape) == (10, 361, 576), "SOURCE_Y_SHAPE_MISMATCH")

    print("RESTRICTED_DATA_VALIDATION=PASS")
    print(f"MODE={'QUICK_SIZE_ONLY' if args.quick else 'FULL_SHA256'}")
    print(f"FILES={len(rows)}")
    print(f"BYTES={checked_bytes}")
    print("FORMAL_SEEDS=42,43,44")
    print("FULL_TRAINING_EXECUTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
