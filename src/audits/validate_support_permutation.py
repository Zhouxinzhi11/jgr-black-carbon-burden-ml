# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
56_validate_meanstate_support_preserving_permutation.py

Read-only Gate P2A for the clean mean-state branch.

Purpose
-------
Validate a deterministic support-preserving spatial permutation of the
8-channel processed emission-prior vectors.

The transformation:
1. defines processed-emission support as cells where any of the eight
   channels has abs(value) > support_epsilon;
2. leaves that support mask exactly unchanged;
3. permutes complete 8-dimensional emission vectors only among supported
   cells, preserving:
   - the non-zero support mask,
   - the exact multiset of 8-D vectors,
   - cross-sector structure within each vector,
   - per-channel value distributions;
4. uses a deterministic key per monthly payload.

This gate does not train a model and does not modify any existing artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_2023_MONTHS = list(range(1, 13))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def support_preserving_permute_2d(
    emission: torch.Tensor,
    *,
    key: str,
    support_epsilon: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Permute complete 8-D emission vectors within the original support mask.

    Returns the transformed tensor and deterministic permutation diagnostics.
    """
    require(emission.ndim == 3, "EMISSION_MUST_BE_C_H_W")
    channels, height, width = emission.shape
    require(channels == 8, f"EXPECTED_8_CHANNELS_GOT_{channels}")
    require(support_epsilon >= 0.0, "SUPPORT_EPSILON_MUST_BE_NONNEGATIVE")

    x = torch.nan_to_num(
        emission.detach().cpu().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    support = torch.any(torch.abs(x) > float(support_epsilon), dim=0)
    flat_support = support.reshape(-1)
    support_indices = torch.nonzero(flat_support, as_tuple=False).squeeze(1)
    n_support = int(support_indices.numel())
    require(n_support >= 2, f"INSUFFICIENT_SUPPORT_CELLS: {n_support}")

    seed = md5_int(f"supportperm::{key}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(n_support, generator=generator)

    # Avoid the pathological identity permutation.
    if bool(torch.equal(permutation, torch.arange(n_support))):
        permutation = torch.roll(permutation, shifts=1)

    flat = x.reshape(channels, -1)
    source_vectors = flat[:, support_indices].clone()
    transformed = flat.clone()
    transformed[:, support_indices] = source_vectors[:, permutation]
    transformed = transformed.reshape(channels, height, width)

    fixed_points = int(torch.count_nonzero(
        permutation == torch.arange(n_support)
    ).item())

    return transformed, {
        "seed": int(seed),
        "n_support_cells": n_support,
        "fixed_points": fixed_points,
        "fixed_point_fraction": float(fixed_points / n_support),
        "permutation_sha256": hashlib.sha256(
            permutation.numpy().tobytes()
        ).hexdigest(),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-epsilon", type=float, default=0.0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifests" / "monthly_manifest.json"

    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(manifest_path.is_file(), f"MANIFEST_NOT_FOUND: {manifest_path}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")
    require(args.support_epsilon >= 0.0, "SUPPORT_EPSILON_MUST_BE_NONNEGATIVE")

    manifest = load_json(manifest_path)
    entries = sorted(
        list(manifest.get("groups", [])),
        key=lambda e: (int(e["year"]), int(e["month"])),
    )
    require(len(entries) == 21, f"EXPECTED_21_GROUPS_GOT_{len(entries)}")

    entries_2022 = [e for e in entries if int(e["year"]) == 2022]
    entries_2023 = [e for e in entries if int(e["year"]) == 2023]
    require(
        sorted(int(e["month"]) for e in entries_2023) == EXPECTED_2023_MONTHS,
        "2023_MONTH_COVERAGE_MISMATCH",
    )

    by_2022_month = {int(e["month"]): e for e in entries_2022}
    proxy_acc = None
    proxy_sources = []
    for month in EXPECTED_TRAIN_MONTHS:
        require(month in by_2022_month, f"MISSING_TRAIN_MONTH_{month:02d}")
        path = Path(by_2022_month[month]["file"]).expanduser().resolve()
        payload = load_pt(path)
        emission = payload.get("emission_log_mean")
        require(isinstance(emission, torch.Tensor), f"MISSING_EMISSION_{month:02d}")
        emission = torch.nan_to_num(
            emission.detach().cpu().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        require(tuple(emission.shape) == (8, 361, 576), "BAD_EMISSION_SHAPE")
        if proxy_acc is None:
            proxy_acc = torch.zeros_like(emission)
        proxy_acc += emission
        proxy_sources.append(
            {"month": month, "file": str(path), "sha256": sha256_file(path)}
        )
    require(proxy_acc is not None, "FIXED_PROXY_BUILD_FAILED")
    fixed_proxy = proxy_acc / float(len(EXPECTED_TRAIN_MONTHS))

    rows: list[dict[str, Any]] = []

    for entry in entries:
        year = int(entry["year"])
        month = int(entry["month"])
        path = Path(entry["file"]).expanduser().resolve()
        payload = load_pt(path)

        if year == 2022:
            emission = payload.get("emission_log_mean")
            require(isinstance(emission, torch.Tensor), f"MISSING_2022_EMISSION: {path}")
            emission_mode = "actual_month_specific"
        elif year == 2023:
            emission = fixed_proxy
            emission_mode = "fixed_2022_train_mean_proxy"
        else:
            raise RuntimeError(f"UNEXPECTED_YEAR: {year}")

        original = torch.nan_to_num(
            emission.detach().cpu().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        key = path.name

        transformed_a, diag_a = support_preserving_permute_2d(
            original,
            key=key,
            support_epsilon=args.support_epsilon,
        )
        transformed_b, diag_b = support_preserving_permute_2d(
            original,
            key=key,
            support_epsilon=args.support_epsilon,
        )

        support_before = torch.any(
            torch.abs(original) > float(args.support_epsilon), dim=0
        )
        support_after = torch.any(
            torch.abs(transformed_a) > float(args.support_epsilon), dim=0
        )

        unsupported = ~support_before
        support_exact = bool(torch.equal(support_before, support_after))
        unsupported_exact = bool(torch.equal(
            original[:, unsupported],
            transformed_a[:, unsupported],
        ))
        deterministic_exact = bool(torch.equal(transformed_a, transformed_b))
        changed = bool(not torch.equal(original, transformed_a))

        support_indices = torch.nonzero(
            support_before.reshape(-1), as_tuple=False
        ).squeeze(1)
        source_vectors = original.reshape(8, -1)[:, support_indices]
        seed = md5_int(f"supportperm::{key}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(
            int(support_indices.numel()), generator=generator
        )
        if bool(torch.equal(
            permutation,
            torch.arange(int(support_indices.numel())),
        )):
            permutation = torch.roll(permutation, shifts=1)
        expected_vectors = source_vectors[:, permutation]
        actual_vectors = transformed_a.reshape(8, -1)[:, support_indices]
        exact_assigned_vector_multiset = bool(torch.equal(
            actual_vectors,
            expected_vectors,
        ))

        channel_sum_max_abs_diff = float(torch.max(torch.abs(
            original.double().sum(dim=(1, 2))
            - transformed_a.double().sum(dim=(1, 2))
        )).item())
        channel_sumsq_max_abs_diff = float(torch.max(torch.abs(
            (original.double() ** 2).sum(dim=(1, 2))
            - (transformed_a.double() ** 2).sum(dim=(1, 2))
        )).item())

        row = {
            "year": year,
            "month": month,
            "season": str(entry.get("season")),
            "split": str(entry.get("split")),
            "emission_mode": emission_mode,
            "cache_file": str(path),
            "cache_file_sha256": sha256_file(path),
            "key": key,
            "original_tensor_sha256": tensor_sha256(original),
            "transformed_tensor_sha256": tensor_sha256(transformed_a),
            "support_cells": int(torch.count_nonzero(support_before).item()),
            "support_mask_exactly_preserved": support_exact,
            "unsupported_values_exactly_preserved": unsupported_exact,
            "deterministic_reproduction_exact": deterministic_exact,
            "transformation_changed_tensor": changed,
            "complete_8d_vector_assignment_exact": exact_assigned_vector_multiset,
            "channel_sum_max_abs_diff": channel_sum_max_abs_diff,
            "channel_sumsq_max_abs_diff": channel_sumsq_max_abs_diff,
            **diag_a,
        }
        require(diag_a == diag_b, f"DIAGNOSTIC_NONDETERMINISM: {key}")
        rows.append(row)

    required_true_fields = [
        "support_mask_exactly_preserved",
        "unsupported_values_exactly_preserved",
        "deterministic_reproduction_exact",
        "transformation_changed_tensor",
        "complete_8d_vector_assignment_exact",
    ]
    for row in rows:
        for field in required_true_fields:
            require(bool(row[field]), f"{field}_FAILED: {row['year']}-{row['month']:02d}")
        require(
            float(row["channel_sum_max_abs_diff"]) <= 1e-6,
            f"CHANNEL_SUM_NOT_PRESERVED: {row['year']}-{row['month']:02d}",
        )
        require(
            float(row["channel_sumsq_max_abs_diff"]) <= 1e-6,
            f"CHANNEL_SUMSQ_NOT_PRESERVED: {row['year']}-{row['month']:02d}",
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "support_preserving_permutation_validation.json"
    csv_path = output_dir / "support_preserving_permutation_monthly.csv"
    txt_path = output_dir / "support_preserving_permutation_summary.txt"

    report = {
        "artifact_name": "clean_meanstate_support_preserving_permutation_gateP2A",
        "artifact_version": 1,
        "inputs": {
            "cache_dir": str(cache_dir),
            "monthly_manifest": str(manifest_path),
            "monthly_manifest_sha256": sha256_file(manifest_path),
            "support_epsilon": float(args.support_epsilon),
            "fixed_proxy_sources": proxy_sources,
        },
        "transformation_contract": {
            "unit_permuted": "complete 8-dimensional processed emission vector",
            "permutation_domain": "cells inside the original non-zero processed-emission support mask",
            "support_mask": "preserved exactly",
            "unsupported_cells": "preserved exactly",
            "cross_sector_structure": "preserved within each 8-dimensional vector",
            "per_channel_distribution": "preserved exactly by permutation",
            "determinism": "md5-derived CPU torch.Generator seed from monthly cache filename",
        },
        "interpretation_boundary": (
            "This control conditions on the processed-emission support geometry. "
            "It tests whether the spatial ordering of emission-vector values adds information "
            "beyond the support mask; it does not test whether the support geometry itself is physical."
        ),
        "monthly": rows,
        "verification": {
            "all_21_monthly_groups_checked": len(rows) == 21,
            "all_support_masks_exactly_preserved": all(
                r["support_mask_exactly_preserved"] for r in rows
            ),
            "all_unsupported_values_exactly_preserved": all(
                r["unsupported_values_exactly_preserved"] for r in rows
            ),
            "all_deterministic_reproductions_exact": all(
                r["deterministic_reproduction_exact"] for r in rows
            ),
            "all_transformations_nonidentity": all(
                r["transformation_changed_tensor"] for r in rows
            ),
            "all_complete_8d_assignments_exact": all(
                r["complete_8d_vector_assignment_exact"] for r in rows
            ),
            "all_channel_sums_and_sumsq_preserved_within_1e_6": True,
            "overall_pass": True,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    max_fixed_fraction = max(float(r["fixed_point_fraction"]) for r in rows)
    lines = [
        "CLEAN_MEANSTATE_SUPPORT_PRESERVING_PERMUTATION_GATEP2A_PASS",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_CSV={csv_path}",
        f"OUTPUT_TXT={txt_path}",
        f"MONTHLY_MANIFEST_SHA256={sha256_file(manifest_path)}",
        f"MONTHLY_GROUPS_CHECKED={len(rows)}",
        "SUPPORT_MASK_EXACTLY_PRESERVED_ALL_MONTHS=True",
        "UNSUPPORTED_VALUES_EXACTLY_PRESERVED_ALL_MONTHS=True",
        "COMPLETE_8D_VECTOR_ASSIGNMENT_EXACT_ALL_MONTHS=True",
        "DETERMINISTIC_REPRODUCTION_EXACT_ALL_MONTHS=True",
        "TRANSFORMATION_NONIDENTITY_ALL_MONTHS=True",
        f"MAX_FIXED_POINT_FRACTION={max_fixed_fraction:.8f}",
        "MODEL_TRAINING_STARTED=False",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
