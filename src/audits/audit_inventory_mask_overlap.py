# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
63_audit_meanstate_mask_overlap_gateP3.py

Read-only Gate P3 audit of overlap among:
1) monthly source availability: obs_count_map > 0;
2) evaluation target mask: valid_mask > 0.5;
3) processed emission support: any(abs(emission_log_mean) > epsilon, dim=0).

For 2023, the script reproduces the paper contract by using the unweighted
mean of the six approved 2022 training-month emission tensors as a fixed proxy.
It never modifies caches and refuses to overwrite an existing output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch

H, W = 361, 576
N_GRID = H * W
TRAIN_2022 = [2, 3, 5, 6, 8, 9]
VAL_2022 = [11, 12]
BLACKLIST_2022 = [7]
EXTERNAL_2023 = list(range(1, 13))
EXPECTED_GROUPS = 21

DEFAULT_CACHE_DIR = (
    r"data/RESTRICTED_INPUT_NOT_INCLUDED"
    r"\07_phase2_meanstate_fullmonth_trainonly_union"
)

CSV_COLUMNS = [
    "analysis_role", "year", "month", "split", "emission_source",
    "cache_file", "cache_file_sha256", "n_days", "n_days_with_emission",
    "obs_any_cells", "obs_any_fraction", "target_eval_cells",
    "target_eval_fraction", "emission_support_cells",
    "emission_support_fraction", "jaccard_obs_emission",
    "jaccard_obs_target", "jaccard_emission_target",
    "p_emission_given_target", "p_target_given_emission",
    "p_obs_given_target", "emission_support_variance_within_target",
    "emission_support_non_degenerate_within_target",
    "emission_support_near_degenerate_within_target",
    "obs_target_exact_equal", "target_subset_of_obs",
    "target_nonfinite_count_within_eval",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON root is not an object: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    require(isinstance(obj, dict), f"PT root is not a dict: {path}")
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json_dump(obj: dict[str, Any], path: Path) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, torch.Tensor):
            return clean(x.detach().cpu().tolist())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else float(num / den)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float | None:
    require(a.dtype == torch.bool and b.dtype == torch.bool, "Masks must be bool")
    union = int(torch.count_nonzero(a | b).item())
    if union == 0:
        return None
    inter = int(torch.count_nonzero(a & b).item())
    return float(inter / union)


def ensure_tensor(payload: dict[str, Any], key: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = payload.get(key)
    require(isinstance(value, torch.Tensor), f"{key} is not a tensor")
    value = value.detach().cpu()
    require(tuple(value.shape) == shape, f"{key} shape {tuple(value.shape)} != {shape}")
    return value


def resolve_cache_file(cache_dir: Path, entry: dict[str, Any]) -> Path:
    manifest_path = Path(str(entry["file"]))
    if manifest_path.is_file():
        return manifest_path.resolve()
    candidate = cache_dir / "monthly_mean_full" / str(entry["split"]) / manifest_path.name
    require(
        candidate.is_file(),
        f"Cache file not found. Manifest={manifest_path}; candidate={candidate}",
    )
    return candidate.resolve()


def role(year: int, month: int) -> str:
    if year == 2022 and month in TRAIN_2022:
        return "2022_train"
    if year == 2022 and month in VAL_2022:
        return "2022_model_selection_validation"
    if year == 2022 and month in BLACKLIST_2022:
        return "2022_blacklisted_weak_group"
    if year == 2023 and month in EXTERNAL_2023:
        return "2023_proxy_external"
    return "other"


def make_emission_support(emission: torch.Tensor, epsilon: float) -> torch.Tensor:
    require(tuple(emission.shape) == (8, H, W), "Emission must be (8,361,576)")
    require(bool(torch.isfinite(emission).all().item()), "Emission contains NaN/Inf")
    return torch.any(torch.abs(emission.float()) > epsilon, dim=0)


def numeric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "max": max(clean),
    }


def pairwise_summary(items: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    pairs = []
    for (name_a, mask_a), (name_b, mask_b) in combinations(items, 2):
        pairs.append({"a": name_a, "b": name_b, "jaccard": jaccard(mask_a, mask_b)})
    return {
        "n_pairs": len(pairs),
        "summary": numeric_summary(p["jaccard"] for p in pairs),
        "pairs": pairs,
    }


def build_proxy(entries: list[dict[str, Any]], cache_dir: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    selected = sorted(
        [e for e in entries if int(e["year"]) == 2022 and int(e["month"]) in TRAIN_2022],
        key=lambda e: int(e["month"]),
    )
    require([int(e["month"]) for e in selected] == TRAIN_2022, "Proxy source months mismatch")
    tensors, provenance = [], []
    for entry in selected:
        path = resolve_cache_file(cache_dir, entry)
        payload = load_pt(path)
        require(bool(payload.get("emission_available", False)), f"Missing emission: {path}")
        emission = ensure_tensor(payload, "emission_log_mean", (8, H, W)).float()
        require(bool(torch.isfinite(emission).all().item()), f"Non-finite emission: {path}")
        tensors.append(emission)
        provenance.append({
            "year": 2022,
            "month": int(entry["month"]),
            "file": str(path),
            "sha256": sha256_file(path),
        })
    proxy = torch.stack(tensors, dim=0).mean(dim=0)
    require(bool(torch.isfinite(proxy).all().item()), "Proxy emission contains NaN/Inf")
    return proxy, provenance


def audit_one(
    entry: dict[str, Any],
    cache_dir: Path,
    proxy: torch.Tensor,
    proxy_hash: str,
    min_obs_days: int,
    epsilon: float,
    near_low: float,
    near_high: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    year, month, split = int(entry["year"]), int(entry["month"]), str(entry["split"])
    path = resolve_cache_file(cache_dir, entry)
    payload = load_pt(path)

    require(int(payload["year"]) == year, f"Year mismatch: {path}")
    require(int(payload["month"]) == month, f"Month mismatch: {path}")
    require(str(payload["split"]) == split, f"Split mismatch: {path}")

    target = ensure_tensor(payload, "target_log_mean_conc", (H, W)).float()
    count = ensure_tensor(payload, "obs_count_map", (H, W))
    valid = ensure_tensor(payload, "valid_mask", (H, W)).float()
    require(bool(torch.all(count >= 0).item()), f"Negative obs_count_map: {path}")

    obs_any = count > 0
    target_eval = valid > 0.5
    expected_target_eval = count >= min_obs_days
    require(
        torch.equal(target_eval, expected_target_eval),
        f"valid_mask != obs_count_map >= {min_obs_days}: {path}",
    )
    require(
        set(float(v) for v in torch.unique(valid).tolist()).issubset({0.0, 1.0}),
        f"valid_mask is not binary: {path}",
    )

    nonfinite_target = int(torch.count_nonzero((~torch.isfinite(target)) & target_eval).item())
    require(nonfinite_target == 0, f"Non-finite target inside eval mask: {path}")

    if year == 2022:
        require(bool(payload.get("emission_available", False)), f"2022 emission unavailable: {path}")
        emission = ensure_tensor(payload, "emission_log_mean", (8, H, W)).float()
        emission_source = "actual_2022_monthly_processed_prior"
        emission_ref_hash = sha256_file(path)
    elif year == 2023:
        require(not bool(payload.get("emission_available", False)), f"2023 real emission unexpectedly present: {path}")
        require(payload.get("emission_log_mean") is None, f"2023 emission_log_mean must be None: {path}")
        emission = proxy
        emission_source = "fixed_2022_train_mean_proxy"
        emission_ref_hash = proxy_hash
    else:
        raise RuntimeError(f"Unexpected year: {year}")

    emis = make_emission_support(emission, epsilon)
    n_obs = int(torch.count_nonzero(obs_any).item())
    n_target = int(torch.count_nonzero(target_eval).item())
    n_emis = int(torch.count_nonzero(emis).item())
    n_emis_target = int(torch.count_nonzero(emis & target_eval).item())
    n_obs_target = int(torch.count_nonzero(obs_any & target_eval).item())

    p_emis_target = ratio(n_emis_target, n_target)
    if p_emis_target is None:
        variance, nondegenerate, near_degenerate = None, False, True
    else:
        variance = p_emis_target * (1.0 - p_emis_target)
        nondegenerate = 0.0 < p_emis_target < 1.0
        near_degenerate = p_emis_target <= near_low or p_emis_target >= near_high

    row = {
        "analysis_role": role(year, month),
        "year": year,
        "month": month,
        "split": split,
        "emission_source": emission_source,
        "emission_reference_sha256": emission_ref_hash,
        "cache_file": str(path),
        "cache_file_sha256": sha256_file(path),
        "n_days": int(payload["n_days"]),
        "n_days_with_emission": int(payload["n_days_with_emission"]),
        "obs_any_cells": n_obs,
        "obs_any_fraction": ratio(n_obs, N_GRID),
        "target_eval_cells": n_target,
        "target_eval_fraction": ratio(n_target, N_GRID),
        "emission_support_cells": n_emis,
        "emission_support_fraction": ratio(n_emis, N_GRID),
        "jaccard_obs_emission": jaccard(obs_any, emis),
        "jaccard_obs_target": jaccard(obs_any, target_eval),
        "jaccard_emission_target": jaccard(emis, target_eval),
        "p_emission_given_target": p_emis_target,
        "p_target_given_emission": ratio(n_emis_target, n_emis),
        "p_obs_given_target": ratio(n_obs_target, n_target),
        "emission_support_variance_within_target": variance,
        "emission_support_non_degenerate_within_target": bool(nondegenerate),
        "emission_support_near_degenerate_within_target": bool(near_degenerate),
        "obs_target_exact_equal": bool(torch.equal(obs_any, target_eval)),
        "target_subset_of_obs": bool(torch.all((~target_eval) | obs_any).item()),
        "target_nonfinite_count_within_eval": nonfinite_target,
    }
    return row, {"obs_any": obs_any, "target_eval": target_eval, "emission_support": emis}


def screening_decision(rows: list[dict[str, Any]], near_duplicate_jaccard: float) -> dict[str, Any]:
    train = [r for r in rows if r["analysis_role"] == "2022_train"]
    val = [r for r in rows if r["analysis_role"] == "2022_model_selection_validation"]
    ext = [r for r in rows if r["analysis_role"] == "2023_proxy_external"]
    require(len(train) == 6 and len(val) == 2 and len(ext) == 12, "Approved split row counts mismatch")
    approved = train + val + ext

    empty = [f"{r['year']}-{r['month']:02d}" for r in approved if int(r["target_eval_cells"]) == 0]
    exact_deg = [f"2022-{r['month']:02d}" for r in train if not r["emission_support_non_degenerate_within_target"]]
    near_deg = [f"2022-{r['month']:02d}" for r in train if r["emission_support_near_degenerate_within_target"]]
    near_dup = [
        f"2022-{r['month']:02d}" for r in train
        if r["jaccard_obs_emission"] is not None
        and float(r["jaccard_obs_emission"]) >= near_duplicate_jaccard
    ]

    if empty:
        code = "STOP_EMPTY_APPROVED_TARGET"
        recommendation = "Do not train. At least one approved group has no evaluation cells."
    elif exact_deg:
        code = "NOT_INFORMATIVE_EXACTLY_DEGENERATE_ON_TRAIN_TARGET"
        recommendation = "Do not train. Binary emission support is constant inside at least one training target mask."
    elif near_deg:
        code = "WEAKLY_INFORMATIVE_NEAR_DEGENERATE_ON_TRAIN_TARGET"
        recommendation = "Do not train yet. Inspect maps/common-support evaluation because support is near constant in at least one training month."
    elif len(near_dup) == len(train):
        code = "DESIGNABLE_BUT_SUPPORT_OBSERVATION_NEAR_DUPLICATE"
        recommendation = "A support-mask control is designable, but emission and source-availability supports are near duplicates in all training months; interpretation must remain limited to the joint processed sampling/support construction."
    else:
        code = "DESIGNABLE"
        recommendation = "A meteorology + binary emission-support control is designable under the current cache contract."

    return {
        "screening_code": code,
        "recommendation": recommendation,
        "operational_thresholds_are_not_significance_tests": True,
        "empty_target_rows": empty,
        "exact_degenerate_training_rows": exact_deg,
        "near_degenerate_training_rows": near_deg,
        "near_duplicate_obs_emission_training_rows": near_dup,
        "all_approved_target_masks_subset_of_obs_any": all(r["target_subset_of_obs"] for r in approved),
        "all_approved_targets_finite_within_eval": all(r["target_nonfinite_count_within_eval"] == 0 for r in approved),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in CSV_COLUMNS})


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(
    path: Path,
    cache_dir: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    summaries: dict[str, Any],
    min_obs_days: int,
    epsilon: float,
) -> None:
    approved = [r for r in rows if r["analysis_role"] != "2022_blacklisted_weak_group"]
    lines = [
        "# Mean-state support-mask overlap audit",
        "",
        "## Gate status",
        "",
        f"- Screening code: `{decision['screening_code']}`",
        f"- Recommendation: {decision['recommendation']}",
        "- Operational thresholds are diagnostics, not significance tests.",
        "",
        "## Mask definitions",
        "",
        f"- Cache: `{cache_dir}`",
        f"- Manifest: `{manifest_path}`",
        "- Source availability: `obs_count_map > 0`.",
        f"- Evaluation target: `valid_mask > 0.5`, verified as `obs_count_map >= {min_obs_days}`.",
        f"- Emission support: `any(abs(emission_log_mean) > {epsilon}, dim=0)`.",
        "- 2023 support uses the fixed mean of 2022 training months 02, 03, 05, 06, 08, and 09.",
        "",
        "## Approved-group results",
        "",
        "| Role | Year-month | Target cells | P(emission support | target) | J(obs, emission) | J(emission, target) | Near-degenerate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in approved:
        lines.append(
            f"| {r['analysis_role']} | {r['year']}-{r['month']:02d} | "
            f"{r['target_eval_cells']} | {fmt(r['p_emission_given_target'])} | "
            f"{fmt(r['jaccard_obs_emission'])} | {fmt(r['jaccard_emission_target'])} | "
            f"{r['emission_support_near_degenerate_within_target']} |"
        )
    lines += ["", "## Aggregate diagnostics", ""]
    for name, value in summaries.items():
        s = value.get("summary", value) if isinstance(value, dict) else value
        lines.append(f"- {name}: `{json.dumps(s, ensure_ascii=False)}`")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This audit uses masks propagated into the clean monthly cache. It does not independently re-derive raw I_380 validity, and a non-zero processed emission cell is not automatically a physically verified non-zero emission cell. If source availability and emission support are nearly identical, the cyclic-shift contrast must be described as sensitivity to the joint processed sampling/support construction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only mean-state mask-overlap audit")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-obs-days", type=int, default=5)
    parser.add_argument("--support-epsilon", type=float, default=0.0)
    parser.add_argument("--near-degenerate-low", type=float, default=0.01)
    parser.add_argument("--near-degenerate-high", type=float, default=0.99)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.95)
    parser.add_argument("--expected-monthly-groups", type=int, default=EXPECTED_GROUPS)
    args = parser.parse_args()

    require(args.min_obs_days >= 1, "min_obs_days must be >= 1")
    require(args.support_epsilon >= 0.0, "support_epsilon must be >= 0")
    require(0.0 <= args.near_degenerate_low < args.near_degenerate_high <= 1.0, "Invalid near-degenerate thresholds")
    require(0.0 <= args.near_duplicate_jaccard <= 1.0, "Invalid Jaccard threshold")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require(cache_dir.is_dir(), f"Cache directory not found: {cache_dir}")
    require(not output_dir.exists(), f"Output directory already exists; refusing overwrite: {output_dir}")

    manifest_path = cache_dir / "manifests" / "monthly_manifest.json"
    require(manifest_path.is_file(), f"Manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    entries = list(manifest.get("groups", []))
    require(len(entries) == args.expected_monthly_groups, f"Expected {args.expected_monthly_groups} groups, found {len(entries)}")

    keys = [(int(e["year"]), str(e["split"]), int(e["month"])) for e in entries]
    require(len(keys) == len(set(keys)), "Duplicate monthly keys")
    months_2022 = sorted(int(e["month"]) for e in entries if int(e["year"]) == 2022)
    months_2023 = sorted(int(e["month"]) for e in entries if int(e["year"]) == 2023)
    require(months_2022 == sorted(TRAIN_2022 + VAL_2022 + BLACKLIST_2022), f"Unexpected 2022 months: {months_2022}")
    require(months_2023 == EXTERNAL_2023, f"Unexpected 2023 months: {months_2023}")

    proxy, proxy_sources = build_proxy(entries, cache_dir)
    proxy_hash = hashlib.sha256(proxy.contiguous().numpy().tobytes(order="C")).hexdigest()

    rows: list[dict[str, Any]] = []
    masks: dict[str, dict[str, torch.Tensor]] = {}
    for entry in sorted(entries, key=lambda e: (int(e["year"]), int(e["month"]))):
        row, month_masks = audit_one(
            entry, cache_dir, proxy, proxy_hash,
            args.min_obs_days, args.support_epsilon,
            args.near_degenerate_low, args.near_degenerate_high,
        )
        label = f"{row['year']}-{row['month']:02d}"
        rows.append(row)
        masks[label] = month_masks

    decision = screening_decision(rows, args.near_duplicate_jaccard)
    train_labels = [f"2022-{m:02d}" for m in TRAIN_2022]
    approved_2022_labels = [f"2022-{m:02d}" for m in TRAIN_2022 + VAL_2022]
    approved_rows = [r for r in rows if r["analysis_role"] != "2022_blacklisted_weak_group"]
    summaries = {
        "approved_p_emission_given_target": numeric_summary(r["p_emission_given_target"] for r in approved_rows),
        "approved_jaccard_obs_emission": numeric_summary(r["jaccard_obs_emission"] for r in approved_rows),
        "approved_jaccard_emission_target": numeric_summary(r["jaccard_emission_target"] for r in approved_rows),
        "training_emission_support_pairwise": pairwise_summary([(x, masks[x]["emission_support"]) for x in train_labels]),
        "approved_2022_emission_support_pairwise": pairwise_summary([(x, masks[x]["emission_support"]) for x in approved_2022_labels]),
        "training_obs_any_pairwise": pairwise_summary([(x, masks[x]["obs_any"]) for x in train_labels]),
        "training_target_eval_pairwise": pairwise_summary([(x, masks[x]["target_eval"]) for x in train_labels]),
    }

    report = {
        "artifact_name": "meanstate_support_mask_overlap_gateP3",
        "artifact_version": 1,
        "status": "PASS_READ_ONLY_AUDIT_COMPLETED",
        "inputs": {
            "cache_dir": str(cache_dir),
            "monthly_manifest": str(manifest_path),
            "monthly_manifest_sha256": sha256_file(manifest_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "contracts": {
            "grid_shape": [H, W],
            "min_obs_days": args.min_obs_days,
            "support_epsilon": args.support_epsilon,
            "obs_any_definition": "obs_count_map > 0",
            "target_eval_definition": "valid_mask > 0.5",
            "valid_mask_contract": f"valid_mask == (obs_count_map >= {args.min_obs_days})",
            "emission_support_definition": "any(abs(emission_log_mean) > support_epsilon, dim=0)",
            "proxy_definition": "unweighted arithmetic mean of 2022 months [2,3,5,6,8,9]",
            "operational_thresholds": {
                "near_degenerate_low": args.near_degenerate_low,
                "near_degenerate_high": args.near_degenerate_high,
                "near_duplicate_jaccard": args.near_duplicate_jaccard,
                "not_significance_thresholds": True,
            },
        },
        "proxy_emission": {
            "tensor_sha256": proxy_hash,
            "source_months": TRAIN_2022,
            "source_files": proxy_sources,
            "support_cells": int(torch.count_nonzero(make_emission_support(proxy, args.support_epsilon)).item()),
        },
        "screening_decision": decision,
        "monthly_rows": rows,
        "aggregate_summaries": summaries,
        "interpretation_boundary": "Propagated clean-cache audit only; raw I_380 support is not independently re-derived.",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "support_mask_overlap_audit.json"
    csv_path = output_dir / "support_mask_overlap_by_month.csv"
    md_path = output_dir / "support_mask_overlap_summary.md"
    strict_json_dump(report, json_path)
    write_csv(rows, csv_path)
    write_markdown(md_path, cache_dir, manifest_path, rows, decision, summaries, args.min_obs_days, args.support_epsilon)

    print("=" * 88)
    print("MEANSTATE_SUPPORT_MASK_OVERLAP_GATEP3_PASS")
    print("=" * 88)
    print(f"screening_code = {decision['screening_code']}")
    print(f"json             = {json_path}")
    print(f"csv              = {csv_path}")
    print(f"summary          = {md_path}")
    print("-" * 88)
    print(decision["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
