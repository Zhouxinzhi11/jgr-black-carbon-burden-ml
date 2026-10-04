# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
76_preflight_s2_meanstate_binary_support_gateS8A.py

Gate S8A -- read-only preflight for the planned binary-support + meteorology
mean-state control.

This script does NOT train a model. It verifies that the new control can be
constructed under exactly the current clean mean-state data/provenance contract:

  2022: support = any(abs(monthly processed emission prior) > 0, sector)
  2023: support = any(abs(fixed 2022 train-mean processed prior) > 0, sector)
  input: [1 binary support channel, 5 monthly-mean meteorology channels]

It also checks that the existing clean MeanStateUNet backbone can accept a
1-channel emission branch without changing the backbone design.

Scientific boundary:
- The support channel is support of the inherited processed prior, not verified
  physical non-zero emissions.
- 2023 remains a proxy external contract for any emission-derived input.
- This is a design/provenance gate only; it produces no scientific skill result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

H = 361
W = 576
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VAL_MONTHS = [11, 12]
EXPECTED_BLACKLIST_MONTHS = [7]
EXPECTED_2023_MONTHS = list(range(1, 13))
EXPECTED_GROUPS = 21
EXPECTED_MODEL_PARAMETER_COUNT = 476449
SUPPORT_EPSILON = 0.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor_bytes(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def strict_json_dump(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def resolve_cache_file(cache_dir: Path, entry: dict[str, Any]) -> Path:
    raw = Path(str(entry["file"]))
    if raw.is_file():
        return raw.resolve()
    candidate = cache_dir / "monthly_mean_full" / str(entry["split"]) / raw.name
    require(candidate.is_file(), f"CACHE_FILE_NOT_FOUND: manifest={raw}; candidate={candidate}")
    return candidate.resolve()


def ensure_tensor(payload: dict[str, Any], key: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = payload.get(key)
    require(isinstance(value, torch.Tensor), f"{key}_NOT_TENSOR")
    require(tuple(value.shape) == shape, f"{key}_SHAPE_MISMATCH: {tuple(value.shape)} != {shape}")
    return value


def make_binary_support(emission: torch.Tensor) -> torch.Tensor:
    require(tuple(emission.shape) == (8, H, W), f"EMISSION_SHAPE_MISMATCH: {tuple(emission.shape)}")
    x = torch.nan_to_num(emission.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    support = torch.any(torch.abs(x) > SUPPORT_EPSILON, dim=0).float().unsqueeze(0)
    require(tuple(support.shape) == (1, H, W), "SUPPORT_SHAPE_MISMATCH")
    require(bool(torch.isfinite(support).all().item()), "SUPPORT_NONFINITE")
    unique = set(float(v) for v in torch.unique(support).tolist())
    require(unique.issubset({0.0, 1.0}), f"SUPPORT_NOT_BINARY: {sorted(unique)}")
    return support


def build_proxy(entries: list[dict[str, Any]], cache_dir: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tensors: list[torch.Tensor] = []
    provenance: list[dict[str, Any]] = []
    by_month = {(int(e["year"]), int(e["month"])): e for e in entries}
    for month in EXPECTED_TRAIN_MONTHS:
        entry = by_month.get((2022, month))
        require(entry is not None, f"MISSING_PROXY_SOURCE_MONTH: 2022-{month:02d}")
        path = resolve_cache_file(cache_dir, entry)
        payload = load_pt(path)
        require(bool(payload.get("emission_available", False)), f"PROXY_SOURCE_EMISSION_UNAVAILABLE: {path}")
        emission = ensure_tensor(payload, "emission_log_mean", (8, H, W)).float()
        require(bool(torch.isfinite(emission).all().item()), f"PROXY_SOURCE_NONFINITE: {path}")
        tensors.append(emission)
        provenance.append({
            "year": 2022,
            "month": month,
            "file": str(path),
            "sha256": sha256_file(path),
        })
    proxy = torch.stack(tensors, dim=0).mean(dim=0)
    require(tuple(proxy.shape) == (8, H, W), "PROXY_SHAPE_MISMATCH")
    require(bool(torch.isfinite(proxy).all().item()), "PROXY_NONFINITE")
    return proxy, provenance


def role(year: int, month: int) -> str:
    if year == 2022 and month in EXPECTED_TRAIN_MONTHS:
        return "2022_train"
    if year == 2022 and month in EXPECTED_VAL_MONTHS:
        return "2022_model_selection_validation"
    if year == 2022 and month in EXPECTED_BLACKLIST_MONTHS:
        return "2022_blacklist"
    if year == 2023:
        return "2023_proxy_external"
    return "unexpected"


def load_base_training_module(path: Path):
    name = "meanstate_clean_base_for_gateS8A"
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "BASE_TRAINING_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def validate_gate11b(cache_dir: Path, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    require(report.get("audit_name") == "phase2_meanstate_fullmonth_clean_gate11b", "GATE11B_AUDIT_NAME_MISMATCH")
    verification = report.get("verification", {})
    for key in [
        "all_21_monthly_groups_reconstructed",
        "all_8_seasonal_groups_reconstructed",
        "primary_2022_train_validation_groups_usable",
        "only_expected_2022_07_group_is_weak",
        "overall_pass",
    ]:
        require(verification.get(key) is True, f"GATE11B_FAILED_OR_MISSING: {key}")
    split = report.get("approved_split", {})
    require(split.get("train_months_2022") == EXPECTED_TRAIN_MONTHS, "GATE11B_TRAIN_SPLIT_CHANGED")
    require(split.get("validation_months_2022") == EXPECTED_VAL_MONTHS, "GATE11B_VAL_SPLIT_CHANGED")
    require(split.get("blacklist_months_2022") == EXPECTED_BLACKLIST_MONTHS, "GATE11B_BLACKLIST_CHANGED")
    require(split.get("external_months_2023") == EXPECTED_2023_MONTHS, "GATE11B_EXTERNAL_SPLIT_CHANGED")
    require(norm_path(report.get("inputs", {}).get("cache_dir", "")) == norm_path(cache_dir), "GATE11B_CACHE_DIR_MISMATCH")
    manifest_hash = sha256_file(manifest_path)
    require(report.get("hashes", {}).get("monthly_manifest_sha256") == manifest_hash, "GATE11B_MANIFEST_HASH_MISMATCH")
    return {
        "gate11b_report": str(report_path),
        "gate11b_report_sha256": sha256_file(report_path),
        "monthly_manifest_sha256": manifest_hash,
    }


def validate_gatep3(cache_dir: Path, report_path: Path, manifest_hash: str) -> dict[str, Any]:
    report = load_json(report_path)
    require(report.get("artifact_name") == "meanstate_support_mask_overlap_gateP3", "GATEP3_ARTIFACT_NAME_MISMATCH")
    require(int(report.get("artifact_version", -1)) == 1, "GATEP3_VERSION_MISMATCH")
    require(report.get("status") == "PASS_READ_ONLY_AUDIT_COMPLETED", "GATEP3_STATUS_NOT_PASS")
    inputs = report.get("inputs", {})
    require(norm_path(inputs.get("cache_dir", "")) == norm_path(cache_dir), "GATEP3_CACHE_DIR_MISMATCH")
    require(inputs.get("monthly_manifest_sha256") == manifest_hash, "GATEP3_MANIFEST_HASH_MISMATCH")
    contracts = report.get("contracts", {})
    require(contracts.get("grid_shape") == [H, W], "GATEP3_GRID_SHAPE_MISMATCH")
    require(int(contracts.get("min_obs_days", -1)) == 5, "GATEP3_MIN_OBS_DAYS_MISMATCH")
    require(math.isclose(float(contracts.get("support_epsilon", -1.0)), 0.0, rel_tol=0.0, abs_tol=0.0), "GATEP3_SUPPORT_EPSILON_CHANGED")
    require(contracts.get("emission_support_definition") == "any(abs(emission_log_mean) > support_epsilon, dim=0)", "GATEP3_SUPPORT_DEFINITION_CHANGED")
    require(contracts.get("proxy_definition") == "unweighted arithmetic mean of 2022 months [2,3,5,6,8,9]", "GATEP3_PROXY_DEFINITION_CHANGED")
    decision = report.get("screening_decision", {})
    require(decision.get("screening_code") == "DESIGNABLE", f"GATEP3_NOT_DESIGNABLE: {decision}")
    require(not decision.get("exact_degenerate_training_rows", []), "GATEP3_EXACT_DEGENERATE_TRAINING_SUPPORT")
    rows = report.get("monthly_rows", [])
    require(isinstance(rows, list) and len(rows) == EXPECTED_GROUPS, "GATEP3_MONTHLY_ROWS_MISMATCH")
    return {
        "report": report,
        "gatep3_report": str(report_path),
        "gatep3_report_sha256": sha256_file(report_path),
        "screening_code": decision.get("screening_code"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate S8A read-only preflight for binary-support + meteorology mean-state control")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--gatep3-report", required=True)
    parser.add_argument("--base-training-script", default="./47_train_s2_meanstate_clean.py")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    gate11b_report = Path(args.gate11b_report).expanduser().resolve()
    gatep3_report = Path(args.gatep3_report).expanduser().resolve()
    base_training_script = Path(args.base_training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(gate11b_report.is_file(), f"GATE11B_REPORT_NOT_FOUND: {gate11b_report}")
    require(gatep3_report.is_file(), f"GATEP3_REPORT_NOT_FOUND: {gatep3_report}")
    require(base_training_script.is_file(), f"BASE_TRAINING_SCRIPT_NOT_FOUND: {base_training_script}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    manifest_path = cache_dir / "manifests" / "monthly_manifest.json"
    require(manifest_path.is_file(), f"MONTHLY_MANIFEST_NOT_FOUND: {manifest_path}")
    manifest = load_json(manifest_path)
    entries = list(manifest.get("groups", []))
    require(len(entries) == EXPECTED_GROUPS, f"MONTHLY_GROUP_COUNT_MISMATCH: {len(entries)}")

    gate11b = validate_gate11b(cache_dir, gate11b_report, manifest_path)
    gatep3 = validate_gatep3(cache_dir, gatep3_report, gate11b["monthly_manifest_sha256"])
    p3_report = gatep3["report"]
    p3_rows = {(int(r["year"]), int(r["month"])): r for r in p3_report["monthly_rows"]}

    proxy, proxy_sources = build_proxy(entries, cache_dir)
    proxy_hash = sha256_tensor_bytes(proxy)
    require(proxy_hash == p3_report.get("proxy_emission", {}).get("tensor_sha256"), "PROXY_TENSOR_HASH_DIFFERS_FROM_GATEP3")
    proxy_support = make_binary_support(proxy)
    proxy_support_cells = int(torch.count_nonzero(proxy_support).item())
    require(proxy_support_cells == int(p3_report.get("proxy_emission", {}).get("support_cells", -1)), "PROXY_SUPPORT_COUNT_DIFFERS_FROM_GATEP3")

    rows: list[dict[str, Any]] = []
    training_inputs: list[torch.Tensor] = []

    for entry in sorted(entries, key=lambda e: (int(e["year"]), int(e["month"]))):
        year = int(entry["year"])
        month = int(entry["month"])
        split = str(entry["split"])
        analysis_role = role(year, month)
        require(analysis_role != "unexpected", f"UNEXPECTED_GROUP: {year}-{month:02d}")

        path = resolve_cache_file(cache_dir, entry)
        payload = load_pt(path)
        require(int(payload["year"]) == year, f"YEAR_MISMATCH: {path}")
        require(int(payload["month"]) == month, f"MONTH_MISMATCH: {path}")
        require(str(payload["split"]) == split, f"SPLIT_MISMATCH: {path}")

        meteo = ensure_tensor(payload, "meteo_mean", (5, H, W)).float()
        require(bool(torch.isfinite(meteo).all().item()), f"METEO_NONFINITE: {path}")
        target = ensure_tensor(payload, "target_log_mean_conc", (H, W)).float()
        count = ensure_tensor(payload, "obs_count_map", (H, W))
        valid = ensure_tensor(payload, "valid_mask", (H, W)).float()
        target_eval = valid > 0.5
        require(torch.equal(target_eval, count >= 5), f"VALID_MASK_CONTRACT_CHANGED: {path}")
        require(int(torch.count_nonzero((~torch.isfinite(target)) & target_eval).item()) == 0, f"TARGET_NONFINITE_IN_EVAL: {path}")

        if year == 2022:
            require(bool(payload.get("emission_available", False)), f"2022_EMISSION_UNAVAILABLE: {path}")
            emission = ensure_tensor(payload, "emission_log_mean", (8, H, W)).float()
            require(bool(torch.isfinite(emission).all().item()), f"2022_EMISSION_NONFINITE: {path}")
            emission_source = "actual_2022_monthly_processed_prior"
        else:
            require(not bool(payload.get("emission_available", False)), f"2023_REAL_EMISSION_UNEXPECTEDLY_PRESENT: {path}")
            require(payload.get("emission_log_mean") is None, f"2023_EMISSION_LOG_MEAN_MUST_BE_NONE: {path}")
            emission = proxy
            emission_source = "fixed_2022_train_mean_proxy"

        support = make_binary_support(emission)
        x = torch.cat([support, meteo], dim=0)
        require(tuple(x.shape) == (6, H, W), f"S8_INPUT_SHAPE_MISMATCH: {path}")
        require(bool(torch.isfinite(x).all().item()), f"S8_INPUT_NONFINITE: {path}")

        support_cells = int(torch.count_nonzero(support).item())
        target_cells = int(torch.count_nonzero(target_eval).item())
        support_inside_target = int(torch.count_nonzero((support[0] > 0.5) & target_eval).item())
        require(target_cells > 0 or analysis_role == "2022_blacklist", f"EMPTY_TARGET_UNEXPECTED: {path}")

        p3_row = p3_rows.get((year, month))
        require(p3_row is not None, f"MISSING_GATEP3_ROW: {year}-{month:02d}")
        require(support_cells == int(p3_row["emission_support_cells"]), f"SUPPORT_COUNT_DIFFERS_FROM_GATEP3: {year}-{month:02d}")
        if year == 2022:
            require(sha256_file(path) == p3_row["cache_file_sha256"], f"CACHE_HASH_DIFFERS_FROM_GATEP3: {year}-{month:02d}")

        frac = (support_inside_target / target_cells) if target_cells > 0 else None
        nondegenerate = bool(0 < support_inside_target < target_cells) if target_cells > 0 else False
        if analysis_role == "2022_train":
            require(nondegenerate, f"TRAINING_SUPPORT_DEGENERATE_WITHIN_TARGET: {year}-{month:02d}")
            training_inputs.append(x)

        rows.append({
            "analysis_role": analysis_role,
            "year": year,
            "month": month,
            "split": split,
            "cache_file": str(path),
            "emission_source": emission_source,
            "support_cells": support_cells,
            "target_cells": target_cells,
            "support_inside_target_cells": support_inside_target,
            "p_support_given_target": frac,
            "support_non_degenerate_within_target": nondegenerate,
            "input_shape": list(x.shape),
            "input_all_finite": True,
        })

    train_months = sorted(r["month"] for r in rows if r["analysis_role"] == "2022_train")
    val_months = sorted(r["month"] for r in rows if r["analysis_role"] == "2022_model_selection_validation")
    blacklist_months = sorted(r["month"] for r in rows if r["analysis_role"] == "2022_blacklist")
    ext_months = sorted(r["month"] for r in rows if r["analysis_role"] == "2023_proxy_external")
    require(train_months == EXPECTED_TRAIN_MONTHS, "S8_TRAIN_MONTHS_MISMATCH")
    require(val_months == EXPECTED_VAL_MONTHS, "S8_VAL_MONTHS_MISMATCH")
    require(blacklist_months == EXPECTED_BLACKLIST_MONTHS, "S8_BLACKLIST_MONTHS_MISMATCH")
    require(ext_months == EXPECTED_2023_MONTHS, "S8_EXTERNAL_MONTHS_MISMATCH")

    base = load_base_training_module(base_training_script)
    s8_spec = base.MeanStateSpec(
        name="mean_binary_support_meteo_clim",
        mapping="Binary Processed-Emission Support + Monthly-Mean Meteo -> Monthly Mean Concentration",
        note="Gate S8 binary processed-prior support control.",
        uses_emission=True,
        uses_meteo=True,
        shuffle_emission=False,
        emission_channels=1,
        meteo_channels=5,
    )
    model = base.MeanStateUNet(spec=s8_spec, base_features=32).cpu().eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(n_params == EXPECTED_MODEL_PARAMETER_COUNT, f"S8_MODEL_PARAMETER_COUNT_UNEXPECTED: {n_params}")
    require(bool(model.dual), "S8_MODEL_NOT_DUAL_STEM")
    require(model.emis_stem[0].in_channels == 1, "S8_EMISSION_STEM_NOT_1_CHANNEL")
    require(model.met_stem[0].in_channels == 5, "S8_METEO_STEM_NOT_5_CHANNEL")

    require(training_inputs, "NO_TRAINING_INPUTS_FOR_MODEL_SMOKE")
    crop = training_inputs[0][:, :128, :128].unsqueeze(0)
    with torch.no_grad():
        pred = model(crop)
    require(tuple(pred.shape) == (1, 1, 128, 128), f"S8_MODEL_OUTPUT_SHAPE_MISMATCH: {tuple(pred.shape)}")
    require(bool(torch.isfinite(pred).all().item()), "S8_MODEL_OUTPUT_NONFINITE")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "binary_support_preflight_gateS8A.json"
    txt_path = output_dir / "binary_support_preflight_gateS8A.txt"

    report = {
        "artifact_name": "s2_meanstate_binary_support_preflight_gateS8A",
        "artifact_version": 1,
        "status": "PASS_READ_ONLY_PREFLIGHT",
        "inputs": {
            "cache_dir": str(cache_dir),
            "monthly_manifest": str(manifest_path),
            "monthly_manifest_sha256": gate11b["monthly_manifest_sha256"],
            "gate11b_report": gate11b["gate11b_report"],
            "gate11b_report_sha256": gate11b["gate11b_report_sha256"],
            "gatep3_report": gatep3["gatep3_report"],
            "gatep3_report_sha256": gatep3["gatep3_report_sha256"],
            "base_training_script": str(base_training_script),
            "base_training_script_sha256": sha256_file(base_training_script),
        },
        "s8_contract": {
            "experiment_name_planned": "mean_binary_support_meteo_clim",
            "support_definition": "any(abs(emission_log_mean) > 0.0, dim=0)",
            "support_channels": 1,
            "meteo_channels": 5,
            "total_input_channels": 6,
            "2022_support_source": "actual monthly processed emission prior",
            "2023_support_source": "fixed unweighted mean of 2022 training-month processed priors, then binarized",
            "train_months_2022": EXPECTED_TRAIN_MONTHS,
            "validation_months_2022": EXPECTED_VAL_MONTHS,
            "blacklist_months_2022": EXPECTED_BLACKLIST_MONTHS,
            "external_months_2023": EXPECTED_2023_MONTHS,
            "planned_seeds": [42, 43, 44],
            "base_features": 32,
            "model_trainable_parameters": n_params,
        },
        "proxy": {
            "tensor_sha256": proxy_hash,
            "support_cells": proxy_support_cells,
            "source_files": proxy_sources,
        },
        "monthly_rows": rows,
        "model_smoke": {
            "dual_stem": True,
            "emission_stem_input_channels": 1,
            "meteo_stem_input_channels": 5,
            "crop_input_shape": list(crop.shape),
            "crop_output_shape": list(pred.shape),
            "output_all_finite": True,
        },
        "verification": {
            "gate11b_provenance_exact": True,
            "gatep3_designability_exact": True,
            "all_21_monthly_groups_checked": True,
            "support_reproduces_gatep3_counts": True,
            "proxy_reproduces_gatep3_hash": True,
            "all_s8_inputs_shape_6x361x576": True,
            "all_s8_inputs_finite": True,
            "all_training_support_masks_nondegenerate_within_target": True,
            "base_backbone_accepts_1plus5_dual_input": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "This preflight validates only the binary-support control construction. "
            "The support channel represents non-zero support in the inherited processed prior and must not be interpreted as verified physical emission geography. "
            "For 2023, the support mask is derived from the same fixed 2022 train-mean processed-prior proxy contract used by existing emission-input evaluations. "
            "No predictive-performance result is produced here."
        ),
    }
    strict_json_dump(report, report_path)

    lines = [
        "GATE_S8A_BINARY_SUPPORT_PREFLIGHT_PASS",
        f"CACHE_DIR={cache_dir}",
        f"MONTHLY_MANIFEST_SHA256={gate11b['monthly_manifest_sha256']}",
        f"GATEP3_SCREENING_CODE={gatep3['screening_code']}",
        f"PROXY_TENSOR_SHA256={proxy_hash}",
        f"PROXY_SUPPORT_CELLS={proxy_support_cells}",
        "S8_INPUT_CHANNELS=6",
        "S8_SUPPORT_CHANNELS=1",
        "S8_METEO_CHANNELS=5",
        f"S8_MODEL_TRAINABLE_PARAMETERS={n_params}",
        f"TRAIN_MONTHS={EXPECTED_TRAIN_MONTHS}",
        f"VALIDATION_MONTHS={EXPECTED_VAL_MONTHS}",
        f"BLACKLIST_MONTHS={EXPECTED_BLACKLIST_MONTHS}",
        f"EXTERNAL_2023_MONTHS={EXPECTED_2023_MONTHS}",
        "SCIENTIFIC_TRAINING_NOT_STARTED=True",
        f"OUTPUT_JSON={report_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
