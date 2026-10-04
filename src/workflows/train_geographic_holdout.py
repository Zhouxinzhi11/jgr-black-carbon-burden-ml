# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
83_train_s2_meanstate_spatial_holdout_gateS9C.py

Gate S9C: spatial-holdout mean-state training.

Scientific question
-------------------
After Gate S8 showed that binary processed-prior support alone did not provide
a seed-stable increment over meteorology, test whether the monthly mean-state
association learned outside East Asia transfers into an East-Asia core that
never participates in:
  1) meteorological normalization fitting,
  2) training crops / training loss, or
  3) checkpoint selection.

This script NEVER evaluates predictive skill inside the East-Asia core.
The final core test must be opened only by a later locked evaluator after the
planned formal training runs have been completed.

Predefined model set
--------------------
- mean_meteo_clim_only
- mean_binary_support_meteo_clim
- mean_emis_meteo_clim

Locked spatial design
---------------------
Core East Asia:
    y=220..290, x=448..520  (20-55 N, 100-145 E)

Training exclusion:
    y=188..322, x=416..552  (core + 32-cell guard)
A complete 128x128 training crop must have ZERO overlap with this exclusion.

Checkpoint-selection exclusion:
    y=156..354, x=384..575  (core + 64-cell exclusion)
November-December 2022 checkpoint selection uses only valid pixels outside
this stricter exclusion.

Meteorological preprocessing
-----------------------------
The monthly cache was normalized using time-train-only global statistics.
Gate S9B computed stricter spatial-train-only statistics. Because z-scoring
and arithmetic averaging are affine, cached monthly meteo_mean is converted by

    z_spatial = scale * z_existing + offset

BEFORE nan_to_num. No cache rebuild is required.

Training contract
-----------------
Formal optimization remains the same as 47_train_s2_meanstate_clean.py:
- train months: 02,03,05,06,08,09 (2022)
- model-selection months: 11,12 (2022), but only on safe selection pixels
- blacklist month: 07
- crop 128, batch 8, 512 samples/epoch
- AdamW lr=3e-4, weight_decay=1e-2
- cosine warm restarts T0=24, Tmult=2, eta_min=1e-6
- gradient clip 5
- maximum 160 epochs, patience 25
- seeds 42,43,44

Interpretation boundary
-----------------------
This is a spatial transfer test of the learned processed-prior / meteorology
association. It does NOT validate native EDGAR geography, physical emissions,
or causal transport. The processed prior remains an inherited,
observation-collocated representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

H = 361
W = 576

EXPECTED_BASE_47_SHA256 = (
    "ff3bec0f270b1f8dd191f43230d9665761fb6f4c65baafd71bdfc56cda6a2fe8"
)
EXPECTED_BINARY_77_SHA256 = (
    "b2de46481d2119a7c4bb49dc698ce27ed34390ee6ca37b47a472e5cfde41b2e1"
)
EXPECTED_MONTHLY_MANIFEST_SHA256 = (
    "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
)
EXPECTED_S9A_ARTIFACT = "s2_meanstate_spatial_holdout_preflight_gateS9A"
EXPECTED_S9B_ARTIFACT = "s2_meanstate_spatial_train_only_meteo_stats_gateS9B"
EXPECTED_PRIMARY_REGION = "East_Asia"

CORE_BBOX = (220, 290, 448, 520)
TRAIN_EXCLUSION_BBOX = (188, 322, 416, 552)
SELECTION_EXCLUSION_BBOX = (156, 354, 384, 575)
GUARD_CELLS = 32
SELECTION_GUARD_CELLS = 64
EXPECTED_RECEPTIVE_FIELD_DIAMETER = 46
EXPECTED_MAX_HALF_RECEPTIVE_FIELD = 23

ALLOWED_EXPERIMENTS = [
    "mean_meteo_clim_only",
    "mean_binary_support_meteo_clim",
    "mean_emis_meteo_clim",
]
ALLOWED_SEEDS = {42, 43, 44}
TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
VAL_MONTHS = [11, 12]
BLACKLIST_MONTHS = [7]
EXTERNAL_2023_MONTHS = list(range(1, 13))
SUPPORT_DEFINITION = "any(abs(emission_log_mean) > 0.0, dim=0)"

EXPECTED_PARAM_COUNTS = {
    "mean_meteo_clim_only": 477025,
    "mean_binary_support_meteo_clim": 476449,
    "mean_emis_meteo_clim": 477457,
}
METEO_ORDER = ["Temp", "RH", "BLH", "Windspeed_U", "Windspeed_V"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: str | Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    require(isinstance(obj, dict), f"PT_ROOT_NOT_DICT: {path}")
    return obj


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    def safe(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, np.generic):
            return safe(v.item())
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe(obj), f, ensure_ascii=True, indent=2, allow_nan=False)


def import_locked_base(path: Path):
    require(path.is_file(), f"BASE_47_NOT_FOUND: {path}")
    actual = sha256_file(path)
    require(
        actual == EXPECTED_BASE_47_SHA256,
        f"BASE_47_SHA256_MISMATCH: {actual}",
    )
    module_name = "gate_s9c_locked_base47"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None, "BASE_47_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def make_rect_mask(bbox: tuple[int, int, int, int]) -> torch.Tensor:
    y0, y1, x0, x1 = bbox
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


CORE_MASK = make_rect_mask(CORE_BBOX)
TRAIN_EXCLUSION_MASK = make_rect_mask(TRAIN_EXCLUSION_BBOX)
SELECTION_EXCLUSION_MASK = make_rect_mask(SELECTION_EXCLUSION_BBOX)
SAFE_SELECTION_MASK = ~SELECTION_EXCLUSION_MASK


def precompute_eligible_top_lefts(crop_size: int) -> list[tuple[int, int]]:
    require(crop_size == 128, "S9C_CROP_SIZE_MUST_BE_128")
    exclusion = TRAIN_EXCLUSION_MASK.numpy().astype(np.int64)
    ii = np.pad(exclusion, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def rect_sum(top: int, left: int) -> int:
        bottom = top + crop_size
        right = left + crop_size
        return int(
            ii[bottom, right]
            - ii[top, right]
            - ii[bottom, left]
            + ii[top, left]
        )

    eligible: list[tuple[int, int]] = []
    for top in range(H - crop_size + 1):
        for left in range(W - crop_size + 1):
            if rect_sum(top, left) == 0:
                eligible.append((top, left))
    require(len(eligible) == 77386, f"ELIGIBLE_PATCH_COUNT_MISMATCH: {len(eligible)}")
    return eligible


def validate_s9a(path: Path, cache_dir: str) -> dict[str, Any]:
    report = load_json(path)
    require(report.get("artifact_name") == EXPECTED_S9A_ARTIFACT, "S9A_ARTIFACT_MISMATCH")
    require(report.get("status") == "DESIGNABLE_PRIMARY_REGION", "S9A_STATUS_MISMATCH")
    require(report.get("verification", {}).get("overall_pass") is True, "S9A_NOT_PASS")
    require(
        normalized_path(report.get("inputs", {}).get("cache_dir", ""))
        == normalized_path(cache_dir),
        "S9A_CACHE_DIR_MISMATCH",
    )
    require(
        report.get("inputs", {}).get("monthly_manifest_sha256")
        == EXPECTED_MONTHLY_MANIFEST_SHA256,
        "S9A_MANIFEST_HASH_MISMATCH",
    )
    contract = report.get("design_contract", {})
    require(contract.get("primary_region") == EXPECTED_PRIMARY_REGION, "S9A_REGION_MISMATCH")
    require(list(contract.get("grid_shape", [])) == [H, W], "S9A_GRID_SHAPE_MISMATCH")
    require(int(contract.get("crop_size", -1)) == 128, "S9A_CROP_SIZE_MISMATCH")
    require(int(contract.get("guard_cells", -1)) == GUARD_CELLS, "S9A_GUARD_MISMATCH")
    require(
        int(contract.get("selection_guard_cells", -1)) == SELECTION_GUARD_CELLS,
        "S9A_SELECTION_GUARD_MISMATCH",
    )
    require(
        contract.get("planned_core_models") == ALLOWED_EXPERIMENTS,
        "S9A_PLANNED_MODELS_MISMATCH",
    )
    require(contract.get("planned_seeds") == [42, 43, 44], "S9A_SEEDS_MISMATCH")

    east = report["region_results"][EXPECTED_PRIMARY_REGION]
    require(
        tuple(east["core_bbox_indices"][k] for k in ["y0", "y1", "x0", "x1"])
        == CORE_BBOX,
        "S9A_CORE_BBOX_MISMATCH",
    )
    require(
        tuple(east["training_exclusion_bbox_indices"][k] for k in ["y0", "y1", "x0", "x1"])
        == TRAIN_EXCLUSION_BBOX,
        "S9A_TRAIN_EXCLUSION_BBOX_MISMATCH",
    )
    require(
        tuple(east["selection_exclusion_bbox_indices"][k] for k in ["y0", "y1", "x0", "x1"])
        == SELECTION_EXCLUSION_BBOX,
        "S9A_SELECTION_EXCLUSION_BBOX_MISMATCH",
    )
    require(east.get("checks", {}).get("overall_designable") is True, "S9A_EAST_ASIA_NOT_DESIGNABLE")
    require(
        math.isclose(
            float(east["patch_geometry"]["eligible_fraction"]),
            0.7365465516913179,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "S9A_ELIGIBLE_FRACTION_MISMATCH",
    )
    return report


def validate_s9b(path: Path, s9a_path: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    report = load_json(path)
    require(report.get("artifact_name") == EXPECTED_S9B_ARTIFACT, "S9B_ARTIFACT_MISMATCH")
    require(report.get("status") == "PASS_SPATIAL_TRAIN_ONLY_PREPROCESSING", "S9B_STATUS_MISMATCH")
    verification = report.get("verification", {})
    for key in [
        "s9a_contract_exact",
        "grid_mapping_matches_clean_rebuild",
        "gate7_time_train_stats_reproduced",
        "spatial_exclusion_removed_rows",
        "all_train_months_retained_rows",
        "spatial_stats_finite_and_nondegenerate",
        "monthly_cache_rebuild_required",
        "overall_pass",
    ]:
        expected = False if key == "monthly_cache_rebuild_required" else True
        require(verification.get(key) is expected, f"S9B_VERIFICATION_MISMATCH_{key}")

    require(
        report.get("inputs", {}).get("s9a_report_sha256") == sha256_file(s9a_path),
        "S9B_S9A_HASH_MISMATCH",
    )
    row = report.get("row_accounting", {})
    require(int(row.get("processed_rows", -1)) == 26075892, "S9B_PROCESSED_ROWS_MISMATCH")
    require(int(row.get("train_month_rows", -1)) == 21423116, "S9B_TRAIN_ROWS_MISMATCH")
    require(int(row.get("excluded_spatial_rows", 0)) > 0, "S9B_NO_EXCLUDED_ROWS")
    require(int(row.get("retained_spatial_rows", 0)) > 0, "S9B_NO_RETAINED_ROWS")

    affine = report.get("old_z_to_spatial_z_affine", {})
    scales = []
    offsets = []
    for var in METEO_ORDER:
        item = affine.get(var)
        require(isinstance(item, dict), f"S9B_AFFINE_MISSING_{var}")
        scale = float(item["old_z_to_spatial_z_scale"])
        offset = float(item["old_z_to_spatial_z_offset"])
        require(math.isfinite(scale) and scale > 0.0, f"S9B_SCALE_INVALID_{var}")
        require(math.isfinite(offset), f"S9B_OFFSET_INVALID_{var}")
        require(item.get("formula") == "z_spatial = scale * z_existing + offset",
                f"S9B_FORMULA_MISMATCH_{var}")
        scales.append(scale)
        offsets.append(offset)

    scale_t = torch.tensor(scales, dtype=torch.float32).view(5, 1, 1)
    offset_t = torch.tensor(offsets, dtype=torch.float32).view(5, 1, 1)
    return report, scale_t, offset_t


def build_specs(base) -> Dict[str, Any]:
    return {
        "mean_meteo_clim_only": base.MeanStateSpec(
            name="mean_meteo_clim_only",
            mapping="Spatial-holdout Monthly-Mean Meteo -> Monthly Mean Concentration",
            note="Gate S9: meteorology-only spatial transfer control.",
            uses_emission=False,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=0,
            meteo_channels=5,
        ),
        "mean_binary_support_meteo_clim": base.MeanStateSpec(
            name="mean_binary_support_meteo_clim",
            mapping="Spatial-holdout Binary Processed-Prior Support + Monthly Meteo -> Monthly Mean Concentration",
            note="Gate S9: binary processed-prior support spatial transfer control.",
            uses_emission=True,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=1,
            meteo_channels=5,
        ),
        "mean_emis_meteo_clim": base.MeanStateSpec(
            name="mean_emis_meteo_clim",
            mapping="Spatial-holdout Processed Emission Prior + Monthly Meteo -> Monthly Mean Concentration",
            note="Gate S9: sector-resolved processed-prior spatial transfer model.",
            uses_emission=True,
            uses_meteo=True,
            shuffle_emission=False,
            emission_channels=8,
            meteo_channels=5,
        ),
    }


def build_input(
    payload: dict[str, Any],
    spec,
    *,
    meteo_scale: torch.Tensor,
    meteo_offset: torch.Tensor,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    if spec.uses_emission:
        emis = payload.get("emission_log_mean")
        require(isinstance(emis, torch.Tensor), "EMISSION_INPUT_MISSING")
        emis = emis.detach().cpu().float()
        require(tuple(emis.shape) == (8, H, W), f"EMISSION_SHAPE_INVALID: {tuple(emis.shape)}")
        require(bool(torch.isfinite(emis).all().item()), "EMISSION_NONFINITE")
        if spec.name == "mean_binary_support_meteo_clim":
            support = torch.any(torch.abs(emis) > 0.0, dim=0, keepdim=True).float()
            require(tuple(support.shape) == (1, H, W), "SUPPORT_SHAPE_INVALID")
            require(
                set(float(v) for v in torch.unique(support).tolist()).issubset({0.0, 1.0}),
                "SUPPORT_NOT_BINARY",
            )
            parts.append(support)
        else:
            parts.append(emis)

    if spec.uses_meteo:
        met = payload.get("meteo_mean")
        require(isinstance(met, torch.Tensor), "METEO_INPUT_MISSING")
        met = met.detach().cpu().float()
        require(tuple(met.shape) == (5, H, W), f"METEO_SHAPE_INVALID: {tuple(met.shape)}")
        # Apply affine transform BEFORE nan_to_num so missing cells remain missing
        # until they are mapped to zero in the NEW normalized coordinate system.
        met = meteo_scale * met + meteo_offset
        met = torch.nan_to_num(met, nan=0.0, posinf=0.0, neginf=0.0)
        require(bool(torch.isfinite(met).all().item()), "TRANSFORMED_METEO_NONFINITE")
        parts.append(met)

    require(parts, "NO_INPUT_PARTS")
    x = torch.cat(parts, dim=0)
    require(tuple(x.shape) == (spec.input_channels, H, W), "INPUT_SHAPE_MISMATCH")
    return x


class SpatialMonthlyPatchDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict[str, Any]],
        spec,
        *,
        meteo_scale: torch.Tensor,
        meteo_offset: torch.Tensor,
        crop_size: int,
        samples_per_epoch: int,
        eligible_top_lefts: Sequence[tuple[int, int]],
    ) -> None:
        self.entries = list(entries)
        self.spec = spec
        self.meteo_scale = meteo_scale
        self.meteo_offset = meteo_offset
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.eligible_top_lefts = list(eligible_top_lefts)
        require(self.entries, "NO_TRAIN_ENTRIES")
        require(self.eligible_top_lefts, "NO_ELIGIBLE_PATCH_POSITIONS")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        entry = self.entries[index % len(self.entries)]
        payload = load_pt(entry["file"])
        x = build_input(
            payload,
            self.spec,
            meteo_scale=self.meteo_scale,
            meteo_offset=self.meteo_offset,
        )
        y = payload["target_log_mean_conc"].float().unsqueeze(0)
        valid_mask = payload["valid_mask"].float().unsqueeze(0)
        count_map = payload["obs_count_map"].float().unsqueeze(0)

        top, left = self.eligible_top_lefts[
            random.randrange(len(self.eligible_top_lefts))
        ]
        ys = slice(top, top + self.crop_size)
        xs = slice(left, left + self.crop_size)
        require(
            not bool(TRAIN_EXCLUSION_MASK[ys, xs].any().item()),
            "TRAINING_CROP_OVERLAPS_SPATIAL_EXCLUSION",
        )

        return (
            x[:, ys, xs],
            y[:, ys, xs],
            valid_mask[:, ys, xs],
            count_map[:, ys, xs],
        )


@torch.no_grad()
def evaluate_safe_selection(
    *,
    model,
    entries: Sequence[dict[str, Any]],
    spec,
    device: torch.device,
    meteo_scale: torch.Tensor,
    meteo_offset: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    all_pred = []
    all_true = []
    per_group = []

    for entry in entries:
        payload = load_pt(entry["file"])
        x = build_input(
            payload,
            spec,
            meteo_scale=meteo_scale,
            meteo_offset=meteo_offset,
        ).unsqueeze(0).to(device)

        pred = model(x).squeeze(0).squeeze(0).detach().cpu().float()
        target = payload["target_log_mean_conc"].float()
        original_valid = payload["valid_mask"].float() > 0.5
        safe_valid = (
            original_valid
            & SAFE_SELECTION_MASK
            & torch.isfinite(target)
            & torch.isfinite(pred)
        )
        n_valid = int(safe_valid.sum().item())
        require(n_valid > 1, f"SAFE_SELECTION_HAS_TOO_FEW_PIXELS_MONTH_{entry['month']}")

        p = pred[safe_valid]
        t = target[safe_valid]
        all_pred.append(p)
        all_true.append(t)

        ss_res = torch.sum((t - p) ** 2)
        ss_tot = torch.sum((t - torch.mean(t)) ** 2)
        r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))
        rmse = float(torch.sqrt(torch.mean((p - t) ** 2)))
        bias = float(torch.mean(p - t))

        per_group.append(
            {
                "year": int(entry["year"]),
                "month": int(entry["month"]),
                "split": entry["split"],
                "n_safe_selection_pixels": n_valid,
                "log_r2": r2,
                "log_rmse": rmse,
                "log_bias": bias,
                "file": entry["file"],
            }
        )

    pred_all = torch.cat(all_pred)
    true_all = torch.cat(all_true)
    ss_res = torch.sum((true_all - pred_all) ** 2)
    ss_tot = torch.sum((true_all - torch.mean(true_all)) ** 2)
    global_r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))
    global_rmse = float(torch.sqrt(torch.mean((pred_all - true_all) ** 2)))
    global_bias = float(torch.mean(pred_all - true_all))

    return {
        "evaluation_role": "checkpoint_selection_only_outside_core_plus_64_cell_exclusion",
        "core_holdout_performance_opened": False,
        "n_groups": len(entries),
        "n_total_safe_selection_pixels": int(true_all.numel()),
        "global": {
            "log_r2": global_r2,
            "log_rmse": global_rmse,
            "log_bias": global_bias,
        },
        "per_group": per_group,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate S9C spatial-holdout mean-state training.")
    parser.add_argument("--experiment", required=True, choices=ALLOWED_EXPERIMENTS)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gate11b-report", required=True)
    parser.add_argument("--s9a-report", required=True)
    parser.add_argument("--s9b-report", required=True)
    parser.add_argument("--base-training-script", required=True)
    parser.add_argument("--binary-training-script", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--val-months", default="11,12")
    parser.add_argument("--blacklist-months", default="7")
    parser.add_argument("--min-train-days", type=int, default=5)
    parser.add_argument("--min-valid-fraction", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    cache_dir = normalized_path(args.cache_dir)
    run_root = normalized_path(args.run_root)
    run_dir = normalized_path(os.path.join(run_root, args.run_name))
    gate11b_path = Path(args.gate11b_report).expanduser().resolve()
    s9a_path = Path(args.s9a_report).expanduser().resolve()
    s9b_path = Path(args.s9b_report).expanduser().resolve()
    base47_path = Path(args.base_training_script).expanduser().resolve()
    binary77_path = Path(args.binary_training_script).expanduser().resolve()

    require(Path(cache_dir).is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(not os.path.exists(run_dir), f"STOP_RUN_DIR_ALREADY_EXISTS: {run_dir}")
    require(binary77_path.is_file(), f"BINARY_77_NOT_FOUND: {binary77_path}")
    require(
        sha256_file(binary77_path) == EXPECTED_BINARY_77_SHA256,
        "BINARY_77_SHA256_MISMATCH",
    )

    base = import_locked_base(base47_path)
    provenance11b = base.validate_gate11b_provenance(
        cache_dir=cache_dir,
        gate11b_report_path=str(gate11b_path),
    )
    require(
        provenance11b["monthly_manifest_sha256"] == EXPECTED_MONTHLY_MANIFEST_SHA256,
        "GATE11B_MANIFEST_HASH_MISMATCH",
    )

    s9a = validate_s9a(s9a_path, cache_dir)
    s9b, meteo_scale, meteo_offset = validate_s9b(s9b_path, s9a_path)

    require(
        EXPECTED_MAX_HALF_RECEPTIVE_FIELD <= GUARD_CELLS,
        "MODEL_RECEPTIVE_FIELD_EXCEEDS_TRAINING_GUARD",
    )

    if args.run_mode == "smoke":
        expected_settings = {
            "epochs": 2,
            "samples_per_epoch": 32,
            "batch_size": 8,
            "crop_size": 128,
            "patience": 2,
            "base_features": 32,
            "num_workers": 0,
        }
    else:
        expected_settings = {
            "epochs": 160,
            "samples_per_epoch": 512,
            "batch_size": 8,
            "crop_size": 128,
            "patience": 25,
            "base_features": 32,
            "num_workers": 0,
        }
    actual_settings = {
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "patience": args.patience,
        "base_features": args.base_features,
        "num_workers": args.num_workers,
    }
    require(actual_settings == expected_settings,
            f"RUN_MODE_SETTINGS_MISMATCH expected={expected_settings} actual={actual_settings}")
    require(args.val_months == "11,12", "VAL_MONTHS_ARGUMENT_MISMATCH")
    require(args.blacklist_months == "7", "BLACKLIST_MONTHS_ARGUMENT_MISMATCH")
    require(args.min_train_days == 5, "MIN_TRAIN_DAYS_MISMATCH")
    require(math.isclose(args.min_valid_fraction, 1e-6, rel_tol=0.0, abs_tol=0.0),
            "MIN_VALID_FRACTION_MISMATCH")
    require(math.isclose(args.learning_rate, 3e-4, rel_tol=0.0, abs_tol=0.0),
            "LEARNING_RATE_MISMATCH")
    require(math.isclose(args.weight_decay, 1e-2, rel_tol=0.0, abs_tol=0.0),
            "WEIGHT_DECAY_MISMATCH")
    require(args.seed in ALLOWED_SEEDS, "SEED_NOT_PREDECLARED")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA_REQUIRED_BUT_UNAVAILABLE")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base.set_seed(args.seed)

    specs = build_specs(base)
    spec = specs[args.experiment]

    monthly_entries = base.load_monthly_manifest(cache_dir)
    kept_2022, dropped_2022 = base.filter_monthly_entries(
        monthly_entries,
        split="2022_full",
        blacklist_months=BLACKLIST_MONTHS,
        min_days=5,
        min_valid_fraction=1e-6,
    )
    train_entries, val_entries = base.split_train_val_months(kept_2022, VAL_MONTHS)
    external_2023, dropped_2023 = base.filter_monthly_entries(
        monthly_entries,
        split="2023_all",
        blacklist_months=[],
        min_days=5,
        min_valid_fraction=1e-6,
    )
    require([int(x["month"]) for x in train_entries] == TRAIN_MONTHS, "TRAIN_MONTHS_MISMATCH")
    require([int(x["month"]) for x in val_entries] == VAL_MONTHS, "VAL_MONTHS_MISMATCH")
    require([int(x["month"]) for x in external_2023] == EXTERNAL_2023_MONTHS,
            "EXTERNAL_2023_MONTHS_MISMATCH")
    require(len(dropped_2022) == 1 and int(dropped_2022[0]["month"]) == 7,
            "DROPPED_2022_CONTRACT_MISMATCH")
    require(not dropped_2023, "DROPPED_2023_MUST_BE_EMPTY")

    eligible = precompute_eligible_top_lefts(args.crop_size)

    # Read-only input preflight on all training and model-selection months.
    input_checks = []
    for entry in [*train_entries, *val_entries]:
        payload = load_pt(entry["file"])
        x = build_input(
            payload,
            spec,
            meteo_scale=meteo_scale,
            meteo_offset=meteo_offset,
        )
        require(bool(torch.isfinite(x).all().item()), "INPUT_PREFLIGHT_NONFINITE")
        input_checks.append(
            {
                "year": int(entry["year"]),
                "month": int(entry["month"]),
                "input_shape": list(x.shape),
                "all_finite": True,
            }
        )

    os.makedirs(run_dir, exist_ok=False)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=False)
    os.makedirs(log_dir, exist_ok=False)

    model = base.MeanStateUNet(spec=spec, base_features=args.base_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(
        n_params == EXPECTED_PARAM_COUNTS[args.experiment],
        f"TRAINABLE_PARAMETER_COUNT_MISMATCH: {n_params}",
    )

    # Synthetic architecture check: output shape only; no holdout target is opened.
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(
            (1, spec.input_channels, 128, 128),
            dtype=torch.float32,
            device=device,
        )
        dummy_out = model(dummy)
    require(tuple(dummy_out.shape) == (1, 1, 128, 128), "MODEL_FORWARD_SHAPE_MISMATCH")

    print("=" * 92)
    print("Gate S9C - spatial-holdout mean-state training")
    print("=" * 92)
    print(f"RUN_MODE={args.run_mode}")
    print(f"EXPERIMENT={args.experiment}")
    print(f"SEED={args.seed}")
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print(f"RUN_DIR={run_dir}")
    print(f"TRAIN_MONTHS={TRAIN_MONTHS}")
    print(f"VALIDATION_MONTHS={VAL_MONTHS}")
    print(f"PRIMARY_HOLDOUT_REGION={EXPECTED_PRIMARY_REGION}")
    print(f"CORE_BBOX={CORE_BBOX}")
    print(f"TRAIN_EXCLUSION_BBOX={TRAIN_EXCLUSION_BBOX}")
    print(f"SELECTION_EXCLUSION_BBOX={SELECTION_EXCLUSION_BBOX}")
    print(f"ELIGIBLE_TRAIN_PATCH_POSITIONS={len(eligible)}")
    print(f"MODEL_RECEPTIVE_FIELD_DIAMETER={EXPECTED_RECEPTIVE_FIELD_DIAMETER}")
    print(f"MODEL_MAX_HALF_RECEPTIVE_FIELD={EXPECTED_MAX_HALF_RECEPTIVE_FIELD}")
    print(f"TRAINING_GUARD_CELLS={GUARD_CELLS}")
    print("SPATIAL_TRAIN_ONLY_METEO_NORMALIZATION=True")
    print("CORE_HOLDOUT_USED_FOR_TRAINING=False")
    print("CORE_HOLDOUT_USED_FOR_CHECKPOINT_SELECTION=False")
    print("CORE_HOLDOUT_PERFORMANCE_OPENED=False")
    print(f"TRAINABLE_PARAMETERS={n_params}")

    manifest = {
        "artifact_name": "s2_meanstate_spatial_holdout_training_gateS9C",
        "artifact_version": 1,
        "run_mode": args.run_mode,
        "experiment": asdict(spec),
        "seed": args.seed,
        "run_name": args.run_name,
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "external_2023_entries_recorded_but_not_evaluated": external_2023,
        "dropped_2022_entries": dropped_2022,
        "dropped_2023_entries": dropped_2023,
        "spatial_contract": {
            "primary_region": EXPECTED_PRIMARY_REGION,
            "core_bbox_indices": list(CORE_BBOX),
            "training_exclusion_bbox_indices": list(TRAIN_EXCLUSION_BBOX),
            "selection_exclusion_bbox_indices": list(SELECTION_EXCLUSION_BBOX),
            "guard_cells": GUARD_CELLS,
            "selection_guard_cells": SELECTION_GUARD_CELLS,
            "crop_size": 128,
            "eligible_patch_positions": len(eligible),
            "receptive_field_diameter": EXPECTED_RECEPTIVE_FIELD_DIAMETER,
            "max_half_receptive_field": EXPECTED_MAX_HALF_RECEPTIVE_FIELD,
            "training_crop_rule": "complete crop has zero overlap with training exclusion",
            "checkpoint_selection_rule": "valid pixels outside core plus 64-cell exclusion only",
            "core_test_opened_during_training": False,
        },
        "normalization_contract": {
            "spatial_train_only": True,
            "variables": METEO_ORDER,
            "affine_scale": [float(x) for x in meteo_scale[:, 0, 0].tolist()],
            "affine_offset": [float(x) for x in meteo_offset[:, 0, 0].tolist()],
            "application_order": "affine transform before nan_to_num",
            "cache_rebuilt": False,
        },
        "provenance": {
            **provenance11b,
            "s9a_report": str(s9a_path),
            "s9a_report_sha256": sha256_file(s9a_path),
            "s9b_report": str(s9b_path),
            "s9b_report_sha256": sha256_file(s9b_path),
            "base_47_script": str(base47_path),
            "base_47_script_sha256": sha256_file(base47_path),
            "binary_77_script": str(binary77_path),
            "binary_77_script_sha256": sha256_file(binary77_path),
            "training_script": normalized_path(__file__),
            "training_script_sha256": sha256_file(__file__),
        },
        "settings": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "base_features": args.base_features,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "require_cuda": bool(args.require_cuda),
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "input_preflight": input_checks,
        "scientific_boundary": (
            "This model is trained without East-Asia core+guard crops and selected without "
            "East-Asia core+64-cell validation targets. East-Asia predictive performance is "
            "not computed in this training artifact. The experiment tests spatial transfer "
            "of the inherited processed-prior / meteorology association, not physical "
            "emission geography or causal attribution."
        ),
    }
    manifest_path = os.path.join(log_dir, "run_manifest.json")
    save_json(manifest, manifest_path)

    dataset = SpatialMonthlyPatchDataset(
        train_entries,
        spec,
        meteo_scale=meteo_scale,
        meteo_offset=meteo_offset,
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch,
        eligible_top_lefts=eligible,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=24,
        T_mult=2,
        eta_min=1e-6,
    )

    history = {
        "train_loss": [],
        "train_patch_r2": [],
        "safe_selection_log_r2": [],
        "safe_selection_log_rmse": [],
        "safe_selection_log_bias": [],
        "lr": [],
    }

    best_val_r2 = -float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        patch_r2_values = []
        n_batches = 0

        for xb, yb, mb, cb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = base.masked_weighted_mse(pred, yb, mb, cb)
            require(bool(torch.isfinite(loss).item()),
                    f"NONFINITE_TRAINING_LOSS_EPOCH_{epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1
            r2 = base.masked_r2(pred.detach(), yb.detach(), mb.detach())
            if math.isfinite(r2):
                patch_r2_values.append(r2)

        scheduler.step()

        train_loss = loss_sum / max(n_batches, 1)
        train_patch_r2 = (
            float(np.mean(patch_r2_values))
            if patch_r2_values
            else float("nan")
        )

        val = evaluate_safe_selection(
            model=model,
            entries=val_entries,
            spec=spec,
            device=device,
            meteo_scale=meteo_scale,
            meteo_offset=meteo_offset,
        )
        val_r2 = float(val["global"]["log_r2"])
        val_rmse = float(val["global"]["log_rmse"])
        val_bias = float(val["global"]["log_bias"])
        require(
            all(math.isfinite(x) for x in [val_r2, val_rmse, val_bias]),
            f"NONFINITE_SAFE_SELECTION_METRIC_EPOCH_{epoch + 1}",
        )

        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_patch_r2"].append(train_patch_r2)
        history["safe_selection_log_r2"].append(val_r2)
        history["safe_selection_log_rmse"].append(val_rmse)
        history["safe_selection_log_bias"].append(val_bias)
        history["lr"].append(lr)

        print(
            f"EPOCH={epoch + 1}/{args.epochs} "
            f"TRAIN_LOSS={train_loss:.6f} "
            f"TRAIN_PATCH_R2={train_patch_r2:.6f} "
            f"SAFE_SELECTION_LOG_R2={val_r2:.6f} "
            f"SAFE_SELECTION_LOG_RMSE={val_rmse:.6f} "
            f"LR={lr:.8f}",
            flush=True,
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_safe_selection_log_r2": best_val_r2,
                    "experiment": args.experiment,
                    "run_name": args.run_name,
                    "run_mode": args.run_mode,
                    "seed": args.seed,
                    "base_features": args.base_features,
                    "spec": asdict(spec),
                    "spatial_contract": manifest["spatial_contract"],
                    "normalization_contract": manifest["normalization_contract"],
                    "s9a_report_sha256": sha256_file(s9a_path),
                    "s9b_report_sha256": sha256_file(s9b_path),
                    "base_47_script_sha256": sha256_file(base47_path),
                    "binary_77_script_sha256": sha256_file(binary77_path),
                    "training_script_sha256": sha256_file(__file__),
                    "core_holdout_performance_opened": False,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        save_json(history, os.path.join(log_dir, "training_history.json"))
        save_json(val, os.path.join(log_dir, "latest_safe_selection_report.json"))

        if patience_counter >= args.patience:
            print(f"EARLY_STOP=True PATIENCE={args.patience}", flush=True)
            break

    required = [
        os.path.join(ckpt_dir, "best_model.pt"),
        manifest_path,
        os.path.join(log_dir, "training_history.json"),
        os.path.join(log_dir, "latest_safe_selection_report.json"),
    ]
    for path in required:
        require(os.path.isfile(path) and os.path.getsize(path) > 0,
                f"MISSING_OR_EMPTY_OUTPUT: {path}")

    print("GATE_S9C_SPATIAL_TRAINING_ARTIFACTS_PASS=True")
    print(f"BEST_EPOCH={best_epoch + 1}")
    print(f"BEST_SAFE_SELECTION_LOG_R2={best_val_r2:.8f}")
    print("CORE_HOLDOUT_PERFORMANCE_OPENED=False")
    if args.run_mode == "smoke":
        print("GATE_S9C_SPATIAL_TRAINING_SMOKE_PASS")
    else:
        print("GATE_S9D_SPATIAL_FORMAL_TRAINING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
