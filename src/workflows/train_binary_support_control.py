# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
77_train_s2_meanstate_binary_support_gateS8.py

Gate S8 dedicated mean-state control: binary processed-prior support + monthly meteorology.

Scientific question
-------------------
The existing cyclic-shift and support-preserving controls implicate broad processed-prior
support/co-location structure, but they do not directly quantify how much predictive skill
can be reproduced by the binary non-zero support mask alone. This script replaces the eight
processed-emission value channels with one binary support channel and retains the approved
five-channel monthly meteorology branch.

Locked support definition
-------------------------
    any(abs(emission_log_mean) > 0.0, dim=0)

Training contract
-----------------
- 2022 training months: 02, 03, 05, 06, 08, 09
- 2022 held-out model-selection validation: 11, 12
- 2022 blacklist: 07
- predeclared seeds: 42, 43, 44
- same U-Net backbone, optimizer, patch sampling, observation-count-weighted loss,
  checkpoint selection, and early stopping contract as 47_train_s2_meanstate_clean.py
- smoke and formal settings are locked separately
- run directories are never overwritten

Provenance gates
----------------
The script independently re-validates Gate-11B provenance and requires a passing Gate-S8A
preflight artifact anchored to the approved 47 training-script SHA256 and the approved
binary-support/proxy contract.

Interpretation boundary
-----------------------
The support channel represents non-zero support in the inherited processed-prior construction.
It is not verified physical emission geography. This training script uses actual 2022 monthly
processed priors to derive support. Later 2023 evaluation must derive support from the fixed
unweighted mean of the six 2022 training-month priors and remains a proxy external test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
H = 361
W = 576
DEFAULT_CACHE_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\07_phase2_meanstate_fullmonth_trainonly_union"
DEFAULT_RUN_ROOT = r"./runs_s2_meanstate_binary_support_gateS8"

S8_EXPERIMENT = "mean_binary_support_meteo_clim"
S8_SUPPORT_DEFINITION = "any(abs(emission_log_mean) > 0.0, dim=0)"
S8_SUPPORT_EPSILON = 0.0
S8_EXPECTED_BASE_TRAINING_SHA256 = "ff3bec0f270b1f8dd191f43230d9665761fb6f4c65baafd71bdfc56cda6a2fe8"
S8_EXPECTED_PROXY_SHA256 = "283867292ed0b51ebce073db35940c4b091d23cacba8ca929f5248982bb0d749"
S8_EXPECTED_PROXY_SUPPORT_CELLS = 42949

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe(obj),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_month_list(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def md5_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(str(path))
        )
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate11b_provenance(
    cache_dir: str,
    gate11b_report_path: str,
) -> dict:
    report_path = Path(gate11b_report_path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if report.get("audit_name") != "phase2_meanstate_fullmonth_clean_gate11b":
        raise RuntimeError("Unexpected Gate-11B audit_name.")
    if report.get("verification", {}).get("overall_pass") is not True:
        raise RuntimeError("Gate-11B overall_pass is not true.")

    required_true = [
        "all_21_monthly_groups_reconstructed",
        "all_8_seasonal_groups_reconstructed",
        "primary_2022_train_validation_groups_usable",
        "only_expected_2022_07_group_is_weak",
        "overall_pass",
    ]
    failed = [
        key
        for key in required_true
        if report.get("verification", {}).get(key) is not True
    ]
    if failed:
        raise RuntimeError(f"Gate-11B verification failed or missing: {failed}")

    approved = report.get("approved_split", {})
    if approved.get("train_months_2022") != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError("Gate-11B train-month definition changed.")
    if approved.get("validation_months_2022") != [11, 12]:
        raise RuntimeError("Gate-11B validation-month definition changed.")
    if approved.get("blacklist_months_2022") != [7]:
        raise RuntimeError("Gate-11B blacklist-month definition changed.")
    if approved.get("external_months_2023") != list(range(1, 13)):
        raise RuntimeError("Gate-11B 2023 external-month definition changed.")

    report_cache_dir = report.get("inputs", {}).get("cache_dir", "")
    if normalized_path(report_cache_dir) != normalized_path(cache_dir):
        raise RuntimeError(
            "Gate-11B cache_dir does not match the requested training cache."
        )

    monthly_manifest = (
        Path(cache_dir).expanduser().resolve()
        / "manifests"
        / "monthly_manifest.json"
    )
    if not monthly_manifest.is_file():
        raise FileNotFoundError(monthly_manifest)

    expected_manifest_hash = report.get("hashes", {}).get(
        "monthly_manifest_sha256"
    )
    actual_manifest_hash = sha256_file(monthly_manifest)
    if expected_manifest_hash != actual_manifest_hash:
        raise RuntimeError(
            "The monthly manifest changed after Gate 11B."
        )

    return {
        "gate11b_report": str(report_path),
        "gate11b_report_sha256": sha256_file(report_path),
        "monthly_manifest": str(monthly_manifest),
        "monthly_manifest_sha256": actual_manifest_hash,
        "approved_split": approved,
    }


# ---------------------------------------------------------------------
# Gate-S8A preflight provenance
# ---------------------------------------------------------------------
def validate_s8a_preflight(
    *,
    cache_dir: str,
    preflight_report_path: str,
) -> dict:
    if not os.path.isfile(preflight_report_path):
        raise FileNotFoundError(preflight_report_path)
    with open(preflight_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if report.get("artifact_name") != "s2_meanstate_binary_support_preflight_gateS8A":
        raise RuntimeError("Unexpected Gate-S8A artifact_name.")
    if report.get("status") != "PASS_READ_ONLY_PREFLIGHT":
        raise RuntimeError("Gate-S8A report is not a PASS_READ_ONLY_PREFLIGHT artifact.")
    verification = report.get("verification", {})
    if verification.get("overall_pass") is not True:
        raise RuntimeError("Gate-S8A overall_pass is not true.")

    inputs = report.get("inputs", {})
    report_cache = normalized_path(inputs.get("cache_dir", ""))
    if report_cache != normalized_path(cache_dir):
        raise RuntimeError(
            f"Gate-S8A cache mismatch: report={report_cache}, current={normalized_path(cache_dir)}"
        )

    manifest_path = os.path.join(cache_dir, "manifests", "monthly_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != inputs.get("monthly_manifest_sha256"):
        raise RuntimeError(
            "monthly_manifest.json changed after Gate-S8A preflight. "
            f"current={manifest_sha}, preflight={inputs.get('monthly_manifest_sha256')}"
        )

    if inputs.get("base_training_script_sha256") != S8_EXPECTED_BASE_TRAINING_SHA256:
        raise RuntimeError("Gate-S8A was not anchored to the approved 47 training script SHA256.")

    contract = report.get("s8_contract", {})
    expected_contract = {
        "experiment_name_planned": S8_EXPERIMENT,
        "support_definition": S8_SUPPORT_DEFINITION,
        "support_channels": 1,
        "meteo_channels": 5,
        "total_input_channels": 6,
        "2022_support_source": "actual monthly processed emission prior",
        "2023_support_source": "fixed unweighted mean of 2022 training-month processed priors, then binarized",
        "train_months_2022": [2, 3, 5, 6, 8, 9],
        "validation_months_2022": [11, 12],
        "blacklist_months_2022": [7],
        "external_months_2023": list(range(1, 13)),
        "planned_seeds": [42, 43, 44],
        "base_features": 32,
        "model_trainable_parameters": 476449,
    }
    for key, expected in expected_contract.items():
        actual = contract.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Gate-S8A contract mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )

    proxy = report.get("proxy", {})
    if proxy.get("tensor_sha256") != S8_EXPECTED_PROXY_SHA256:
        raise RuntimeError("Gate-S8A fixed 2023 proxy tensor SHA256 mismatch.")
    if int(proxy.get("support_cells", -1)) != S8_EXPECTED_PROXY_SUPPORT_CELLS:
        raise RuntimeError("Gate-S8A fixed 2023 proxy support-cell count mismatch.")

    required_verification = {
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
    }
    for key, expected in required_verification.items():
        if verification.get(key) is not expected:
            raise RuntimeError(f"Gate-S8A verification flag failed: {key}")

    return {
        "s8a_preflight_report": normalized_path(preflight_report_path),
        "s8a_preflight_report_sha256": sha256_file(preflight_report_path),
        "monthly_manifest_sha256": manifest_sha,
        "base_training_script_sha256": inputs.get("base_training_script_sha256"),
        "support_definition": S8_SUPPORT_DEFINITION,
        "proxy_tensor_sha256": proxy.get("tensor_sha256"),
        "proxy_support_cells": int(proxy.get("support_cells")),
    }


# ---------------------------------------------------------------------
# Experiment specs
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MeanStateSpec:
    name: str
    mapping: str
    note: str
    uses_emission: bool
    uses_meteo: bool
    shuffle_emission: bool
    binary_support_emission: bool
    emission_channels: int
    meteo_channels: int

    @property
    def input_channels(self) -> int:
        return self.emission_channels + self.meteo_channels


EXPERIMENT_SPECS: Dict[str, MeanStateSpec] = {
    S8_EXPERIMENT: MeanStateSpec(
        name=S8_EXPERIMENT,
        mapping="Binary Processed-Prior Support + Monthly-Mean Meteo -> Monthly Mean Concentration",
        note=(
            "Gate S8 control: replace the eight processed-emission value channels with one "
            "binary non-zero-support channel while preserving the approved monthly meteorology branch."
        ),
        uses_emission=True,
        uses_meteo=True,
        shuffle_emission=False,
        binary_support_emission=True,
        emission_channels=1,
        meteo_channels=5,
    ),
}

# ---------------------------------------------------------------------
# Data manifest helpers
# ---------------------------------------------------------------------
def load_monthly_manifest(cache_dir: str) -> List[dict]:
    path = os.path.join(cache_dir, "manifests", "monthly_manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    groups = obj["groups"]
    if not isinstance(groups, list):
        raise ValueError("monthly_manifest.json missing groups list")
    relocated = []
    for original in groups:
        entry = dict(original)
        embedded = Path(str(entry["file"]))
        candidate = (
            Path(cache_dir) / "monthly_mean_full" / str(entry["split"]) / embedded.name
        ).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                "Relocated monthly cache file not found. Configure --cache-dir to the "
                f"restricted 07_phase2_meanstate_fullmonth_trainonly_union directory: {candidate}"
            )
        entry["file"] = str(candidate)
        relocated.append(entry)
    return relocated


def filter_monthly_entries(
    entries: Sequence[dict],
    *,
    split: str,
    blacklist_months: Sequence[int],
    min_days: int,
    min_valid_fraction: float,
) -> Tuple[List[dict], List[dict]]:
    kept: List[dict] = []
    dropped: List[dict] = []
    bad = set(int(m) for m in blacklist_months)
    for e in entries:
        if e.get("split") != split:
            continue
        month = int(e["month"])
        n_days = int(e["n_days"])
        valid_frac = float(e["valid_pixel_fraction"])
        reason = []
        if month in bad:
            reason.append("blacklist_month")
        if n_days < min_days:
            reason.append("low_n_days")
        if valid_frac <= min_valid_fraction:
            reason.append("low_valid_fraction")
        item = dict(e)
        item["drop_reasons"] = reason
        if reason:
            dropped.append(item)
        else:
            kept.append(item)
    kept = sorted(kept, key=lambda x: (int(x["year"]), int(x["month"])))
    dropped = sorted(dropped, key=lambda x: (int(x["year"]), int(x["month"])))
    return kept, dropped


def split_train_val_months(
    entries_2022: Sequence[dict],
    val_months: Sequence[int],
) -> Tuple[List[dict], List[dict]]:
    val_set = set(int(m) for m in val_months)
    train_entries, val_entries = [], []
    for e in entries_2022:
        if int(e["month"]) in val_set:
            val_entries.append(e)
        else:
            train_entries.append(e)
    train_entries = sorted(train_entries, key=lambda x: (int(x["year"]), int(x["month"])))
    val_entries = sorted(val_entries, key=lambda x: (int(x["year"]), int(x["month"])))
    return train_entries, val_entries


# ---------------------------------------------------------------------
# Input construction and shuffling
# ---------------------------------------------------------------------
def deterministic_roll_2d(x: torch.Tensor, key: str) -> torch.Tensor:
    """Deterministic spatial roll to destroy correspondence while preserving distribution."""
    if x.numel() == 0:
        return x
    seed = md5_int(key)
    shift_y = (seed % (H - 1)) + 1
    shift_x = ((seed // 4096) % (W - 1)) + 1
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))


def build_input_from_payload(payload: dict, spec: MeanStateSpec, *, key: str) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    if spec.uses_emission:
        emis = payload.get("emission_log_mean", None)
        if emis is None:
            raise ValueError("Emission-derived support requested but emission_log_mean is None.")
        emis = emis.float()
        if tuple(emis.shape) != (8, H, W):
            raise ValueError(f"Emission shape invalid: {tuple(emis.shape)}")
        if not bool(torch.isfinite(emis).all().item()):
            raise ValueError("Emission tensor contains NaN/Inf; stop rather than altering support by nan_to_num.")
        if not spec.binary_support_emission:
            raise RuntimeError("Gate-S8 script permits only the binary-support emission representation.")
        support = torch.any(torch.abs(emis) > S8_SUPPORT_EPSILON, dim=0, keepdim=True).float()
        if tuple(support.shape) != (1, H, W):
            raise RuntimeError(f"Binary-support shape invalid: {tuple(support.shape)}")
        unique_values = set(float(v) for v in torch.unique(support).tolist())
        if not unique_values.issubset({0.0, 1.0}):
            raise RuntimeError(f"Binary-support channel is not binary: {sorted(unique_values)}")
        parts.append(support)
    if spec.uses_meteo:
        met = payload["meteo_mean"].float()
        if tuple(met.shape) != (5, H, W):
            raise ValueError(f"Meteo mean shape invalid: {tuple(met.shape)}")
        met = torch.nan_to_num(met, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(met)
    if not parts:
        raise RuntimeError("No input branch selected.")
    x = torch.cat(parts, dim=0)
    if tuple(x.shape) != (spec.input_channels, H, W):
        raise RuntimeError(f"Constructed input shape mismatch: {tuple(x.shape)} vs {spec.input_channels}")
    return x

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class MonthlyPatchDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict],
        spec: MeanStateSpec,
        crop_size: int,
        samples_per_epoch: int = 512,
    ) -> None:
        self.entries = list(entries)
        self.spec = spec
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        if len(self.entries) == 0:
            raise ValueError("No monthly entries available for training.")
        if not (16 <= self.crop_size <= min(H, W)):
            raise ValueError(f"crop_size invalid: {self.crop_size}")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        entry = self.entries[index % len(self.entries)]
        payload = load_pt(entry["file"])
        key = Path(entry["file"]).name
        x = build_input_from_payload(payload, self.spec, key=key)
        y = payload["target_log_mean_conc"].float().unsqueeze(0)
        valid_mask = payload["valid_mask"].float().unsqueeze(0)
        count_map = payload["obs_count_map"].float().unsqueeze(0)

        max_y = H - self.crop_size
        max_x = W - self.crop_size
        top = random.randint(0, max_y) if max_y > 0 else 0
        left = random.randint(0, max_x) if max_x > 0 else 0
        ys = slice(top, top + self.crop_size)
        xs = slice(left, left + self.crop_size)

        x = x[:, ys, xs]
        y = y[:, ys, xs]
        valid_mask = valid_mask[:, ys, xs]
        count_map = count_map[:, ys, xs]
        return x, y, valid_mask, count_map


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanStateUNet(nn.Module):
    def __init__(self, spec: MeanStateSpec, base_features: int = 32):
        super().__init__()
        self.spec = spec
        bf = base_features
        self.dual = spec.uses_emission and spec.uses_meteo

        if self.dual:
            bf_half = max(8, bf // 2)
            self.emis_stem = nn.Sequential(
                nn.Conv2d(spec.emission_channels, bf_half, 3, padding=1, bias=False),
                nn.BatchNorm2d(bf_half),
                nn.GELU(),
            )
            self.met_stem = nn.Sequential(
                nn.Conv2d(spec.meteo_channels, bf_half, 3, padding=1, bias=False),
                nn.BatchNorm2d(bf_half),
                nn.GELU(),
            )
            stem_ch = bf_half * 2
        else:
            self.single_stem = nn.Sequential(
                nn.Conv2d(spec.input_channels, bf, 3, padding=1, bias=False),
                nn.BatchNorm2d(bf),
                nn.GELU(),
            )
            stem_ch = bf

        self.enc1 = ConvBlock(stem_ch, bf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(bf, bf * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(bf * 2, bf * 4)
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(bf * 4, bf * 2)
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(bf * 2, bf)
        self.out = nn.Conv2d(bf, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual:
            e = x[:, : self.spec.emission_channels]
            m = x[:, self.spec.emission_channels :]
            x0 = torch.cat([self.emis_stem(e), self.met_stem(m)], dim=1)
        else:
            x0 = self.single_stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------
def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    count_map: torch.Tensor,
) -> torch.Tensor:
    valid_mask = (valid_mask > 0.5).float()
    if valid_mask.sum() <= 0:
        return (pred - target).pow(2).mean() * 0.0
    max_count = torch.clamp(count_map.max(), min=1.0)
    weights = valid_mask * torch.clamp(count_map / max_count, min=0.0, max=1.0)
    weights = torch.where(valid_mask > 0, 0.25 + 0.75 * weights, torch.zeros_like(weights))
    loss = ((pred - target) ** 2) * weights
    return loss.sum() / torch.clamp(weights.sum(), min=1.0)


def masked_r2(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> float:
    mask = valid_mask > 0.5
    if mask.sum().item() <= 1:
        return float("nan")
    p = pred[mask]
    t = target[mask]
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - torch.mean(t)) ** 2)
    if ss_tot.item() <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def evaluate_full_monthly_set(
    model: nn.Module,
    entries: Sequence[dict],
    spec: MeanStateSpec,
    device: torch.device,
) -> dict:
    model.eval()
    all_pred = []
    all_true = []
    per_group = []
    for entry in entries:
        payload = load_pt(entry["file"])
        key = Path(entry["file"]).name
        x = build_input_from_payload(payload, spec, key=key).unsqueeze(0).to(device)
        pred = model(x).squeeze(0).squeeze(0).cpu()
        target = payload["target_log_mean_conc"].float()
        valid_mask = payload["valid_mask"].float() > 0.5
        if valid_mask.sum().item() <= 1:
            r2 = float("nan")
            rmse = float("nan")
            n_valid = int(valid_mask.sum().item())
        else:
            p = pred[valid_mask]
            t = target[valid_mask]
            all_pred.append(p)
            all_true.append(t)
            ss_res = torch.sum((t - p) ** 2)
            ss_tot = torch.sum((t - torch.mean(t)) ** 2)
            r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))
            rmse = float(torch.sqrt(torch.mean((p - t) ** 2)))
            n_valid = int(valid_mask.sum().item())
        per_group.append(
            {
                "year": int(entry["year"]),
                "split": entry["split"],
                "month": int(entry["month"]),
                "season": entry["season"],
                "n_days": int(entry["n_days"]),
                "valid_pixel_fraction": float(entry["valid_pixel_fraction"]),
                "log_r2": r2,
                "log_rmse": rmse,
                "n_valid_pixels": n_valid,
                "file": entry["file"],
            }
        )

    if all_pred:
        pred_all = torch.cat(all_pred)
        true_all = torch.cat(all_true)
        ss_res = torch.sum((true_all - pred_all) ** 2)
        ss_tot = torch.sum((true_all - torch.mean(true_all)) ** 2)
        global_r2 = float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))
        global_rmse = float(torch.sqrt(torch.mean((pred_all - true_all) ** 2)))
        global_bias = float(torch.mean(pred_all - true_all))
        n_pixels = int(true_all.numel())
    else:
        global_r2 = float("nan")
        global_rmse = float("nan")
        global_bias = float("nan")
        n_pixels = 0

    return {
        "n_groups": len(entries),
        "n_total_valid_pixels": n_pixels,
        "global": {
            "log_r2": global_r2,
            "log_rmse": global_rmse,
            "log_bias": global_bias,
        },
        "per_group": per_group,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Gate-S8 binary processed-prior support + meteorology with "
            "Gate-11B and Gate-S8A provenance plus overwrite protection."
        )
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=sorted(EXPERIMENT_SPECS.keys()),
    )
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--gate11b-report", type=str, required=True)
    parser.add_argument("--s8a-preflight-report", type=str, required=True)
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--run-mode",
        type=str,
        required=True,
        choices=["smoke", "formal"],
    )
    parser.add_argument("--val-months", type=str, default="11,12")
    parser.add_argument("--blacklist-months", type=str, default="7")
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
    gate11b_report = normalized_path(args.gate11b_report)
    s8a_preflight_report = normalized_path(args.s8a_preflight_report)
    run_root = normalized_path(args.run_root)
    run_dir = normalized_path(os.path.join(run_root, args.run_name))

    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(cache_dir)
    provenance = validate_gate11b_provenance(
        cache_dir=cache_dir,
        gate11b_report_path=gate11b_report,
    )
    s8a_provenance = validate_s8a_preflight(
        cache_dir=cache_dir,
        preflight_report_path=s8a_preflight_report,
    )

    if os.path.exists(run_dir):
        raise FileExistsError(
            "Run directory already exists. This script never overwrites: "
            f"{run_dir}"
        )

    # Smoke and formal modes have explicit contracts.
    if args.run_mode == "smoke":
        expected = {
            "epochs": 2,
            "samples_per_epoch": 32,
            "batch_size": 8,
            "crop_size": 128,
            "patience": 2,
            "base_features": 32,
            "num_workers": 0,
        }
        actual = {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "patience": args.patience,
            "base_features": args.base_features,
            "num_workers": args.num_workers,
        }
        if actual != expected:
            raise RuntimeError(
                f"Smoke settings mismatch. expected={expected}, actual={actual}"
            )
    else:
        expected = {
            "epochs": 160,
            "samples_per_epoch": 512,
            "batch_size": 8,
            "crop_size": 128,
            "patience": 25,
            "base_features": 32,
            "num_workers": 0,
        }
        actual = {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "patience": args.patience,
            "base_features": args.base_features,
            "num_workers": args.num_workers,
        }
        if actual != expected:
            raise RuntimeError(
                f"Formal settings mismatch. expected={expected}, actual={actual}"
            )

    val_months = parse_month_list(args.val_months)
    blacklist_months = parse_month_list(args.blacklist_months)
    if val_months != [11, 12]:
        raise RuntimeError(f"Unexpected validation months: {val_months}")
    if blacklist_months != [7]:
        raise RuntimeError(f"Unexpected blacklist months: {blacklist_months}")
    if args.min_train_days != 5:
        raise RuntimeError("--min-train-days must remain 5.")
    if not math.isclose(args.min_valid_fraction, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("--min-valid-fraction must remain 1e-6.")
    if not math.isclose(args.learning_rate, 3e-4, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("--learning-rate must remain 3e-4.")
    if not math.isclose(args.weight_decay, 1e-2, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("--weight-decay must remain 1e-2.")
    if args.seed not in {42, 43, 44}:
        raise RuntimeError("Only predeclared seeds 42, 43, and 44 are allowed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required but unavailable. Stop rather than silently training on CPU."
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)

    spec = EXPERIMENT_SPECS[args.experiment]
    if spec.name != S8_EXPERIMENT:
        raise RuntimeError(f"This dedicated Gate-S8 script only permits {S8_EXPERIMENT}.")
    monthly_entries = load_monthly_manifest(cache_dir)
    kept_2022, dropped_2022 = filter_monthly_entries(
        monthly_entries,
        split="2022_full",
        blacklist_months=blacklist_months,
        min_days=args.min_train_days,
        min_valid_fraction=args.min_valid_fraction,
    )
    train_entries, val_entries = split_train_val_months(
        kept_2022,
        val_months,
    )
    external_2023, dropped_2023 = filter_monthly_entries(
        monthly_entries,
        split="2023_all",
        blacklist_months=[],
        min_days=args.min_train_days,
        min_valid_fraction=args.min_valid_fraction,
    )

    train_months = [int(entry["month"]) for entry in train_entries]
    validation_months = [int(entry["month"]) for entry in val_entries]
    external_months = [int(entry["month"]) for entry in external_2023]
    if train_months != [2, 3, 5, 6, 8, 9]:
        raise RuntimeError(f"Unexpected training groups: {train_months}")
    if validation_months != [11, 12]:
        raise RuntimeError(f"Unexpected validation groups: {validation_months}")
    if external_months != list(range(1, 13)):
        raise RuntimeError(f"Unexpected 2023 groups: {external_months}")
    if len(dropped_2022) != 1 or int(dropped_2022[0]["month"]) != 7:
        raise RuntimeError(
            f"Only 2022-07 may be dropped; found={dropped_2022}"
        )
    if dropped_2023:
        raise RuntimeError(f"No 2023 month may be dropped: {dropped_2023}")

    os.makedirs(run_dir, exist_ok=False)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    print("=" * 88)
    print("Gate S8B - binary-support + meteorology mean-state training")
    print("=" * 88)
    print(f"RUN_MODE={args.run_mode}")
    print(f"EXPERIMENT={spec.name}")
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print(f"CACHE_DIR={cache_dir}")
    print(f"RUN_DIR={run_dir}")
    print(f"TRAIN_MONTHS={train_months}")
    print(f"VALIDATION_MONTHS={validation_months}")
    print(f"EXTERNAL_2023_MONTHS={external_months}")
    print("GATE11B_PROVENANCE_ACCEPTED=True")
    print("GATE_S8A_PREFLIGHT_ACCEPTED=True")
    print(f"S8_SUPPORT_DEFINITION={S8_SUPPORT_DEFINITION}")
    print("S8_SUPPORT_CHANNELS=1")
    print("S8_METEO_CHANNELS=5")
    print("S8_INPUT_CHANNELS=6")

    manifest = {
        "artifact_name": "s2_meanstate_binary_support_training_gateS8",
        "artifact_version": 1,
        "run_mode": args.run_mode,
        "run_name": args.run_name,
        "run_dir": run_dir,
        "experiment": asdict(spec),
        "cache_dir": cache_dir,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "external_2023_entries": external_2023,
        "dropped_2022_entries": dropped_2022,
        "dropped_2023_entries": dropped_2023,
        "provenance": {
            **provenance,
            **s8a_provenance,
            "training_script": normalized_path(__file__),
            "training_script_sha256": sha256_file(__file__),
        },
        "binary_support_contract": {
            "support_definition": S8_SUPPORT_DEFINITION,
            "support_epsilon": S8_SUPPORT_EPSILON,
            "support_channels": 1,
            "meteo_channels": 5,
            "total_input_channels": 6,
            "2022_support_source": "actual monthly processed emission prior",
            "2023_support_source_for_later_eval": (
                "fixed unweighted mean of 2022 training-month processed priors, then binarized"
            ),
            "expected_2023_proxy_tensor_sha256": S8_EXPECTED_PROXY_SHA256,
            "expected_2023_proxy_support_cells": S8_EXPECTED_PROXY_SUPPORT_CELLS,
        },
        "settings": {
            "val_months": val_months,
            "blacklist_months": blacklist_months,
            "min_train_days": args.min_train_days,
            "min_valid_fraction": args.min_valid_fraction,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "epochs": args.epochs,
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
        "scientific_boundary": (
            "This Gate-S8 model uses one binary channel indicating non-zero support in the inherited "
            "processed emission prior plus five monthly meteorological channels. The support mask is "
            "a property of the processed-prior construction and must not be interpreted as verified "
            "physical emission geography. November-December 2022 remains held-out model-selection "
            "validation, not an independent test. Any later 2023 emission-derived support evaluation "
            "must use the fixed 2022 train-mean processed-prior proxy and remains a proxy external test."
        ),
    }
    save_json(manifest, os.path.join(log_dir, "run_manifest.json"))

    train_ds = MonthlyPatchDataset(
        entries=train_entries,
        spec=spec,
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = MeanStateUNet(
        spec=spec,
        base_features=args.base_features,
    ).to(device)
    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"TRAINABLE_PARAMETERS={n_params}")
    if n_params != 476449:
        raise RuntimeError(f"Unexpected Gate-S8 model parameter count: {n_params} != 476449")

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
        "val_monthly_log_r2": [],
        "val_monthly_log_rmse": [],
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

        for xb, yb, mb, cb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_weighted_mse(pred, yb, mb, cb)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1
            r2 = masked_r2(
                pred.detach(),
                yb.detach(),
                mb.detach(),
            )
            if math.isfinite(r2):
                patch_r2_values.append(r2)

        scheduler.step()
        train_loss = loss_sum / max(n_batches, 1)
        train_patch_r2 = (
            float(np.mean(patch_r2_values))
            if patch_r2_values
            else float("nan")
        )

        val_result = evaluate_full_monthly_set(
            model=model,
            entries=val_entries,
            spec=spec,
            device=device,
        )
        val_log_r2 = float(val_result["global"]["log_r2"])
        val_log_rmse = float(val_result["global"]["log_rmse"])
        if not math.isfinite(val_log_r2) or not math.isfinite(val_log_rmse):
            raise RuntimeError(
                f"Non-finite validation metric at epoch {epoch + 1}: "
                f"R2={val_log_r2}, RMSE={val_log_rmse}"
            )

        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_patch_r2"].append(train_patch_r2)
        history["val_monthly_log_r2"].append(val_log_r2)
        history["val_monthly_log_rmse"].append(val_log_rmse)
        history["lr"].append(lr)

        print(
            f"EPOCH={epoch + 1}/{args.epochs} "
            f"TRAIN_LOSS={train_loss:.6f} "
            f"TRAIN_PATCH_R2={train_patch_r2:.6f} "
            f"VAL_LOG_R2={val_log_r2:.6f} "
            f"VAL_LOG_RMSE={val_log_rmse:.6f} "
            f"LR={lr:.8f}",
            flush=True,
        )

        if val_log_r2 > best_val_r2:
            best_val_r2 = val_log_r2
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_log_r2": best_val_r2,
                    "experiment": spec.name,
                    "mapping": spec.mapping,
                    "run_name": args.run_name,
                    "run_mode": args.run_mode,
                    "base_features": args.base_features,
                    "spec": asdict(spec),
                    "cache_dir": cache_dir,
                    "gate11b_report_sha256": provenance[
                        "gate11b_report_sha256"
                    ],
                    "training_script_sha256": sha256_file(__file__),
                    "s8a_preflight_report_sha256": s8a_provenance["s8a_preflight_report_sha256"],
                    "binary_support_definition": S8_SUPPORT_DEFINITION,
                    "binary_support_epsilon": S8_SUPPORT_EPSILON,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        save_json(
            history,
            os.path.join(log_dir, "training_history.json"),
        )
        save_json(
            val_result,
            os.path.join(log_dir, "latest_val_monthly_report.json"),
        )

        if patience_counter >= args.patience:
            print(
                f"EARLY_STOP=True PATIENCE={args.patience}",
                flush=True,
            )
            break

    checkpoint_path = os.path.join(ckpt_dir, "best_model.pt")
    history_path = os.path.join(log_dir, "training_history.json")
    manifest_path = os.path.join(log_dir, "run_manifest.json")
    validation_path = os.path.join(
        log_dir,
        "latest_val_monthly_report.json",
    )

    required_files = [
        checkpoint_path,
        history_path,
        manifest_path,
        validation_path,
    ]
    missing = [
        path
        for path in required_files
        if not os.path.isfile(path) or os.path.getsize(path) <= 0
    ]
    if missing:
        raise RuntimeError(f"Missing or empty training artifacts: {missing}")
    if best_epoch < 0 or not math.isfinite(best_val_r2):
        raise RuntimeError("No valid best checkpoint was selected.")

    print("GATE_S8_BINARY_SUPPORT_TRAINING_ARTIFACTS_PASS=True")
    print(f"BEST_EPOCH={best_epoch + 1}")
    print(f"BEST_VAL_LOG_R2={best_val_r2:.8f}")
    print(f"CHECKPOINT={checkpoint_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"HISTORY={history_path}")
    print(f"VALIDATION_REPORT={validation_path}")

    if args.run_mode == "smoke":
        print("GATE_S8B_BINARY_SUPPORT_TRAINING_SMOKE_PASS")
    else:
        print("GATE_S8C_BINARY_SUPPORT_FORMAL_TRAINING_PASS")


if __name__ == "__main__":
    main()
