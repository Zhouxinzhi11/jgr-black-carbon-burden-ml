# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
67b_train_s2_anomaly_clean.py

Clean second-stage daily-anomaly training script.

This script consumes the frozen lean cache produced by
65_build_clean_anomaly_cache.py and accepted by
66_audit_clean_anomaly_cache_gateA3.py.

Locked scientific scope
-----------------------
Experiments:
1. anom_meteo_only
2. anom_meteo_fire
3. anom_meteo_aef

No emission input is allowed in the anomaly branch.

The same architecture and core optimization choices are retained from the
legacy anomaly baseline so that the clean rerun isolates the corrected data
source and split contract rather than introducing a new model family.

Run modes
---------
smoke:
- engineering/dataflow validation only;
- seed 42, anom_meteo_only;
- 12 training files, 4 validation files, 2 epochs;
- result is not scientifically interpretable.

formal:
- complete 2022 training and validation splits;
- seeds 42, 43 or 44;
- exact locked hyperparameters;
- best checkpoint selected by pooled 2022 validation anomaly log-R2.

The script never overwrites an existing run directory.

Early-stopping contract
-----------------------
The CLI value --patience is measured in training epochs, not validation
events. Validation is still performed every --val-every epochs; stopping is
checked only when a validation result becomes available. This prevents the
combination patience=25 and val_every=5 from silently becoming 125 epochs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


H = 361
W = 576
X_CHANNELS = 112
AEF_SLICE = slice(48, 112)

EXPECTED_CACHE_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\08_phase2_anomaly_clean_trainonly_union"
EXPECTED_COUNTS = {
    "2022_train": 178,
    "2022_val": 61,
    "2023_all": 361,
}
EXPECTED_MANIFEST_NAMES = [
    "split_manifest.json",
    "monthly_manifest.json",
    "anomaly_manifest.json",
    "build_report.json",
]

FORMAL_CONTRACT = {
    "batch_size": 4,
    "num_workers": 0,
    "patch_size": 96,
    "patches_per_file": 4,
    "base_features": 32,
    "max_epochs": 140,
    "patience": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_every": 5,
    "train_limit": 0,
    "val_limit": 0,
}
SMOKE_CONTRACT = {
    "batch_size": 2,
    "num_workers": 0,
    "patch_size": 96,
    "patches_per_file": 2,
    "base_features": 32,
    "max_epochs": 2,
    "patience": 2,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_every": 1,
    "train_limit": 12,
    "val_limit": 4,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)


def load_pt(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"PT_NOT_FOUND: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(obj, dict), f"PT_ROOT_NOT_DICT: {path}")
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finite_number(value: float, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return masked_mean(loss, mask)


def masked_r2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    valid = mask > 0.5
    if int(valid.sum().item()) < 16:
        return float("nan")
    p = pred[valid]
    t = target[valid]
    sse = torch.sum((p - t) ** 2)
    sst = torch.sum((t - t.mean()) ** 2)
    if float(sst.item()) < eps:
        return 0.0
    return float((1.0 - sse / (sst + eps)).item())


@dataclass(frozen=True)
class AnomalySpec:
    name: str
    mapping: str
    note: str
    uses_fire: bool
    uses_aef: bool
    use_meteo_month_context: bool = True
    use_fire_month_context: bool = True

    @property
    def dynamic_channels(self) -> int:
        return 5 + (1 if self.uses_fire else 0)

    @property
    def context_channels(self) -> int:
        channels = 0
        if self.use_meteo_month_context:
            channels += 5
        if self.uses_fire and self.use_fire_month_context:
            channels += 1
        if self.uses_aef:
            channels += 64
        return channels

    @property
    def input_channels(self) -> int:
        return self.dynamic_channels + self.context_channels


EXPERIMENT_SPECS: Dict[str, AnomalySpec] = {
    "anom_meteo_only": AnomalySpec(
        name="anom_meteo_only",
        mapping="Meteo Anomaly + Monthly Meteo Context -> Daily Concentration Anomaly",
        note=(
            "Clean anomaly baseline using meteorological anomalies and monthly "
            "meteorological context."
        ),
        uses_fire=False,
        uses_aef=False,
    ),
    "anom_meteo_fire": AnomalySpec(
        name="anom_meteo_fire",
        mapping=(
            "Meteo Anomaly + Fire Anomaly + Monthly Meteo/Fire Context "
            "-> Daily Concentration Anomaly"
        ),
        note=(
            "Adds fire anomaly and monthly fire context to the meteorological "
            "anomaly baseline."
        ),
        uses_fire=True,
        uses_aef=False,
    ),
    "anom_meteo_aef": AnomalySpec(
        name="anom_meteo_aef",
        mapping=(
            "Meteo Anomaly + Monthly Meteo Context + AEF Context "
            "-> Daily Concentration Anomaly"
        ),
        note=(
            "Adds static AEF context loaded from the immutable source tensor; "
            "AEF is not duplicated in the lean anomaly cache."
        ),
        uses_fire=False,
        uses_aef=True,
    ),
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnomalyUNet(nn.Module):
    def __init__(self, spec: AnomalySpec, base_features: int = 32):
        super().__init__()
        self.spec = spec
        bf = int(base_features)
        self.dual = spec.context_channels > 0

        if self.dual:
            half = max(8, bf // 2)
            self.dynamic_stem = nn.Sequential(
                nn.Conv2d(spec.dynamic_channels, half, 3, padding=1, bias=False),
                nn.BatchNorm2d(half),
                nn.GELU(),
            )
            self.context_stem = nn.Sequential(
                nn.Conv2d(spec.context_channels, half, 3, padding=1, bias=False),
                nn.BatchNorm2d(half),
                nn.GELU(),
            )
            stem_channels = half * 2
        else:
            self.single_stem = nn.Sequential(
                nn.Conv2d(spec.input_channels, bf, 3, padding=1, bias=False),
                nn.BatchNorm2d(bf),
                nn.GELU(),
            )
            stem_channels = bf

        self.enc1 = ConvBlock(stem_channels, bf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(bf, bf * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(bf * 2, bf * 4, dropout=0.1)
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = ConvBlock(bf * 4, bf * 2)
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = ConvBlock(bf * 2, bf)
        self.out = nn.Conv2d(bf, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual:
            dynamic = x[:, : self.spec.dynamic_channels]
            context = x[:, self.spec.dynamic_channels :]
            x0 = torch.cat(
                [self.dynamic_stem(dynamic), self.context_stem(context)],
                dim=1,
            )
        else:
            x0 = self.single_stem(x)

        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool1(e1))
        bottleneck = self.bottleneck(self.pool2(e2))

        d2 = self.up2(bottleneck)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class LeanAnomalyReader:
    def __init__(
        self,
        cache_dir: Path,
        *,
        aef_source_root_2022: Path | None = None,
        aef_source_root_2023: Path | None = None,
    ):
        self.cache_dir = cache_dir.resolve()
        self.aef_source_roots = {
            2022: None if aef_source_root_2022 is None else aef_source_root_2022.resolve(),
            2023: None if aef_source_root_2023 is None else aef_source_root_2023.resolve(),
        }
        self._background_cache: dict[str, dict[str, Any]] = {}

    def resolve_daily_path(self, entry: dict[str, Any]) -> Path:
        embedded = Path(str(entry["file"]))
        candidate = (
            self.cache_dir / "daily_anomaly" / str(entry["split"]) / embedded.name
        ).resolve()
        if candidate.is_file():
            return candidate
        embedded = embedded.expanduser().resolve()
        require(
            embedded.is_file(),
            "DAILY_FILE_NOT_FOUND: configure --cache-dir to the restricted "
            f"08_phase2_anomaly_clean_trainonly_union directory; missing={candidate}",
        )
        return embedded

    def load_background(self, daily_payload: dict[str, Any]) -> dict[str, Any]:
        path_value = str(daily_payload["background_month_file"])
        embedded = Path(path_value)
        candidate = (
            self.cache_dir
            / "monthly_background"
            / str(daily_payload["split"])
            / embedded.name
        ).resolve()
        path = candidate if candidate.is_file() else embedded.expanduser().resolve()
        require(
            path.is_file(),
            "BACKGROUND_NOT_FOUND: configure --cache-dir to the restricted anomaly cache; "
            f"missing={candidate}",
        )
        require(
            canonical(path).startswith(canonical(self.cache_dir)),
            f"BACKGROUND_OUTSIDE_CACHE: {path}",
        )
        key = canonical(path)
        if key not in self._background_cache:
            payload = load_pt(path)
            require(
                payload.get("artifact_name") == "clean_anomaly_monthly_background",
                f"BACKGROUND_ARTIFACT_MISMATCH: {path}",
            )
            required = [
                "target_log_month_mean",
                "month_valid_mask",
                "meteo_month_mean",
                "fire_month_mean",
            ]
            for field in required:
                require(field in payload, f"BACKGROUND_MISSING_{field}: {path}")
            self._background_cache[key] = payload
        return self._background_cache[key]

    def load_aef(self, daily_payload: dict[str, Any]) -> torch.Tensor:
        year = int(daily_payload["year"])
        filename = str(daily_payload.get("source_file_name") or Path(str(daily_payload["source_file"])).name)
        root = self.aef_source_roots.get(year)
        embedded = Path(str(daily_payload["source_file"]))
        source_path = (root / filename).resolve() if root is not None else embedded.expanduser().resolve()
        require(
            source_path.is_file(),
            f"AEF_SOURCE_NOT_FOUND_YEAR_{year}: pass --aef-source-root-{year}; missing={source_path}",
        )
        expected_hash = str(daily_payload.get("source_file_sha256", ""))
        if expected_hash:
            require(sha256_file(source_path) == expected_hash, f"AEF_SOURCE_SHA256_MISMATCH: {source_path}")
        source = load_pt(source_path)
        require("x" in source, f"SOURCE_MISSING_X: {source_path}")
        x = source["x"]
        require(
            isinstance(x, torch.Tensor) and tuple(x.shape) == (X_CHANNELS, H, W),
            f"SOURCE_X_CONTRACT_FAIL: {source_path}",
        )
        aef = x[AEF_SLICE].float()
        require(tuple(aef.shape) == (64, H, W), f"AEF_SHAPE_FAIL: {source_path}")
        require(bool(torch.isfinite(aef).all().item()), f"AEF_NONFINITE: {source_path}")
        return aef

    def load_example(
        self,
        entry: dict[str, Any],
        spec: AnomalySpec,
    ) -> dict[str, torch.Tensor]:
        daily_path = self.resolve_daily_path(entry)
        require(
            canonical(daily_path).startswith(canonical(self.cache_dir)),
            f"DAILY_FILE_OUTSIDE_CACHE: {daily_path}",
        )
        payload = load_pt(daily_path)
        require(
            payload.get("artifact_name") == "clean_daily_anomaly",
            f"DAILY_ARTIFACT_MISMATCH: {daily_path}",
        )

        background = self.load_background(payload)

        dynamic_parts: list[torch.Tensor] = [payload["meteo_anomaly"].float()]
        context_parts: list[torch.Tensor] = []

        if spec.uses_fire:
            dynamic_parts.append(payload["fire_anomaly"].float())
        if spec.use_meteo_month_context:
            context_parts.append(background["meteo_month_mean"].float())
        if spec.uses_fire and spec.use_fire_month_context:
            context_parts.append(background["fire_month_mean"].float())
        if spec.uses_aef:
            context_parts.append(self.load_aef(payload))

        dynamic = torch.cat(dynamic_parts, dim=0)
        context = torch.cat(context_parts, dim=0) if context_parts else None
        x = torch.cat([dynamic, context], dim=0) if context is not None else dynamic

        target = payload["target_log_anomaly"].float()
        mask = payload["anomaly_valid_mask"].float()
        background_target = background["target_log_month_mean"].float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if background_target.ndim == 2:
            background_target = background_target.unsqueeze(0)

        require(tuple(x.shape) == (spec.input_channels, H, W),
                f"INPUT_SHAPE_FAIL_{spec.name}: {tuple(x.shape)}")
        require(tuple(target.shape) == (1, H, W), f"TARGET_SHAPE_FAIL: {daily_path}")
        require(tuple(mask.shape) == (1, H, W), f"MASK_SHAPE_FAIL: {daily_path}")
        require(
            tuple(background_target.shape) == (1, H, W),
            f"TARGET_BACKGROUND_SHAPE_FAIL: {daily_path}",
        )
        require(bool(torch.isfinite(x).all().item()), f"INPUT_NONFINITE: {daily_path}")
        require(bool(torch.isfinite(target).all().item()), f"TARGET_NONFINITE: {daily_path}")
        require(bool(torch.isfinite(mask).all().item()), f"MASK_NONFINITE: {daily_path}")
        require(
            int((mask > 0.5).sum().item()) > 0,
            f"EXAMPLE_HAS_NO_VALID_PIXELS: {daily_path}",
        )

        return {
            "x": torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            "target": target,
            "mask": mask,
            "target_background": background_target,
        }


class AnomalyPatchDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict[str, Any]],
        reader: LeanAnomalyReader,
        spec: AnomalySpec,
        patch_size: int,
        patches_per_file: int,
        min_valid_fraction: float = 0.03,
    ):
        self.entries = list(entries)
        self.reader = reader
        self.spec = spec
        self.patch_size = int(patch_size)
        self.patches_per_file = int(patches_per_file)
        self.min_valid_fraction = float(min_valid_fraction)
        require(bool(self.entries), "EMPTY_ANOMALY_PATCH_DATASET")
        require(1 <= self.patch_size <= min(H, W), "PATCH_SIZE_OUT_OF_RANGE")
        require(self.patches_per_file >= 1, "PATCHES_PER_FILE_INVALID")

    def __len__(self) -> int:
        return len(self.entries)

    def sample_patch(self, mask: torch.Tensor) -> tuple[int, int]:
        max_y = H - self.patch_size
        max_x = W - self.patch_size
        best = (0, 0, -1.0)
        for _ in range(24):
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)
            fraction = float(
                mask[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
                .mean()
                .item()
            )
            if fraction >= self.min_valid_fraction:
                return y0, x0
            if fraction > best[2]:
                best = (y0, x0, fraction)
        return int(best[0]), int(best[1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.reader.load_example(self.entries[index], self.spec)
        x = example["x"]
        target = example["target"]
        mask = example["mask"]

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for _ in range(self.patches_per_file):
            y0, x0 = self.sample_patch(mask)
            y1 = y0 + self.patch_size
            x1 = x0 + self.patch_size
            xs.append(x[:, y0:y1, x0:x1].contiguous())
            ys.append(target[:, y0:y1, x0:x1].contiguous())
            masks.append(mask[:, y0:y1, x0:x1].contiguous())

        return {
            "x": torch.stack(xs, dim=0),
            "y": torch.stack(ys, dim=0),
            "mask": torch.stack(masks, dim=0),
        }


def flatten_patch_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    x = torch.stack([sample["x"] for sample in batch], dim=0)
    y = torch.stack([sample["y"] for sample in batch], dim=0)
    mask = torch.stack([sample["mask"] for sample in batch], dim=0)
    require(x.ndim == 5, f"EXPECTED_X_5D_GOT={tuple(x.shape)}")
    batch_size, n_patches = x.shape[:2]
    return {
        "x": x.reshape(batch_size * n_patches, *x.shape[2:]).contiguous(),
        "y": y.reshape(batch_size * n_patches, *y.shape[2:]).contiguous(),
        "mask": mask.reshape(batch_size * n_patches, *mask.shape[2:]).contiguous(),
    }


class StreamingRegression:
    def __init__(self) -> None:
        self.n = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_error = 0.0
        self.sum_squared_error = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        require(pred.shape == true.shape, "STREAMING_METRIC_SHAPE_MISMATCH")
        if pred.size == 0:
            return
        p = pred.astype(np.float64, copy=False).ravel()
        t = true.astype(np.float64, copy=False).ravel()
        error = p - t
        self.n += int(p.size)
        self.sum_pred += float(p.sum())
        self.sum_true += float(t.sum())
        self.sum_pred2 += float(np.dot(p, p))
        self.sum_true2 += float(np.dot(t, t))
        self.sum_pred_true += float(np.dot(p, t))
        self.sum_error += float(error.sum())
        self.sum_squared_error += float(np.dot(error, error))

    def finalize(self) -> dict[str, float | int]:
        require(self.n > 0, "STREAMING_METRIC_EMPTY")
        n = float(self.n)
        true_sst = self.sum_true2 - (self.sum_true * self.sum_true) / n
        pred_ss = self.sum_pred2 - (self.sum_pred * self.sum_pred) / n
        covariance = self.sum_pred_true - (self.sum_pred * self.sum_true) / n

        r2 = 0.0 if true_sst <= 1e-12 else 1.0 - self.sum_squared_error / true_sst
        rmse = math.sqrt(self.sum_squared_error / n)
        bias = self.sum_error / n
        corr_denom = math.sqrt(max(pred_ss, 0.0) * max(true_sst, 0.0))
        corr_defined = corr_denom > 1e-12
        corr = covariance / corr_denom if corr_defined else 0.0

        return {
            "n_values": self.n,
            "log_r2": finite_number(r2, "R2"),
            "log_rmse": finite_number(rmse, "RMSE"),
            "log_bias": finite_number(bias, "BIAS"),
            "log_corr": finite_number(corr, "CORR"),
            "log_corr_defined": corr_defined,
        }


@torch.no_grad()
def evaluate_full_image(
    model: nn.Module,
    entries: Sequence[dict[str, Any]],
    reader: LeanAnomalyReader,
    spec: AnomalySpec,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    anomaly_metrics = StreamingRegression()
    reconstruction_metrics = StreamingRegression()
    valid_days = 0
    coverage: list[float] = []

    for entry in entries:
        example = reader.load_example(entry, spec)
        x = example["x"].unsqueeze(0).to(device, non_blocking=True)
        target = example["target"]
        mask = example["mask"]
        target_background = example["target_background"]

        pred_anomaly = model(x).detach().cpu().squeeze(0)
        valid = (mask > 0.5).squeeze(0).numpy()
        if int(valid.sum()) < 16:
            continue

        true_anomaly = target.squeeze(0).numpy()[valid]
        pred_anomaly_values = pred_anomaly.squeeze(0).numpy()[valid]
        true_daily = (target + target_background).squeeze(0).numpy()[valid]
        pred_daily = (pred_anomaly + target_background).squeeze(0).numpy()[valid]

        anomaly_metrics.update(pred_anomaly_values, true_anomaly)
        reconstruction_metrics.update(pred_daily, true_daily)
        valid_days += 1
        coverage.append(float(mask.mean().item()))

        del x, pred_anomaly

    require(valid_days > 0, "NO_VALID_DAYS_IN_EVALUATION")
    anomaly = anomaly_metrics.finalize()
    reconstruction = reconstruction_metrics.finalize()

    return {
        "valid_days": valid_days,
        "obs_coverage_mean": float(np.mean(coverage)),
        "log_r2_anom": anomaly["log_r2"],
        "log_rmse_anom": anomaly["log_rmse"],
        "log_bias_anom": anomaly["log_bias"],
        "log_corr_anom": anomaly["log_corr"],
        "log_r2_recon": reconstruction["log_r2"],
        "log_rmse_recon": reconstruction["log_rmse"],
        "log_bias_recon": reconstruction["log_bias"],
        "n_values": anomaly["n_values"],
    }


def validate_cache_provenance(
    cache_dir: Path,
    audit_path: Path,
) -> dict[str, Any]:
    audit = load_json(audit_path)
    require(
        audit.get("artifact_name") == "clean_anomaly_cache_independent_audit",
        "CACHE_AUDIT_ARTIFACT_MISMATCH",
    )
    require(int(audit.get("artifact_version", -1)) == 1, "CACHE_AUDIT_VERSION_MISMATCH")
    require(
        audit.get("verification", {}).get("overall_pass") is True,
        "CACHE_AUDIT_NOT_PASS",
    )
    require(
        audit.get("decision") == "PASS_FREEZE_CLEAN_ANOMALY_CACHE",
        "CACHE_AUDIT_DECISION_MISMATCH",
    )
    require(canonical(audit.get("cache_dir", "")) == canonical(cache_dir),
            "CACHE_AUDIT_DIR_MISMATCH")

    counts = audit.get("counts", {})
    require(int(counts.get("2022_train", -1)) == EXPECTED_COUNTS["2022_train"],
            "AUDIT_TRAIN_COUNT_MISMATCH")
    require(int(counts.get("2022_val", -1)) == EXPECTED_COUNTS["2022_val"],
            "AUDIT_VAL_COUNT_MISMATCH")
    require(int(counts.get("2023_external", -1)) == EXPECTED_COUNTS["2023_all"],
            "AUDIT_EXTERNAL_COUNT_MISMATCH")
    require(int(counts.get("2022_blacklist_excluded", -1)) == 1,
            "AUDIT_BLACKLIST_COUNT_MISMATCH")

    build_seal_path = cache_dir / "manifests" / "build_seal.json"
    build_seal = load_json(build_seal_path)
    expected_build_seal_sha = (
        audit.get("input_provenance", {})
        .get("build_seal", {})
        .get("sha256")
    )
    require(sha256_file(build_seal_path) == expected_build_seal_sha,
            "BUILD_SEAL_CHANGED_AFTER_AUDIT")

    manifest_hashes = build_seal.get("manifest_sha256", {})
    for name in EXPECTED_MANIFEST_NAMES:
        path = cache_dir / "manifests" / name
        require(sha256_file(path) == manifest_hashes.get(name),
                f"MANIFEST_CHANGED_AFTER_AUDIT_{name}")

    return {
        "audit": audit,
        "audit_sha256": sha256_file(audit_path),
        "build_seal_path": str(build_seal_path),
        "build_seal_sha256": sha256_file(build_seal_path),
        "manifest_sha256": manifest_hashes,
    }


def load_entries(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    manifest_path = cache_dir / "manifests" / "anomaly_manifest.json"
    manifest = load_json(manifest_path)
    require(
        manifest.get("artifact_name") == "clean_anomaly_daily_manifest",
        "ANOMALY_MANIFEST_ARTIFACT_MISMATCH",
    )
    entries: dict[str, list[dict[str, Any]]] = {}
    for group in manifest.get("groups", []):
        split = str(group["split"])
        files = list(group.get("files", []))
        require(int(group.get("n_files", -1)) == len(files),
                f"ANOMALY_GROUP_COUNT_MISMATCH_{split}")
        entries[split] = files

    require(len(entries.get("2022_train", [])) == EXPECTED_COUNTS["2022_train"],
            "TRAIN_ENTRY_COUNT_MISMATCH")
    require(len(entries.get("2022_val", [])) == EXPECTED_COUNTS["2022_val"],
            "VAL_ENTRY_COUNT_MISMATCH")
    require(len(entries.get("2023_all", [])) == EXPECTED_COUNTS["2023_all"],
            "EXTERNAL_ENTRY_COUNT_MISMATCH")
    return entries


def validate_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    expected = SMOKE_CONTRACT if args.run_mode == "smoke" else FORMAL_CONTRACT
    actual = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "patch_size": args.patch_size,
        "patches_per_file": args.patches_per_file,
        "base_features": args.base_features,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_every": args.val_every,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
    }

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            require(
                math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12),
                f"{args.run_mode.upper()}_CONTRACT_MISMATCH_{key}: "
                f"{actual_value} != {expected_value}",
            )
        else:
            require(
                actual_value == expected_value,
                f"{args.run_mode.upper()}_CONTRACT_MISMATCH_{key}: "
                f"{actual_value} != {expected_value}",
            )

    if args.run_mode == "smoke":
        require(args.experiment == "anom_meteo_only", "SMOKE_EXPERIMENT_LOCKED")
        require(args.seed == 42, "SMOKE_SEED_LOCKED_TO_42")
    else:
        require(args.seed in (42, 43, 44), "FORMAL_SEED_NOT_ALLOWED")

    return expected


def probe_all_input_modes(
    reader: LeanAnomalyReader,
    entries: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    selected = {
        "2022_train": entries["2022_train"][0],
        "2022_val": entries["2022_val"][0],
        "2023_all": entries["2023_all"][0],
    }
    for split, entry in selected.items():
        for name, spec in EXPERIMENT_SPECS.items():
            example = reader.load_example(entry, spec)
            probes.append(
                {
                    "split": split,
                    "experiment": name,
                    "input_shape": list(example["x"].shape),
                    "target_shape": list(example["target"].shape),
                    "mask_shape": list(example["mask"].shape),
                    "n_valid_pixels": int((example["mask"] > 0.5).sum().item()),
                    "finite": True,
                }
            )
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train clean anomaly baseline.")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENT_SPECS))
    parser.add_argument("--run-mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--cache-dir", default=EXPECTED_CACHE_DIR)
    parser.add_argument("--aef-source-root-2022")
    parser.add_argument("--aef-source-root-2023")
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patches-per-file", type=int, required=True)
    parser.add_argument("--base-features", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--val-every", type=int, required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--val-limit", type=int, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    audit_path = Path(args.cache_audit).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    require(cache_dir.name == "08_phase2_anomaly_clean_trainonly_union",
            f"UNEXPECTED_CACHE_DIRECTORY_NAME: {cache_dir.name}")
    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(audit_path.is_file(), f"CACHE_AUDIT_NOT_FOUND: {audit_path}")
    require(not run_dir.exists(), f"STOP_RUN_DIR_EXISTS: {run_dir}")
    require(run_dir.parent.is_dir(), f"RUN_PARENT_NOT_FOUND: {run_dir.parent}")

    contract = validate_run_contract(args)
    provenance = validate_cache_provenance(cache_dir, audit_path)

    if args.require_cuda:
        require(torch.cuda.is_available(), "CUDA_REQUIRED_BUT_UNAVAILABLE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CLEAN_ANOMALY_TRAINING_REQUIRES_CUDA")

    set_seed(args.seed)
    spec = EXPERIMENT_SPECS[args.experiment]
    entries = load_entries(cache_dir)
    reader = LeanAnomalyReader(
        cache_dir,
        aef_source_root_2022=(None if args.aef_source_root_2022 is None else Path(args.aef_source_root_2022).expanduser()),
        aef_source_root_2023=(None if args.aef_source_root_2023 is None else Path(args.aef_source_root_2023).expanduser()),
    )
    input_probes = probe_all_input_modes(reader, entries)

    train_entries = list(entries["2022_train"])
    val_entries = list(entries["2022_val"])
    external_entries = list(entries["2023_all"])

    if args.train_limit > 0:
        train_entries = train_entries[: args.train_limit]
    if args.val_limit > 0:
        val_entries = val_entries[: args.val_limit]

    require(bool(train_entries), "EMPTY_EFFECTIVE_TRAIN_SPLIT")
    require(bool(val_entries), "EMPTY_EFFECTIVE_VAL_SPLIT")

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    train_dataset = AnomalyPatchDataset(
        train_entries,
        reader,
        spec,
        patch_size=args.patch_size,
        patches_per_file=args.patches_per_file,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=flatten_patch_batch,
        generator=generator,
    )

    model = AnomalyUNet(spec, base_features=args.base_features).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=24,
        T_mult=1,
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 96)
    print("Clean anomaly training")
    print("=" * 96)
    print(f"EXPERIMENT={spec.name}")
    print(f"RUN_MODE={args.run_mode}")
    print(f"SEED={args.seed}")
    print(f"RUN_DIR={run_dir}")
    print(f"CACHE_DIR={cache_dir}")
    print(f"CACHE_AUDIT={audit_path}")
    print(f"DEVICE={device.type}")
    print(f"CUDA_DEVICE_NAME={torch.cuda.get_device_name(0)}")
    print(f"TRAIN_FILES_EFFECTIVE={len(train_entries)}")
    print(f"VAL_FILES_EFFECTIVE={len(val_entries)}")
    print(f"EXTERNAL_FILES_MANIFEST={len(external_entries)}")
    print(f"INPUT_CHANNELS={spec.input_channels}")
    print(f"TRAINABLE_PARAMETERS={trainable_parameters}")
    print(f"EARLY_STOPPING_PATIENCE_EPOCHS={args.patience}")
    print(f"VALIDATION_FREQUENCY_EPOCHS={args.val_every}")
    print(f"INPUT_MODE_PROBES={len(input_probes)}")
    print("MODEL_ARCHITECTURE=legacy_anomaly_unet_preserved")
    print("EMISSION_INPUT_USED=False")
    print("SCIENTIFIC_RESULT_INTERPRETABLE=False" if args.run_mode == "smoke"
          else "SCIENTIFIC_RESULT_INTERPRETABLE_AFTER_EVALUATION_ONLY=True")
    print()

    best_epoch = -1
    best_val_r2 = -float("inf")
    epochs_since_best = 0
    history: list[dict[str, Any]] = []
    latest_val_report: dict[str, Any] | None = None
    training_started = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_r2s: list[float] = []

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_smooth_l1(pred, y, mask)
            require(bool(torch.isfinite(loss).item()), f"NONFINITE_TRAIN_LOSS_EPOCH={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            patch_r2 = masked_r2(pred.detach(), y, mask)
            if math.isfinite(patch_r2):
                train_r2s.append(patch_r2)

        scheduler.step(epoch + 1)

        train_loss = finite_number(float(np.mean(train_losses)), f"TRAIN_LOSS_EPOCH_{epoch}")
        train_patch_r2 = (
            finite_number(float(np.mean(train_r2s)), f"TRAIN_R2_EPOCH_{epoch}")
            if train_r2s
            else 0.0
        )

        did_validate = epoch == 1 or epoch % args.val_every == 0
        if did_validate:
            val_metrics = evaluate_full_image(
                model,
                val_entries,
                reader,
                spec,
                device,
            )
            latest_val_report = {
                "epoch": epoch,
                "experiment": spec.name,
                "run_mode": args.run_mode,
                "seed": args.seed,
                "metrics": val_metrics,
                "validation_entries": len(val_entries),
            }
            save_json(latest_val_report, log_dir / "latest_val_full_report.json")
            val_r2 = finite_number(val_metrics["log_r2_anom"], f"VAL_R2_EPOCH_{epoch}")
        else:
            val_metrics = {
                "valid_days": 0,
                "obs_coverage_mean": None,
                "log_r2_anom": None,
                "log_rmse_anom": None,
                "log_bias_anom": None,
                "log_corr_anom": None,
                "log_corr_anom_defined": None,
                "log_r2_recon": None,
                "log_rmse_recon": None,
                "log_bias_recon": None,
                "n_values": 0,
            }
            val_r2 = float("nan")

        lr_now = float(optimizer.param_groups[0]["lr"])
        print(
            f"EPOCH={epoch}/{args.max_epochs} "
            f"TRAIN_LOSS={train_loss:.6f} "
            f"TRAIN_PATCH_R2={train_patch_r2:.6f} "
            f"VAL_ANOM_LOG_R2={val_metrics['log_r2_anom']} "
            f"VAL_ANOM_LOG_RMSE={val_metrics['log_rmse_anom']} "
            f"VAL_RECON_LOG_R2={val_metrics['log_r2_recon']} "
            f"LR={lr_now:.8f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_patch_loss": train_loss,
                "train_patch_r2": train_patch_r2,
                "did_validate": did_validate,
                "validation": val_metrics,
                "lr": lr_now,
            }
        )

        if did_validate:
            improved = val_r2 > best_val_r2 + 1e-6
            if improved:
                best_epoch = epoch
                best_val_r2 = val_r2
                epochs_since_best = 0
                checkpoint = {
                    "artifact_name": "clean_anomaly_best_checkpoint",
                    "artifact_version": 1,
                    "model_state": model.state_dict(),
                    "spec": asdict(spec),
                    "base_features": args.base_features,
                    "run_mode": args.run_mode,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_log_r2_anom": best_val_r2,
                    "trainable_parameters": trainable_parameters,
                    "cache_audit_sha256": provenance["audit_sha256"],
                    "anomaly_manifest_sha256": provenance["manifest_sha256"][
                        "anomaly_manifest.json"
                    ],
                }
                torch.save(checkpoint, checkpoint_dir / "best_model.pt")
                print(
                    f"BEST_CHECKPOINT_SAVED=True "
                    f"BEST_EPOCH={best_epoch} "
                    f"BEST_VAL_ANOM_LOG_R2={best_val_r2:.8f}"
                )
            else:
                require(best_epoch >= 1, "BEST_EPOCH_INVALID_DURING_EARLY_STOP")
                epochs_since_best = epoch - best_epoch

            if best_epoch >= 1 and (epoch - best_epoch) >= args.patience:
                print(
                    f"EARLY_STOP=True "
                    f"PATIENCE_EPOCHS={args.patience} "
                    f"EPOCHS_SINCE_BEST={epoch - best_epoch} "
                    f"VALIDATION_FREQUENCY_EPOCHS={args.val_every}"
                )
                break

    completed_epochs = len(history)
    require(completed_epochs >= 1, "NO_TRAINING_EPOCH_COMPLETED")
    require(best_epoch >= 1, "BEST_CHECKPOINT_NOT_CREATED")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    require(checkpoint_path.is_file(), "BEST_CHECKPOINT_FILE_MISSING")
    require(latest_val_report is not None, "LATEST_VALIDATION_REPORT_MISSING")

    history_path = log_dir / "training_history.json"
    manifest_path = log_dir / "run_manifest.json"

    save_json(
        {
            "artifact_name": "clean_anomaly_training_history",
            "artifact_version": 1,
            "history": history,
        },
        history_path,
    )

    manifest = {
        "artifact_name": "clean_anomaly_training_run",
        "artifact_version": 1,
        "task": "s2_clean_daily_anomaly",
        "experiment": spec.name,
        "mapping": spec.mapping,
        "note": spec.note,
        "run_mode": args.run_mode,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "cache_audit": {
            "path": str(audit_path),
            "sha256": provenance["audit_sha256"],
            "accepted": True,
        },
        "build_seal": {
            "path": provenance["build_seal_path"],
            "sha256": provenance["build_seal_sha256"],
        },
        "manifest_sha256": provenance["manifest_sha256"],
        "training_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "spec": asdict(spec),
        "contract": dict(contract),
        "effective_counts": {
            "train_files": len(train_entries),
            "val_files": len(val_entries),
            "external_manifest_files": len(external_entries),
        },
        "input_mode_probes": input_probes,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "external_entries": external_entries,
        "device": {
            "type": device.type,
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
        "determinism": {
            "torch_use_deterministic_algorithms_warn_only": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "optimizer": "AdamW",
        "scheduler": {
            "name": "CosineAnnealingWarmRestarts",
            "T_0": 24,
            "T_mult": 1,
        },
        "early_stopping": {
            "monitor": "held-out_2022_validation_pooled_log_r2_anom",
            "minimum_delta": 1e-6,
            "patience": args.patience,
            "patience_unit": "epochs",
            "validation_frequency_epochs": args.val_every,
            "stopping_check": "validation_epochs_only",
        },
        "loss": {
            "name": "masked_smooth_l1",
            "beta": 0.05,
        },
        "gradient_clip_max_norm": 3.0,
        "trainable_parameters": trainable_parameters,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_val_log_r2_anom": best_val_r2,
        "scientific_result_interpretable": args.run_mode == "formal",
        "formal_result_ready_for_paper": False,
    }
    save_json(manifest, manifest_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        checkpoint.get("artifact_name") == "clean_anomaly_best_checkpoint",
        "CHECKPOINT_ARTIFACT_MISMATCH",
    )
    require(int(checkpoint.get("seed", -1)) == args.seed, "CHECKPOINT_SEED_MISMATCH")
    require(checkpoint.get("spec", {}).get("name") == spec.name, "CHECKPOINT_SPEC_MISMATCH")
    require(
        int(checkpoint.get("best_epoch", -1)) == best_epoch,
        "CHECKPOINT_BEST_EPOCH_MISMATCH",
    )
    require(
        math.isclose(
            float(checkpoint.get("best_val_log_r2_anom")),
            best_val_r2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "CHECKPOINT_BEST_METRIC_MISMATCH",
    )

    print()
    pass_label = (
        "GATE_A4_CLEAN_ANOMALY_TRAINING_SMOKE_PASS"
        if args.run_mode == "smoke"
        else f"CLEAN_ANOMALY_FORMAL_TRAINING_SEED{args.seed}_PASS"
    )
    print(pass_label)
    print(f"EXPERIMENT={spec.name}")
    print(f"RUN_MODE={args.run_mode}")
    print(f"SEED={args.seed}")
    print(f"COMPLETED_EPOCHS={completed_epochs}")
    print(f"BEST_EPOCH={best_epoch}")
    print(f"BEST_VAL_ANOM_LOG_R2={best_val_r2:.8f}")
    print(f"TRAINABLE_PARAMETERS={trainable_parameters}")
    print(f"TRAIN_FILES_EFFECTIVE={len(train_entries)}")
    print(f"VAL_FILES_EFFECTIVE={len(val_entries)}")
    print(f"EARLY_STOPPING_PATIENCE_UNIT=epochs")
    print(f"EARLY_STOPPING_PATIENCE={args.patience}")
    print(f"VALIDATION_FREQUENCY_EPOCHS={args.val_every}")
    print(f"INPUT_MODE_PROBES_PASS={len(input_probes) == 9}")
    print("CACHE_AUDIT_PROVENANCE_PASS=True")
    print("LEAN_CACHE_DATAFLOW_PASS=True")
    print("CHECKPOINT_ARTIFACT_PASS=True")
    print(
        "SCIENTIFIC_RESULT_INTERPRETABLE=False"
        if args.run_mode == "smoke"
        else "SCIENTIFIC_EVALUATION_NOT_YET_RUN=True"
    )
    print(f"ELAPSED_MINUTES={(time.time() - training_started) / 60.0:.2f}")
    print(f"CHECKPOINT={checkpoint_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"HISTORY={history_path}")
    print(f"LATEST_VALIDATION_REPORT={log_dir / 'latest_val_full_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
