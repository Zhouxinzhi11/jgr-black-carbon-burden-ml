# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
52_eval_s2_meanstate_clean_formal_run.py

Reusable provenance-locked formal evaluation for any clean mean-state run
created by 47_train_s2_meanstate_clean.py and audited by
51_audit_s2_meanstate_clean_formal_run.py.

Evaluation contract
-------------------
2022:
- held-out model-selection validation;
- months 11 and 12;
- actual cached inputs are used.

2023:
- months 01-12;
- meteorology-only: ordinary external evaluation, no emission proxy;
- emission-input configurations: proxy external evaluation using the
  unweighted arithmetic mean of the six 2022 training-month processed
  emission-prior tensors;
- shuffled configuration applies the same deterministic 2-D roll rule as
  training after selecting the actual/proxy emission input.

Outputs:
- pooled, per-month, seasonal, and regional log-space metrics;
- strict JSON with no NaN/Infinity;
- provenance hashes and proxy-emission diagnostics;
- no artifact overwrite.

No training artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


H = 361
W = 576
EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VALIDATION_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
ALLOWED_SEEDS = {42, 43, 44}
ALLOWED_EXPERIMENTS = {
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
}

REGIONS = {
    "North_America": {
        "lat": (15.0, 75.0),
        "lon": (-170.0, -50.0),
    },
    "Europe": {
        "lat": (35.0, 72.0),
        "lon": (-10.0, 40.0),
    },
    "East_Asia": {
        "lat": (20.0, 55.0),
        "lon": (100.0, 145.0),
    },
    "South_Asia": {
        "lat": (5.0, 35.0),
        "lon": (65.0, 100.0),
    },
    "Africa": {
        "lat": (-35.0, 37.0),
        "lon": (-20.0, 55.0),
    },
}

LAT = np.linspace(-90.0, 90.0, H, dtype=np.float32)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float32)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
    )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(str(path))
        )
    )


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name} is None")
    number = float(value)
    require(math.isfinite(number), f"{name} is non-finite: {value}")
    return number


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json_strict(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(obj),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def md5_int(text: str) -> int:
    return int(
        hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid month: {month}")


def region_mask(region_name: str) -> torch.Tensor:
    cfg = REGIONS[region_name]
    lat0, lat1 = cfg["lat"]
    lon0, lon1 = cfg["lon"]
    mask = (
        (LAT_GRID >= lat0)
        & (LAT_GRID <= lat1)
        & (LON_GRID >= lon0)
        & (LON_GRID <= lon1)
    )
    return torch.from_numpy(mask).bool()


@dataclass(frozen=True)
class MeanStateSpec:
    name: str
    mapping: str
    note: str
    uses_emission: bool
    uses_meteo: bool
    shuffle_emission: bool
    emission_channels: int
    meteo_channels: int

    @property
    def input_channels(self) -> int:
        return self.emission_channels + self.meteo_channels


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanStateUNet(nn.Module):
    def __init__(
        self,
        spec: MeanStateSpec,
        base_features: int = 32,
    ):
        super().__init__()
        self.spec = spec
        bf = int(base_features)
        self.dual = spec.uses_emission and spec.uses_meteo

        if self.dual:
            bf_half = max(8, bf // 2)
            self.emis_stem = nn.Sequential(
                nn.Conv2d(
                    spec.emission_channels,
                    bf_half,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bf_half),
                nn.GELU(),
            )
            self.met_stem = nn.Sequential(
                nn.Conv2d(
                    spec.meteo_channels,
                    bf_half,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bf_half),
                nn.GELU(),
            )
            stem_ch = bf_half * 2
        else:
            self.single_stem = nn.Sequential(
                nn.Conv2d(
                    spec.input_channels,
                    bf,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bf),
                nn.GELU(),
            )
            stem_ch = bf

        self.enc1 = ConvBlock(stem_ch, bf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(bf, bf * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(bf * 2, bf * 4)
        self.up2 = nn.ConvTranspose2d(
            bf * 4,
            bf * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(bf * 4, bf * 2)
        self.up1 = nn.ConvTranspose2d(
            bf * 2,
            bf,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(bf * 2, bf)
        self.out = nn.Conv2d(
            bf,
            1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual:
            emission = x[
                :,
                : self.spec.emission_channels,
            ]
            meteorology = x[
                :,
                self.spec.emission_channels :,
            ]
            x0 = torch.cat(
                [
                    self.emis_stem(emission),
                    self.met_stem(meteorology),
                ],
                dim=1,
            )
        else:
            x0 = self.single_stem(x)

        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool1(e1))
        bottleneck = self.bottleneck(
            self.pool2(e2)
        )

        d2 = self.up2(bottleneck)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        d2 = self.dec2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )
        return self.out(d1)


def deterministic_roll_2d(
    x: torch.Tensor,
    key: str,
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    seed = md5_int(key)
    shift_y = (seed % (H - 1)) + 1
    shift_x = ((seed // 4096) % (W - 1)) + 1
    return torch.roll(
        x,
        shifts=(shift_y, shift_x),
        dims=(-2, -1),
    )


def build_proxy_emission_baseline(
    train_entries: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    expected_months = [
        int(entry["month"])
        for entry in train_entries
    ]
    require(
        expected_months == EXPECTED_TRAIN_MONTHS,
        f"Unexpected proxy source months: {expected_months}",
    )

    tensors: list[torch.Tensor] = []
    source_records: list[dict[str, Any]] = []

    for entry in train_entries:
        payload_path = Path(
            entry["file"]
        ).expanduser().resolve()
        require(
            payload_path.is_file(),
            f"Proxy source payload not found: {payload_path}",
        )
        payload = load_pt(payload_path)
        emission = payload.get(
            "emission_log_mean"
        )
        require(
            isinstance(emission, torch.Tensor),
            f"Proxy source lacks emission tensor: {payload_path}",
        )
        emission = emission.detach().cpu().float()
        require(
            tuple(emission.shape) == (8, H, W),
            f"Invalid proxy emission shape: {tuple(emission.shape)}",
        )
        require(
            bool(torch.isfinite(emission).all().item()),
            f"Proxy source emission contains NaN/Inf: {payload_path}",
        )
        tensors.append(emission)
        source_records.append(
            {
                "month": int(entry["month"]),
                "file": str(payload_path),
                "file_sha256": sha256_file(payload_path),
                "tensor_sha256": sha256_tensor(emission),
            }
        )

    require(
        len(tensors) == 6,
        f"Expected 6 proxy source tensors, found {len(tensors)}",
    )
    proxy = torch.stack(
        tensors,
        dim=0,
    ).mean(dim=0)
    require(
        bool(torch.isfinite(proxy).all().item()),
        "Proxy emission baseline contains NaN/Inf.",
    )

    diagnostics = {
        "construction": (
            "Unweighted arithmetic mean of the six 2022 "
            "training-month processed emission-prior tensors."
        ),
        "source_months": expected_months,
        "source_records": source_records,
        "proxy_tensor_sha256": sha256_tensor(proxy),
        "shape": list(proxy.shape),
        "dtype": str(proxy.dtype),
        "minimum": float(proxy.min().item()),
        "maximum": float(proxy.max().item()),
        "mean": float(proxy.mean().item()),
        "std_population": float(
            proxy.std(unbiased=False).item()
        ),
    }
    return proxy, diagnostics


def build_input_from_payload(
    payload: dict[str, Any],
    spec: MeanStateSpec,
    *,
    key: str,
    proxy_emission_log_mean: Optional[torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    if spec.uses_emission:
        emission = payload.get(
            "emission_log_mean"
        )
        if emission is None:
            require(
                proxy_emission_log_mean is not None,
                "Emission input requested but neither actual "
                "nor proxy emission is available.",
            )
            emission = proxy_emission_log_mean.clone()

        require(
            isinstance(emission, torch.Tensor),
            "Emission input is not a tensor.",
        )
        emission = emission.detach().cpu().float()
        require(
            tuple(emission.shape) == (8, H, W),
            f"Invalid emission input shape: {tuple(emission.shape)}",
        )
        require(
            bool(torch.isfinite(emission).all().item()),
            "Emission input contains NaN/Inf.",
        )
        emission = torch.nan_to_num(
            emission,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if spec.shuffle_emission:
            emission = deterministic_roll_2d(
                emission,
                key,
            )
        parts.append(emission)

    if spec.uses_meteo:
        meteorology = payload[
            "meteo_mean"
        ].detach().cpu().float()
        require(
            tuple(meteorology.shape) == (5, H, W),
            f"Invalid meteorology shape: {tuple(meteorology.shape)}",
        )
        require(
            bool(torch.isfinite(meteorology).all().item()),
            "Meteorology input contains NaN/Inf.",
        )
        meteorology = torch.nan_to_num(
            meteorology,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        parts.append(meteorology)

    require(parts, "No input branch selected.")
    x = torch.cat(parts, dim=0)
    require(
        tuple(x.shape)
        == (spec.input_channels, H, W),
        f"Input shape mismatch: {tuple(x.shape)}",
    )
    return x


def pooled_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Optional[float]]:
    prediction = prediction.reshape(-1).float()
    target = target.reshape(-1).float()

    if prediction.numel() <= 1:
        return {
            "log_r2": None,
            "log_rmse": None,
            "log_bias": None,
        }

    require(
        prediction.numel() == target.numel(),
        "Prediction and target lengths differ.",
    )
    require(
        bool(torch.isfinite(prediction).all().item()),
        "Prediction vector contains NaN/Inf.",
    )
    require(
        bool(torch.isfinite(target).all().item()),
        "Target vector contains NaN/Inf.",
    )

    ss_res = torch.sum(
        (target - prediction) ** 2
    )
    ss_tot = torch.sum(
        (target - torch.mean(target)) ** 2
    )
    r2 = float(
        1.0
        - ss_res
        / torch.clamp(ss_tot, min=1e-12)
    )
    rmse = float(
        torch.sqrt(
            torch.mean(
                (prediction - target) ** 2
            )
        )
    )
    bias = float(
        torch.mean(prediction - target)
    )
    require(
        all(
            math.isfinite(value)
            for value in [r2, rmse, bias]
        ),
        "Computed metric is non-finite.",
    )
    return {
        "log_r2": r2,
        "log_rmse": rmse,
        "log_bias": bias,
    }


@torch.no_grad()
def evaluate_entries(
    model: nn.Module,
    entries: Sequence[dict[str, Any]],
    spec: MeanStateSpec,
    device: torch.device,
    *,
    proxy_emission_log_mean: Optional[torch.Tensor],
) -> dict[str, Any]:
    model.eval()

    all_prediction: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    per_group: list[dict[str, Any]] = []

    season_buckets: Dict[
        str,
        List[Tuple[torch.Tensor, torch.Tensor]],
    ] = {
        season: []
        for season in ["DJF", "MAM", "JJA", "SON"]
    }
    region_buckets: Dict[
        str,
        List[Tuple[torch.Tensor, torch.Tensor]],
    ] = {
        region: []
        for region in REGIONS
    }
    region_masks = {
        name: region_mask(name)
        for name in REGIONS
    }

    for entry in entries:
        payload_path = Path(
            entry["file"]
        ).expanduser().resolve()
        require(
            payload_path.is_file(),
            f"Monthly payload not found: {payload_path}",
        )
        payload = load_pt(payload_path)
        key = payload_path.name

        x = build_input_from_payload(
            payload,
            spec,
            key=key,
            proxy_emission_log_mean=proxy_emission_log_mean,
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )
        prediction = (
            model(x)
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .float()
        )
        target = (
            payload["target_log_mean_conc"]
            .detach()
            .cpu()
            .float()
        )
        valid_mask = (
            payload["valid_mask"]
            .detach()
            .cpu()
            .float()
            > 0.5
        )

        require(
            tuple(prediction.shape) == (H, W),
            "Prediction shape mismatch.",
        )
        require(
            tuple(target.shape) == (H, W),
            "Target shape mismatch.",
        )
        require(
            tuple(valid_mask.shape) == (H, W),
            "Valid-mask shape mismatch.",
        )
        require(
            bool(torch.isfinite(prediction).all().item()),
            f"Prediction contains NaN/Inf: {payload_path.name}",
        )
        require(
            bool(torch.isfinite(target).all().item()),
            f"Target contains NaN/Inf: {payload_path.name}",
        )

        n_valid = int(
            torch.count_nonzero(valid_mask).item()
        )
        require(
            n_valid > 1,
            f"Group has <=1 valid pixels: {payload_path}",
        )

        group_prediction = prediction[valid_mask]
        group_target = target[valid_mask]
        group_metrics = pooled_metrics(
            group_prediction,
            group_target,
        )

        all_prediction.append(group_prediction)
        all_target.append(group_target)

        month = int(entry["month"])
        season = str(
            entry.get(
                "season",
                month_to_season(month),
            )
        )
        require(
            season == month_to_season(month),
            f"Season-month mismatch for month {month}.",
        )
        season_buckets[season].append(
            (group_prediction, group_target)
        )

        for region_name, mask in region_masks.items():
            region_valid = valid_mask & mask
            if int(
                torch.count_nonzero(
                    region_valid
                ).item()
            ) > 1:
                region_buckets[region_name].append(
                    (
                        prediction[region_valid],
                        target[region_valid],
                    )
                )

        per_group.append(
            {
                "year": int(entry["year"]),
                "split": str(entry["split"]),
                "month": month,
                "season": season,
                "n_days": int(entry["n_days"]),
                "valid_pixel_fraction": float(
                    entry["valid_pixel_fraction"]
                ),
                "n_valid_pixels": n_valid,
                **group_metrics,
                "file": str(payload_path),
                "file_sha256": sha256_file(payload_path),
            }
        )

        print(
            f"EVAL_GROUP={int(entry['year'])}-{month:02d} "
            f"N_VALID={n_valid} "
            f"LOG_R2={group_metrics['log_r2']:.8f} "
            f"LOG_RMSE={group_metrics['log_rmse']:.8f} "
            f"LOG_BIAS={group_metrics['log_bias']:.8f}",
            flush=True,
        )

    global_metrics = pooled_metrics(
        torch.cat(all_prediction),
        torch.cat(all_target),
    )

    season_summary: dict[str, Any] = {}
    for season, pairs in season_buckets.items():
        if pairs:
            season_summary[season] = {
                "n_groups": len(pairs),
                **pooled_metrics(
                    torch.cat(
                        [
                            prediction
                            for prediction, _ in pairs
                        ]
                    ),
                    torch.cat(
                        [
                            target
                            for _, target in pairs
                        ]
                    ),
                ),
            }
        else:
            season_summary[season] = {
                "n_groups": 0,
                "log_r2": None,
                "log_rmse": None,
                "log_bias": None,
            }

    region_summary: dict[str, Any] = {}
    for region, pairs in region_buckets.items():
        if pairs:
            region_summary[region] = {
                "n_groups_with_valid_pixels": len(pairs),
                **pooled_metrics(
                    torch.cat(
                        [
                            prediction
                            for prediction, _ in pairs
                        ]
                    ),
                    torch.cat(
                        [
                            target
                            for _, target in pairs
                        ]
                    ),
                ),
            }
        else:
            region_summary[region] = {
                "n_groups_with_valid_pixels": 0,
                "log_r2": None,
                "log_rmse": None,
                "log_bias": None,
            }

    return {
        "n_groups": len(entries),
        "n_total_valid_pixels": int(
            sum(
                item["n_valid_pixels"]
                for item in per_group
            )
        ),
        "global": global_metrics,
        "season_summary": season_summary,
        "region_summary": region_summary,
        "per_group": per_group,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reusable formal evaluation for an audited "
            "clean mean-state run."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--metric-reproduction-tolerance",
        type=float,
        default=1e-6,
    )
    args = parser.parse_args()

    require(
        args.expected_experiment in ALLOWED_EXPERIMENTS,
        f"Unsupported experiment: {args.expected_experiment}",
    )
    require(
        args.expected_seed in ALLOWED_SEEDS,
        f"Unsupported seed: {args.expected_seed}",
    )

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()
    audit_path = Path(
        args.training_audit
    ).expanduser().resolve()
    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    require(
        run_dir.is_dir(),
        f"Run directory not found: {run_dir}",
    )
    require(
        audit_path.is_file(),
        f"Training audit not found: {audit_path}",
    )
    require(
        not output_dir.exists(),
        "Evaluation output directory already exists. "
        "This script never overwrites.",
    )

    manifest_path = (
        run_dir / "logs" / "run_manifest.json"
    )
    history_path = (
        run_dir / "logs" / "training_history.json"
    )
    checkpoint_path = (
        run_dir / "checkpoints" / "best_model.pt"
    )
    for path in [
        manifest_path,
        history_path,
        checkpoint_path,
    ]:
        require(
            path.is_file()
            and path.stat().st_size > 0,
            f"Missing or empty training artifact: {path}",
        )

    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    history = load_json(history_path)

    require(
        audit.get("audit_name")
        == "s2_meanstate_clean_formal_training_generic",
        "Unexpected training audit_name.",
    )
    require(
        audit.get("verification", {}).get(
            "overall_pass"
        ) is True,
        "Training audit overall_pass is not true.",
    )
    require(
        audit.get("experiment")
        == args.expected_experiment,
        "Training audit experiment mismatch.",
    )
    require(
        int(audit.get("seed", -1))
        == args.expected_seed,
        "Training audit seed mismatch.",
    )
    require(
        normalized_path(
            audit.get("inputs", {}).get(
                "run_dir",
                "",
            )
        )
        == normalized_path(run_dir),
        "Training audit run_dir mismatch.",
    )

    hashes = audit.get(
        "artifact_hashes",
        {},
    )
    require(
        hashes.get("manifest_sha256")
        == sha256_file(manifest_path),
        "Run manifest changed after training audit.",
    )
    require(
        hashes.get("history_sha256")
        == sha256_file(history_path),
        "Training history changed after training audit.",
    )
    require(
        hashes.get("checkpoint_sha256")
        == sha256_file(checkpoint_path),
        "Checkpoint changed after training audit.",
    )

    require(
        manifest.get("run_mode") == "formal",
        "Manifest run_mode is not formal.",
    )
    require(
        manifest["experiment"]["name"]
        == args.expected_experiment,
        "Manifest experiment mismatch.",
    )
    require(
        int(manifest["settings"]["seed"])
        == args.expected_seed,
        "Manifest seed mismatch.",
    )

    spec = MeanStateSpec(
        **manifest["experiment"]
    )
    train_entries = list(
        manifest["train_entries"]
    )
    validation_entries = list(
        manifest["val_entries"]
    )
    external_entries = list(
        manifest["external_2023_entries"]
    )

    train_months = [
        int(entry["month"])
        for entry in train_entries
    ]
    validation_months = [
        int(entry["month"])
        for entry in validation_entries
    ]
    external_months = [
        int(entry["month"])
        for entry in external_entries
    ]
    require(
        train_months == EXPECTED_TRAIN_MONTHS,
        f"Unexpected train months: {train_months}",
    )
    require(
        validation_months
        == EXPECTED_VALIDATION_MONTHS,
        f"Unexpected validation months: {validation_months}",
    )
    require(
        external_months
        == EXPECTED_EXTERNAL_MONTHS,
        f"Unexpected external months: {external_months}",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if args.require_cuda:
        require(
            device.type == "cuda",
            "CUDA is required but unavailable.",
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if spec.uses_emission:
        (
            proxy_emission,
            proxy_diagnostics,
        ) = build_proxy_emission_baseline(
            train_entries
        )
        external_type = "proxy_external"
        emission_proxy_used = True
    else:
        proxy_emission = None
        proxy_diagnostics = None
        external_type = "ordinary_external"
        emission_proxy_used = False

    print("=" * 88)
    print(
        "Clean mean-state generic formal evaluation"
    )
    print("=" * 88)
    print(f"EXPERIMENT={spec.name}")
    print(f"SEED={args.expected_seed}")
    print(f"RUN_DIR={run_dir}")
    print(f"DEVICE={device}")
    if device.type == "cuda":
        print(
            "CUDA_DEVICE_NAME="
            f"{torch.cuda.get_device_name(0)}"
        )
    print(f"VALIDATION_MONTHS={validation_months}")
    print(f"EXTERNAL_2023_MONTHS={external_months}")
    print(f"EXTERNAL_2023_TYPE={external_type}")
    print(
        f"EXTERNAL_2023_EMISSION_PROXY_USED="
        f"{emission_proxy_used}"
    )
    print("TRAINING_AUDIT_PROVENANCE_ACCEPTED=True")

    model = MeanStateUNet(
        spec=spec,
        base_features=int(
            manifest["settings"]["base_features"]
        ),
    ).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )
    require(
        checkpoint.get("experiment")
        == args.expected_experiment,
        "Checkpoint experiment mismatch.",
    )
    checkpoint_epoch = int(
        checkpoint.get("epoch", -1)
    ) + 1
    checkpoint_best_r2 = finite_float(
        checkpoint.get("best_val_log_r2"),
        "checkpoint.best_val_log_r2",
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    audit_summary = audit.get(
        "training_summary",
        {},
    )
    audit_best_epoch = int(
        audit_summary.get("best_epoch", -1)
    )
    audit_best_r2 = finite_float(
        audit_summary.get("best_val_log_r2"),
        "audit.best_val_log_r2",
    )
    audit_best_rmse = finite_float(
        audit_summary.get(
            "best_val_log_rmse_at_same_epoch"
        ),
        "audit.best_val_log_rmse",
    )
    require(
        checkpoint_epoch == audit_best_epoch,
        "Checkpoint epoch differs from training audit.",
    )
    require(
        math.isclose(
            checkpoint_best_r2,
            audit_best_r2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Checkpoint best R2 differs from training audit.",
    )

    print("\n[1/2] 2022 held-out validation")
    evaluation_2022 = evaluate_entries(
        model,
        validation_entries,
        spec,
        device,
        proxy_emission_log_mean=None,
    )
    reproduced_r2 = finite_float(
        evaluation_2022["global"]["log_r2"],
        "evaluation_2022.global.log_r2",
    )
    reproduced_rmse = finite_float(
        evaluation_2022["global"]["log_rmse"],
        "evaluation_2022.global.log_rmse",
    )
    tolerance = float(
        args.metric_reproduction_tolerance
    )
    require(
        abs(reproduced_r2 - audit_best_r2)
        <= tolerance,
        "2022 R2 does not reproduce training audit best R2.",
    )
    require(
        abs(reproduced_rmse - audit_best_rmse)
        <= tolerance,
        "2022 RMSE does not reproduce training audit best RMSE.",
    )

    print("\n[2/2] 2023 external evaluation")
    evaluation_2023 = evaluate_entries(
        model,
        external_entries,
        spec,
        device,
        proxy_emission_log_mean=proxy_emission,
    )

    external_r2 = finite_float(
        evaluation_2023["global"]["log_r2"],
        "evaluation_2023.global.log_r2",
    )
    external_rmse = finite_float(
        evaluation_2023["global"]["log_rmse"],
        "evaluation_2023.global.log_rmse",
    )
    external_bias = finite_float(
        evaluation_2023["global"]["log_bias"],
        "evaluation_2023.global.log_bias",
    )

    report = {
        "artifact_name": (
            "s2_meanstate_clean_formal_evaluation_generic"
        ),
        "artifact_version": 1,
        "experiment": manifest["experiment"],
        "seed": args.expected_seed,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selected_epoch": checkpoint_epoch,
            "best_val_log_r2": checkpoint_best_r2,
            "trainable_parameters": trainable_parameters,
        },
        "provenance": {
            "training_audit": str(audit_path),
            "training_audit_sha256": sha256_file(
                audit_path
            ),
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(
                manifest_path
            ),
            "training_history": str(history_path),
            "training_history_sha256": sha256_file(
                history_path
            ),
            "cache_dir": manifest["cache_dir"],
            "monthly_manifest": manifest[
                "provenance"
            ]["monthly_manifest"],
            "monthly_manifest_sha256": manifest[
                "provenance"
            ]["monthly_manifest_sha256"],
        },
        "evaluation_design": {
            "2022_validation_months": validation_months,
            "2022_status": (
                "held-out model-selection validation"
            ),
            "2023_external_months": external_months,
            "2023_status": external_type,
            "emission_proxy_used": emission_proxy_used,
            "shuffle_emission": bool(
                spec.shuffle_emission
            ),
            "metric_space": (
                "processed log-BC monthly mean-state"
            ),
            "pooling": (
                "all valid prediction-target pairs "
                "pooled within each reported split"
            ),
            "regions": REGIONS,
        },
        "proxy_emission": proxy_diagnostics,
        "metric_reproduction": {
            "tolerance": tolerance,
            "audit_best_epoch": audit_best_epoch,
            "audit_best_log_r2": audit_best_r2,
            "audit_best_log_rmse": audit_best_rmse,
            "recomputed_2022_log_r2": reproduced_r2,
            "recomputed_2022_log_rmse": reproduced_rmse,
            "pass": True,
        },
        "eval_2022_validation": evaluation_2022,
        "eval_2023_external": evaluation_2023,
        "verification": {
            "training_audit_provenance_pass": True,
            "checkpoint_strict_load_pass": True,
            "2022_best_metric_reproduction_pass": True,
            "2023_all_12_months_evaluated": True,
            "2023_emission_proxy_used": emission_proxy_used,
            "all_reported_global_metrics_finite": True,
            "overall_pass": True,
        },
        "scientific_boundary": (
            "The 2022 result is held-out model-selection "
            "validation. The 2023 result is ordinary external "
            "evaluation only for meteorology-only; all "
            "emission-input configurations are proxy external "
            "tests. A single seed is not a final manuscript estimate."
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    report_path = output_dir / "eval_report.json"
    summary_path = output_dir / "eval_summary.txt"
    save_json_strict(report, report_path)

    summary_path.write_text(
        "\n".join(
            [
                "Clean mean-state generic formal evaluation",
                f"Experiment: {spec.name}",
                f"Seed: {args.expected_seed}",
                (
                    "2022 pooled log R2: "
                    f"{reproduced_r2:.8f}"
                ),
                (
                    "2022 pooled log RMSE: "
                    f"{reproduced_rmse:.8f}"
                ),
                (
                    "2022 pooled log bias: "
                    f"{evaluation_2022['global']['log_bias']:.8f}"
                ),
                f"2023 status: {external_type}",
                (
                    "2023 emission proxy used: "
                    f"{emission_proxy_used}"
                ),
                (
                    "2023 pooled log R2: "
                    f"{external_r2:.8f}"
                ),
                (
                    "2023 pooled log RMSE: "
                    f"{external_rmse:.8f}"
                ),
                (
                    "2023 pooled log bias: "
                    f"{external_bias:.8f}"
                ),
                "Single-seed result ready for paper: False",
                "Overall pass: True",
            ]
        ),
        encoding="utf-8",
    )

    reloaded = load_json(report_path)
    require(
        reloaded.get("verification", {}).get(
            "overall_pass"
        ) is True,
        "Written report failed strict reload check.",
    )

    print(
        "\nGATE_GENERIC_CLEAN_MEANSTATE_FORMAL_EVALUATION_PASS"
    )
    print(f"EXPERIMENT={spec.name}")
    print(f"SEED={args.expected_seed}")
    print(f"EVAL_REPORT={report_path}")
    print(f"EVAL_SUMMARY={summary_path}")
    print("TRAINING_AUDIT_PROVENANCE_PASS=True")
    print("CHECKPOINT_STRICT_LOAD_PASS=True")
    print(
        "BEST_CHECKPOINT_METRIC_REPRODUCTION_PASS=True"
    )
    print(f"VAL_2022_LOG_R2={reproduced_r2:.8f}")
    print(f"VAL_2022_LOG_RMSE={reproduced_rmse:.8f}")
    print(f"EXTERNAL_2023_TYPE={external_type}")
    print(
        "EXTERNAL_2023_EMISSION_PROXY_USED="
        f"{emission_proxy_used}"
    )
    print(f"EXTERNAL_2023_LOG_R2={external_r2:.8f}")
    print(
        f"EXTERNAL_2023_LOG_RMSE={external_rmse:.8f}"
    )
    print(
        f"EXTERNAL_2023_LOG_BIAS={external_bias:.8f}"
    )
    print("SINGLE_SEED_RESULT_READY_FOR_PAPER=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
