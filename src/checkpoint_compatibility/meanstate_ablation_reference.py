# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
13_train_s2_meanstate_ablation.py

【第二阶段：mean-state / 慢变量主实验统一训练脚本】

一、这个脚本要回答什么问题
--------------------------------------------------------------------------
第一阶段已经证明：
1. pooled skill 主要来自空间平均态，而不是逐像元时间异常；
2. 年排放清单在日异常任务中几乎没有稳定贡献；
3. 如果还想认真检验“排放是否更适合解释慢变量背景态”，就必须把目标改写为 month-level mean-state。

因此，本脚本的任务不是继续做 daily concentration regression，而是：
    在月平均背景态任务中，比较排放、气候态气象以及二者组合的解释能力。

二、为什么这里仍然采用“核心消融”而不是复杂模型堆叠
--------------------------------------------------------------------------
本脚本只做四个实验：
- mean_emis_only
- mean_meteo_clim_only
- mean_emis_meteo_clim
- mean_emis_meteo_clim_shuffled

原因很简单：
- 第二阶段刚开始，最重要的是先回答“慢变量尺度上排放是否终于开始有用”；
- 如果这一步都站不住，再往模型里塞 AEF / fire / fancy tricks 只会重复第一阶段的问题：分数可能变了，但你不知道是谁贡献的。

三、这个脚本在整个第二阶段中的位置
--------------------------------------------------------------------------
11b_build_phase2_meanstate_fullmonth.py  -> 构造 full-month mean-state cache
12b_audit_phase2_meanstate_fullmonth.py  -> 审计 full-month mean-state cache
13_train_s2_meanstate_ablation.py        -> 训练四个 mean-state 核心消融实验
14_eval_s2_meanstate_ablation.py         -> 统一评估 2022/2023 monthly mean-state 表现
15/16                                    -> anomaly 线
17/18                                    -> bridge / residual 线

四、这里借鉴了哪些有效方法
--------------------------------------------------------------------------
1. 时间尺度分离：
   先把 mean-state 和 anomaly 分开，这是来自第一阶段负结果与相关文献共同支持的方向。
2. 轻量的多模态分支融合：
   对 mean_emis_meteo_clim，不使用简单 raw concat 后直接送入网络，而是先做 emission stem 与 meteo stem，
   再融合进入主干。这样做是为了尊重“静态排放”和“气候态气象”在统计属性上的异质性。
3. observation-confidence weighting：
   month-level target 是由不同 obs_count_map 聚合而来。训练时使用 obs_count 作为置信度权重，
   让观测更充分的像元在损失中稍占更高权重。

五、重要限制
--------------------------------------------------------------------------
1. 本脚本默认只训练 monthly groups，不拿 seasonal groups 做主训练样本；
2. 会自动过滤：
   - n_days 过少的月组
   - valid_pixel_fraction <= 0 的月组
   - 用户指定黑名单月份（默认 2022-07）
3. 2023 对 uses_emission=True 的实验不能称为严格跨年验证，只能在 14 中标成 proxy external test。
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_CACHE_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\07_phase2_meanstate_fullmonth"
DEFAULT_RUN_ROOT = r"./runs_s2_meanstate"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


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
    emission_channels: int
    meteo_channels: int

    @property
    def input_channels(self) -> int:
        return self.emission_channels + self.meteo_channels


EXPERIMENT_SPECS: Dict[str, MeanStateSpec] = {
    "mean_emis_only": MeanStateSpec(
        name="mean_emis_only",
        mapping="Annual Emission -> Monthly Mean Concentration",
        note="检验年排放清单在慢变量背景态任务中的单独解释力。",
        uses_emission=True,
        uses_meteo=False,
        shuffle_emission=False,
        emission_channels=8,
        meteo_channels=0,
    ),
    "mean_meteo_clim_only": MeanStateSpec(
        name="mean_meteo_clim_only",
        mapping="Monthly-Mean Meteo -> Monthly Mean Concentration",
        note="检验气候态/月平均气象对背景态的解释力。",
        uses_emission=False,
        uses_meteo=True,
        shuffle_emission=False,
        emission_channels=0,
        meteo_channels=5,
    ),
    "mean_emis_meteo_clim": MeanStateSpec(
        name="mean_emis_meteo_clim",
        mapping="Annual Emission + Monthly-Mean Meteo -> Monthly Mean Concentration",
        note="第二阶段主实验。检验排放在慢变量尺度上是否在气候态气象控制下提供独立增益。",
        uses_emission=True,
        uses_meteo=True,
        shuffle_emission=False,
        emission_channels=8,
        meteo_channels=5,
    ),
    "mean_emis_meteo_clim_shuffled": MeanStateSpec(
        name="mean_emis_meteo_clim_shuffled",
        mapping="ShuffledEmission + Monthly-Mean Meteo -> Monthly Mean Concentration",
        note="排放负对照。保留数值分布但破坏空间对应关系，用于检验真实排放空间格局是否真的被利用。",
        uses_emission=True,
        uses_meteo=True,
        shuffle_emission=True,
        emission_channels=8,
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
    return groups


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
            raise ValueError("Emission input requested but emission_log_mean is None.")
        emis = emis.float()
        if tuple(emis.shape) != (8, H, W):
            raise ValueError(f"Emission shape invalid: {tuple(emis.shape)}")
        emis = torch.nan_to_num(emis, nan=0.0, posinf=0.0, neginf=0.0)
        if spec.shuffle_emission:
            emis = deterministic_roll_2d(emis, key)
        parts.append(emis)
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
    parser = argparse.ArgumentParser(description="Train phase-2 mean-state ablation experiment.")
    parser.add_argument("--experiment", type=str, required=True, choices=sorted(EXPERIMENT_SPECS.keys()))
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=str, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
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
    args = parser.parse_args()

    spec = EXPERIMENT_SPECS[args.experiment]
    run_name = args.run_name or f"{spec.name}_v1"
    run_dir = os.path.join(args.run_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print("=" * 88)
    print("🧪 第二阶段 mean-state 核心消融训练")
    print("=" * 88)
    print(f"Experiment : {spec.name}")
    print(f"Mapping    : {spec.mapping}")
    print(f"Note       : {spec.note}")
    print(f"Device     : {device}")
    print(f"Run name   : {run_name}")

    monthly_entries = load_monthly_manifest(args.cache_dir)
    kept_2022, dropped_2022 = filter_monthly_entries(
        monthly_entries,
        split="2022_full",
        blacklist_months=parse_month_list(args.blacklist_months),
        min_days=args.min_train_days,
        min_valid_fraction=args.min_valid_fraction,
    )
    train_entries, val_entries = split_train_val_months(kept_2022, parse_month_list(args.val_months))
    external_2023, dropped_2023 = filter_monthly_entries(
        monthly_entries,
        split="2023_all",
        blacklist_months=[],
        min_days=args.min_train_days,
        min_valid_fraction=args.min_valid_fraction,
    )

    if len(train_entries) == 0 or len(val_entries) == 0:
        raise RuntimeError(
            f"Invalid monthly split: train={len(train_entries)}, val={len(val_entries)}. "
            "Please adjust --val-months or inspect cache quality."
        )

    print("\n📊 Monthly group split (2022_full):")
    print(f"   train groups = {len(train_entries)} | months = {[int(e['month']) for e in train_entries]}")
    print(f"   val groups   = {len(val_entries)} | months = {[int(e['month']) for e in val_entries]}")
    print(f"   dropped 2022 = {len(dropped_2022)}")
    print(f"   2023 groups  = {len(external_2023)} | months = {[int(e['month']) for e in external_2023]}")

    manifest = {
        "run_name": run_name,
        "experiment": asdict(spec),
        "cache_dir": args.cache_dir,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "external_2023_entries": external_2023,
        "dropped_2022_entries": dropped_2022,
        "dropped_2023_entries": dropped_2023,
        "settings": {
            "val_months": parse_month_list(args.val_months),
            "blacklist_months": parse_month_list(args.blacklist_months),
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
            "seed": args.seed,
        },
        "note": (
            "2023 monthly groups are retained for later external evaluation. "
            "For experiments using emission input, 2023 evaluation must be reported as proxy external test because 2023 real emission labels are unavailable in this cache."
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
        pin_memory=True,
    )

    model = MeanStateUNet(spec=spec, base_features=args.base_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

    print(f"\n🎯 开始训练: max_epochs={args.epochs}, patience={args.patience}\n")
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1
            r2 = masked_r2(pred.detach(), yb.detach(), mb.detach())
            if not math.isnan(r2):
                patch_r2_values.append(r2)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = loss_sum / max(n_batches, 1)
        train_patch_r2 = float(np.mean(patch_r2_values)) if patch_r2_values else float("nan")

        val_result = evaluate_full_monthly_set(
            model=model,
            entries=val_entries,
            spec=spec,
            device=device,
        )
        val_log_r2 = val_result["global"]["log_r2"]
        val_log_rmse = val_result["global"]["log_rmse"]

        history["train_loss"].append(train_loss)
        history["train_patch_r2"].append(train_patch_r2)
        history["val_monthly_log_r2"].append(val_log_r2)
        history["val_monthly_log_rmse"].append(val_log_rmse)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"TrainPatch Loss={train_loss:.4f}, R²={train_patch_r2:.4f} | "
            f"ValMonthly logR²={val_log_r2:.4f}, logRMSE={val_log_rmse:.4f} | "
            f"LR={lr:.2e}"
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
                    "run_name": run_name,
                    "base_features": args.base_features,
                    "spec": asdict(spec),
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            print(f"   ✅ Save best checkpoint: val_monthly_log_r2={best_val_r2:.4f}")
        else:
            patience_counter += 1

        save_json(history, os.path.join(log_dir, "training_history.json"))
        save_json(val_result, os.path.join(log_dir, "latest_val_monthly_report.json"))

        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stop triggered: patience={args.patience}")
            break

    print("\n" + "=" * 88)
    print("📌 训练完成")
    print(f"Best epoch           : {best_epoch + 1}")
    print(f"Best val_monthly logR²: {best_val_r2:.4f}")
    print(f"Checkpoint           : {os.path.join(ckpt_dir, 'best_model.pt')}")
    print(f"Manifest             : {os.path.join(log_dir, 'run_manifest.json')}")
    print(f"History              : {os.path.join(log_dir, 'training_history.json')}")
    print("=" * 88)


if __name__ == "__main__":
    main()
