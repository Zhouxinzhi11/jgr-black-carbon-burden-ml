# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
11b_build_phase2_meanstate_fullmonth.py

【这个脚本专门为第二阶段的 mean-state / 慢变量任务服务】

一、为什么需要 11b，而不是继续沿用 11_build_phase2_cache.py 里的 monthly cache
--------------------------------------------------------------------------
第一阶段和 11/12 的结果已经说明：
1. daily anomaly 这条线使用“沿用第一阶段 2022 train/val day-level split”的背景态定义是合理的；
2. 但 mean-state / 月平均任务如果继续继承 day-level split，会把完整 month 打碎；
3. 一旦 month 被切碎，就会出现某些 month-group 只有 1 天、甚至 valid_pixel_fraction=0 的情况；
4. 这样的缓存虽然在数学上可以自洽，但不适合作为 mean-state 主训练数据。

因此，本脚本的目标很明确：
- 不再按第一阶段的 day-level split 构造 mean-state group；
- 直接按“完整 calendar month / season”聚合；
- 为第二阶段 mean-state 训练准备统计上更合理的缓存。

二、这个脚本会生成什么
--------------------------------------------------------------------------
1. monthly_mean_full/
   - 2022_full: 使用 2022 全部可用天，按完整月份聚合；
   - 2023_all : 使用 2023 全部可用天，按完整月份聚合；

2. seasonal_mean_full/
   - 2022_full: 使用 2022 全部可用天，按季节聚合；
   - 2023_all : 使用 2023 全部可用天，按季节聚合；

3. manifests/
   - meanstate_build_report.json
   - monthly_manifest.json
   - seasonal_manifest.json

三、这个脚本在第二阶段中的位置
--------------------------------------------------------------------------
11_build_phase2_cache.py               -> 为 anomaly 任务构造 split-aware cache
12_audit_phase2_cache.py               -> 审计 anomaly cache
11b_build_phase2_meanstate_fullmonth.py -> 为 mean-state 任务构造 full-month cache
12b_audit_phase2_meanstate_fullmonth.py -> 审计 full-month mean-state cache
13/14                                  -> 基于 11b/12b 的 mean-state 训练与评估

四、为什么 full-month 更适合 mean-state
--------------------------------------------------------------------------
你第一阶段已经证明：
- 年排放更像慢变量 / 空间背景项；
- pooled skill 主要来自空间平均态，而不是逐像元时间异常；
- anomaly 任务里排放几乎没有稳定贡献。

因此，如果第二阶段还要认真检验“排放是否更适合解释慢变量背景态”，
就必须让目标的时间尺度与排放更匹配。
按完整 month 聚合，比按 day-level split 切碎后的 pseudo-month 更合理。

五、输出目录结构（默认）
--------------------------------------------------------------------------
data/RESTRICTED_INPUT_NOT_INCLUDED\07_phase2_meanstate_fullmonth\
  manifests\
    meanstate_build_report.json
    monthly_manifest.json
    seasonal_manifest.json
  monthly_mean_full\
    2022_full\
    2023_all\
  seasonal_mean_full\
    2022_full\
    2023_all\

六、如果这个脚本失败，意味着什么
--------------------------------------------------------------------------
- 如果 monthly groups 太少，说明原始 daily tensor 的月份覆盖本身不足；
- 如果 monthly groups 存在大量 valid_pixel_fraction≈0，说明 mean-state 任务的目标定义或数据可用性有问题；
- 此时不应该继续训练 13/14，而应先回头检查数据覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
H = 361
W = 576
X_CHANNELS = 112
Y_CHANNELS = 10
METEO_SLICE = slice(43, 48)
AEF_SLICE = slice(48, 112)

DEFAULT_OUT_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\07_phase2_meanstate_fullmonth_trainonly_union"


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Daily tensor accessors
# ---------------------------------------------------------------------
def get_x(data: dict) -> torch.Tensor:
    x = data["x"].float()
    if tuple(x.shape) != (X_CHANNELS, H, W):
        raise ValueError(f"x shape invalid: {tuple(x.shape)}")
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def get_y(data: dict) -> torch.Tensor:
    y = data["y"].float()
    if tuple(y.shape) != (Y_CHANNELS, H, W):
        raise ValueError(f"y shape invalid: {tuple(y.shape)}")
    return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def get_obs_mask(data: dict) -> torch.Tensor:
    m = data["obs_mask"].float()
    if tuple(m.shape) != (H, W):
        raise ValueError(f"obs_mask shape invalid: {tuple(m.shape)}")
    return torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def get_concentration_log(data: dict) -> torch.Tensor:
    return get_y(data)[0:1]


def get_emission_log(data: dict) -> torch.Tensor:
    emis = get_y(data)[1:9]
    if tuple(emis.shape) != (8, H, W):
        raise ValueError(f"emission shape invalid: {tuple(emis.shape)}")
    return emis


def get_fire_log(data: dict) -> torch.Tensor:
    fire = get_y(data)[9:10]
    if tuple(fire.shape) != (1, H, W):
        raise ValueError(f"fire shape invalid: {tuple(fire.shape)}")
    return fire


def get_meteo(data: dict) -> torch.Tensor:
    return get_x(data)[METEO_SLICE]


def get_aef(data: dict) -> torch.Tensor:
    aef = get_x(data)[AEF_SLICE]
    if tuple(aef.shape) != (64, H, W):
        raise ValueError(f"AEF shape invalid: {tuple(aef.shape)}")
    return aef


def has_real_emission_label(data: dict) -> bool:
    flag = data.get("has_emission", False)
    if isinstance(flag, torch.Tensor):
        return bool(flag.item())
    return bool(flag)


# ---------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------
def parse_doy_from_filename(fname: str) -> Optional[int]:
    try:
        return int(Path(fname).stem.split("_")[1])
    except Exception:
        return None


def doy_to_month_nonleap(doy: int) -> Optional[int]:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = 0
    for month, nd in enumerate(month_days, start=1):
        if doy <= cum + nd:
            return month
        cum += nd
    return None


def month_to_season(month: int) -> Optional[str]:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    return None


def stable_mean_from_sum_count(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sum_arr, dtype=np.float32)
    valid = count_arr > 0
    out[valid] = (sum_arr[valid] / count_arr[valid]).astype(np.float32)
    return out


# ---------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------
class MeanAccumulator:
    def __init__(self) -> None:
        self.sum_target = np.zeros((H, W), dtype=np.float64)
        self.count_target = np.zeros((H, W), dtype=np.int32)
        self.sum_meteo = np.zeros((5, H, W), dtype=np.float64)
        self.sum_fire = np.zeros((1, H, W), dtype=np.float64)
        self.sum_aef = np.zeros((64, H, W), dtype=np.float64)
        self.sum_emis = np.zeros((8, H, W), dtype=np.float64)
        self.n_days = 0
        self.n_days_with_emission = 0
        self.file_list: List[str] = []
        self.year: Optional[int] = None
        self.split: Optional[str] = None
        self.month: Optional[int] = None
        self.season: Optional[str] = None

    def update(self, *, data: dict, fname: str, year: int, split: str, month: int, season: str) -> None:
        target = tensor_to_np(get_concentration_log(data))[0]
        obs_mask = tensor_to_np(get_obs_mask(data)).astype(bool)
        meteo = tensor_to_np(get_meteo(data))
        fire = tensor_to_np(get_fire_log(data))
        aef = tensor_to_np(get_aef(data))

        self.sum_target[obs_mask] += target[obs_mask]
        self.count_target[obs_mask] += 1
        self.sum_meteo += meteo
        self.sum_fire += fire
        self.sum_aef += aef
        self.n_days += 1
        self.file_list.append(fname)
        self.year = year
        self.split = split
        self.month = month
        self.season = season

        if has_real_emission_label(data):
            emis = tensor_to_np(get_emission_log(data))
            self.sum_emis += emis
            self.n_days_with_emission += 1

    def finalize(self, min_obs_days_per_pixel: int) -> dict:
        if self.n_days <= 0:
            raise RuntimeError("Accumulator is empty.")

        target_log_mean = stable_mean_from_sum_count(self.sum_target, self.count_target)
        valid_mask = (self.count_target >= min_obs_days_per_pixel)
        meteo_mean = (self.sum_meteo / float(self.n_days)).astype(np.float32)
        fire_mean = (self.sum_fire / float(self.n_days)).astype(np.float32)
        aef_mean = (self.sum_aef / float(self.n_days)).astype(np.float32)

        if self.n_days_with_emission > 0:
            emis_mean = (self.sum_emis / float(self.n_days_with_emission)).astype(np.float32)
            emission_available = True
        else:
            emis_mean = None
            emission_available = False

        return {
            "target_log_mean_conc": torch.from_numpy(target_log_mean),
            "obs_count_map": torch.from_numpy(self.count_target.astype(np.int32)),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32)),
            "meteo_mean": torch.from_numpy(meteo_mean),
            "fire_mean": torch.from_numpy(fire_mean),
            "aef_mean": torch.from_numpy(aef_mean),
            "emission_log_mean": None if emis_mean is None else torch.from_numpy(emis_mean),
            "emission_available": bool(emission_available),
            "n_days": int(self.n_days),
            "n_days_with_emission": int(self.n_days_with_emission),
            "year": int(self.year),
            "split": str(self.split),
            "month": None if self.month is None else int(self.month),
            "season": self.season,
            "source_files": list(self.file_list),
        }


# ---------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------
def discover_daily_files(data_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(data_dir) if f.startswith('day_') and f.endswith('.pt')])


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def validate_gate_artifacts(
    tensor_dir_2022: str,
    tensor_dir_2023: str,
    gate8b_manifest_path: str,
    gate10c_report_path: str,
) -> dict:
    with open(gate8b_manifest_path, 'r', encoding='utf-8') as f:
        gate8b = json.load(f)
    with open(gate10c_report_path, 'r', encoding='utf-8') as f:
        gate10c = json.load(f)

    if gate8b.get('artifact_name') != 'daily_tensor_meteo_trainonly_direct_raw_gate8b':
        raise RuntimeError('Unexpected Gate-8B artifact name.')
    if int(gate8b.get('source_file_count', -1)) != 240:
        raise RuntimeError('Gate-8B source_file_count is not 240.')
    if int(gate8b.get('output_file_count', -1)) != 240:
        raise RuntimeError('Gate-8B output_file_count is not 240.')
    if int(gate8b.get('processed_netcdf_rows', -1)) != int(gate8b.get('matched_netcdf_rows', -2)):
        raise RuntimeError('Gate-8B processed/matched row counts differ.')
    if int(gate8b.get('invalid_coordinate_rows', -1)) != 0:
        raise RuntimeError('Gate-8B contains invalid coordinate rows.')
    if normalized_path(gate8b.get('output_dir', '')) != normalized_path(tensor_dir_2022):
        raise RuntimeError('Gate-8B output_dir does not match --tensor-dir-2022.')

    verification = gate10c.get('verification', {})
    required_true = [
        'raw_union_reconstruction_pass',
        'preserved_fields_exactly_unchanged',
        'nonoverlap_source_content_reproduction_pass',
        'overlap_source_content_change_verified',
        'cross_file_key_diagnostic_pass',
        'overall_pass',
    ]
    failed = [key for key in required_true if verification.get(key) is not True]
    if failed:
        raise RuntimeError(f'Gate-10C verification failed or missing: {failed}')
    if int(gate10c.get('audit_coverage', {}).get('overlap_day_count', -1)) != 44:
        raise RuntimeError('Gate-10C overlap day count is not 44.')
    key_diag = gate10c.get('cross_file_key_diagnostic', {})
    if int(key_diag.get('total_nc2_unique_keys_on_overlap_days', -1)) != 546:
        raise RuntimeError('Gate-10C complementary NC2 key count is not 546.')
    if int(key_diag.get('total_common_exact_keys', -1)) != 0:
        raise RuntimeError('Gate-10C found common exact cross-file keys.')
    if normalized_path(gate10c.get('inputs', {}).get('new_dir', '')) != normalized_path(tensor_dir_2023):
        raise RuntimeError('Gate-10C new_dir does not match --tensor-dir-2023.')

    return {
        'gate8b_manifest': gate8b_manifest_path,
        'gate8b_manifest_sha256': sha256_file(gate8b_manifest_path),
        'gate10c_report': gate10c_report_path,
        'gate10c_report_sha256': sha256_file(gate10c_report_path),
    }


def build_fullmonth_accumulators(
    tensor_dir_2022: str,
    tensor_dir_2023: str,
) -> Tuple[Dict[Tuple[int, str, int], MeanAccumulator], Dict[Tuple[int, str, str], MeanAccumulator], dict]:
    data_dirs = {
        2022: tensor_dir_2022,
        2023: tensor_dir_2023,
    }
    split_names = {2022: "2022_full", 2023: "2023_all"}

    monthly_groups: Dict[Tuple[int, str, int], MeanAccumulator] = {}
    seasonal_groups: Dict[Tuple[int, str, str], MeanAccumulator] = {}
    source_index = {"2022_full": [], "2023_all": []}

    for year, data_dir in data_dirs.items():
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"tensor dir not found: {data_dir}")
        split = split_names[year]
        for fname in discover_daily_files(data_dir):
            doy = parse_doy_from_filename(fname)
            if doy is None:
                continue
            month = doy_to_month_nonleap(doy)
            if month is None:
                continue
            season = month_to_season(month)
            if season is None:
                continue
            data = load_pt(os.path.join(data_dir, fname))
            source_index[split].append(fname)

            mkey = (year, split, month)
            skey = (year, split, season)
            if mkey not in monthly_groups:
                monthly_groups[mkey] = MeanAccumulator()
            if skey not in seasonal_groups:
                seasonal_groups[skey] = MeanAccumulator()

            monthly_groups[mkey].update(data=data, fname=fname, year=year, split=split, month=month, season=season)
            seasonal_groups[skey].update(data=data, fname=fname, year=year, split=split, month=month, season=season)

    return monthly_groups, seasonal_groups, source_index


def save_fullmonth_groups(
    out_dir: str,
    monthly_groups: Dict[Tuple[int, str, int], MeanAccumulator],
    seasonal_groups: Dict[Tuple[int, str, str], MeanAccumulator],
    min_obs_days_per_pixel: int,
) -> Tuple[dict, dict]:
    monthly_root = os.path.join(out_dir, "monthly_mean_full")
    seasonal_root = os.path.join(out_dir, "seasonal_mean_full")
    ensure_dir(monthly_root)
    ensure_dir(seasonal_root)

    monthly_manifest = {"groups": []}
    seasonal_manifest = {"groups": []}

    for (year, split, month), acc in sorted(monthly_groups.items()):
        payload = acc.finalize(min_obs_days_per_pixel=min_obs_days_per_pixel)
        subdir = os.path.join(monthly_root, split)
        ensure_dir(subdir)
        fname = f"mean_{year}_{month:02d}_{split}.pt"
        path = os.path.join(subdir, fname)
        torch.save(payload, path)
        monthly_manifest["groups"].append(
            {
                "year": year,
                "split": split,
                "month": month,
                "season": payload["season"],
                "n_days": payload["n_days"],
                "n_days_with_emission": payload["n_days_with_emission"],
                "emission_available": bool(payload["emission_available"]),
                "valid_pixel_fraction": float(payload["valid_mask"].float().mean().item()),
                "file": path,
                "source_files": payload["source_files"],
            }
        )

    for (year, split, season), acc in sorted(seasonal_groups.items()):
        payload = acc.finalize(min_obs_days_per_pixel=min_obs_days_per_pixel)
        subdir = os.path.join(seasonal_root, split)
        ensure_dir(subdir)
        fname = f"mean_{year}_{season}_{split}.pt"
        path = os.path.join(subdir, fname)
        torch.save(payload, path)
        seasonal_manifest["groups"].append(
            {
                "year": year,
                "split": split,
                "season": season,
                "n_days": payload["n_days"],
                "n_days_with_emission": payload["n_days_with_emission"],
                "emission_available": bool(payload["emission_available"]),
                "valid_pixel_fraction": float(payload["valid_mask"].float().mean().item()),
                "file": path,
                "source_files": payload["source_files"],
            }
        )

    return monthly_manifest, seasonal_manifest


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean full-month mean-state cache from Gate-8B and Gate-10C tensors."
    )
    parser.add_argument("--tensor-dir-2022", type=str, required=True)
    parser.add_argument("--tensor-dir-2023", type=str, required=True)
    parser.add_argument("--gate8b-manifest", type=str, required=True)
    parser.add_argument("--gate10c-report", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-obs-days-per-pixel", type=int, default=5)
    parser.add_argument("--expected-files-2022", type=int, default=240)
    parser.add_argument("--expected-files-2023", type=int, default=361)
    args = parser.parse_args()

    tensor_dir_2022 = os.path.abspath(os.path.normpath(args.tensor_dir_2022))
    tensor_dir_2023 = os.path.abspath(os.path.normpath(args.tensor_dir_2023))
    gate8b_manifest = os.path.abspath(os.path.normpath(args.gate8b_manifest))
    gate10c_report = os.path.abspath(os.path.normpath(args.gate10c_report))
    out_dir = os.path.abspath(os.path.normpath(args.out_dir))

    print("=" * 88)
    print("Gate 11A - clean mean-state full-month cache build")
    print("=" * 88)
    print(f"tensor_dir_2022 = {tensor_dir_2022}")
    print(f"tensor_dir_2023 = {tensor_dir_2023}")
    print(f"out_dir         = {out_dir}")
    print(f"min_obs         = {args.min_obs_days_per_pixel}")

    for directory in [tensor_dir_2022, tensor_dir_2023]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Tensor directory not found: {directory}")
    for path in [gate8b_manifest, gate10c_report]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    files_2022 = discover_daily_files(tensor_dir_2022)
    files_2023 = discover_daily_files(tensor_dir_2023)
    if len(files_2022) != args.expected_files_2022:
        raise RuntimeError(
            f"Expected {args.expected_files_2022} 2022 tensors, found {len(files_2022)}."
        )
    if len(files_2023) != args.expected_files_2023:
        raise RuntimeError(
            f"Expected {args.expected_files_2023} 2023 tensors, found {len(files_2023)}."
        )

    provenance = validate_gate_artifacts(
        tensor_dir_2022=tensor_dir_2022,
        tensor_dir_2023=tensor_dir_2023,
        gate8b_manifest_path=gate8b_manifest,
        gate10c_report_path=gate10c_report,
    )

    monthly_root = os.path.join(out_dir, "monthly_mean_full")
    seasonal_root = os.path.join(out_dir, "seasonal_mean_full")
    manifests_dir = os.path.join(out_dir, "manifests")

    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(
            "Output directory is not empty. Use a new empty directory; this script never overwrites."
        )

    ensure_dir(out_dir)
    ensure_dir(manifests_dir)

    print("\n[1/3] Aggregating clean full-month and full-season groups...")
    monthly_groups, seasonal_groups, source_index = build_fullmonth_accumulators(
        tensor_dir_2022=tensor_dir_2022,
        tensor_dir_2023=tensor_dir_2023,
    )
    print(f"MONTHLY_GROUPS_DISCOVERED={len(monthly_groups)}")
    print(f"SEASONAL_GROUPS_DISCOVERED={len(seasonal_groups)}")

    print("\n[2/3] Writing cache payloads...")
    monthly_manifest, seasonal_manifest = save_fullmonth_groups(
        out_dir=out_dir,
        monthly_groups=monthly_groups,
        seasonal_groups=seasonal_groups,
        min_obs_days_per_pixel=args.min_obs_days_per_pixel,
    )

    print("\n[3/3] Writing manifests and provenance...")
    build_report = {
        "artifact_name": "phase2_meanstate_fullmonth_trainonly_union_gate11a",
        "version": 1,
        "tensor_dir_2022": tensor_dir_2022,
        "tensor_dir_2023": tensor_dir_2023,
        "out_dir": out_dir,
        "min_obs_days_per_pixel": int(args.min_obs_days_per_pixel),
        "source_file_count_2022": len(files_2022),
        "source_file_count_2023": len(files_2023),
        "n_monthly_groups": len(monthly_manifest["groups"]),
        "n_seasonal_groups": len(seasonal_manifest["groups"]),
        "source_index": source_index,
        "provenance": provenance,
        "construction_contract": {
            "target": "Per-pixel arithmetic mean of daily processed log-BC over days with obs_mask=1.",
            "valid_mask": "obs_count_map >= min_obs_days_per_pixel.",
            "meteorology": "Unweighted arithmetic mean over all source day tensors.",
            "fire": "Unweighted arithmetic mean over all source day tensors.",
            "aef": "Unweighted arithmetic mean over all source day tensors.",
            "emission_2022": "Unweighted arithmetic mean over days with has_emission=True.",
            "emission_2023": "Unavailable; no real 2023 emission channels are fabricated.",
        },
        "note": (
            "This cache is for mean-state / slow-variable tasks. It uses complete calendar-month "
            "groups and does not inherit the phase-1 day-level split. Training/validation month "
            "selection remains the responsibility of script 13."
        ),
    }

    save_json(monthly_manifest, os.path.join(manifests_dir, "monthly_manifest.json"))
    save_json(seasonal_manifest, os.path.join(manifests_dir, "seasonal_manifest.json"))
    save_json(build_report, os.path.join(manifests_dir, "meanstate_build_report.json"))

    print("\nGATE11A_CLEAN_MEANSTATE_CACHE_BUILD_PASS")
    print(f"MANIFEST_DIR={manifests_dir}")
    print(f"MONTHLY_DIR={monthly_root}")
    print(f"SEASONAL_DIR={seasonal_root}")
    print(f"SOURCE_FILES_2022={len(files_2022)}")
    print(f"SOURCE_FILES_2023={len(files_2023)}")
    print(f"MONTHLY_GROUPS={len(monthly_manifest['groups'])}")
    print(f"SEASONAL_GROUPS={len(seasonal_manifest['groups'])}")


if __name__ == "__main__":
    main()
