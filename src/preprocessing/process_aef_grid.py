# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""
🚀 [Step 3] 双年份 AEF 合并脚本
功能：分别读取 2022 和 2023 的 CSV，生成对应的 .npy 文件
"""

import numpy as np
import pandas as pd
import os
import glob
import re

# ================= 配置 =================
# CSV 输入目录 (请从 Google Drive 下载后解压到这里)
CSV_DIR_2022 = r"data/RESTRICTED_INPUT_NOT_INCLUDED\02_source_nc\AEF_MERRA2_2022"
CSV_DIR_2023 = r"data/RESTRICTED_INPUT_NOT_INCLUDED\02_source_nc\AEF_MERRA2_2023"

# 输出目录
OUTPUT_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\02_source_nc\AEF_Grid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 网格参数
TARGET_H, TARGET_W = 361, 576
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 179.375
LAT_STEP = 0.5
LON_STEP = 0.625

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def process_year(year, csv_dir):
    print(f"\n🔄 正在处理 {year} 年数据...")

    if not os.path.exists(csv_dir):
        print(f"❌ 目录不存在: {csv_dir}")
        print("   请确保已从 Google Drive 下载并解压了对应年份的数据！")
        return

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "AEF_*.csv")), key=natural_sort_key)
    if not csv_files:
        print(f"❌ {year} 目录下没有 CSV 文件！")
        return

    # 1. 读取
    dfs = []
    print(f"   📥 读取 {len(csv_files)} 个 CSV...")
    for f in csv_files:
        try:
            if os.path.getsize(f) > 100:
                df = pd.read_csv(f)
                if not df.empty: dfs.append(df)
        except: pass

    if not dfs: return
    full_df = pd.concat(dfs, ignore_index=True)

    # 2. 列名识别
    cols = {c.lower(): c for c in full_df.columns}
    lat_col = cols.get('lat') or cols.get('latitude')
    lon_col = cols.get('lon') or cols.get('longitude')

    # 3. 映射网格
    print("   ⚡ 映射到 MERRA-2 网格...")
    lat_idx = ((full_df[lat_col].values - LAT_MIN) / LAT_STEP).round().astype(int)
    lon_idx = ((full_df[lon_col].values - LON_MIN) / LON_STEP).round().astype(int)

    # 过滤越界
    mask = (lat_idx >= 0) & (lat_idx < TARGET_H) & (lon_idx >= 0) & (lon_idx < TARGET_W)
    lat_idx, lon_idx = lat_idx[mask], lon_idx[mask]

    # 提取 AEF 列
    aef_cols = [c for c in full_df.columns if c.startswith('A') and c[1:].isdigit()]
    aef_cols.sort()
    values = full_df[aef_cols].values[mask]
    # 将 GEE 强制填充的海洋无数据区域 (-9999) 还原为 NaN
    values[values == -9999] = np.nan

    # 4. 网格平均
    aef_grid = np.zeros((TARGET_H, TARGET_W, 64), dtype=np.float32)
    flat_idx = lat_idx * TARGET_W + lon_idx
    grid_size = TARGET_H * TARGET_W
    counts = np.bincount(flat_idx, minlength=grid_size)
    counts[counts == 0] = 1

    for i in range(64):
        sum_grid = np.bincount(flat_idx, weights=values[:, i], minlength=grid_size)
        aef_grid[:, :, i] = (sum_grid / counts).reshape(TARGET_H, TARGET_W)

    # 5. 保存
    filename = f"AEF_Global_0.5deg_{year}.npy"
    save_path = os.path.join(OUTPUT_DIR, filename)
    np.save(save_path, aef_grid)
    print(f"   ✅ {year} 年数据已保存: {filename}")

def main():
    print("="*60)
    print("🚀 双年份 AEF 合并程序")
    print("="*60)

    process_year(2022, CSV_DIR_2022)
    process_year(2023, CSV_DIR_2023)

if __name__ == '__main__':
    main()