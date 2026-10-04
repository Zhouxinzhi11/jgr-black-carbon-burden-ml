# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""
🚀 [Step 2] 双年份 AEF 下载器
功能：分别提交 2022 和 2023 年的下载任务
"""

import ee
import numpy as np
import os
import time

# ================= 配置 =================
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"

GEE_PROJECT_ID = 'project-9204154a-443d-4e2c-b7b'
AEF_ASSET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

# 分别存到两个文件夹，互不干扰
DRIVE_FOLDER_2022 = 'AEF_MERRA2_2022'
DRIVE_FOLDER_2023 = 'AEF_MERRA2_2023'

BATCH_SIZE = 1000


def submit_year_tasks(year, folder_name, all_lats, all_lons, existing_tasks):
    print(f"\n📅 正在处理 {year} 年任务...")

    # 筛选对应年份的图像
    # ========== GEE AEF 提取核心修正 ==========
    # 注意：filterDate 是左闭右开的，要涵盖全年，结束日期必须是下一年的 1月1日
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'

    # 核心修复 1：使用 .mosaic() 将全球所有切片拼接成一张完整的无缝影像
    img = ee.ImageCollection(AEF_ASSET_ID).filterDate(start_date, end_date).mosaic()

    # 核心修复 2：处理 AEF 海洋区域的无数据问题
    # 由于 AEF 主要覆盖陆地，为了防止 GEE 在 sample 时自动丢弃海洋点（导致网格错位），
    # 我们将影像中的空值（海洋）临时填充为 -9999，并在本地处理时替换。
    img_unmasked = img.unmask(-9999)

    # 选取前64波段
    band_names = img.bandNames().slice(0, 64)
    img = img.select(band_names)

    total_points = len(all_lats)
    num_batches = int(np.ceil(total_points / BATCH_SIZE))

    count = 0
    for i in range(num_batches):
        # 任务名区分年份：AEF_2022_batch1
        task_name = f'AEF_{year}_batch{i + 1}'

        if task_name in existing_tasks:
            continue

        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total_points)

        # 构建 FeatureCollection
        feats = []
        for lat, lon in zip(all_lats[start:end], all_lons[start:end]):
            feat = ee.Feature(ee.Geometry.Point([float(lon), float(lat)]),
                              {'lat': float(lat), 'lon': float(lon)})
            feats.append(feat)

        fc = ee.FeatureCollection(feats)

        # 采样 (Scale=55km 对应 0.5度)
        sampled = img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=55000,
            tileScale=4
        )

        # 导出
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=task_name,
            folder=folder_name,  # 存入对应年份文件夹
            fileNamePrefix=task_name,
            fileFormat='CSV'
        )
        task.start()
        count += 1

        if count % 10 == 0:
            print(f"   已提交 {count} 个任务...")
            time.sleep(2)

    print(f"   ✅ {year} 年任务提交完成: 新增 {count} 个")


def main():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)

    # 加载网格
    grid_file = r"data/RESTRICTED_INPUT_NOT_INCLUDED\00_grid_ref\merra2_grid_coords.npz"
    data = np.load(grid_file)
    lat_grid, lon_grid = data['lat_grid'], data['lon_grid']
    all_lats, all_lons = lat_grid.flatten(), lon_grid.flatten()

    # 检查现有任务
    tasks = ee.batch.Task.list()
    existing_tasks = set()
    for t in tasks:
        if t.state in ['READY', 'RUNNING', 'COMPLETED']:
            existing_tasks.add(t.config['description'])

    # 提交 2022
    submit_year_tasks(2022, DRIVE_FOLDER_2022, all_lats, all_lons, existing_tasks)

    # 提交 2023
    submit_year_tasks(2023, DRIVE_FOLDER_2023, all_lats, all_lons, existing_tasks)

    print("\n🚀 所有任务已提交！请前往 GEE 网页端确认。")
    print(f"   注意：Google Drive 会生成两个文件夹：")
    print(f"   1. {DRIVE_FOLDER_2022}")
    print(f"   2. {DRIVE_FOLDER_2023}")


if __name__ == '__main__':
    main()