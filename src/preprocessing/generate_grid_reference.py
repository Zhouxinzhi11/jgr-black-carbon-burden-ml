# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
import numpy as np
import os

# ================= 配置 =================
OUTPUT_DIR = r"data/RESTRICTED_INPUT_NOT_INCLUDED\00_grid_ref"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_grid():
    """
    生成 MERRA-2 标准网格参考系
    网格规格：
    - 纬度：-90 到 90°，步长 0.5°，共 361 点
    - 经度：-180 到 179.375°，步长 0.625°，共 576 点
    """
    print("🌍 [Step 1] 生成 MERRA-2 标准网格参考系...")

    # 1. 生成坐标轴
    lat_vals = np.linspace(-90, 90, 361)
    lon_vals = np.linspace(-180, 179.375, 576)

    # 2. 严格验证网格参数
    print("   🔍 执行网格参数校验...")

    # 验证点数
    assert len(lat_vals) == 361, f"❌ 纬度点数错误: 应为 361, 实为 {len(lat_vals)}"
    assert len(lon_vals) == 576, f"❌ 经度点数错误: 应为 576, 实为 {len(lon_vals)}"

    # 验证步长 (允许浮点误差 1e-5)
    lat_step = lat_vals[1] - lat_vals[0]
    lon_step = lon_vals[1] - lon_vals[0]
    assert np.isclose(lat_step, 0.5, atol=1e-5), \
        f"❌ 纬度步长错误: 应为 0.5, 实为 {lat_step:.6f}"
    assert np.isclose(lon_step, 0.625, atol=1e-5), \
        f"❌ 经度步长错误: 应为 0.625, 实为 {lon_step:.6f}"

    # 验证边界值
    assert np.isclose(lat_vals[0], -90, atol=1e-5), "❌ 纬度起点应为 -90"
    assert np.isclose(lat_vals[-1], 90, atol=1e-5), "❌ 纬度终点应为 90"
    assert np.isclose(lon_vals[0], -180, atol=1e-5), "❌ 经度起点应为 -180"
    assert np.isclose(lon_vals[-1], 179.375, atol=1e-5), "❌ 经度终点应为 179.375"

    print("   ✅ 校验通过！网格参数严格符合 MERRA-2 标准。")

    # 3. 生成 2D 网格矩阵 (meshgrid)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    # 4. 保存为 npz 文件
    save_path = os.path.join(OUTPUT_DIR, "merra2_grid_coords.npz")
    np.savez_compressed(
        save_path,
        lats=lat_vals,  # 1D 纬度数组
        lons=lon_vals,  # 1D 经度数组
        lat_grid=lat_grid,  # 2D 纬度网格
        lon_grid=lon_grid  # 2D 经度网格
    )

    print(f"🎉 网格参考文件已保存: {save_path}")
    print(f"   网格形状: {lat_grid.shape}")
    print(f"   文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")


if __name__ == '__main__':
    generate_grid()