# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
66_render_figure5_spatial_diagnostics_v4.py

Render-only refinement of Figure 5 from the already generated Figure 5 v3 NPZ
and JSON artifacts. No model loading, training, or inference is performed.

Main-figure design:
Rows:
  November 2022
  December 2022

Columns:
  1. Reanalysis-derived target
  2. Three-seed ensemble-mean joint prediction
  3. Local absolute-error difference between cyclic-shift and joint

For column 3:
  advantage = |error_cyclic| - |error_joint|
Positive values indicate lower local absolute error for the joint model.

The meteorology-only spatial-advantage panels are deliberately omitted from the
main figure to improve readability. Their arrays and diagnostics remain in the
v3 NPZ/JSON artifacts and can be used later for a supplementary figure.

Outputs:
  Figure5_meanstate_spatial_diagnostics_v4_word.svg
  Figure5_meanstate_spatial_diagnostics_v4_editable.svg
  Figure5_meanstate_spatial_diagnostics_v4.pdf
  Figure5_meanstate_spatial_diagnostics_v4.png   (600 dpi)
  Figure5_meanstate_spatial_diagnostics_v4_render_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


H = 361
W = 576
LAT = np.linspace(-90.0, 90.0, H, dtype=np.float64)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float64)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)
MONTHS = (11, 12)


class RenderError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderError(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def save_json(payload: dict, path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def try_basemap_coastlines(ax) -> bool:
    """Use Basemap coastlines only if Basemap is already installed."""
    try:
        from mpl_toolkits.basemap import Basemap  # type: ignore
    except Exception:
        return False
    try:
        m = Basemap(
            projection="cyl",
            llcrnrlon=-180,
            urcrnrlon=180,
            llcrnrlat=-90,
            urcrnrlat=90,
            resolution="c",
            ax=ax,
        )
        m.drawcoastlines(linewidth=0.30, color="0.28")
        return True
    except Exception:
        return False


def setup_geo_axis(ax, *, show_ylabels: bool, show_xlabels: bool) -> None:
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)

    # Fewer ticks than v3 to avoid crowding at 180-mm final width.
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-60, 0, 60])

    if show_xlabels:
        ax.set_xticklabels(["120°W", "0°", "120°E"])
    else:
        ax.set_xticklabels([])

    if show_ylabels:
        ax.set_yticklabels(["60°S", "0°", "60°N"])
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", labelsize=7.1, length=2.4, width=0.65)
    ax.grid(True, linewidth=0.30, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.70)
    ax.spines["bottom"].set_linewidth(0.70)
    try_basemap_coastlines(ax)


def validate_artifacts(npz, meta: dict) -> None:
    require(meta.get("artifact_name") == "Figure5_meanstate_spatial_diagnostics",
            "Unexpected Figure 5 JSON artifact.")
    require(bool(meta.get("verification", {}).get("overall_pass", False)),
            "v3 Figure 5 verification did not pass.")
    require(meta.get("verification", {}).get("training_performed") is False,
            "Unexpected training flag in v3 JSON.")
    require(meta.get("verification", {}).get("checkpoints_modified") is False,
            "Unexpected checkpoint modification flag.")

    expected = []
    for m in MONTHS:
        expected.extend([
            f"m{m:02d}_target",
            f"m{m:02d}_valid_mask",
            f"m{m:02d}_joint_ensemble",
            f"m{m:02d}_advantage_vs_cyclic",
        ])

    missing = [k for k in expected if k not in npz.files]
    require(not missing, f"Missing NPZ arrays: {missing}")

    for m in MONTHS:
        target = np.asarray(npz[f"m{m:02d}_target"])
        mask = np.asarray(npz[f"m{m:02d}_valid_mask"]).astype(bool)
        joint = np.asarray(npz[f"m{m:02d}_joint_ensemble"])
        advantage = np.asarray(npz[f"m{m:02d}_advantage_vs_cyclic"])

        for name, arr in [
            ("target", target),
            ("mask", mask),
            ("joint", joint),
            ("advantage", advantage),
        ]:
            require(arr.shape == (H, W), f"Bad {name} shape for month {m}: {arr.shape}")

        require(mask.any(), f"No valid cells for month {m}")
        require(np.isfinite(target[mask]).all(), f"Non-finite target values for month {m}")
        require(np.isfinite(joint[mask]).all(), f"Non-finite joint values for month {m}")
        require(np.isfinite(advantage[mask]).all(), f"Non-finite advantage values for month {m}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a less-crowded 2x3 main Figure 5 from sealed v3 NPZ/JSON artifacts."
    )
    parser.add_argument("--npz", required=True, help="Figure5 v3 data NPZ")
    parser.add_argument("--json", required=True, help="Figure5 v3 JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(npz_path.is_file(), f"NPZ not found: {npz_path}")
    require(json_path.is_file(), f"JSON not found: {json_path}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    print("=" * 88, flush=True)
    print("FIGURE5_SPATIAL_RENDER_V4_START", flush=True)
    print(f"NPZ={npz_path}", flush=True)
    print(f"JSON={json_path}", flush=True)
    print("TRAINING_PERFORMED=False", flush=True)
    print("INFERENCE_PERFORMED=False", flush=True)
    print("=" * 88, flush=True)

    meta = load_json(json_path)
    npz = np.load(npz_path, allow_pickle=False)
    validate_artifacts(npz, meta)
    print("SEALED_V3_ARTIFACT_CHECK=PASS", flush=True)

    targets = {}
    masks = {}
    joints = {}
    advantages = {}

    for m in MONTHS:
        targets[m] = np.asarray(npz[f"m{m:02d}_target"], dtype=np.float32)
        masks[m] = np.asarray(npz[f"m{m:02d}_valid_mask"]).astype(bool)
        joints[m] = np.asarray(npz[f"m{m:02d}_joint_ensemble"], dtype=np.float32)
        advantages[m] = np.asarray(npz[f"m{m:02d}_advantage_vs_cyclic"], dtype=np.float32)

    # Shared sequential limits for target and joint prediction.
    seq_values = np.concatenate([
        targets[m][masks[m]].astype(np.float64) for m in MONTHS
    ] + [
        joints[m][masks[m]].astype(np.float64) for m in MONTHS
    ])
    vmin, vmax = np.nanpercentile(seq_values, [2.0, 98.0])
    require(math.isfinite(float(vmin)) and math.isfinite(float(vmax)) and vmax > vmin,
            "Invalid sequential colour limits.")

    # The main figure now contains only the cyclic-shift spatial contrast, so the
    # diverging limit is determined from that contrast alone.
    diff_values = np.concatenate([
        np.abs(advantages[m][masks[m]].astype(np.float64)) for m in MONTHS
    ])
    diff_limit = float(np.nanpercentile(diff_values, 98.0))
    require(math.isfinite(diff_limit) and diff_limit > 0.0,
            "Invalid difference colour limit.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    # 180 mm two-column width; fewer columns allow each map to be materially larger.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(180.0 / 25.4, 4.55),
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.080,
        right=0.985,
        top=0.905,
        bottom=0.175,
        wspace=0.12,
        hspace=0.24,
    )

    column_titles = (
        "Reanalysis-derived target",
        "Joint prediction\n(three-seed ensemble)",
        "Local absolute-error difference\n(cyclic − joint)",
    )
    panel_letters = list("abcdef")

    seq_cmap = plt.get_cmap("viridis").copy()
    seq_cmap.set_bad("white")
    diff_cmap = plt.get_cmap("RdBu_r").copy()
    diff_cmap.set_bad("white")
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)

    seq_mappable = None
    diff_mappable = None

    for row, m in enumerate(MONTHS):
        fields = (targets[m], joints[m], advantages[m])
        valid = masks[m]

        for col, field in enumerate(fields):
            ax = axes[row, col]
            masked = np.ma.masked_where(~valid, field)

            if col < 2:
                im = ax.pcolormesh(
                    LON_GRID, LAT_GRID, masked,
                    shading="auto",
                    cmap=seq_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
                seq_mappable = im
            else:
                im = ax.pcolormesh(
                    LON_GRID, LAT_GRID, masked,
                    shading="auto",
                    cmap=diff_cmap,
                    norm=diff_norm,
                    rasterized=True,
                )
                diff_mappable = im

            setup_geo_axis(
                ax,
                show_ylabels=(col == 0),
                show_xlabels=(row == 1),
            )

            if row == 0:
                ax.set_title(column_titles[col], pad=5.5, fontweight="normal")

            ax.text(
                0.012, 0.985,
                f"({panel_letters[row * 3 + col]})",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8.6,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8),
            )

            if col == 0:
                ax.text(
                    -0.20, 0.50,
                    f"{['November', 'December'][row]} 2022",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center", va="center",
                    fontsize=7.8,
                    fontweight="bold",
                )

    require(seq_mappable is not None and diff_mappable is not None,
            "Plot mappables were not created.")

    # Two compact colour bars only; explanatory prose is moved to the caption.
    cax1 = fig.add_axes([0.19, 0.072, 0.36, 0.022])
    cb1 = fig.colorbar(seq_mappable, cax=cax1, orientation="horizontal")
    cb1.set_label("Monthly mean transformed BC burden (log space)", labelpad=1.8)
    cb1.ax.tick_params(labelsize=7.0, length=2.4)

    cax2 = fig.add_axes([0.72, 0.072, 0.20, 0.022])
    cb2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal")
    cb2.set_label(
        r"$|e_{\mathrm{cyclic}}|-|e_{\mathrm{joint}}|$",
        labelpad=1.8,
    )
    cb2.ax.tick_params(labelsize=7.0, length=2.4)

    output_dir.mkdir(parents=True, exist_ok=False)
    base = output_dir / "Figure5_meanstate_spatial_diagnostics_v4"

    word_svg = base.with_name(base.name + "_word").with_suffix(".svg")
    editable_svg = base.with_name(base.name + "_editable").with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    render_json = base.with_name(base.name + "_render_manifest").with_suffix(".json")

    original_svg_fonttype = plt.rcParams.get("svg.fonttype", "path")

    plt.rcParams["svg.fonttype"] = "path"
    fig.savefig(word_svg, facecolor="white")

    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(editable_svg, facecolor="white")

    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")

    plt.rcParams["svg.fonttype"] = original_svg_fonttype
    plt.close(fig)
    npz.close()

    render_manifest = {
        "artifact_name": "Figure5_meanstate_spatial_diagnostics_v4",
        "render_only": True,
        "training_performed": False,
        "inference_performed": False,
        "source_npz": str(npz_path),
        "source_json": str(json_path),
        "layout": "2 rows x 3 columns",
        "months": [11, 12],
        "columns": [
            "reanalysis-derived target",
            "three-seed ensemble joint prediction",
            "local absolute-error difference (cyclic - joint)",
        ],
        "difference_formula": "abs(error_cyclic) - abs(error_joint)",
        "positive_difference_meaning": "joint has lower local absolute error",
        "target_prediction_limits": [float(vmin), float(vmax)],
        "difference_symmetric_limit": float(diff_limit),
        "style": {
            "final_width_mm": 180.0,
            "font_family": "Arial/Helvetica/DejaVu Sans fallback",
            "png_dpi": 600,
            "word_safe_svg": True,
            "editable_svg": True,
        },
        "scientific_boundary": (
            "This is a render-only spatial diagnostic of the 2022 held-out "
            "model-selection months. It does not constitute an independent test "
            "or physical source-attribution map."
        ),
        "outputs": {
            "word_svg": str(word_svg),
            "editable_svg": str(editable_svg),
            "pdf": str(pdf_path),
            "png_600dpi": str(png_path),
        },
    }
    save_json(render_manifest, render_json)

    print("=" * 88, flush=True)
    print("FIGURE5_SPATIAL_RENDER_V4_PASS", flush=True)
    print(f"word_svg     = {word_svg}", flush=True)
    print(f"editable_svg = {editable_svg}", flush=True)
    print(f"pdf          = {pdf_path}", flush=True)
    print(f"png_600dpi   = {png_path}", flush=True)
    print(f"manifest     = {render_json}", flush=True)
    print("TRAINING_PERFORMED=False", flush=True)
    print("INFERENCE_PERFORMED=False", flush=True)
    print("=" * 88, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
