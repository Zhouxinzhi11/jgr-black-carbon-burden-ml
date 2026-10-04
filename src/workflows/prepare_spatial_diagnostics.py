# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
65_plot_meanstate_spatial_diagnostics_figure5_v3.py

Generate a publication-oriented spatial diagnostic figure from the existing
formal 2022 mean-state checkpoints. The script performs read-only inference;
it does not train any model or modify any existing artifact.

Figure design (rows: November and December 2022):
1. reanalysis-derived target monthly log-BC burden;
2. three-seed ensemble-mean joint prediction;
3. absolute-error advantage of joint over meteorology-only;
4. absolute-error advantage of joint over cyclic-shift control.

For the difference panels:
    advantage = |error_control| - |error_joint|
so positive values indicate lower absolute error for the joint model.

Outputs:
- Figure5_meanstate_spatial_diagnostics.png (600 dpi)
- Figure5_meanstate_spatial_diagnostics.pdf
- Figure5_meanstate_spatial_diagnostics.svg
- Figure5_meanstate_spatial_diagnostics_data.npz
- Figure5_meanstate_spatial_diagnostics.json
- Figure5_meanstate_spatial_diagnostics_summary.md

Expected project root:
.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


SEEDS = (42, 43, 44)
H = 361
W = 576
LAT = np.linspace(-90.0, 90.0, H, dtype=np.float64)
LON = np.linspace(-180.0, 179.375, W, dtype=np.float64)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)

EXPERIMENTS = {
    "joint": "mean_emis_meteo_clim",
    "meteorology": "mean_meteo_clim_only",
    "cyclic": "mean_emis_meteo_clim_shuffled",
}


class FigureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureError(message)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def save_json_strict(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module_from_path(name: str, path: Path):
    require(path.is_file(), f"Required evaluation script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def manifest_experiment_name(manifest: Mapping[str, Any]) -> str:
    experiment = manifest.get("experiment", {})
    if isinstance(experiment, dict):
        return str(experiment.get("name", ""))
    return str(experiment or "")


def manifest_seed(manifest: Mapping[str, Any]) -> int:
    settings = manifest.get("settings", {})
    if isinstance(settings, dict) and "seed" in settings:
        return int(settings["seed"])
    return int(manifest.get("seed", -1))


def discover_run(root: Path, experiment: str, seed: int) -> Path:
    """Resolve the canonical sealed formal run, excluding interrupted copies."""
    canonical = (
        root
        / "runs_s2_meanstate_clean"
        / "formal"
        / f"seed{seed}"
        / f"{experiment}_seed{seed}"
    )
    manifest_path = canonical / "logs" / "run_manifest.json"
    checkpoint_path = canonical / "checkpoints" / "best_model.pt"
    if manifest_path.is_file() and checkpoint_path.is_file():
        manifest = load_json(manifest_path)
        require(str(manifest.get("run_mode", "")) == "formal", f"Canonical run is not formal: {canonical}")
        require(manifest_experiment_name(manifest) == experiment, f"Canonical experiment mismatch: {canonical}")
        require(manifest_seed(manifest) == seed, f"Canonical seed mismatch: {canonical}")
        return canonical.resolve()

    formal_root = root / "runs_s2_meanstate_clean" / "formal"
    require(formal_root.is_dir(), f"Formal run root not found: {formal_root}")
    excluded_tokens = ("interrupted", "archive", "archived", "backup", "bak", "old")
    candidates: List[Path] = []
    for mp in formal_root.rglob("run_manifest.json"):
        if mp.parent.name != "logs":
            continue
        run_dir = mp.parent.parent.resolve()
        lower = str(run_dir).lower()
        if any(token in lower for token in excluded_tokens):
            continue
        try:
            manifest = load_json(mp)
        except Exception:
            continue
        if str(manifest.get("run_mode", "")) != "formal":
            continue
        if manifest_experiment_name(manifest) != experiment:
            continue
        if manifest_seed(manifest) != seed:
            continue
        if (run_dir / "checkpoints" / "best_model.pt").is_file():
            candidates.append(run_dir)
    unique = sorted({normalized(p): p for p in candidates}.values(), key=str)
    require(
        len(unique) == 1,
        f"Could not resolve one canonical run for {experiment}, seed {seed}; found {len(unique)}:\n"
        + "\n".join(str(p) for p in unique),
    )
    return unique[0]


def find_entry(entries: Sequence[Mapping[str, Any]], year: int, month: int) -> Mapping[str, Any]:
    matched = [
        entry for entry in entries
        if int(entry["year"]) == year and int(entry["month"]) == month
    ]
    require(len(matched) == 1, f"Expected one entry for {year}-{month:02d}; found {len(matched)}")
    return matched[0]


def area_weight_grid() -> np.ndarray:
    return np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, W), dtype=np.float64)


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    weighted: bool,
) -> Dict[str, float]:
    p = np.asarray(prediction, dtype=np.float64)[valid_mask]
    t = np.asarray(target, dtype=np.float64)[valid_mask]
    require(p.size > 1 and p.shape == t.shape, "Metric vectors invalid")
    require(np.isfinite(p).all() and np.isfinite(t).all(), "Metric vectors non-finite")
    if weighted:
        w = area_weight_grid()[valid_mask]
    else:
        w = np.ones_like(t, dtype=np.float64)
    w_sum = float(np.sum(w))
    require(w_sum > 0.0, "Metric weights sum to zero")
    mean_t = float(np.sum(w * t) / w_sum)
    residual = p - t
    ss_res = float(np.sum(w * residual ** 2))
    ss_tot = float(np.sum(w * (t - mean_t) ** 2))
    require(ss_tot > 0.0, "Target variance is zero")
    return {
        "log_r2": float(1.0 - ss_res / ss_tot),
        "log_rmse": float(np.sqrt(np.sum(w * residual ** 2) / w_sum)),
        "log_bias": float(np.sum(w * residual) / w_sum),
    }


def predict_month(
    module,
    model: torch.nn.Module,
    spec,
    entry: Mapping[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    payload_path = Path(str(entry["file"])).expanduser().resolve()
    require(payload_path.is_file(), f"Monthly payload missing: {payload_path}")
    payload = module.load_pt(payload_path)
    x = module.build_input_from_payload(
        payload,
        spec,
        key=payload_path.name,
        proxy_emission_log_mean=None,
    ).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        prediction = (
            model(x).squeeze(0).squeeze(0).detach().cpu().float().numpy()
        )
    target = payload["target_log_mean_conc"].detach().cpu().float().numpy()
    valid = (payload["valid_mask"].detach().cpu().float().numpy() > 0.5)
    require(prediction.shape == (H, W), "Prediction shape mismatch")
    require(target.shape == (H, W), "Target shape mismatch")
    require(valid.shape == (H, W), "Mask shape mismatch")
    require(np.isfinite(prediction).all(), f"Prediction non-finite: {payload_path}")
    require(np.isfinite(target).all(), f"Target non-finite: {payload_path}")
    return prediction, target, valid, payload_path


def try_basemap_coastlines(ax) -> bool:
    """Draw coastlines only when Basemap is already installed; never download data."""
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
        m.drawcoastlines(linewidth=0.35, color="0.25")
        return True
    except Exception:
        return False


def setup_geo_axis(ax, show_ylabels: bool, show_xlabels: bool) -> None:
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-60, -30, 0, 30, 60])
    if show_xlabels:
        ax.set_xticklabels(["120°W", "60°W", "0°", "60°E", "120°E"])
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=7.5, length=2.5, width=0.6)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    try_basemap_coastlines(ax)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a two-month ensemble spatial diagnostic figure from locked formal runs."
    )
    parser.add_argument(
        "--root",
        default=r".",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--months", default="11,12")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require(root.is_dir(), f"Project root not found: {root}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")
    try:
        months = tuple(int(v.strip()) for v in args.months.split(",") if v.strip())
    except ValueError as exc:
        raise FigureError("--months must be comma-separated integers") from exc
    require(months == (11, 12), "Primary manuscript figure is locked to months 11,12")

    module = import_module_from_path(
        "meanstate_eval_spatial_figure5",
        root / "52_eval_s2_meanstate_clean_formal_run.py",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda:
        require(device.type == "cuda", "CUDA is required but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("=" * 88)
    print("MEANSTATE_SPATIAL_DIAGNOSTICS_FIGURE5_START", flush=True)
    print(f"ROOT={root}")
    print(f"DEVICE={device}")
    print("TRAINING_PERFORMED=False")
    print("=" * 88)

    predictions: Dict[int, Dict[str, List[np.ndarray]]] = {
        month: {key: [] for key in EXPERIMENTS} for month in months
    }
    targets: Dict[int, np.ndarray] = {}
    masks: Dict[int, np.ndarray] = {}
    provenance: List[Dict[str, Any]] = []

    for key, experiment in EXPERIMENTS.items():
        for seed in SEEDS:
            run_dir = discover_run(root, experiment, seed)
            manifest_path = run_dir / "logs" / "run_manifest.json"
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            manifest = load_json(manifest_path)
            spec = module.MeanStateSpec(**manifest["experiment"])
            require(not bool(spec.uses_emission) or experiment != "mean_meteo_clim_only", "Bad spec")
            model = module.MeanStateUNet(
                spec=spec,
                base_features=int(manifest["settings"]["base_features"]),
            ).to(device)
            checkpoint = module.load_checkpoint(checkpoint_path, device)
            require(str(checkpoint.get("experiment", "")) == experiment, "Checkpoint mismatch")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            validation_entries = list(manifest["val_entries"])
            for month in months:
                entry = find_entry(validation_entries, 2022, month)
                pred, target, valid, payload_path = predict_month(
                    module, model, spec, entry, device
                )
                predictions[month][key].append(pred)
                if month not in targets:
                    targets[month] = target
                    masks[month] = valid
                else:
                    require(
                        np.array_equal(masks[month], valid),
                        f"Valid mask differs across runs for 2022-{month:02d}",
                    )
                    require(
                        np.allclose(targets[month], target, rtol=0.0, atol=0.0),
                        f"Target differs across runs for 2022-{month:02d}",
                    )
                provenance.append(
                    {
                        "experiment": experiment,
                        "role": key,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "month": month,
                        "payload": str(payload_path),
                        "payload_sha256": sha256_file(payload_path),
                    }
                )
                print(
                    f"INFERENCE_PASS role={key} seed={seed} month=2022-{month:02d}",
                    flush=True,
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ensemble: Dict[int, Dict[str, np.ndarray]] = {}
    figure_arrays: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {}
    for month in months:
        ensemble[month] = {}
        for key in EXPERIMENTS:
            require(len(predictions[month][key]) == len(SEEDS), f"Missing seeds: {month} {key}")
            ensemble[month][key] = np.mean(
                np.stack(predictions[month][key], axis=0), axis=0
            ).astype(np.float32)
        target = targets[month].astype(np.float32)
        valid = masks[month]
        joint = ensemble[month]["joint"]
        meteo = ensemble[month]["meteorology"]
        cyclic = ensemble[month]["cyclic"]
        advantage_meteo = np.abs(meteo - target) - np.abs(joint - target)
        advantage_cyclic = np.abs(cyclic - target) - np.abs(joint - target)

        prefix = f"m{month:02d}"
        figure_arrays[f"{prefix}_target"] = target
        figure_arrays[f"{prefix}_valid_mask"] = valid.astype(np.uint8)
        figure_arrays[f"{prefix}_joint_ensemble"] = joint
        figure_arrays[f"{prefix}_meteorology_ensemble"] = meteo
        figure_arrays[f"{prefix}_cyclic_ensemble"] = cyclic
        figure_arrays[f"{prefix}_advantage_vs_meteorology"] = advantage_meteo.astype(np.float32)
        figure_arrays[f"{prefix}_advantage_vs_cyclic"] = advantage_cyclic.astype(np.float32)

        diagnostics[str(month)] = {
            "year": 2022,
            "month": month,
            "n_valid_pixels": int(valid.sum()),
            "ensemble_metrics": {
                key: {
                    "unweighted": metrics(ensemble[month][key], target, valid, False),
                    "area_weighted": metrics(ensemble[month][key], target, valid, True),
                }
                for key in EXPERIMENTS
            },
            "joint_absolute_error_advantage": {
                "versus_meteorology": {
                    "mean_unweighted": float(np.mean(advantage_meteo[valid])),
                    "mean_area_weighted": float(
                        np.average(advantage_meteo[valid], weights=area_weight_grid()[valid])
                    ),
                    "fraction_valid_cells_positive": float(np.mean(advantage_meteo[valid] > 0)),
                },
                "versus_cyclic": {
                    "mean_unweighted": float(np.mean(advantage_cyclic[valid])),
                    "mean_area_weighted": float(
                        np.average(advantage_cyclic[valid], weights=area_weight_grid()[valid])
                    ),
                    "fraction_valid_cells_positive": float(np.mean(advantage_cyclic[valid] > 0)),
                },
            },
        }

    target_values = np.concatenate(
        [targets[m][masks[m]].astype(np.float64) for m in months]
        + [ensemble[m]["joint"][masks[m]].astype(np.float64) for m in months]
    )
    vmin, vmax = np.nanpercentile(target_values, [2.0, 98.0])
    if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(target_values)), float(np.nanmax(target_values))
    difference_chunks: List[np.ndarray] = []
    for month in months:
        difference_chunks.append(
            np.abs(
                figure_arrays[
                    f"m{month:02d}_advantage_vs_meteorology"
                ][masks[month]]
            )
        )
        difference_chunks.append(
            np.abs(
                figure_arrays[
                    f"m{month:02d}_advantage_vs_cyclic"
                ][masks[month]]
            )
        )
    difference_values = np.concatenate(difference_chunks)
    diff_limit = float(np.nanpercentile(difference_values, 98.0))
    require(math.isfinite(diff_limit) and diff_limit > 0.0, "Difference color limit invalid")

    # Match the publication style used for Figures 1–4.
    # Final reproduction width: 180 mm (two-column width).
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(180.0 / 25.4, 4.45),
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.070,
        right=0.985,
        top=0.905,
        bottom=0.155,
        wspace=0.105,
        hspace=0.275,
    )

    column_titles = (
        "Reanalysis-derived target",
        "Joint prediction\n(three-seed ensemble)",
        "Joint advantage over\nmeteorology-only",
        "Joint advantage over\ncyclic-shift",
    )
    panel_letters = list("abcdefgh")
    seq_mappable = None
    diff_mappable = None
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)

    seq_cmap = plt.get_cmap("viridis").copy()
    seq_cmap.set_bad("white")
    diff_cmap = plt.get_cmap("RdBu_r").copy()
    diff_cmap.set_bad("white")

    for row, month in enumerate(months):
        valid = masks[month]
        fields = (
            targets[month],
            ensemble[month]["joint"],
            figure_arrays[f"m{month:02d}_advantage_vs_meteorology"],
            figure_arrays[f"m{month:02d}_advantage_vs_cyclic"],
        )
        for col, field in enumerate(fields):
            ax = axes[row, col]
            masked = np.ma.masked_where(~valid, field)
            if col < 2:
                image = ax.pcolormesh(
                    LON_GRID,
                    LAT_GRID,
                    masked,
                    shading="auto",
                    cmap=seq_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
                seq_mappable = image
            else:
                image = ax.pcolormesh(
                    LON_GRID,
                    LAT_GRID,
                    masked,
                    shading="auto",
                    cmap=diff_cmap,
                    norm=diff_norm,
                    rasterized=True,
                )
                diff_mappable = image
            setup_geo_axis(
                ax,
                show_ylabels=(col == 0),
                show_xlabels=(row == len(months) - 1),
            )
            if row == 0:
                ax.set_title(column_titles[col], pad=5.0, fontweight="normal")
            ax.text(
                0.012,
                0.985,
                f"({panel_letters[row * 4 + col]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.6,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )
            if col == 0:
                ax.text(
                    -0.23,
                    0.50,
                    f"{['November', 'December'][row]} 2022",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                )

    require(seq_mappable is not None and diff_mappable is not None, "Plot mappables missing")
    cax1 = fig.add_axes([0.19, 0.065, 0.29, 0.026])
    cb1 = fig.colorbar(seq_mappable, cax=cax1, orientation="horizontal")
    cb1.set_label("Monthly mean transformed BC burden (log space)", labelpad=2.0)
    cb1.ax.tick_params(labelsize=7.2, length=2.5)

    cax2 = fig.add_axes([0.61, 0.065, 0.29, 0.026])
    cb2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal")
    cb2.set_label(
        r"Absolute-error advantage, $|e_{control}|-|e_{joint}|$",
        labelpad=2.0,
    )
    cb2.ax.tick_params(labelsize=7.2, length=2.5)

    fig.text(
        0.755,
        0.018,
        "Positive values indicate lower absolute error for the joint model.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.25",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    base = output_dir / "Figure5_meanstate_spatial_diagnostics_v3"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    word_svg_path = base.with_name(base.name + "_word").with_suffix(".svg")
    editable_svg_path = base.with_name(base.name + "_editable").with_suffix(".svg")
    npz_path = base.with_name(base.name + "_data").with_suffix(".npz")
    json_path = base.with_suffix(".json")
    md_path = base.with_name(base.name + "_summary").with_suffix(".md")

    original_svg_fonttype = plt.rcParams.get("svg.fonttype", "path")

    # Word-safe SVG: all text becomes vector paths, avoiding Word SVG
    # font/baseline substitution problems.
    plt.rcParams["svg.fonttype"] = "path"
    fig.savefig(word_svg_path, facecolor="white")

    # Editable SVG for Inkscape / Illustrator.
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(editable_svg_path, facecolor="white")

    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")

    plt.rcParams["svg.fonttype"] = original_svg_fonttype
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        latitude=LAT.astype(np.float32),
        longitude=LON.astype(np.float32),
        **figure_arrays,
    )

    payload = {
        "artifact_name": "Figure5_meanstate_spatial_diagnostics",
        "artifact_version": 3,
        "root": str(root),
        "device": str(device),
        "evaluation_status": "2022 held-out model-selection validation",
        "months": list(months),
        "seeds": list(SEEDS),
        "experiments": EXPERIMENTS,
        "figure_definition": {
            "target_and_prediction_color_limits_2nd_98th_percentile": [
                float(vmin),
                float(vmax),
            ],
            "difference_symmetric_98th_percentile_limit": diff_limit,
            "difference_formula": "abs(error_control) - abs(error_joint)",
            "positive_difference_meaning": "joint model has lower absolute error",
            "invalid_cells": "masked",
            "coastline_behavior": (
                "Basemap coastlines are drawn only if the package is already installed; "
                "no online map data are downloaded."
            ),
        },
        "diagnostics": diagnostics,
        "provenance": provenance,
        "outputs": {
            "png_600dpi": str(png_path),
            "pdf_vector": str(pdf_path),
            "svg_word_safe": str(word_svg_path),
            "svg_editable": str(editable_svg_path),
            "npz": str(npz_path),
        },
        "figure_style_contract": {
            "final_width_mm": 180.0,
            "font_family": "Arial/Helvetica/DejaVu Sans fallback",
            "panel_labels": "bold lowercase letters",
            "target_prediction_colormap": "viridis sequential",
            "signed_difference_colormap": "RdBu_r zero-centred",
            "png_dpi": 600,
            "word_safe_svg_text": "converted to vector paths",
            "editable_svg_text": "preserved",
        },
        "scientific_boundary": (
            "All mapped target and prediction fields are real arrays from the locked formal "
            "2022 validation payloads and checkpoints; no schematic or AI-generated research "
            "field is used. The figure uses the three pre-specified formal checkpoints and "
            "the two held-out model-selection months. It is a spatial diagnostic, not an "
            "independent test or physical source-attribution map."
        ),
        "verification": {
            "training_performed": False,
            "checkpoints_modified": False,
            "targets_identical_across_runs": True,
            "valid_masks_identical_across_runs": True,
            "all_outputs_finite_on_valid_cells": True,
            "overall_pass": True,
        },
    }
    save_json_strict(payload, json_path)

    md_lines = [
        "# Mean-state spatial diagnostics",
        "",
        "- Gate: `MEANSTATE_SPATIAL_DIAGNOSTICS_FIGURE5_V3_PASS`",
        "- Training performed: `False`",
        "- Evaluation setting: `2022 held-out model-selection validation`",
        "- Months: November and December 2022",
        "- Predictions: arithmetic ensemble mean across seeds 42, 43, and 44",
        "",
        "## Panel interpretation",
        "",
        "The first two columns show the reanalysis-derived monthly target and the joint-model "
        "ensemble prediction. The last two columns show `|error_control| - |error_joint|`; positive "
        "values indicate cells where the joint model has lower absolute error than the selected control.",
        "",
        "## Scientific boundary",
        "",
        "These maps diagnose where the existing formal models differ within the held-out model-selection "
        "months. They do not constitute an independent spatial test, causal source attribution, or "
        "validation of native emission-inventory geography.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("MEANSTATE_SPATIAL_DIAGNOSTICS_FIGURE5_V3_PASS")
    print(f"png_600dpi   = {png_path}")
    print(f"pdf_vector   = {pdf_path}")
    print(f"svg_word     = {word_svg_path}")
    print(f"svg_editable = {editable_svg_path}")
    print(f"npz          = {npz_path}")
    print(f"json         = {json_path}")
    print(f"summary      = {md_path}")
    print("TRAINING_PERFORMED=False")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FigureError as exc:
        print("=" * 88, file=sys.stderr)
        print("MEANSTATE_SPATIAL_DIAGNOSTICS_FIGURE5_V3_FAIL", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 88, file=sys.stderr)
        raise SystemExit(2)
