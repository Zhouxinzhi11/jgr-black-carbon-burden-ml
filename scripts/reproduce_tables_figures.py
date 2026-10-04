#!/usr/bin/env python3
"""Rebuild public tables and aggregate-only figures from packaged CSV files.

The utility does not access restricted observations. Spatial panels in main
Figure 3 and SI Figure S3 are represented by their explicitly public summary
statistics because pointwise grids were intentionally withheld.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mean_sd(value: str) -> tuple[float, float]:
    cleaned = value.strip().replace("−", "-")
    if "±" in cleaned:
        mean, sd = cleaned.split("±", 1)
        return float(mean.strip()), float(sd.strip())
    return float(cleaned), 0.0


def copy_tables(release_root: Path, output_dir: Path) -> int:
    source = release_root / "results"
    destination = output_dir / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob("*.csv")):
        shutil.copy2(path, destination / path.name)
        count += 1
    return count


def setup_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
    return plt


def save_figure(fig, path: Path, plt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def workflow_figure(rows, path, plt):
    fig, axis = plt.subplots(figsize=(9, 2.8))
    axis.axis("off")
    positions = [0.18, 0.5, 0.82]
    for x, row in zip(positions, rows):
        text = f"{row['branch'].replace('_', ' ').title()}\n\n{row['target_or_output']}\n{row['reported_elements']}"
        axis.text(
            x, 0.5, text, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#e8f1f8", "edgecolor": "#2d5f8b"},
        )
    axis.annotate("", xy=(0.4, 0.5), xytext=(0.28, 0.5), arrowprops={"arrowstyle": "->"})
    axis.annotate("", xy=(0.72, 0.5), xytext=(0.6, 0.5), arrowprops={"arrowstyle": "->"})
    axis.set_title("Public workflow summary (aggregate reconstruction)")
    save_figure(fig, path, plt)


def grouped_metric_figure(rows, metric, title, path, plt):
    splits = list(dict.fromkeys(row["evaluation_split"] for row in rows))
    models = list(dict.fromkeys(row["Model configuration"] for row in rows))
    width = 0.8 / max(len(models), 1)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for model_index, model in enumerate(models):
        values, errors = [], []
        for split in splits:
            row = next(item for item in rows if item["evaluation_split"] == split and item["Model configuration"] == model)
            value, error = mean_sd(row[metric])
            values.append(value)
            errors.append(error)
        x = [index - 0.4 + width / 2 + model_index * width for index in range(len(splits))]
        axis.bar(x, values, width, yerr=errors, capsize=2, label=model)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(range(len(splits)), [item.replace("_", "\n") for item in splits])
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.legend(fontsize=7, ncol=2)
    save_figure(fig, path, plt)


def bounded_figure(rows, path, plt):
    labels = [row["Setting"].replace(" adjustment", "\nadjustment") for row in rows]
    values = [float(row["R²"].replace("−", "-")) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.2))
    bars = axis.bar(range(len(rows)), values, color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"])
    axis.set_xticks(range(len(rows)), labels, rotation=10, ha="right")
    axis.set_ylabel("R²")
    axis.set_title("Bounded-adjustment comparison from public aggregates")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    save_figure(fig, path, plt)


def monthly_support_figure(rows, path, plt):
    filtered = [row for row in rows if row["Source days"] not in {"", "NA", "0"}]
    labels = [row["Year-month"] for row in filtered]
    days = [float(row["Source days"].replace(",", "")) for row in filtered]
    coverage = [float(row["Evaluation-grid coverage (%)"].replace(",", "")) for row in filtered]
    fig, first = plt.subplots(figsize=(10, 4.2))
    second = first.twinx()
    first.plot(labels, days, marker="o", color="#2d5f8b", label="Source days")
    second.plot(labels, coverage, marker="s", color="#c44e52", label="Evaluation-grid coverage")
    first.set_ylabel("Source days")
    second.set_ylabel("Coverage (%)")
    first.tick_params(axis="x", rotation=55)
    first.set_title("Monthly sampling support from public aggregates")
    first.legend(loc="upper left")
    second.legend(loc="upper right")
    save_figure(fig, path, plt)


def histogram_figure(rows, path, plt):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["year_month"]].append(row)
    fig, axis = plt.subplots(figsize=(10, 5))
    for month, month_rows in grouped.items():
        month_rows.sort(key=lambda row: int(row["valid_day_count"]))
        axis.plot(
            [int(row["valid_day_count"]) for row in month_rows],
            [float(row["grid_cell_fraction"]) for row in month_rows],
            alpha=0.65,
            linewidth=1,
            label=month,
        )
    axis.set_xlabel("Valid day count")
    axis.set_ylabel("Grid-cell fraction")
    axis.set_yscale("log")
    axis.set_title("Valid-day-count distributions from aggregated histograms")
    axis.legend(fontsize=6, ncol=3)
    save_figure(fig, path, plt)


def summary_only_figure(rows, title, path, plt):
    fig, axis = plt.subplots(figsize=(9, max(2.6, 0.38 * len(rows) + 1.2)))
    axis.axis("off")
    headers = list(rows[0]) if rows else []
    values = [[row[header] for header in headers] for row in rows]
    axis.table(cellText=values, colLabels=headers, loc="center", cellLoc="center")
    axis.set_title(title + "\n(pointwise spatial arrays intentionally withheld)")
    save_figure(fig, path, plt)


def reproduce_figures(release_root: Path, output_dir: Path) -> int:
    source = release_root / "figure_source_data"
    destination = output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt = setup_plotting()
    workflow_figure(read_rows(source / "figure_1_workflow.csv"), destination / "figure_1_workflow.png", plt)
    grouped_metric_figure(
        read_rows(source / "figure_2_meanstate_performance.csv"), "R²",
        "Mean-state performance from public aggregates", destination / "figure_2_meanstate_performance.png", plt,
    )
    summary_only_figure(
        read_rows(source / "figure_3_spatial_summary_only.csv"),
        "Figure 3 public spatial summary", destination / "figure_3_spatial_summary.png", plt,
    )
    grouped_metric_figure(
        read_rows(source / "figure_4_daily_anomaly_performance.csv"), "Anomaly R²",
        "Daily-anomaly performance from public aggregates", destination / "figure_4_daily_anomaly_performance.png", plt,
    )
    bounded_figure(read_rows(source / "figure_5_bounded_adjustment.csv"), destination / "figure_5_bounded_adjustment.png", plt)
    monthly_support_figure(read_rows(source / "figure_s1_monthly_sampling_support.csv"), destination / "figure_s1_monthly_sampling_support.png", plt)
    histogram_figure(read_rows(source / "figure_s2_valid_day_count_histograms.csv"), destination / "figure_s2_valid_day_count_histograms.png", plt)
    summary_only_figure(
        read_rows(source / "figure_s3_spatial_common_support_summary_only.csv"),
        "Figure S3 public common-support summary", destination / "figure_s3_common_support_summary.png", plt,
    )
    return 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables-only", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    release_root = args.release_root.resolve()
    output_dir = (args.output_dir or (release_root / "reproduced")).resolve()
    table_count = 0 if args.figures_only else copy_tables(release_root, output_dir)
    figure_count = 0 if args.tables_only else reproduce_figures(release_root, output_dir)
    print(f"REPRODUCED_TABLES={table_count}")
    print(f"REPRODUCED_FIGURES={figure_count}")
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
