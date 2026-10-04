# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
import json
from pathlib import Path
import csv

ROOT = Path(r".")
OUT = ROOT / "runs_directionB_inverse" / "b0_two_month_inverse_summary_v1"
OUT.mkdir(parents=True, exist_ok=True)

REPORTS = {
    "2022-11": ROOT / "runs_directionB_inverse" / "b0_inverse_2022_11_sector_offset_sweep_v1" / "b0_inverse_2022_11_sector_offset_sweep_report.json",
    "2022-12": ROOT / "runs_directionB_inverse" / "b0_inverse_2022_12_sector_offset_sweep_v1" / "b0_inverse_2022_12_sector_offset_sweep_report.json",
}

def row_id(r):
    if r["method"] != "bounded_sector_offset":
        return r["method"]
    return f"bounded_max{r['max_delta_bound']}_lambda{r['lambda_delta']}"

def get_train(rows):
    return next(r for r in rows if r["method"] == "train_mean_prior")

def get_true(rows):
    return next(r for r in rows if r["method"] == "true_edgar_reference")

def best_bounded(rows, max_delta_limit=None):
    cand = [r for r in rows if r["method"] == "bounded_sector_offset"]
    if max_delta_limit is not None:
        cand = [r for r in cand if float(r["max_delta_bound"]) <= max_delta_limit]
    return max(cand, key=lambda r: r["log_r2"])

records = []
compact = {}

for month, path in REPORTS.items():
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj["rows"]
    train = get_train(rows)
    true = get_true(rows)

    selections = {
        "train_mean_prior": train,
        "true_edgar_reference": true,
        "best_delta_le_0.25": best_bounded(rows, 0.25),
        "best_delta_le_0.5": best_bounded(rows, 0.5),
        "best_delta_le_1.0": best_bounded(rows, 1.0),
        "best_overall": best_bounded(rows, None),
    }

    compact[month] = {}

    for label, r in selections.items():
        rec = {
            "month": month,
            "selection": label,
            "method": r["method"],
            "max_delta_bound": r.get("max_delta_bound"),
            "lambda_delta": r.get("lambda_delta"),
            "log_r2": r["log_r2"],
            "log_rmse": r["log_rmse"],
            "log_bias": r["log_bias"],
            "emis_log_rmse": r["emis_log_rmse"],
            "mean_abs_channel_mean_error": r["mean_abs_channel_mean_error"],
            "delta_abs_max": r.get("delta_abs_max"),
            "delta_abs_mean": r.get("delta_abs_mean"),
            "delta_values": r.get("delta_values"),
            "delta_log_r2_vs_train": r["log_r2"] - train["log_r2"],
            "delta_log_rmse_vs_train": r["log_rmse"] - train["log_rmse"],
            "delta_emis_rmse_vs_train": r["emis_log_rmse"] - train["emis_log_rmse"],
            "delta_channel_mean_error_vs_train": r["mean_abs_channel_mean_error"] - train["mean_abs_channel_mean_error"],
            "delta_log_r2_vs_true_edgar": r["log_r2"] - true["log_r2"],
        }
        records.append(rec)
        compact[month][label] = rec

# Save JSON
json_path = OUT / "b0_two_month_inverse_summary.json"
json_path.write_text(json.dumps({
    "status": "B0_TWO_MONTH_INVERSE_SUMMARY_OK",
    "boundary": [
        "Two validation months only: 2022-11 and 2022-12.",
        "This summarizes bounded sector-offset inverse feasibility.",
        "This is not validated physical emission inventory retrieval."
    ],
    "compact": compact,
    "records": records,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# Save CSV
csv_path = OUT / "b0_two_month_inverse_summary.csv"
fields = [
    "month", "selection", "method", "max_delta_bound", "lambda_delta",
    "log_r2", "log_rmse", "emis_log_rmse", "mean_abs_channel_mean_error",
    "delta_abs_max", "delta_abs_mean",
    "delta_log_r2_vs_train", "delta_log_rmse_vs_train",
    "delta_emis_rmse_vs_train", "delta_channel_mean_error_vs_train",
    "delta_log_r2_vs_true_edgar",
    "delta_values",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({k: r.get(k) for k in fields})

# Markdown
md = []
md.append("# B0 two-month inverse feasibility summary")
md.append("")
md.append("## Boundary")
md.append("")
md.append("- Only validation months 2022-11 and 2022-12 are summarized.")
md.append("- This is a bounded sector-offset inverse feasibility analysis.")
md.append("- Optimized variables are emission_log_mean-like sector offsets, not validated physical emissions.")
md.append("")
md.append("## Selected results")
md.append("")
md.append("| month | selection | max_delta | lambda | log_R2 | ΔR2 vs train | log_RMSE | ΔRMSE vs train | emis_log_RMSE | Δemis_RMSE vs train | mean_abs_channel_mean_error | delta_abs_max | delta_abs_mean |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in records:
    md.append(
        f"| {r['month']} | {r['selection']} | {r['max_delta_bound']} | {r['lambda_delta']} | "
        f"{r['log_r2']:.6f} | {r['delta_log_r2_vs_train']:.6f} | "
        f"{r['log_rmse']:.6f} | {r['delta_log_rmse_vs_train']:.6f} | "
        f"{r['emis_log_rmse']:.6f} | {r['delta_emis_rmse_vs_train']:.6f} | "
        f"{r['mean_abs_channel_mean_error']:.6f} | "
        f"{r['delta_abs_max']:.6f} | {r['delta_abs_mean']:.6f} |"
    )

md.append("")
md.append("## Conservative interpretation")
md.append("")
md.append("- Bounded sector offsets improve concentration reconstruction in both validation months.")
md.append("- Small offsets are more interpretable, but gains are modest.")
md.append("- Larger offsets yield stronger concentration gains but weaken emission-prior plausibility.")
md.append("- This supports constrained inverse feasibility, not validated inventory retrieval.")

md_path = OUT / "b0_two_month_inverse_summary.md"
md_path.write_text("\n".join(md), encoding="utf-8")

print("===== saved =====")
print(json_path)
print(csv_path)
print(md_path)
print()
print(md_path.read_text(encoding="utf-8"))
print()
print("B0_TWO_MONTH_INVERSE_SUMMARY_OK")
