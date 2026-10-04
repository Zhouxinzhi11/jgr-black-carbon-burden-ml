# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
import json
import csv
from pathlib import Path

ROOT = Path(r".")
OUT = ROOT / "runs_directionB_inverse" / "b0_fixed_setting_inverse_summary_v1"
OUT.mkdir(parents=True, exist_ok=True)

REPORTS = {
    "2022-11": ROOT / "runs_directionB_inverse" / "b0_inverse_2022_11_sector_offset_sweep_v1" / "b0_inverse_2022_11_sector_offset_sweep_report.json",
    "2022-12": ROOT / "runs_directionB_inverse" / "b0_inverse_2022_12_sector_offset_sweep_v1" / "b0_inverse_2022_12_sector_offset_sweep_report.json",
}

FIXED_SETTINGS = {
    "ultra_conservative_delta0.25_lambda0.1": (0.25, 0.1),
    "balanced_conservative_delta0.5_lambda0.1": (0.5, 0.1),
}

def find_row(rows, method=None, max_delta=None, lam=None):
    for r in rows:
        if method is not None and r["method"] != method:
            continue
        if max_delta is not None:
            if r["method"] != "bounded_sector_offset":
                continue
            if abs(float(r["max_delta_bound"]) - float(max_delta)) > 1e-12:
                continue
            if abs(float(r["lambda_delta"]) - float(lam)) > 1e-12:
                continue
        return r
    raise RuntimeError(f"row not found: method={method}, max_delta={max_delta}, lambda={lam}")

records = []

for month, path in REPORTS.items():
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj["rows"]

    train = find_row(rows, method="train_mean_prior")
    true = find_row(rows, method="true_edgar_reference")

    selected = {
        "train_mean_prior": train,
        "true_edgar_reference": true,
    }

    for label, (max_delta, lam) in FIXED_SETTINGS.items():
        selected[label] = find_row(rows, max_delta=max_delta, lam=lam)

    for label, r in selected.items():
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
            "delta_log_r2_vs_train": r["log_r2"] - train["log_r2"],
            "delta_log_rmse_vs_train": r["log_rmse"] - train["log_rmse"],
            "delta_emis_rmse_vs_train": r["emis_log_rmse"] - train["emis_log_rmse"],
            "delta_channel_mean_error_vs_train": r["mean_abs_channel_mean_error"] - train["mean_abs_channel_mean_error"],
            "delta_log_r2_vs_true_edgar": r["log_r2"] - true["log_r2"],
            "delta_values": r.get("delta_values"),
        }
        records.append(rec)

# Aggregate by selection
agg = {}
for sel in sorted(set(r["selection"] for r in records)):
    subset = [r for r in records if r["selection"] == sel]
    agg[sel] = {
        "n_months": len(subset),
        "mean_log_r2": sum(r["log_r2"] for r in subset) / len(subset),
        "mean_delta_log_r2_vs_train": sum(r["delta_log_r2_vs_train"] for r in subset) / len(subset),
        "mean_log_rmse": sum(r["log_rmse"] for r in subset) / len(subset),
        "mean_delta_log_rmse_vs_train": sum(r["delta_log_rmse_vs_train"] for r in subset) / len(subset),
        "mean_emis_log_rmse": sum(r["emis_log_rmse"] for r in subset) / len(subset),
        "mean_delta_emis_rmse_vs_train": sum(r["delta_emis_rmse_vs_train"] for r in subset) / len(subset),
        "mean_channel_mean_error": sum(r["mean_abs_channel_mean_error"] for r in subset) / len(subset),
        "mean_delta_channel_mean_error_vs_train": sum(r["delta_channel_mean_error_vs_train"] for r in subset) / len(subset),
        "mean_delta_abs_max": sum(r["delta_abs_max"] for r in subset) / len(subset),
        "mean_delta_abs_mean": sum(r["delta_abs_mean"] for r in subset) / len(subset),
    }

json_path = OUT / "b0_fixed_setting_inverse_summary.json"
json_path.write_text(json.dumps({
    "status": "B0_FIXED_SETTING_INVERSE_SUMMARY_OK",
    "boundary": [
        "Fixed settings across both validation months.",
        "This avoids per-month cherry-picking.",
        "This remains constrained inverse feasibility, not validated physical emission retrieval."
    ],
    "fixed_settings": FIXED_SETTINGS,
    "records": records,
    "aggregate": agg,
}, ensure_ascii=False, indent=2), encoding="utf-8")

csv_path = OUT / "b0_fixed_setting_inverse_summary.csv"
fields = [
    "month", "selection", "method", "max_delta_bound", "lambda_delta",
    "log_r2", "delta_log_r2_vs_train",
    "log_rmse", "delta_log_rmse_vs_train",
    "emis_log_rmse", "delta_emis_rmse_vs_train",
    "mean_abs_channel_mean_error", "delta_channel_mean_error_vs_train",
    "delta_abs_max", "delta_abs_mean",
    "delta_log_r2_vs_true_edgar",
    "delta_values",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({k: r.get(k) for k in fields})

md = []
md.append("# B0 fixed-setting inverse summary")
md.append("")
md.append("## Boundary")
md.append("")
md.append("- Fixed settings are applied to both validation months.")
md.append("- This avoids per-month cherry-picking.")
md.append("- Optimized variables are bounded sector-level emission_log_mean-like offsets.")
md.append("- This is not validated physical emission inventory retrieval.")
md.append("")
md.append("## Per-month results")
md.append("")
md.append("| month | selection | max_delta | lambda | log_R2 | ΔR2 vs train | log_RMSE | ΔRMSE vs train | emis_log_RMSE | Δemis_RMSE vs train | mean_abs_channel_mean_error | Δchannel_mean_error vs train | delta_abs_max | delta_abs_mean |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in records:
    md.append(
        f"| {r['month']} | {r['selection']} | {r['max_delta_bound']} | {r['lambda_delta']} | "
        f"{r['log_r2']:.6f} | {r['delta_log_r2_vs_train']:.6f} | "
        f"{r['log_rmse']:.6f} | {r['delta_log_rmse_vs_train']:.6f} | "
        f"{r['emis_log_rmse']:.6f} | {r['delta_emis_rmse_vs_train']:.6f} | "
        f"{r['mean_abs_channel_mean_error']:.6f} | {r['delta_channel_mean_error_vs_train']:.6f} | "
        f"{r['delta_abs_max']:.6f} | {r['delta_abs_mean']:.6f} |"
    )

md.append("")
md.append("## Aggregate over two validation months")
md.append("")
md.append("| selection | mean_log_R2 | mean_ΔR2_vs_train | mean_log_RMSE | mean_ΔRMSE_vs_train | mean_emis_log_RMSE | mean_Δemis_RMSE_vs_train | mean_channel_mean_error | mean_Δchannel_mean_error_vs_train | mean_delta_abs_max | mean_delta_abs_mean |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for sel, a in agg.items():
    md.append(
        f"| {sel} | "
        f"{a['mean_log_r2']:.6f} | {a['mean_delta_log_r2_vs_train']:.6f} | "
        f"{a['mean_log_rmse']:.6f} | {a['mean_delta_log_rmse_vs_train']:.6f} | "
        f"{a['mean_emis_log_rmse']:.6f} | {a['mean_delta_emis_rmse_vs_train']:.6f} | "
        f"{a['mean_channel_mean_error']:.6f} | {a['mean_delta_channel_mean_error_vs_train']:.6f} | "
        f"{a['mean_delta_abs_max']:.6f} | {a['mean_delta_abs_mean']:.6f} |"
    )

md.append("")
md.append("## Conservative interpretation")
md.append("")
md.append("- Both fixed bounded settings improve concentration reconstruction in both validation months.")
md.append("- The balanced setting gives larger concentration gains but also larger emission-prior deviation.")
md.append("- The ultra-conservative setting is more interpretable but yields smaller gains.")
md.append("- This supports constrained inverse feasibility only.")

md_path = OUT / "b0_fixed_setting_inverse_summary.md"
md_path.write_text("\n".join(md), encoding="utf-8")

print("===== saved =====")
print(json_path)
print(csv_path)
print(md_path)
print()
print(md_path.read_text(encoding="utf-8"))
print()
print("B0_FIXED_SETTING_INVERSE_SUMMARY_OK")
