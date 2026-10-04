# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
import json
import csv
from pathlib import Path

ROOT = Path(r".")
SRC_DIR = ROOT / "runs_directionB_inverse" / "b0_fixed_setting_inverse_summary_v1"
SRC_JSON = SRC_DIR / "b0_fixed_setting_inverse_summary.json"

OUT_DIR = ROOT / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_CSV = OUT_DIR / "table3_inverse_fixed_setting_main_v1.csv"
MAIN_MD = OUT_DIR / "table3_inverse_fixed_setting_main_v1.md"
PM_CSV = OUT_DIR / "table3_inverse_fixed_setting_per_month_v1.csv"
PM_MD = OUT_DIR / "table3_inverse_fixed_setting_per_month_v1.md"

if not SRC_JSON.exists():
    raise FileNotFoundError(SRC_JSON)

obj = json.loads(SRC_JSON.read_text(encoding="utf-8"))

records = obj["records"]
agg = obj["aggregate"]

ORDER = [
    "train_mean_prior",
    "true_edgar_reference",
    "ultra_conservative_delta0.25_lambda0.1",
    "balanced_conservative_delta0.5_lambda0.1",
]

DISPLAY = {
    "train_mean_prior": "Train-mean emission prior",
    "true_edgar_reference": "True EDGAR reference",
    "ultra_conservative_delta0.25_lambda0.1": "Ultra-conservative bounded inverse",
    "balanced_conservative_delta0.5_lambda0.1": "Balanced bounded inverse",
}

ROLE = {
    "train_mean_prior": "Baseline prior",
    "true_edgar_reference": "Reference, not an inverse method",
    "ultra_conservative_delta0.25_lambda0.1": "Fixed bounded sector-offset inverse; smaller deviation",
    "balanced_conservative_delta0.5_lambda0.1": "Fixed bounded sector-offset inverse; larger gain with larger deviation",
}

def fmt(x, n=6):
    if x is None:
        return ""
    return f"{float(x):.{n}f}"

def setting_params(selection):
    if selection == "ultra_conservative_delta0.25_lambda0.1":
        return 0.25, 0.1
    if selection == "balanced_conservative_delta0.5_lambda0.1":
        return 0.5, 0.1
    return None, None

# -----------------------------
# Main aggregate table
# -----------------------------
main_rows = []

for sel in ORDER:
    if sel not in agg:
        raise KeyError(f"Missing aggregate selection: {sel}")

    a = agg[sel]
    max_delta, lam = setting_params(sel)

    main_rows.append({
        "selection": sel,
        "display_name": DISPLAY[sel],
        "role": ROLE[sel],
        "max_delta": max_delta,
        "lambda_delta": lam,

        "mean_log_r2": a["mean_log_r2"],
        "mean_delta_log_r2_vs_train": a["mean_delta_log_r2_vs_train"],
        "mean_log_rmse": a["mean_log_rmse"],
        "mean_delta_log_rmse_vs_train": a["mean_delta_log_rmse_vs_train"],

        "mean_emis_log_rmse": a["mean_emis_log_rmse"],
        "mean_delta_emis_rmse_vs_train": a["mean_delta_emis_rmse_vs_train"],
        "mean_channel_mean_error": a["mean_channel_mean_error"],
        "mean_delta_channel_mean_error_vs_train": a["mean_delta_channel_mean_error_vs_train"],

        "mean_delta_abs_max": a["mean_delta_abs_max"],
        "mean_delta_abs_mean": a["mean_delta_abs_mean"],
    })

main_fields = [
    "selection", "display_name", "role", "max_delta", "lambda_delta",
    "mean_log_r2", "mean_delta_log_r2_vs_train",
    "mean_log_rmse", "mean_delta_log_rmse_vs_train",
    "mean_emis_log_rmse", "mean_delta_emis_rmse_vs_train",
    "mean_channel_mean_error", "mean_delta_channel_mean_error_vs_train",
    "mean_delta_abs_max", "mean_delta_abs_mean",
]

with MAIN_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=main_fields)
    w.writeheader()
    for r in main_rows:
        w.writerow({k: r.get(k) for k in main_fields})

md = []
md.append("# Table 3. Fixed-setting constrained inverse summary")
md.append("")
md.append("## Boundary")
md.append("")
md.append("- Results are averaged over the two held-out validation months: 2022-11 and 2022-12.")
md.append("- The bounded inverse settings are fixed across months to avoid per-month cherry-picking.")
md.append("- Optimized variables are bounded sector-level emission_log_mean-like offsets.")
md.append("- This is constrained inverse feasibility, not validated physical emission inventory retrieval.")
md.append("- The true EDGAR row is a reference, not an inverse method.")
md.append("")
md.append("## Main aggregate table")
md.append("")
md.append("| Setting | Role | max_delta | lambda | mean log R² | ΔR² vs train | mean log RMSE | ΔRMSE vs train | mean emission log RMSE | Δemission RMSE vs train | mean channel-mean error | Δchannel error vs train | mean |δ|max | mean |δ|mean |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in main_rows:
    md.append(
        f"| {r['display_name']} | {r['role']} | "
        f"{'' if r['max_delta'] is None else r['max_delta']} | {'' if r['lambda_delta'] is None else r['lambda_delta']} | "
        f"{fmt(r['mean_log_r2'])} | {fmt(r['mean_delta_log_r2_vs_train'])} | "
        f"{fmt(r['mean_log_rmse'])} | {fmt(r['mean_delta_log_rmse_vs_train'])} | "
        f"{fmt(r['mean_emis_log_rmse'])} | {fmt(r['mean_delta_emis_rmse_vs_train'])} | "
        f"{fmt(r['mean_channel_mean_error'])} | {fmt(r['mean_delta_channel_mean_error_vs_train'])} | "
        f"{fmt(r['mean_delta_abs_max'])} | {fmt(r['mean_delta_abs_mean'])} |"
    )

md.append("")
md.append("## Strict interpretation")
md.append("")
md.append("- Both fixed bounded inverse settings improve concentration reconstruction relative to the train-mean prior.")
md.append("- The balanced setting nearly matches the true-EDGAR reference in mean log R², but it also introduces larger emission-prior deviation.")
md.append("- The ultra-conservative setting is more interpretable, with smaller deviation and smaller reconstruction gain.")
md.append("- These results support weak but consistent top-down constraints on low-dimensional emission-prior adjustment.")
md.append("- They must not be described as successful physical emission inventory retrieval.")

MAIN_MD.write_text("\n".join(md), encoding="utf-8")

# -----------------------------
# Per-month table
# -----------------------------
pm_rows = []
for r in records:
    if r["selection"] not in ORDER:
        continue

    max_delta, lam = setting_params(r["selection"])
    pm_rows.append({
        "month": r["month"],
        "selection": r["selection"],
        "display_name": DISPLAY[r["selection"]],
        "role": ROLE[r["selection"]],
        "max_delta": max_delta,
        "lambda_delta": lam,

        "log_r2": r["log_r2"],
        "delta_log_r2_vs_train": r["delta_log_r2_vs_train"],
        "log_rmse": r["log_rmse"],
        "delta_log_rmse_vs_train": r["delta_log_rmse_vs_train"],

        "emis_log_rmse": r["emis_log_rmse"],
        "delta_emis_rmse_vs_train": r["delta_emis_rmse_vs_train"],
        "mean_abs_channel_mean_error": r["mean_abs_channel_mean_error"],
        "delta_channel_mean_error_vs_train": r["delta_channel_mean_error_vs_train"],

        "delta_abs_max": r["delta_abs_max"],
        "delta_abs_mean": r["delta_abs_mean"],
    })

pm_rows = sorted(pm_rows, key=lambda x: (x["month"], ORDER.index(x["selection"])))

pm_fields = [
    "month", "selection", "display_name", "role", "max_delta", "lambda_delta",
    "log_r2", "delta_log_r2_vs_train",
    "log_rmse", "delta_log_rmse_vs_train",
    "emis_log_rmse", "delta_emis_rmse_vs_train",
    "mean_abs_channel_mean_error", "delta_channel_mean_error_vs_train",
    "delta_abs_max", "delta_abs_mean",
]

with PM_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=pm_fields)
    w.writeheader()
    for r in pm_rows:
        w.writerow({k: r.get(k) for k in pm_fields})

pmd = []
pmd.append("# Supplementary Table. Per-month fixed-setting constrained inverse results")
pmd.append("")
pmd.append("## Boundary")
pmd.append("")
pmd.append("- Per-month results are shown for transparency.")
pmd.append("- Main text should emphasize the fixed-setting aggregate table, not per-month cherry-picked optima.")
pmd.append("")
pmd.append("| Month | Setting | max_delta | lambda | log R² | ΔR² vs train | log RMSE | ΔRMSE vs train | emission log RMSE | Δemission RMSE vs train | channel-mean error | Δchannel error vs train | |δ|max | |δ|mean |")
pmd.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in pm_rows:
    pmd.append(
        f"| {r['month']} | {r['display_name']} | "
        f"{'' if r['max_delta'] is None else r['max_delta']} | {'' if r['lambda_delta'] is None else r['lambda_delta']} | "
        f"{fmt(r['log_r2'])} | {fmt(r['delta_log_r2_vs_train'])} | "
        f"{fmt(r['log_rmse'])} | {fmt(r['delta_log_rmse_vs_train'])} | "
        f"{fmt(r['emis_log_rmse'])} | {fmt(r['delta_emis_rmse_vs_train'])} | "
        f"{fmt(r['mean_abs_channel_mean_error'])} | {fmt(r['delta_channel_mean_error_vs_train'])} | "
        f"{fmt(r['delta_abs_max'])} | {fmt(r['delta_abs_mean'])} |"
    )

PM_MD.write_text("\n".join(pmd), encoding="utf-8")

print("===== saved =====")
print(MAIN_CSV)
print(MAIN_MD)
print(PM_CSV)
print(PM_MD)

print("\n===== Table 3 main =====")
print(MAIN_MD.read_text(encoding="utf-8"))

print("\n===== Table 3 per-month =====")
print(PM_MD.read_text(encoding="utf-8"))

print("\nTABLE3_FIXED_SETTING_INVERSE_SUMMARY_OK")
