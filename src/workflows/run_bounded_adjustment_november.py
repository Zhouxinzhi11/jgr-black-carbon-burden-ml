# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
import json
import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(r".")
SCRIPT = ROOT / "src" / "checkpoint_compatibility" / "meanstate_ablation_reference.py"
RUN = ROOT / "runs_s2_meanstate" / "mean_emis_meteo_clim_v1"
CKPT = RUN / "checkpoints" / "best_model.pt"
MANIFEST = RUN / "logs" / "run_manifest.json"

OUT_DIR = ROOT / "runs_directionB_inverse" / "b0_inverse_2022_11_sector_offset_sweep_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2022
TARGET_MONTH = 11
SEED = 42
STEPS = 150
LR = 0.05

MAX_DELTAS = [0.25, 0.5, 1.0, 2.0]
LAMBDA_DELTAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

torch.manual_seed(SEED)

def load_module():
    spec_mod = importlib.util.spec_from_file_location("s2mean_train", SCRIPT)
    m = importlib.util.module_from_spec(spec_mod)
    sys.modules["s2mean_train"] = m
    spec_mod.loader.exec_module(m)
    return m

def masked_metrics(pred, target, mask):
    mask = mask > 0.5
    p = pred[mask]
    t = target[mask]
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - torch.mean(t)) ** 2)
    return {
        "n_valid": int(mask.sum().item()),
        "log_r2": float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)),
        "log_rmse": float(torch.sqrt(torch.mean((p - t) ** 2))),
        "log_bias": float(torch.mean(p - t)),
    }

def corr_1d(a, b, mask):
    m = mask > 0.5
    a = a.float()[m]
    b = b.float()[m]
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt(torch.sum(a*a) * torch.sum(b*b)).clamp_min(1e-12)
    return float(torch.sum(a*b) / den)

def emission_error(candidate, truth, mask):
    m = mask > 0.5
    diff = candidate[:, m] - truth[:, m]
    channel_mean_error = []
    channel_corr = []
    for c in range(truth.shape[0]):
        channel_mean_error.append(float(candidate[c][m].mean() - truth[c][m].mean()))
        channel_corr.append(corr_1d(candidate[c], truth[c], mask))
    return {
        "mean_channel_corr": float(sum(channel_corr) / len(channel_corr)),
        "aggregate_corr": corr_1d(candidate.sum(dim=0), truth.sum(dim=0), mask),
        "emis_log_rmse": float(torch.sqrt(torch.mean(diff ** 2))),
        "emis_log_mae": float(torch.mean(torch.abs(diff))),
        "mean_abs_channel_mean_error": float(sum(abs(x) for x in channel_mean_error) / len(channel_mean_error)),
        "channel_mean_error": channel_mean_error,
    }

def forward_pred(model, emis, meteo, device):
    x = torch.cat([emis, meteo], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).squeeze(0).squeeze(0).detach().cpu()

def evaluate(model, emis, meteo, target, mask, true_emis, device):
    pred = forward_pred(model, emis, meteo, device)
    cm = masked_metrics(pred, target, mask)
    em = emission_error(emis.detach().cpu(), true_emis, mask)
    return {**cm, **em}

print("===== B0-3 bounded sector-offset sweep =====")
print("OUT_DIR:", OUT_DIR)

m = load_module()
ckpt = torch.load(CKPT, map_location="cpu")
spec = m.MeanStateSpec(**ckpt["spec"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = m.MeanStateUNet(spec, base_features=int(ckpt.get("base_features", 32)))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

target_entry = None
for e in manifest["val_entries"]:
    if int(e["year"]) == TARGET_YEAR and int(e["month"]) == TARGET_MONTH:
        target_entry = e
        break
if target_entry is None:
    raise RuntimeError("target val entry not found")

payload = m.load_pt(target_entry["file"])
true_emis = payload["emission_log_mean"].float()
meteo = payload["meteo_mean"].float()
target = payload["target_log_mean_conc"].float()
mask = payload["valid_mask"].float()

train_emis = []
for e in manifest["train_entries"]:
    p = m.load_pt(e["file"])
    train_emis.append(p["emission_log_mean"].float())
train_mean = torch.stack(train_emis, dim=0).mean(dim=0)

# Baselines
rows = []

for name, emis in [
    ("true_edgar_reference", true_emis),
    ("train_mean_prior", train_mean),
]:
    r = evaluate(model, emis, meteo, target, mask, true_emis, device)
    r.update({
        "method": name,
        "max_delta_bound": None,
        "lambda_delta": None,
        "delta_abs_max": 0.0,
        "delta_abs_mean": 0.0,
        "delta_values": [],
    })
    rows.append(r)
    print(name, r)

# Sweep
train_mean_d = train_mean.unsqueeze(0).to(device)
meteo_d = meteo.unsqueeze(0).to(device)
target_d = target.to(device)
mask_d = (mask > 0.5).to(device)

for max_delta in MAX_DELTAS:
    for lambda_delta in LAMBDA_DELTAS:
        torch.manual_seed(SEED)
        theta = torch.nn.Parameter(torch.zeros(1, 8, 1, 1, device=device))
        opt = torch.optim.Adam([theta], lr=LR)

        for step in range(1, STEPS + 1):
            opt.zero_grad()
            delta = max_delta * torch.tanh(theta)
            emis_opt = train_mean_d + delta
            x = torch.cat([emis_opt, meteo_d], dim=1)
            pred = model(x).squeeze(0).squeeze(0)

            recon_loss = torch.mean((pred[mask_d] - target_d[mask_d]) ** 2)
            delta_loss = torch.mean(delta ** 2)
            loss = recon_loss + lambda_delta * delta_loss
            loss.backward()
            opt.step()

        with torch.no_grad():
            delta = (max_delta * torch.tanh(theta)).detach().cpu().view(8, 1, 1)
            emis_final = train_mean + delta
            r = evaluate(model, emis_final, meteo, target, mask, true_emis, device)
            dvals = [float(x) for x in delta.view(-1)]
            r.update({
                "method": "bounded_sector_offset",
                "max_delta_bound": max_delta,
                "lambda_delta": lambda_delta,
                "delta_abs_max": float(torch.max(torch.abs(delta))),
                "delta_abs_mean": float(torch.mean(torch.abs(delta))),
                "delta_values": dvals,
            })
            rows.append(r)
            print("max_delta", max_delta, "lambda", lambda_delta, r)

# Save JSON
report = {
    "status": "B0_BOUNDED_SECTOR_OFFSET_SWEEP_OK",
    "boundary": [
        "One-month bounded sector-offset sweep.",
        "This tests whether concentration can improve under constrained sector-level emission-like offsets.",
        "This is not physical emission inventory retrieval."
    ],
    "target": {
        "year": TARGET_YEAR,
        "month": TARGET_MONTH,
        "file": target_entry["file"],
    },
    "settings": {
        "steps": STEPS,
        "lr": LR,
        "max_deltas": MAX_DELTAS,
        "lambda_deltas": LAMBDA_DELTAS,
        "seed": SEED,
        "device": str(device),
    },
    "rows": rows,
}
json_path = OUT_DIR / "b0_inverse_2022_11_sector_offset_sweep_report.json"
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

# Save CSV
csv_path = OUT_DIR / "b0_inverse_2022_11_sector_offset_sweep.csv"
fieldnames = [
    "method", "max_delta_bound", "lambda_delta",
    "log_r2", "log_rmse", "log_bias",
    "emis_log_rmse", "emis_log_mae", "mean_abs_channel_mean_error",
    "mean_channel_corr", "aggregate_corr",
    "delta_abs_max", "delta_abs_mean", "delta_values"
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})

# Markdown summary sorted by log_r2 descending
summary_rows = sorted(rows, key=lambda x: x["log_r2"], reverse=True)

md = []
md.append("# B0-3 bounded sector-offset sweep")
md.append("")
md.append(f"Target: {TARGET_YEAR}-{TARGET_MONTH:02d}")
md.append("")
md.append("## Boundary")
md.append("")
md.append("- One-month bounded sector-offset sweep.")
md.append("- Offset is constrained by `delta = max_delta * tanh(theta)`.")
md.append("- This is not physical emission inventory retrieval.")
md.append("")
md.append("## Results sorted by concentration log_R2")
md.append("")
md.append("| method | max_delta | lambda | log_R2 | log_RMSE | emis_log_RMSE | mean_abs_channel_mean_error | delta_abs_max | delta_abs_mean |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in summary_rows:
    md.append(
        f"| {r['method']} | {r['max_delta_bound']} | {r['lambda_delta']} | "
        f"{r['log_r2']:.6f} | {r['log_rmse']:.6f} | "
        f"{r['emis_log_rmse']:.6f} | {r['mean_abs_channel_mean_error']:.6f} | "
        f"{r['delta_abs_max']:.6f} | {r['delta_abs_mean']:.6f} |"
    )

md_path = OUT_DIR / "b0_inverse_2022_11_sector_offset_sweep_summary.md"
md_path.write_text("\n".join(md), encoding="utf-8")

print("\n===== saved =====")
print(json_path)
print(csv_path)
print(md_path)
print("\n" + md_path.read_text(encoding="utf-8"))
print("\nB0_BOUNDED_SECTOR_OFFSET_SWEEP_OK")
