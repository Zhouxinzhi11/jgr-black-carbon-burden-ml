#!/usr/bin/env python3
"""Run the manuscript bounded sector-offset experiment with explicit paths.

This is a release-safe entry point for the November/December historical
workflows.  It preserves their optimization, metrics, seed, and default sweep,
but performs no work at import time and contains no local absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return values


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("jgr_meanstate_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def masked_metrics(pred, target, mask):
    selected = mask > 0.5
    p = pred[selected]
    t = target[selected]
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - torch.mean(t)) ** 2)
    return {
        "n_valid": int(selected.sum().item()),
        "log_r2": float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)),
        "log_rmse": float(torch.sqrt(torch.mean((p - t) ** 2))),
        "log_bias": float(torch.mean(p - t)),
    }


def corr_1d(a, b, mask):
    selected = mask > 0.5
    a = a.float()[selected]
    b = b.float()[selected]
    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)).clamp_min(1e-12)
    return float(torch.sum(a * b) / denominator)


def emission_error(candidate, truth, mask):
    selected = mask > 0.5
    difference = candidate[:, selected] - truth[:, selected]
    channel_mean_error = []
    channel_corr = []
    for channel in range(truth.shape[0]):
        channel_mean_error.append(
            float(candidate[channel][selected].mean() - truth[channel][selected].mean())
        )
        channel_corr.append(corr_1d(candidate[channel], truth[channel], mask))
    return {
        "mean_channel_corr": float(sum(channel_corr) / len(channel_corr)),
        "aggregate_corr": corr_1d(candidate.sum(dim=0), truth.sum(dim=0), mask),
        "emis_log_rmse": float(torch.sqrt(torch.mean(difference ** 2))),
        "emis_log_mae": float(torch.mean(torch.abs(difference))),
        "mean_abs_channel_mean_error": float(
            sum(abs(value) for value in channel_mean_error) / len(channel_mean_error)
        ),
        "channel_mean_error": channel_mean_error,
    }


def forward_prediction(model, emissions, meteorology, device):
    inputs = torch.cat([emissions, meteorology], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(inputs).squeeze(0).squeeze(0).detach().cpu()


def evaluate(model, emissions, meteorology, target, mask, true_emissions, device):
    prediction = forward_prediction(model, emissions, meteorology, device)
    return {
        **masked_metrics(prediction, target, mask),
        **emission_error(emissions.detach().cpu(), true_emissions, mask),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-module", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, choices=(11, 12), required=True)
    parser.add_argument("--max-deltas", type=parse_floats, default=parse_floats("0.25,0.5,1,2"))
    parser.add_argument(
        "--lambda-deltas",
        type=parse_floats,
        default=parse_floats("0,0.0001,0.001,0.01,0.1"),
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    model_module = args.model_module.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    manifest_path = run_dir / "logs" / "run_manifest.json"
    for required in (model_module, checkpoint_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required authorized-input artifact is missing: {required}. "
                "Full execution requires authorized restricted fused inputs and run artifacts."
            )

    torch.manual_seed(args.seed)
    module = load_module(model_module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_spec = module.MeanStateSpec(**checkpoint["spec"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is not available")

    model = module.MeanStateUNet(
        model_spec, base_features=int(checkpoint.get("base_features", 32))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_entry = next(
        (
            entry
            for entry in manifest["val_entries"]
            if int(entry["year"]) == args.year and int(entry["month"]) == args.month
        ),
        None,
    )
    if target_entry is None:
        raise RuntimeError(f"No validation entry for {args.year}-{args.month:02d}")

    payload = module.load_pt(target_entry["file"])
    true_emissions = payload["emission_log_mean"].float()
    meteorology = payload["meteo_mean"].float()
    target = payload["target_log_mean_conc"].float()
    mask = payload["valid_mask"].float()
    training_emissions = [
        module.load_pt(entry["file"])["emission_log_mean"].float()
        for entry in manifest["train_entries"]
    ]
    training_mean = torch.stack(training_emissions, dim=0).mean(dim=0)

    rows = []
    for name, emissions in (
        ("true_edgar_reference", true_emissions),
        ("train_mean_prior", training_mean),
    ):
        result = evaluate(
            model, emissions, meteorology, target, mask, true_emissions, device
        )
        result.update(
            {
                "method": name,
                "max_delta_bound": None,
                "lambda_delta": None,
                "delta_abs_max": 0.0,
                "delta_abs_mean": 0.0,
                "delta_values": [],
            }
        )
        rows.append(result)

    training_mean_device = training_mean.unsqueeze(0).to(device)
    meteorology_device = meteorology.unsqueeze(0).to(device)
    target_device = target.to(device)
    mask_device = (mask > 0.5).to(device)
    for max_delta in args.max_deltas:
        for lambda_delta in args.lambda_deltas:
            torch.manual_seed(args.seed)
            theta = torch.nn.Parameter(torch.zeros(1, 8, 1, 1, device=device))
            optimizer = torch.optim.Adam([theta], lr=args.learning_rate)
            for _ in range(args.steps):
                optimizer.zero_grad()
                delta = max_delta * torch.tanh(theta)
                optimized_emissions = training_mean_device + delta
                inputs = torch.cat([optimized_emissions, meteorology_device], dim=1)
                prediction = model(inputs).squeeze(0).squeeze(0)
                reconstruction_loss = torch.mean(
                    (prediction[mask_device] - target_device[mask_device]) ** 2
                )
                delta_loss = torch.mean(delta ** 2)
                loss = reconstruction_loss + lambda_delta * delta_loss
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                delta = (max_delta * torch.tanh(theta)).detach().cpu().view(8, 1, 1)
                final_emissions = training_mean + delta
                result = evaluate(
                    model,
                    final_emissions,
                    meteorology,
                    target,
                    mask,
                    true_emissions,
                    device,
                )
                result.update(
                    {
                        "method": "bounded_sector_offset",
                        "max_delta_bound": max_delta,
                        "lambda_delta": lambda_delta,
                        "delta_abs_max": float(torch.max(torch.abs(delta))),
                        "delta_abs_mean": float(torch.mean(torch.abs(delta))),
                        "delta_values": [float(value) for value in delta.view(-1)],
                    }
                )
                rows.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bounded_adjustment_{args.year}_{args.month:02d}"
    report = {
        "status": "BOUNDED_SECTOR_OFFSET_SWEEP_COMPLETE",
        "boundary": [
            "One-month bounded sector-offset sweep.",
            "This tests constrained emission-like offsets.",
            "This is not physical emission inventory retrieval.",
        ],
        "target": {"year": args.year, "month": args.month},
        "settings": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "max_deltas": args.max_deltas,
            "lambda_deltas": args.lambda_deltas,
            "seed": args.seed,
            "device": str(device),
        },
        "rows": rows,
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "method", "max_delta_bound", "lambda_delta", "log_r2", "log_rmse",
        "log_bias", "emis_log_rmse", "emis_log_mae",
        "mean_abs_channel_mean_error", "mean_channel_corr", "aggregate_corr",
        "delta_abs_max", "delta_abs_mean", "delta_values",
    ]
    with (output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    print(f"WROTE={output_dir / stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
