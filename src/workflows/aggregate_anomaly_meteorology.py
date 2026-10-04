# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
71_seal_anomaly_meteo_three_seed.py

One-command closure for the clean meteo-only anomaly baseline.

The script:
1. verifies the existing seed42/43 audit and evaluation artifacts;
2. creates the missing seed44 formal audit and external evaluation by invoking
   the already frozen audit/evaluation scripts;
3. verifies all three run manifests, checkpoints, cache provenance, contracts,
   evaluation reports and hashes;
4. aggregates seed42/43/44 global and grouped diagnostics;
5. seals the meteo-only anomaly baseline without starting any fire, AEF, Atlas
   or new-model experiment.

No training is performed by this script.
No existing artifact is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SEEDS = [42, 43, 44]
EXPERIMENT = "anom_meteo_only"
EXPECTED_TRAIN_FILES = 178
EXPECTED_VAL_FILES = 61
EXPECTED_EXTERNAL_FILES = 361
EXPECTED_PARAMETERS = 477025

EXPECTED_CONTRACT = {
    "batch_size": 4,
    "num_workers": 0,
    "patch_size": 96,
    "patches_per_file": 4,
    "base_features": 32,
    "max_epochs": 140,
    "patience": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_every": 5,
    "train_limit": 0,
    "val_limit": 0,
}

GLOBAL_METRIC_PATHS = {
    "anom_log_r2": ("model_anomaly", "r2"),
    "anom_log_rmse": ("model_anomaly", "rmse"),
    "anom_log_bias": ("model_anomaly", "bias"),
    "anom_log_corr": ("model_anomaly", "corr"),
    "zero_anom_log_rmse": ("zero_anomaly_baseline", "rmse"),
    "anom_mse_skill_vs_zero": ("anomaly_mse_skill_vs_zero",),
    "recon_log_r2": ("model_reconstruction", "r2"),
    "month_background_log_r2": ("month_background_baseline", "r2"),
    "recon_r2_delta_vs_background": (
        "reconstruction_r2_delta_vs_month_background",
    ),
}

GROUPED_METRIC_PATHS = {
    "anom_log_r2": ("model_anomaly", "r2"),
    "anom_log_rmse": ("model_anomaly", "rmse"),
    "anom_log_bias": ("model_anomaly", "bias"),
    "anom_log_corr": ("model_anomaly", "corr"),
    "anom_mse_skill_vs_zero": ("anomaly_mse_skill_vs_zero",),
    "recon_r2_delta_vs_background": (
        "reconstruction_r2_delta_vs_month_background",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)


def finite_float(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def nested_get(obj: dict[str, Any], path: tuple[str, ...], label: str) -> Any:
    value: Any = obj
    for key in path:
        require(isinstance(value, dict) and key in value,
                f"MISSING_{label}_{'_'.join(path)}")
        value = value[key]
    return value


def summarize(values: list[float], label: str) -> dict[str, Any]:
    require(len(values) == len(SEEDS),
            f"SUMMARY_SEED_COUNT_MISMATCH_{label}")
    for index, value in enumerate(values):
        finite_float(value, f"{label}_SEED{SEEDS[index]}")
    return {
        "seeds": SEEDS,
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "all_positive": all(value > 0.0 for value in values),
        "all_nonnegative": all(value >= 0.0 for value in values),
    }


def invoke(command: list[str], label: str) -> None:
    print("=" * 96, flush=True)
    print(f"INVOKE_{label}=True", flush=True)
    print("COMMAND=" + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, check=False)
    require(
        completed.returncode == 0,
        f"{label}_FAILED_EXIT_CODE={completed.returncode}",
    )


def verify_contract(actual: dict[str, Any], seed: int) -> None:
    require(set(actual) == set(EXPECTED_CONTRACT),
            f"FORMAL_CONTRACT_KEY_SET_MISMATCH_SEED{seed}")
    for key, expected in EXPECTED_CONTRACT.items():
        value = actual[key]
        if isinstance(expected, float):
            require(
                math.isclose(
                    float(value),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"FORMAL_CONTRACT_MISMATCH_SEED{seed}_{key}",
            )
        else:
            require(
                value == expected,
                f"FORMAL_CONTRACT_MISMATCH_SEED{seed}_{key}",
            )


def aggregate_grouped(
    reports: dict[int, dict[str, Any]],
    split_key: str,
    group_key: str,
) -> dict[str, Any]:
    group_sets = [
        set(
            reports[seed]["results"][split_key][group_key]
        )
        for seed in SEEDS
    ]
    require(
        all(group_set == group_sets[0] for group_set in group_sets[1:]),
        f"GROUP_KEY_SET_MISMATCH_{split_key}_{group_key}",
    )

    result: dict[str, Any] = {}
    for group_name in sorted(group_sets[0]):
        result[group_name] = {}
        for metric_name, metric_path in GROUPED_METRIC_PATHS.items():
            values = [
                finite_float(
                    nested_get(
                        reports[seed]["results"][split_key][group_key][group_name],
                        metric_path,
                        f"{split_key}_{group_key}_{group_name}_{metric_name}",
                    ),
                    f"{split_key}_{group_key}_{group_name}_{metric_name}_SEED{seed}",
                )
                for seed in SEEDS
            ]
            result[group_name][metric_name] = summarize(
                values,
                f"{split_key}_{group_key}_{group_name}_{metric_name}",
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seal clean meteo-only anomaly baseline across three seeds."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--sanity-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    require(
        math.isclose(args.metric_tolerance, 1e-6, rel_tol=0.0, abs_tol=0.0),
        "METRIC_TOLERANCE_LOCKED_TO_1E-6",
    )

    run_root = Path(args.run_root).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    audit_script = Path(args.audit_script).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()
    cache_audit_path = Path(args.cache_audit).expanduser().resolve()
    sanity_root = Path(args.sanity_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_root.is_dir(), f"RUN_ROOT_NOT_FOUND: {run_root}")
    require(training_script.is_file(),
            f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(audit_script.is_file(),
            f"AUDIT_SCRIPT_NOT_FOUND: {audit_script}")
    require(eval_script.is_file(),
            f"EVAL_SCRIPT_NOT_FOUND: {eval_script}")
    require(cache_audit_path.is_file(),
            f"CACHE_AUDIT_NOT_FOUND: {cache_audit_path}")
    require(sanity_root.is_dir(), f"SANITY_ROOT_NOT_FOUND: {sanity_root}")
    require(not output_dir.exists(),
            f"STOP_OUTPUT_DIR_EXISTS: {output_dir}")

    run_dirs: dict[int, Path] = {}
    audit_paths: dict[int, Path] = {}
    eval_paths: dict[int, Path] = {}

    for seed in SEEDS:
        run_dir = run_root / f"seed{seed}" / f"{EXPERIMENT}_seed{seed}"
        audit_dir = sanity_root / f"anomaly_clean_formal_seed{seed}_audit_v01"
        audit_path = (
            audit_dir /
            f"anomaly_clean_formal_seed{seed}_audit_v01.json"
        )
        eval_dir = run_dir / "eval_v01"
        eval_path = eval_dir / "eval_report.json"

        require(run_dir.is_dir(), f"RUN_DIR_NOT_FOUND_SEED{seed}: {run_dir}")

        if not audit_path.is_file():
            require(
                seed == 44,
                f"EXPECTED_EXISTING_AUDIT_MISSING_SEED{seed}: {audit_path}",
            )
            require(
                not audit_dir.exists(),
                f"PARTIAL_AUDIT_DIR_EXISTS_SEED{seed}: {audit_dir}",
            )
            command = [
                sys.executable,
                "-u",
                str(audit_script),
                "--run-dir",
                str(run_dir),
                "--expected-seed",
                str(seed),
                "--training-script",
                str(training_script),
                "--cache-audit",
                str(cache_audit_path),
                "--output-dir",
                str(audit_dir),
                "--metric-tolerance",
                f"{args.metric_tolerance:.12g}",
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            invoke(command, f"SEED{seed}_FORMAL_AUDIT")

        require(audit_path.is_file(),
                f"AUDIT_REPORT_NOT_FOUND_SEED{seed}: {audit_path}")

        if not eval_path.is_file():
            require(
                seed == 44,
                f"EXPECTED_EXISTING_EVAL_MISSING_SEED{seed}: {eval_path}",
            )
            require(
                not eval_dir.exists(),
                f"PARTIAL_EVAL_DIR_EXISTS_SEED{seed}: {eval_dir}",
            )
            command = [
                sys.executable,
                "-u",
                str(eval_script),
                "--run-dir",
                str(run_dir),
                "--expected-seed",
                str(seed),
                "--training-script",
                str(training_script),
                "--formal-audit",
                str(audit_path),
                "--output-dir",
                str(eval_dir),
                "--metric-tolerance",
                f"{args.metric_tolerance:.12g}",
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            invoke(command, f"SEED{seed}_EXTERNAL_EVALUATION")

        require(eval_path.is_file(),
                f"EVAL_REPORT_NOT_FOUND_SEED{seed}: {eval_path}")

        run_dirs[seed] = run_dir
        audit_paths[seed] = audit_path
        eval_paths[seed] = eval_path

    manifests: dict[int, dict[str, Any]] = {}
    audits: dict[int, dict[str, Any]] = {}
    eval_reports: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}

    shared_cache_dir: str | None = None
    shared_cache_manifest_hashes: dict[str, str] | None = None
    shared_contract: dict[str, Any] | None = None

    for seed in SEEDS:
        run_dir = run_dirs[seed]
        manifest_path = run_dir / "logs" / "run_manifest.json"
        history_path = run_dir / "logs" / "training_history.json"
        checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

        manifest = load_json(manifest_path)
        audit = load_json(audit_paths[seed])
        evaluation = load_json(eval_paths[seed])

        require(manifest.get("artifact_name") == "clean_anomaly_training_run",
                f"MANIFEST_ARTIFACT_MISMATCH_SEED{seed}")
        require(manifest.get("experiment") == EXPERIMENT,
                f"MANIFEST_EXPERIMENT_MISMATCH_SEED{seed}")
        require(manifest.get("run_mode") == "formal",
                f"MANIFEST_RUN_MODE_MISMATCH_SEED{seed}")
        require(int(manifest.get("seed", -1)) == seed,
                f"MANIFEST_SEED_MISMATCH_SEED{seed}")
        require(int(manifest.get("trainable_parameters", -1)) ==
                EXPECTED_PARAMETERS,
                f"PARAMETER_COUNT_MISMATCH_SEED{seed}")
        verify_contract(manifest.get("contract", {}), seed)

        effective = manifest.get("effective_counts", {})
        require(int(effective.get("train_files", -1)) ==
                EXPECTED_TRAIN_FILES,
                f"TRAIN_COUNT_MISMATCH_SEED{seed}")
        require(int(effective.get("val_files", -1)) ==
                EXPECTED_VAL_FILES,
                f"VAL_COUNT_MISMATCH_SEED{seed}")
        require(int(effective.get("external_manifest_files", -1)) ==
                EXPECTED_EXTERNAL_FILES,
                f"EXTERNAL_COUNT_MISMATCH_SEED{seed}")

        require(audit.get("artifact_name") ==
                "clean_anomaly_formal_run_audit",
                f"AUDIT_ARTIFACT_MISMATCH_SEED{seed}")
        require(int(audit.get("seed", -1)) == seed,
                f"AUDIT_SEED_MISMATCH_SEED{seed}")
        require(audit.get("verification", {}).get("overall_pass") is True,
                f"AUDIT_NOT_PASS_SEED{seed}")
        require(
            audit.get("decision") ==
            f"PASS_PROCEED_TO_SEED{seed}_EXTERNAL_EVALUATION",
            f"AUDIT_DECISION_MISMATCH_SEED{seed}",
        )

        require(
            evaluation.get("artifact_name") in {
                "clean_anomaly_seed42_evaluation",
                "clean_anomaly_formal_run_evaluation",
            },
            f"EVAL_ARTIFACT_MISMATCH_SEED{seed}",
        )
        require(evaluation.get("experiment") == EXPERIMENT,
                f"EVAL_EXPERIMENT_MISMATCH_SEED{seed}")
        require(int(evaluation.get("seed", -1)) == seed,
                f"EVAL_SEED_MISMATCH_SEED{seed}")
        require(evaluation.get("verification", {}).get("overall_pass") is True,
                f"EVAL_NOT_PASS_SEED{seed}")
        require(
            evaluation.get("verification", {}).get(
                "validation_metric_matches_formal_audit"
            ) is True,
            f"EVAL_VALIDATION_REPRODUCTION_FAIL_SEED{seed}",
        )
        require(
            evaluation.get("verification", {}).get(
                "zero_anomaly_baseline_reported"
            ) is True,
            f"ZERO_BASELINE_MISSING_SEED{seed}",
        )
        require(
            evaluation.get("verification", {}).get(
                "month_background_reconstruction_baseline_reported"
            ) is True,
            f"BACKGROUND_BASELINE_MISSING_SEED{seed}",
        )

        eval_provenance = evaluation.get("input_provenance", {})
        require(
            eval_provenance.get("formal_audit", {}).get("sha256") ==
            sha256_file(audit_paths[seed]),
            f"EVAL_AUDIT_HASH_MISMATCH_SEED{seed}",
        )
        require(
            evaluation.get("best_checkpoint", {}).get("sha256") ==
            sha256_file(checkpoint_path),
            f"EVAL_CHECKPOINT_HASH_MISMATCH_SEED{seed}",
        )

        cache_dir = canonical(manifest.get("cache_dir", ""))
        cache_hashes = manifest.get("manifest_sha256", {})
        contract = manifest.get("contract", {})
        require(isinstance(cache_hashes, dict) and cache_hashes,
                f"CACHE_MANIFEST_HASHES_EMPTY_SEED{seed}")

        if shared_cache_dir is None:
            shared_cache_dir = cache_dir
            shared_cache_manifest_hashes = cache_hashes
            shared_contract = contract
        else:
            require(cache_dir == shared_cache_dir,
                    f"CACHE_DIR_DIFFERS_SEED{seed}")
            require(cache_hashes == shared_cache_manifest_hashes,
                    f"CACHE_HASHES_DIFFER_SEED{seed}")
            require(contract == shared_contract,
                    f"CONTRACT_DIFFERS_SEED{seed}")

        manifests[seed] = manifest
        audits[seed] = audit
        eval_reports[seed] = evaluation
        provenance[str(seed)] = {
            "run_dir": str(run_dir),
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "training_history": {
                "path": str(history_path),
                "sha256": sha256_file(history_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "formal_audit": {
                "path": str(audit_paths[seed]),
                "sha256": sha256_file(audit_paths[seed]),
            },
            "evaluation": {
                "path": str(eval_paths[seed]),
                "sha256": sha256_file(eval_paths[seed]),
            },
        }

    global_aggregate: dict[str, Any] = {}
    for split_key in ("2022_val", "2023_external"):
        global_aggregate[split_key] = {}
        for metric_name, metric_path in GLOBAL_METRIC_PATHS.items():
            values = [
                finite_float(
                    nested_get(
                        eval_reports[seed]["results"][split_key]["global"],
                        metric_path,
                        f"{split_key}_{metric_name}",
                    ),
                    f"{split_key}_{metric_name}_SEED{seed}",
                )
                for seed in SEEDS
            ]
            global_aggregate[split_key][metric_name] = summarize(
                values,
                f"{split_key}_{metric_name}",
            )

    grouped_aggregate = {
        split_key: {
            "by_month": aggregate_grouped(
                eval_reports,
                split_key,
                "by_month",
            ),
            "by_season": aggregate_grouped(
                eval_reports,
                split_key,
                "by_season",
            ),
            "by_region": aggregate_grouped(
                eval_reports,
                split_key,
                "by_region",
            ),
        }
        for split_key in ("2022_val", "2023_external")
    }

    all_val_positive = global_aggregate["2022_val"][
        "anom_log_r2"
    ]["all_positive"]
    all_external_positive = global_aggregate["2023_external"][
        "anom_log_r2"
    ]["all_positive"]

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "anomaly_meteo_three_seed_closure_v01.json"
    txt_path = output_dir / "anomaly_meteo_three_seed_closure_v01.txt"

    report = {
        "artifact_name": "clean_anomaly_meteo_three_seed_closure",
        "artifact_version": 1,
        "experiment": EXPERIMENT,
        "seeds": SEEDS,
        "input_provenance": {
            "per_seed": provenance,
            "training_script": {
                "path": str(training_script),
                "sha256": sha256_file(training_script),
            },
            "audit_script": {
                "path": str(audit_script),
                "sha256": sha256_file(audit_script),
            },
            "evaluation_script": {
                "path": str(eval_script),
                "sha256": sha256_file(eval_script),
            },
            "cache_audit": {
                "path": str(cache_audit_path),
                "sha256": sha256_file(cache_audit_path),
            },
            "closure_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "shared_contract": shared_contract,
        "shared_cache_dir": shared_cache_dir,
        "shared_cache_manifest_sha256": shared_cache_manifest_hashes,
        "global_aggregate": global_aggregate,
        "grouped_aggregate": grouped_aggregate,
        "scientific_interpretation_boundary": {
            "2022_validation_type":
                "held-out_model-selection_validation",
            "2023_evaluation_type":
                "ordinary_external_anomaly_evaluation",
            "emission_input_used": False,
            "reconstruction_requires_month_background_baseline": True,
            "three_seed_summary_type": "descriptive_mean_and_sample_sd",
            "statistical_significance_established": False,
            "pixelwise_independence_assumed": False,
            "paper_ready_for_meteo_baseline_only": True,
            "final_anomaly_contrast_requires_fire_results": True,
        },
        "verification": {
            "all_three_training_runs_present": True,
            "all_three_formal_audits_pass": True,
            "all_three_external_evaluations_pass": True,
            "shared_training_contract_exact": True,
            "shared_cache_provenance_exact": True,
            "all_checkpoint_hashes_sealed": True,
            "all_eval_report_hashes_sealed": True,
            "all_validation_anomaly_r2_positive": all_val_positive,
            "all_external_anomaly_r2_positive": all_external_positive,
            "overall_pass": True,
        },
        "decision": "CLOSE_METEO_ONLY_PROCEED_TO_METEO_FIRE",
        "atlas_direction_closed": True,
        "new_atlas_experiments_allowed": False,
    }
    save_json(report, json_path)

    val = global_aggregate["2022_val"]
    ext = global_aggregate["2023_external"]
    lines = [
        "CLEAN_ANOMALY_METEO_THREE_SEED_CLOSURE_PASS",
        f"EXPERIMENT={EXPERIMENT}",
        f"SEEDS={SEEDS}",
        f"VAL_2022_ANOM_LOG_R2_MEAN_SD="
        f"{val['anom_log_r2']['mean']:.8f}+/-"
        f"{val['anom_log_r2']['sample_sd']:.8f}",
        f"VAL_2022_ANOM_LOG_RMSE_MEAN_SD="
        f"{val['anom_log_rmse']['mean']:.8f}+/-"
        f"{val['anom_log_rmse']['sample_sd']:.8f}",
        f"VAL_2022_ANOM_LOG_CORR_MEAN_SD="
        f"{val['anom_log_corr']['mean']:.8f}+/-"
        f"{val['anom_log_corr']['sample_sd']:.8f}",
        f"VAL_2022_ANOM_SKILL_VS_ZERO_MEAN_SD="
        f"{val['anom_mse_skill_vs_zero']['mean']:.8f}+/-"
        f"{val['anom_mse_skill_vs_zero']['sample_sd']:.8f}",
        f"VAL_2022_RECON_R2_DELTA_VS_BACKGROUND_MEAN_SD="
        f"{val['recon_r2_delta_vs_background']['mean']:.8f}+/-"
        f"{val['recon_r2_delta_vs_background']['sample_sd']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_R2_MEAN_SD="
        f"{ext['anom_log_r2']['mean']:.8f}+/-"
        f"{ext['anom_log_r2']['sample_sd']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_RMSE_MEAN_SD="
        f"{ext['anom_log_rmse']['mean']:.8f}+/-"
        f"{ext['anom_log_rmse']['sample_sd']:.8f}",
        f"EXTERNAL_2023_ANOM_LOG_CORR_MEAN_SD="
        f"{ext['anom_log_corr']['mean']:.8f}+/-"
        f"{ext['anom_log_corr']['sample_sd']:.8f}",
        f"EXTERNAL_2023_ANOM_SKILL_VS_ZERO_MEAN_SD="
        f"{ext['anom_mse_skill_vs_zero']['mean']:.8f}+/-"
        f"{ext['anom_mse_skill_vs_zero']['sample_sd']:.8f}",
        f"EXTERNAL_2023_RECON_R2_DELTA_VS_BACKGROUND_MEAN_SD="
        f"{ext['recon_r2_delta_vs_background']['mean']:.8f}+/-"
        f"{ext['recon_r2_delta_vs_background']['sample_sd']:.8f}",
        f"VAL_2022_ANOM_LOG_R2_ALL_POSITIVE={all_val_positive}",
        f"EXTERNAL_2023_ANOM_LOG_R2_ALL_POSITIVE={all_external_positive}",
        "THREE_SEED_SUMMARY_TYPE=descriptive_mean_and_sample_sd",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
        "PIXELWISE_INDEPENDENCE_ASSUMED=False",
        "METEO_ONLY_LINE_CLOSED=True",
        "NEXT_CONFIG=anom_meteo_fire",
        "ATLAS_DIRECTION_CLOSED=True",
        "NEW_ATLAS_EXPERIMENTS_ALLOWED=False",
        "DECISION=CLOSE_METEO_ONLY_PROCEED_TO_METEO_FIRE",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print()
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
