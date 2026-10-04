#!/usr/bin/env python3
"""Synthetic-data smoke tests for release interfaces; no project data used."""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    path = RELEASE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"PyTorch unavailable: {exc}")
        cls.torch = torch

    def test_meanstate_model_synthetic_forward(self):
        module = load_module("src/workflows/train_meanstate.py", "jgr_smoke_meanstate")
        spec = module.EXPERIMENT_SPECS["mean_emis_meteo_clim"]
        model = module.MeanStateUNet(spec, base_features=8).eval()
        tensor = self.torch.randn(2, spec.input_channels, 16, 16)
        with self.torch.no_grad():
            output = model(tensor)
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))
        self.assertTrue(bool(self.torch.isfinite(output).all()))

    def test_anomaly_model_synthetic_forward(self):
        module = load_module("src/workflows/train_daily_anomaly.py", "jgr_smoke_anomaly")
        spec = module.EXPERIMENT_SPECS["anom_meteo_fire"]
        model = module.AnomalyUNet(spec, base_features=8).eval()
        tensor = self.torch.randn(2, spec.input_channels, 16, 16)
        with self.torch.no_grad():
            output = model(tensor)
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))
        self.assertTrue(bool(self.torch.isfinite(output).all()))

    def test_restricted_path_relocation_contract(self):
        module = load_module("src/workflows/train_daily_anomaly.py", "jgr_smoke_relocation")
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "08_phase2_anomaly_clean_trainonly_union"
            daily = cache / "daily_anomaly" / "2022_train"
            daily.mkdir(parents=True)
            candidate = daily / "day_032.pt"
            candidate.write_bytes(b"synthetic-path-resolution-only")
            reader = module.LeanAnomalyReader(cache)
            resolved = reader.resolve_daily_path(
                {
                    "file": "/historical/absolute/day_032.pt",
                    "split": "2022_train",
                    "year": 2022,
                }
            )
            self.assertEqual(resolved, candidate.resolve())

    def test_bounded_adjustment_metrics_on_synthetic_data(self):
        module = load_module(
            "src/workflows/run_bounded_adjustment.py", "jgr_smoke_bounded"
        )
        target = self.torch.arange(16, dtype=self.torch.float32).reshape(4, 4)
        prediction = target + 0.1
        mask = self.torch.ones(4, 4)
        metrics = module.masked_metrics(prediction, target, mask)
        self.assertEqual(metrics["n_valid"], 16)
        self.assertTrue(all(math.isfinite(metrics[key]) for key in ("log_r2", "log_rmse", "log_bias")))

    def test_public_result_csvs_are_rectangular(self):
        for folder in ("results", "figure_source_data"):
            for path in sorted((RELEASE_ROOT / folder).glob("*.csv")):
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.reader(stream))
                self.assertGreaterEqual(len(rows), 2, path.name)
                width = len(rows[0])
                self.assertTrue(all(len(row) == width for row in rows), path.name)

    def test_table_reproduction_uses_packaged_aggregates_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RELEASE_ROOT / "scripts" / "reproduce_tables_figures.py"),
                    "--release-root",
                    str(RELEASE_ROOT),
                    "--output-dir",
                    temporary,
                    "--tables-only",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(len(list((Path(temporary) / "tables").glob("*.csv"))), 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
