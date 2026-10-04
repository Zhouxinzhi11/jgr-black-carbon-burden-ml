# Restricted input interface

The public code and restricted fused-data archive are separate records. Configure paths with a private copy of `configs/paths.example.yaml`; never commit the private file.

The mean-state, control, holdout, area-weighted, and bounded-adjustment branches use `inputs/meanstate_cache/07_phase2_meanstate_fullmonth_trainonly_union`. The daily-anomaly branches use `inputs/anomaly_cache/08_phase2_anomaly_clean_trainonly_union`. The AEF anomaly branch additionally requires `inputs/anomaly_aef_source_tensors/2022` and `.../2023`.

The original `.pt` payloads are byte-preserved and retain historical path strings. rc3 resolves cache members against the configured cache directory and resolves AEF tensors against the two explicit yearly roots, then checks the recorded source SHA-256. No numerical tensor content, split, seed, or algorithm was changed.

Representative anomaly command:

```bash
python src/workflows/train_daily_anomaly.py --experiment anom_meteo_aef --run-mode formal --cache-dir <RESTRICTED_ROOT>/inputs/anomaly_cache/08_phase2_anomaly_clean_trainonly_union --aef-source-root-2022 <RESTRICTED_ROOT>/inputs/anomaly_aef_source_tensors/2022 --aef-source-root-2023 <RESTRICTED_ROOT>/inputs/anomaly_aef_source_tensors/2023 --cache-audit <AUTHORIZED_CACHE_AUDIT_JSON> --run-dir <PRIVATE_RUN_DIR> --seed 42 --batch-size 4 --num-workers 0 --patch-size 96 --patches-per-file 4 --base-features 32 --max-epochs 140 --patience 25 --lr 0.0003 --weight-decay 0.0001 --val-every 5 --train-limit 0 --val-limit 0 --require-cuda
```

Repeat only with the formal seeds `42`, `43`, and `44`. Full training was not rerun during rc3 preparation.
