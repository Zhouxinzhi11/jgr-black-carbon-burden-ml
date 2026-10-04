# Zhou et al. JGR: Atmospheres code release (v1.0.0)

This archive accompanies the manuscript *Cross-Year Associations Between a Lagged Emission Inventory and Reanalysis-Derived Black Carbon Column Burden: A Multiscale Machine Learning Assessment*. It packages the inspected code paths for mean-state, geographic-holdout, daily-anomaly, negative-control, provenance-audit, and bounded-adjustment analyses, together with public aggregate tables and figure source data.

This is the validated v1.0.0 release package prepared in the private GitHub repository [https://github.com/Zhouxinzhi11/jgr-black-carbon-burden-ml](https://github.com/Zhouxinzhi11/jgr-black-carbon-burden-ml). The corresponding Zenodo software record is an unpublished draft with reserved DOI `10.5281/zenodo.23139187`. Author-written code is licensed under MIT; public aggregate results and public table/figure source-data CSV files are licensed under CC BY 4.0. The repository and Zenodo record remain non-public until explicit release authorization.

## Scientific scope

The mean-state branch models monthly mean log burden using an eight-channel processed inventory reference and/or monthly meteorology. Controls include cyclic displacement, support-preserving vector permutation, and binary support. The daily-anomaly branch models daily deviations using meteorological anomalies with optional fire or annual AlphaEarth Foundations context. Geographic holdout tests spatial transfer. Cosine-latitude weighting, burden-scale back transformation, BCCMASS provenance, sampling/common-mask support, and the bounded sector-offset experiment are separately audited.

The bounded-adjustment experiment is a numerical flexibility diagnostic. It is not an emission inversion or physical retrieval. Likewise, the three training seeds quantify optimization variability and are not a significance test.

The detailed manuscript-to-code crosswalk is in `docs/manuscript_code_map.csv`. The confirmed formal-output baseline is Main Tables 1–4, Main Figures 1–5, SI Tables S1–S12, and SI Figures S1–S3. `docs/table_figure_reproduction_map.csv` gives one unique record for every one of these 24 logical outputs.

## Directory layout

- `src/workflows/`: preprocessing-independent training, evaluation, control, aggregation, and diagnostic workflows.
- `src/audits/`: provenance, support, scale, cache, run, and common-mask audits.
- `src/preprocessing/`: authorized-data preprocessing utilities.
- `src/checkpoint_compatibility/`: one retained model definition required solely to load the historical bounded-adjustment checkpoint.
- `scripts/`: figure/table reproduction and release validation utilities.
- `configs/`: path template containing no local absolute paths.
- `results/`: final manuscript/SI aggregate tables only.
- `figure_source_data/`: public aggregate source data; no pointwise restricted grid is included.
- `data/`: data-access and input-schema documentation only.
- `tests/`: synthetic-data smoke tests.
- `docs/`: per-file audit, exclusion/review lists, crosswalk, source records, and technical release report.
- `manifests/`: copied-source provenance and release checksums.
- `metadata/`: GitHub/Zenodo deposit metadata, citation text, and availability text.


## Public and restricted data boundary

GF-5(02) POSP Level 1 is a public data source available from CRESDA (<https://data.cresda.cn/>); registration may be required. POSP I380 is used only to define record availability. This study did not use the public POSP Level 2 product or DOI `10.57760/sciencedb.14748` as its I380 source.

The complete POSP-centered analysis-ready fused derivative is not included here. It is prepared as a separate restricted-review archive. Without authorized access, users can run the synthetic smoke tests and reproduce the 16 tables and eight aggregate figure products from public packaged results, but cannot rerun full training or regenerate pointwise spatial products. Full training was not rerun during v1.0.0 release preparation.

Formal training seeds are **42, 43, and 44**. The formal checkpoint files are not included; their project-relative identifiers and SHA-256 values are recorded in `manifests/formal_checkpoint_manifest.csv`.

Figure 3 pointwise target/prediction/error grids and Figure S3 cellwise support grids are restricted. Their irreversible summary CSVs remain public. AlphaEarth Foundations / Google Satellite Embedding is a public upstream source; AEF-containing source tensors are in the restricted package only because those composite tensors also contain restricted fused targets, predictors, and exact support.

Author-written code is licensed under MIT. Public aggregate results and public table/figure source-data CSV files are licensed under CC BY 4.0. Neither license applies to the restricted derivative; see `LICENSES.md` and the restricted record's custom rights statement.

## Environment

`requirements.txt` and `environment.yml` provide the minimal imported dependency set pinned to the prior full release-preparation snapshot (Python 3.9.21). The v1.0.0 lightweight checks were executed with the available Python 3.9.21 runtime; exact observed versions are in `docs/current_environment.csv`. Neither environment is claimed to be the exact historical training environment because no complete locked formal-run environment was found.

Create an environment with either:

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

or:

```text
conda env create -f environment.yml
conda activate zhou-jgr-bc
```

CPU execution is sufficient for validation and table/figure reproduction. Full training may require a suitable CUDA-enabled PyTorch installation selected by the authorized user.

## Configure paths

Copy `configs/paths.example.yaml` outside version control or create an equivalent local configuration. Replace every placeholder only in your private working environment. Do not commit restricted paths, credentials, or source data.

The original scientific scripts use explicit CLI paths rather than silently discovering data. The example commands below use angle-bracket placeholders and must be supplied by a researcher authorized to access the fused inputs.

## Public aggregate reproduction

Recreate the 17 packaged physical table CSVs and eight aggregate-only figure renderings:

```text
python scripts/reproduce_tables_figures.py --release-root . --output-dir reproduced
```

Tables only:

```text
python scripts/reproduce_tables_figures.py --release-root . --output-dir reproduced --tables-only
```

The 17 table files represent 16 logical tables: SI Table S7 is one formal table with verified Part A and Part B CSVs. Main Figure 3 and SI Figure S3 are intentionally summary-only in this archive because pointwise/cellwise source arrays are available only through the restricted-access record, subject to the approved access conditions. The generated PNGs are transparent aggregate reconstructions, not claims of pixel-identical reproduction of the manuscript layouts.

Verify the formal-output coverage, Table S7 split, and public aggregate closure:

```text
python scripts/verify_formal_outputs.py --release-root .
```

## Authorized mean-state workflow

Build/audit the cache first, then train fixed experiments. A representative joint mean-state command is:

```text
python src/workflows/train_meanstate.py --experiment mean_emis_meteo_clim --cache-dir <AUTHORIZED_CACHE_DIR> --gate11b-report <AUTHORIZED_CACHE_AUDIT_JSON> --run-root <PRIVATE_RUN_ROOT> --run-name mean_emis_meteo_clim_seed42 --val-months 11,12 --blacklist-months 7 --seed 42
```

The inventory-only, meteorology-only, and cyclic-displacement configurations are selected through `--experiment`; see `--help` for the choices. Formal results used seeds 42, 43, and 44. Evaluation and aggregation entry points are `src/workflows/evaluate_meanstate.py` and `src/workflows/aggregate_meanstate.py`.

## Authorized controls and geographic holdout

Support-preserving control:

```text
python src/workflows/train_support_preserving_control.py --experiment mean_emis_meteo_clim_supportperm --cache-dir <AUTHORIZED_CACHE_DIR> --gate11b-report <AUTHORIZED_CACHE_AUDIT_JSON> --gatep2a-report <AUTHORIZED_CONTROL_AUDIT_JSON> --run-root <PRIVATE_RUN_ROOT> --run-name supportperm_seed42 --seed 42
```

Binary support uses `src/workflows/train_binary_support_control.py`; geographic holdout uses `src/workflows/train_geographic_holdout.py`. Their required preflight reports are documented in `docs/manuscript_code_map.csv` and exposed by each script's `--help` output. Evaluation preserves the manuscript's cosine-latitude-weighted and common-mask definitions where indicated.

## Authorized daily-anomaly workflow

Representative meteorology-only command:

```text
python src/workflows/train_daily_anomaly.py --experiment anom_meteo_only --run-mode formal --cache-dir <AUTHORIZED_ANOMALY_CACHE> --cache-audit <AUTHORIZED_CACHE_AUDIT_JSON> --run-dir <PRIVATE_RUN_DIR> --seed 42 --batch-size 4 --num-workers 0 --patch-size 96 --patches-per-file 4 --base-features 32 --max-epochs 140 --patience 25 --lr 0.0003 --weight-decay 0.0001 --val-every 5 --train-limit 0 --val-limit 0 --require-cuda
```

The fire and AEF branches use `anom_meteo_fire` and `anom_meteo_aef`. The AEF branch additionally requires `--aef-source-root-2022` and `--aef-source-root-2023` pointing to the authorized, year-specific source-tensor roots documented in `docs/restricted_input_interface.md`. Formal results used seeds 42, 43, and 44. The anomaly target is daily log-burden deviation; no emission-anomaly target is introduced.

## Authorized audits

Sampling/common-mask support:

```text
python src/audits/audit_sampling_support.py --nc-2022 <AUTHORIZED_2022_FUSED_FILE> --nc-2023 <AUTHORIZED_2023_FUSED_PART1> <AUTHORIZED_2023_FUSED_PART2> --grid-ref <AUTHORIZED_GRID_REFERENCE> --cache-dir <AUTHORIZED_CACHE_DIR> --output-dir <PRIVATE_AUDIT_OUTPUT>
```

Additional entry points include `src/audits/audit_burden_scale.py`, `src/audits/audit_bccmass_provenance.py`, and `src/audits/summarize_bccmass_provenance.py`. Their detailed roles and manuscript links are in the crosswalk.

## Authorized bounded-adjustment workflow

The safe wrapper preserves the historical optimization defaults (150 steps, learning rate 0.05, seed 42, original bound and penalty grids) while requiring explicit paths:

```text
python src/workflows/run_bounded_adjustment.py --model-module src/checkpoint_compatibility/meanstate_ablation_reference.py --run-dir <AUTHORIZED_SEED42_JOINT_RUN> --output-dir <PRIVATE_ADJUSTMENT_OUTPUT> --year 2022 --month 11 --seed 42
```

Run again with `--month 12`, then use the packaged summary scripts. The historical month-specific scripts are retained for provenance, but the safe wrapper is the recommended release entry point.

## Inputs and outputs

Expected inputs include gridded NetCDF variables, PyTorch monthly/daily cache tensors, and JSON audit/manifests described in `data/README.md`. Private runs write checkpoints, run manifests, aggregate metrics, and audit reports only to user-supplied output directories. The public package contains no training caches, temporary tensors, raw POSP records, row-level fused records, sampled coordinate/time/value tuples, or pointwise restricted maps.

## Reproducibility boundary

Public aggregate tables and non-spatial summary figures can be regenerated directly with `scripts/reproduce_tables_figures.py`. Source-data tables for Main Figures 1–5 and SI Figures S1–S3 are provided, but Main Figure 3 and SI Figure S3 are limited to publishable summary statistics. The authored figure directory was used only as a read-only verification reference; no manuscript image or machine-specific path is stored in this archive.

For all data-dependent scientific workflows: **Code path inspected and release-packaged; full execution requires authorized access to the restricted fused input data.** Full training, large preprocessing, and restricted-data evaluation were not re-run during release preparation.

## Validation

Run synthetic smoke tests and the archive validator:

```text
python tests/smoke_test.py
python scripts/validate_release.py --release-root .
python scripts/validate_restricted_data.py --restricted-root <AUTHORIZED_RESTRICTED_ARCHIVE_ROOT> --quick
```

The validator checks required files, Python syntax, internal imports, CSV structure, local absolute paths, credential/account indicators, forbidden restricted filenames, code-map references, README options, the 1,222/405 code-audit counts, the separate 10-item withheld-derived-data scope, formal-output coverage, the release manifest, and SHA-256 values.

## Citation and licenses

Citation metadata are in `CITATION.cff`, `.zenodo.json`, and `metadata/software_citation_draft.md`. The reserved software DOI is `10.5281/zenodo.23139187` and the separate restricted-data DOI is `10.5281/zenodo.23139189`; neither record has been published. Author-written code is MIT licensed through `LICENSE`. Public aggregate results and public table/figure source-data CSV files are CC BY 4.0 through `LICENSE_PUBLIC_RESULTS_CC_BY_4.0.txt`. `LICENSES.md` defines the directory-level scopes. The separate restricted fused-data archive has no public redistribution license.

## Audit status

The conservative code-tree audit covered 1,222 files: 33 CORE, 61 SUPPORTING, 723 LEGACY_EXCLUDED, and 405 REVIEW_REQUIRED. All 405 review rows received a P0/P1/P2 rc2 disposition, with every P0 resolved, without changing the authoritative classification counts. Ten additional restricted derived-data candidates are tracked separately as `withheld-derived-data`; they are outside the code tree and outside the four classification totals. Across both scopes, 1,232 objects were considered. See `docs/review_required_count_reconciliation.md` and `docs/final_release_report.md`.
