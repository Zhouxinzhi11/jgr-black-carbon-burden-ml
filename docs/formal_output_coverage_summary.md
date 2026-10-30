# Formal output coverage summary

Current sealed-manuscript baseline, clarified on 2026-10-30: Main Tables 1–4, Main Figures 1–4, SI Tables S1–S12, and SI Figures S1–S3. There are 23 current formal outputs.

- Formal logical tables: **16/16 mapped**; 17 physical CSV files because Table S7 is split into verified Part A and Part B files.
- Formal logical figures: **7/7 mapped**; the package retains eight public figure-source CSV files, with the additional bounded-adjustment visualization separately classified in `docs/additional_archive_outputs.csv`.
- Reference figure directory files inventoried: **44**; no image was copied into the release.
- Main Figures 1–4 correspond to the current manuscript. SI Figures S1–S3 were checked against current SI captions and public source data.
- The retained legacy Figure 5 visualization corresponds to Main Table 4 and SI Table S6. Its public CSV is byte-identical to the Main Table 4 CSV; it is not a fifth current manuscript figure. The unchanged reproduction script still emits eight aggregate figure products.
- The historical Atlas-enhanced `SupplementaryFigureS1_Earth2_Atlas_seedwise_final` was explicitly rejected because it does not match the current SI Figure S1 sampling-support content.

Exact tabular reproduction is available from the packaged CSVs. Public figure regeneration reproduces the reported aggregate values, not the manuscript's pixel layout. Main Figure 3 and SI Figure S3 remain summary-only because pointwise spatial arrays are withheld.

The original v1.0.0 ZIP/tag remain unchanged. See `docs/publication_and_output_map_correction_20261030.md` for current publication status and interpretation of the frozen preparation documents.
