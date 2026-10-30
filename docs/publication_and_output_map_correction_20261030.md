# Publication status and manuscript output mapping correction

Date: 2026-10-30. This correction updates current documentation and output-coverage validation only. It does not change the manuscript, scientific code, model algorithms, splits, seeds, numeric source data or data rights.

## Published records

The GitHub repository and [v1.0.0 release](https://github.com/Zhouxinzhi11/jgr-black-carbon-burden-ml/releases/tag/v1.0.0) are public. The software and public aggregate source data are published at <https://doi.org/10.5281/zenodo.23139187>.

The main fused-data record (<https://doi.org/10.5281/zenodo.23139189>) and its transport recovery supplement (<https://doi.org/10.5281/zenodo.23463888>) have published metadata and restricted files. Together they cover 96 logical partitions through 99 data ZIPs. Neither has an automatic public-release embargo. Software is MIT licensed and public aggregate results/source-data CSVs are CC BY 4.0; neither license applies to the restricted files. Their custom restricted-rights statement remains in force.

Older release-preparation documents contained statements describing then-unpublished drafts. Those statements do not describe the current publication state. Editorial approval of the requested restricted-access exception is a separate matter and is not claimed here.

## Current manuscript outputs

The sealed manuscript contains Main Tables 1–4 and Main Figures 1–4. Its SI contains Tables S1–S12 and Figures S1–S3. This is **23 formal outputs: 16 tables and 7 figures**, mapped one to one in `docs/table_figure_reproduction_map.csv`.

There are still 17 physical table CSVs because SI Table S7 has verified Part A and Part B files. The eight retained figure-source CSVs support seven current-manuscript products and one additional archived bounded-adjustment visualization. Figure 3 and Figure S3 remain summary-only in the public package; their pointwise/cellwise arrays require authorized restricted access.

The additional visualization is identified in `docs/additional_archive_outputs.csv`. Its legacy source filename, `figure_source_data/figure_5_bounded_adjustment.csv`, remains byte-identical to `results/main_table_4_bounded_adjustment.csv`. It corresponds to current Main Table 4 and SI Table S6, not to a current Main Figure 5. Historical reference images also retain pre-renumbering filenames; these filenames do not establish extra current manuscript figures.

The reproduction script is unchanged and continues to emit eight aggregate figure products. Two validation helpers now distinguish the 23 formal outputs from the separately classified additional product. No scientific result has been recomputed or altered by changing the output count.

## Frozen release files

The v1.0.0 Git tag, release assets, deposited ZIPs, their hashes and DOIs are unchanged. Frozen files retain their original preparation text. Use this correction and the current branch crosswalk to interpret that text; a documentation correction is not a new scientific release. Current-branch manifests describe an export of that branch, rather than changing the manifests inside a frozen published ZIP.

No reviewer secret link or login credential is included here. Research access continues to be decided by the corresponding author, in consultation with the data creators, subject to the record's access conditions and the applicable editorial exception decision.
