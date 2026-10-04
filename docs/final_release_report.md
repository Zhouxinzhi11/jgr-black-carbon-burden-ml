# v1.0.0 technical release report

## Scope inherited without re-audit

The formal package inherits the completed conservative audit. The code-tree scope is 1,222 files: CORE 33, SUPPORTING 61, LEGACY_EXCLUDED 723, and REVIEW_REQUIRED 405. Ten additional withheld-derived-data objects are outside the code tree and outside those four class totals. The combined consideration scope is 1,232 objects under two distinct statistical denominators. All P0 decisions remain resolved and unchanged.

## Scientific and release invariants

- Formal seeds remain 42, 43, and 44.
- The output baseline remains Main Tables 1–4, Main Figures 1–5, SI Tables S1–S12, and SI Figures S1–S3.
- Seventeen physical table CSVs represent 16 logical tables because Table S7 is verified as Part A plus Part B.
- Figure 3 pointwise arrays and Figure S3 cellwise arrays are restricted; only irreversible summaries are public.
- No scientific result, model algorithm, split, seed, or restricted `.pt` payload was changed.
- Full training and large preprocessing were not rerun.

## Public/restricted records

Private GitHub repository: <https://github.com/Zhouxinzhi11/jgr-black-carbon-burden-ml>  
Unpublished public-software Zenodo draft: <https://zenodo.org/deposit/23139187> (reserved DOI `10.5281/zenodo.23139187`)  
Unpublished restricted-data Zenodo draft: <https://zenodo.org/deposit/23139189> (reserved DOI `10.5281/zenodo.23139189`)

The reviewer preview link is stored outside both archives and is not reproduced here. The repository remains private, no tag or GitHub release has been created, and neither Zenodo record has been published.

## Data rights and provenance

GF-5(02) POSP Level 1 is public through CRESDA and may require account registration; I380 is used only for record availability. POSP Level 2 DOI `10.57760/sciencedb.14748` is not the source. EDGAR v8.1_AP is identified by DOI `10.2905/JRC.E1ZGQER`. Exact original POSP scene/product identifiers and the original download checksum were not retained in the inherited fused derivative; no identifiers were invented.

Author-written code is MIT licensed. Public aggregate results and public table/figure source-data CSV files are CC BY 4.0. The restricted fused derivative has a custom restricted-rights statement and no public redistribution license.

## Validation statement

The final external readiness report records syntax/import checks, six synthetic smoke tests, 16/16 table and 8/8 figure mapping/reproduction, sensitivity and leakage scans, license/DOI/version checks, ZIP CRC/path safety, complete extraction, and checksum verification. Archive SHA-256 values are maintained in their adjacent sidecar files to avoid a circular self-hash inside the ZIP.
