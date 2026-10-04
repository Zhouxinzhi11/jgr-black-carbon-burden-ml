# JGR Atmospheres Code Release Archive Design

## Scope

This staging archive is built from the final manuscript and Supporting Information, then verified against source-code call relationships, run manifests, evaluation reports, and generated outputs. The original project is read-only.

## Classification policy

- CORE: required to generate a reported model, analysis, table, or figure.
- SUPPORTING: required utilities, preprocessing, configuration, audit, or packaging code.
- LEGACY_EXCLUDED: superseded, failed, duplicate, exploratory, or unused material.
- REVIEW_REQUIRED: evidence is insufficient or public release could expose restricted information.

Only evidence-confirmed CORE and necessary SUPPORTING files may be copied into the public staging archive. REVIEW_REQUIRED candidates are listed separately and are not incorporated into the formal package.

## Restricted-data boundary

POSP records and fused row-level data are not redistributable. Files containing or potentially reconstructing row-level coordinates, timestamps, targets, or restricted source values are excluded pending authorization. Public results are limited to confirmed aggregate outputs.

## Verification boundary

No model training or large-scale preprocessing will be rerun. Static inspection, syntax compilation, internal-import checks, synthetic-data smoke tests, aggregate result reconstruction, reference validation, sensitive-string scanning, and checksum verification are permitted.

## Packaging policy

Absolute personal paths are replaced only in release copies. Scientific algorithms, data splits, seeds, evaluation masks, metrics, and reported results are not altered. Unknown licenses, custodians, access mechanisms, and DOI values remain explicitly unconfirmed.
