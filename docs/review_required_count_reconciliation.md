# Review-required count reconciliation

This rc2 uses two deliberately separate scopes.

- Code-tree audit: **1,222 files**.
- Code-tree `REVIEW_REQUIRED`: **405 files**. This remains the sole authoritative `REVIEW_REQUIRED` count and is not altered by priority resolution.
- Restricted derived-data candidates outside the code tree: **10 items**, recorded as `withheld-derived-data` and excluded from the four code classifications.
- Total objects considered across both scopes: **1,232**, calculated as 1,222 code-tree files + 10 outside-tree derived-data candidates.

The previous 415-entry review-only manifest combined these scopes for logistics. Rc2 splits it into `manifests/review_required_manifest.csv` (405) and `manifests/withheld_derived_data_manifest.csv` (10). No derived-data candidate is counted as CORE, SUPPORTING, LEGACY_EXCLUDED, or REVIEW_REQUIRED.
