# Formal output numeric verification

The rc2 review performed real lightweight recomputation against the selected final-figure evidence files; it did not rerun training or restricted-data preprocessing.

Current-scope clarification (2026-10-30): the historical Figure 5 below is an additional archived bounded-adjustment visualization corresponding to current Main Table 4 and SI Table S6. The current manuscript has four main figures and three SI figures. This note preserves the historical numeric checks; it does not claim a new experiment or restore a fifth main-text figure.

- 4 public table/figure-source pairs were byte-compared and matched exactly (Main Table 2/Figure 2; Main Table 3/Figure 4; Main Table 4/additional legacy Figure 5; Table S9/Figure S1).
- Figure 2: mean and sample SD were recomputed from each of 10 three-seed groups and matched the selected final summary CSV; rounded values matched Main Table 2.
- Figure 4: mean and sample SD were recomputed from each of 6 three-seed groups and matched the selected final summary CSV; rounded values matched Main Table 3.
- Additional archived visualization (legacy Figure 5): November–December arithmetic means were recomputed for 4 settings and matched the selected summary CSV; rounded values matched current Main Table 4.
- Table S7 Parts A and B were checked cell by cell against the final SI values and extraction code.

All checks passed. Spatial pointwise arrays for Main Figure 3 and SI Figure S3 were not opened or repackaged; their public scope remains aggregate-summary-only.
