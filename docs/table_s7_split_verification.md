# Table S7 split verification

The additional physical table CSV was directly verified and is not an internal stray output.

- The final SI page 17 labels one logical **Table S7** with **Part A. Area-weighted mean-state performance** and **Part B. Paired area-weighted ΔR² contrasts**.
- `results/si_table_s7a_area_weighted_performance.csv` matches Part A headers and all two-row values.
- `results/si_table_s7b_area_weighted_contrasts.csv` matches Part B headers and all two-row values.
- `scripts/prepare_public_results.py` extracts these from adjacent SI table objects into the two named files.

Consequently, the release contains **16 logical formal tables** represented by **17 physical CSV files**. The sole split is Table S7 Part A/Part B.
