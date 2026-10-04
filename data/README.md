# Data boundary

No row-level, pixel-level, or cell-level POSP-centered fused observations are included in this public code archive. The complete analysis-ready fused derivative is a separate restricted-review archive.

GF-5(02) POSP Level 1 is a public source available through the CRESDA portal (<https://data.cresda.cn/>), where account registration may be required. In this study, POSP I380 is used only to define record availability. The public POSP Level 2 product and DOI `10.57760/sciencedb.14748` are not the I380 source used here.

The restricted input interface expects a 361 x 576 global array, monthly mean-state tensors, daily anomaly tensors, and the source tensors needed for the AEF branch. See `configs/paths.example.yaml` and `docs/restricted_input_interface.md`. The raw downloadable EDGAR, MERRA-2, ERA5, MODIS/FIRMS, POSP Level 1, and AlphaEarth Foundations products are not duplicated in either archive.

Without authorized access to the restricted fused derivative, users can run the synthetic smoke tests and reproduce the 16 tables and eight aggregate figure products from the packaged non-sensitive results. Full training, restricted-data evaluation, spatial-map regeneration, and provenance resampling require the separate restricted archive and appropriate access authorization.

Public aggregate results and source-data CSV files are licensed under CC BY 4.0. Author-written code is licensed under MIT. Neither license applies to the restricted fused derivative.
