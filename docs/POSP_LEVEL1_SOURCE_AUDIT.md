# GF-5(02) POSP Level 1 source-identifier audit

Status: exact scene/product identifiers were not recovered.

The audit inspected the complete restricted staging inventory (1,280 files), the inherited release-review manifest, the available sampling/provenance audit directory (98 files), and the public audit/provenance manifests. File names and text metadata were searched for GF-5(02), GF5/GF5B, POSP, CRESDA, I380, scene-ID, and product-ID patterns. For PyTorch and NumPy ZIP containers, archive member names and serialized metadata members were inspected without deserializing or exporting numeric tensor arrays.

Only generic workflow names, instrument names, I380 field names, and inherited fused-file identifiers were found. No native CRESDA scene name, scene identifier, or exact Level 1 product identifier was present. Therefore, exact original scene identifiers were not retained in the inherited fused derivative. The original POSP Level 1 download checksum was not retained.

This negative finding does not change the public status of GF-5(02) POSP Level 1 or its CRESDA source. It records a provenance limitation of the inherited fused derivative and prevents an identifier from being guessed or reconstructed without evidence.
