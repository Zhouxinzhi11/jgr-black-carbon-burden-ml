# Checkpoint compatibility resolution

The retained mean-state model definition is necessary only to instantiate the architecture expected by the historical seed-42 bounded-adjustment checkpoint. It is not an active legacy experiment and does not define an additional paper result.

Rc2 therefore renames `src/legacy_models/` to `src/checkpoint_compatibility/`. The Python source is unchanged apart from release-path references elsewhere in the package. The bounded-adjustment safe wrapper, README command, provenance mapping, tests, and manifests use `src/checkpoint_compatibility/meanstate_ablation_reference.py`.

This is a packaging/terminology correction only. No model layer, parameter shape, algorithm, seed, data split, or metric changed.
