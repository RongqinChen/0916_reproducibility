# GNNPlus components used by this study

This directory is a modified copy of GNNPlus. See the root `THIRD_PARTY.md`
and the preserved `LICENSE` for attribution.

Use `python tools/run_gnnplus_matched.py` from the **release root** for this
study's matched structural-encoding prediction settings. See
`docs/experiments.md`. Dataset preparation also reuses this package's loaders.
The package modules remain together because their registration imports are eager.
