# Computational protocols and implementation

| Experiment | Main code | Numerics and readouts |
|---|---|---|
| Exact mechanism controls | `rebuttal_experiments/mechanisms.py` | All ordered triples at Q=1; exact fractions; multisets and mean moments |
| BREC | `brec/float32/` | Float32 sorted-channel gap at \(10^{-8}\), with order-specific exact reference cohorts |
| TU prediction | `protein.py`, `protein_encodings.py`, `protein_gnnplus.py` | Co-Sparsify data and ten-fold training; GNNPlus GINE receives structural loop/pair channels |
| Additional object discrimination tasks | `discriminability/` | Specified widths, orbit/FWL references, comparisons with a shared orientation, and MIS |

## Shared scope of the supplementary experiments

- Full multisets retain row multiplicities. Overlapping intervals do not prove exact equality; absence of a compatible matching certifies a multiset difference.
- Unresolved, resource-limited, and unexecuted BREC cases remain in the denominator. A larger lower bound on detected differences does not by itself establish greater true expressive power.
- Object domains use the same orientation swap across all channels, rather than selecting orientations independently per channel. MIS optimality certifies that the numerical similarity graph has been solved, not that the true structural feature values have been certified.

- TU uses the data, ten-fold training, optimizer, and evaluator from Co-Sparsify commit `505f6b50f019dc78edc88653d771667c0c561f5f`: PROTEINS_full and ENZYMES, 9 encodings, widths 8/12/14/16, and 10 folds, for 720 training runs. The backbone is replaced by GINE from `gnnplus_official` (hidden dimension 256, 3 layers), a SAN graph head, and mean pooling. `pair_x` is mapped to the original directed edges, and `loop_x` is concatenated with the node inputs. The physical batch size is explicitly set to 64.
- The same held-out fold is used for checkpoint selection and final scoring, with no separate independent test split, as explicitly requested by the user. Means and population standard deviations follow the original ten-fold reporting convention. The adapter replaces the encoding transform and adds local records; it does not use the earlier custom GNN, 64-slot setup, or independent-test pipeline.
- `joint` uses a fixed pair readout of a triple shared-pivot construction, connected through the original `pair_x`/`loop_x` interface without adding trainable modules. `joint` retains `endpoint`; `powers` contains elementwise squares of existing channels; `joint_mix` applies a full invertible change of channel basis. This finite readout is not equivalent to the full triple dictionary or full 3-FWL.

## BREC reliability

For 2- and 3-FWL, both official isomorphic inputs are evaluated independently and their channels compared after verified vertex alignment. The [BREC run guide](brec_suite.md) lists the tested cohorts and record counts.
