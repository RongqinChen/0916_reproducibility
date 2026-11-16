# Computational protocols and implementation

| Experiment | Main code | Numerics and readouts |
|---|---|---|
| Exact mechanism controls | `rebuttal_experiments/mechanisms.py` | All ordered triples at Q=1; exact fractions; multisets and mean moments |
| Supplementary BREC | `rebuttal_experiments/brec.py` | Full pair domain; fractions for regular graphs and intervals otherwise; multisets or mean moments |
| TU prediction | `protein.py`, `protein_encodings.py`, `protein_gnnplus.py` | Co-Sparsify data and ten-fold training; GNNPlus GINE receives structural loop/pair channels |
| PPGN | `brec/ppgn_models.py`, `ppgn_realization.py` | Specified symmetric matrix-power inputs, pair/triple backbones, and neural readouts |
| Additional object discrimination tasks | `discriminability/` | Specified widths, orbit/FWL references, comparisons with a shared orientation, and MIS |

## Shared scope of the supplementary experiments

- Full multisets retain row multiplicities. Overlapping intervals do not prove exact equality; absence of a compatible matching certifies a multiset difference.
- Unresolved, resource-limited, and unexecuted BREC cases remain in the denominator. A larger lower bound on detected differences does not by itself establish greater true expressive power.
- Object domains use the same orientation swap across all channels, rather than selecting orientations independently per channel. MIS optimality certifies that the numerical similarity graph has been solved, not that the true structural feature values have been certified.

- TU uses the data, ten-fold training, optimizer, and evaluator from Co-Sparsify commit `505f6b50f019dc78edc88653d771667c0c561f5f`: PROTEINS_full and ENZYMES, 9 encodings, widths 8/12/14/16, and 10 folds, for 720 training runs. The backbone is replaced by GINE from `gnnplus_official` (hidden dimension 256, 3 layers), a SAN graph head, and mean pooling. `pair_x` is mapped to the original directed edges, and `loop_x` is concatenated with the node inputs. The physical batch size is explicitly set to 64.
- The same held-out fold is used for checkpoint selection and final scoring, with no separate independent test split, as explicitly requested by the user. Means and population standard deviations follow the original ten-fold reporting convention. The adapter replaces the encoding transform and adds local records; it does not use the earlier custom GNN, 64-slot setup, or independent-test pipeline.
- `joint` uses a fixed pair readout of a triple shared-pivot construction, connected through the original `pair_x`/`loop_x` interface without adding trainable modules. `joint` retains `endpoint`; `powers` contains elementwise squares of existing channels; `joint_mix` applies a full invertible change of channel basis. This finite readout is not equivalent to the full triple dictionary or full 3-FWL.

## PPGN inputs and model

`FixedInput.array` constructs I,S,…,S⁸, where S=D⁻¹ᐟ²AD⁻¹ᐟ². Triple inputs concatenate the (u,v), (u,w), and (v,w) planes; the input itself has no shared-pivot operation. This differs from the fixed encoding dictionary containing all six ordered coordinate lifts.

Each model layer applies affine maps to its branches. Pair branches use matrix multiplication, and triple branches contract over a shared pivot. These operations are followed by the specified signed-root transform, concatenation, affine projection, and BatchNorm. Graph readouts use the diagonal and marginal statistics defined in the code. See `configs/ppgn_protocol.json` for the exact settings.

## Determinism and relabeling

`brec/determinism.py` applies only to BREC and fixes numerical-library thread counts, random seeds, and PyTorch's strict deterministic policy. Agreement across repeated runs with the same input ordering does not imply agreement after relabeling. Floating-point results also do not directly establish exact expressive power. Supplementary BREC separately retains same-graph relabeling and official reliability controls; see the [run guide](rebuttal_experiments.md) for their scope.
