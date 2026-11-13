# Third-party code and attribution

TU encoding comparisons execute the unmodified
[Co-Sparsify](https://github.com/RongqinChen/Co-Sparsify) checkout at commit
`505f6b50f019dc78edc88653d771667c0c561f5f`. The checkout is an external
dependency, not redistributed as part of this repository. The adapter replaces
only the `poly` transform registry entry and adds local result capture; the
upstream dataset, SPPGN1 model, training loop and evaluators remain unchanged.
The original environment and any upstream license terms apply.

`gnnplus_official/` is a locally modified copy of
[GNNPlus](https://github.com/LUOyk1999/GNNPlus), accompanying
*Can Classic GNNs Be Strong Baselines for Graph-level Tasks?* (ICML 2025).
The upstream MIT license is preserved in `gnnplus_official/LICENSE`.
Local additions include structural encoders, normalization/width controls,
dataset compatibility, prediction export, and output handling. This is not an
unmodified upstream release. GNNPlus itself builds on GraphGPS; existing source
attributions are retained.

The PPGN model is the implementation used by this study's archived campaign,
following the CSDGSE reference components named in its source docstring and the
PPGN regular-block construction. BREC data and benchmark conventions are from
the [official BREC project](https://github.com/GraphPKU/BREC).

No new blanket license has been assigned to original research code in this
packaging step. Third-party license notices remain applicable to their files.
This folder is prepared for sharing and inspection; a project-wide redistribution
license for original code must be selected by its copyright holders.
