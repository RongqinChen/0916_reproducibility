"""Compatibility module: the supported experiment is input-versus-trained PPGN.

Run `python -m brec.ppgn_realization --help` for the new arguments.
BN-only, independent pivots and input/basis ablations have been retired.
"""
from .ppgn_realization import main

if __name__ == '__main__':
    main()
