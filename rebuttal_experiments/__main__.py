"""Explicit supplementary commands; no action runs on import."""
from __future__ import annotations

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('experiment', choices=['mechanisms', 'brec', 'objects', 'protein'])
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    if args.experiment == 'brec':
        from brec.determinism import ensure_startup
        ensure_startup()
    module = importlib.import_module('rebuttal_experiments.' + args.experiment)
    module.main(remaining)


if __name__ == '__main__':
    main()
