"""Run one order of the BREC float32 experiment."""

from argparse import ArgumentParser
from pathlib import Path

from brec.suite_resources import enforce_address_space_limit
from rebuttal_experiments.brec_data import load_brec_data

from .protocol import EXPECTED_SHA256, TEST_PAIRS


def add_order_arguments(parser, order):
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    if order in (2, 3):
        parser.add_argument('--workers', default=1, type=int)
    if order > 2:
        parser.add_argument('--lower-order-out', type=Path, help=f'verify completed {order-1}-FWL results')
    if order == 4:
        parser.add_argument('--ids', nargs='+', type=int, help='subset of the exact 4-FWL target pairs')
        parser.add_argument('--memory-gib', type=float, default=58)
        parser.add_argument('--devices', nargs=2, type=int, metavar=('LEFT', 'RIGHT'))


def main(argv=None, *, order=None):
    parser = ArgumentParser(description=__doc__)
    if order is None:
        commands = parser.add_subparsers(dest='order', required=True)
        for k in TEST_PAIRS:
            add_order_arguments(commands.add_parser(str(k), help=f'run {k}-FWL'), k)
    else:
        add_order_arguments(parser, order)
    args = parser.parse_args(argv)
    order = int(args.order) if order is None else order
    if order in (2, 3) and args.workers < 1:
        parser.error('--workers must be positive')
    if order == 4:
        ids = TEST_PAIRS[4] if args.ids is None else set(args.ids)
        if not ids <= TEST_PAIRS[4]:
            parser.error('--ids must belong to the 4-FWL-positive cohort')
        if args.memory_gib <= 0:
            parser.error('--memory-gib must be positive')
        if args.devices and len(set(args.devices)) != 2:
            parser.error('--devices requires two distinct GPU IDs')
        enforce_address_space_limit(args.memory_gib)
    data = load_brec_data(args.brec_file, EXPECTED_SHA256)
    if order == 2:
        from . import pair

        pair.run(data, args.out, args.workers)
    elif order == 3:
        from . import triple

        triple.run(data, args.out, args.workers, args.lower_order_out)
    else:
        from . import quadruple

        quadruple.run(data, args.out, ids, args.memory_gib, args.devices, args.lower_order_out)


if __name__ == '__main__':
    main()
