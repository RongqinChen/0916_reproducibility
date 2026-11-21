"""Command-line entry point for the 2-FWL float32 experiment."""

from brec.float32.__main__ import main as run_brec_order


def main(argv=None):
    return run_brec_order(argv, order=2)


if __name__ == '__main__':
    main()
