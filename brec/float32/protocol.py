"""BREC v3 reference cohorts and common experimental settings."""

from brec.suite_data import EXPECTED_SHA256
from rebuttal_experiments.brec_data import PAIR_COUNT


TOLERANCE = 1e-8
ALL_PAIRS = frozenset(range(PAIR_COUNT))
# These frozen labels are tied to the SHA-256 of the official graph6 array.
TARGETS = {
    2: frozenset((*range(110), *range(160, 320))),
    3: frozenset((*range(110, 160), *range(320, 340), *range(360, 400))),
    4: frozenset(range(340, 360)) - {345, 356},
}

WALK_DEGREE = 24
ROUNDS = 6
BRANCH_DEGREES = (769, 997, 1237, 1597)
PIVOT_FRACTION_BITS = 39

TEST_PAIRS = {2: ALL_PAIRS, 3: ALL_PAIRS - TARGETS[2], 4: TARGETS[4]}


def describe_protocol(**settings):
    return {
        'dataset_sha256': EXPECTED_SHA256,
        'dtype': 'float32',
        'threshold': TOLERANCE,
        'walk_degree': WALK_DEGREE,
        'rounds': ROUNDS,
        **settings,
    }
