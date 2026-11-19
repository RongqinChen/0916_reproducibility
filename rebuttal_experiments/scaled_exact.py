"""Shared-denominator integer arrays for exact BREC computations."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math

import numpy as np

from .intervals import number_json


class ScaledIntegerArray(np.ndarray):
    """An object ndarray of integer numerators with one scalar denominator."""

    denominator: int

    def __new__(cls, values, denominator):
        if type(denominator) is not int or denominator <= 0:
            raise ValueError('Positive integer denominator required')
        result = np.asarray(values, dtype=object).view(cls)
        result.denominator = denominator
        return result

    def __array_finalize__(self, source):
        if source is not None:
            self.denominator = getattr(source, 'denominator', None)


def common_denominator(*values):
    denominators = [getattr(value, 'denominator', None) for value in values]
    if any(denominator is None for denominator in denominators):
        return None
    return math.lcm(*denominators)


def scaled_array_equal(left, right):
    """Compare exact scaled arrays, allowing different denominators."""
    denominator = common_denominator(left, right)
    if denominator is None:
        return np.array_equal(left, right)
    left_scale = denominator // left.denominator
    right_scale = denominator // right.denominator
    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)
    if left_scale != 1:
        left_values = left_values * left_scale
    if right_scale != 1:
        right_values = right_values * right_scale
    return np.array_equal(left_values, right_values)


def scaled_counter(values, denominator):
    """Counter of scaled integer rows with all values on one denominator."""
    scale = denominator // values.denominator
    if scale == 1:
        return Counter(map(tuple, values))
    return Counter(
        tuple(int(value) * scale for value in row) for row in values)


def scaled_row_json(row, denominator):
    return [number_json(Fraction(int(value), denominator)) for value in row]


def scaled_mean_decision(moment_pairs, columns):
    """Compare two exact mean vectors without materializing Fractions per value."""
    if isinstance(columns, slice):
        first = moment_pairs[0][0]
        columns = range(*columns.indices(len(first)))
    left_values = []
    right_values = []
    for left, right in moment_pairs:
        for channel in columns:
            denominator = common_denominator(left, right)
            if denominator is None:
                raise TypeError('Scaled moments require integer denominators')
            left_value = int(left[channel])
            right_value = int(right[channel])
            left_scaled = left_value * (denominator // left.denominator)
            right_scaled = right_value * (denominator // right.denominator)
            if left_scaled != right_scaled:
                left_fraction = Fraction(left_value, left.denominator)
                right_fraction = Fraction(right_value, right.denominator)
                return {
                    'status': 'certified_different',
                    'evidence': {
                        'kind': 'mean_difference',
                        'channel': len(left_values),
                        'left': number_json(left_fraction),
                        'right': number_json(right_fraction),
                        'difference': number_json(left_fraction - right_fraction),
                    },
                }
            left_values.append(Fraction(left_value, left.denominator))
            right_values.append(Fraction(right_value, right.denominator))
    return {
        'status': 'certified_equal',
        'evidence': {
            'kind': 'exact_mean',
            'values': [number_json(value) for value in left_values],
        },
    }
