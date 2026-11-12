"""Real intervals on a fixed dyadic grid, with exact outward rounding.

An interval (lo, hi, bits) denotes [lo/2**bits, hi/2**bits]. Python integers
do not overflow. Each primitive encloses its exact real result. This small
reference backend favors inspectable error bounds over BLAS performance.
It is not a wrapper around already-rounded NumPy features.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from operator import index


def ceildiv(n, d):
    if d <= 0:
        raise ValueError('Positive denominator required')
    return -((-n) // d)


@dataclass(frozen=True, slots=True)
class Interval:
    lo: int
    hi: int
    bits: int

    def __post_init__(self):
        if self.lo > self.hi or self.bits < 2:
            raise ValueError('Invalid interval')

    @classmethod
    def exact(cls, value, bits):
        try:
            value = int(index(value))
        except TypeError:
            pass
        value = Fraction(value)
        numerator = int(value.numerator) << bits
        denominator = int(value.denominator)
        return cls(numerator // denominator, ceildiv(numerator, denominator), bits)

    @classmethod
    def sqrt_integer(cls, value, bits):
        if value < 0:
            raise ValueError('Negative square root')
        scaled = int(value) << (2 * bits)
        low = isqrt(scaled)
        return cls(low, low if low * low == scaled else low + 1, bits)

    def coerce(self, other):
        if isinstance(other, Interval):
            if other.bits != self.bits:
                raise ValueError('Mixed interval precisions')
            return other
        return Interval.exact(other, self.bits)

    def __add__(self, other):
        other = self.coerce(other)
        return Interval(self.lo + other.lo, self.hi + other.hi, self.bits)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo, self.bits)

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        products = (self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi)
        scale = 1 << self.bits
        return Interval(min(products) // scale, ceildiv(max(products), scale), self.bits)

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError('Interval contains zero')
        candidates = (Fraction(1 << (2 * self.bits), self.lo),
                      Fraction(1 << (2 * self.bits), self.hi))
        low, high = min(candidates), max(candidates)
        return Interval(low.numerator // low.denominator,
                        ceildiv(high.numerator, high.denominator), self.bits)

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def __pow__(self, power):
        try:
            power = int(index(power))
        except TypeError as exc:
            raise ValueError('Nonnegative integer powers only') from exc
        if power < 0:
            raise ValueError('Nonnegative integer powers only')
        result, base = self.exact(1, self.bits), self
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power //= 2
        return result

    def overlaps(self, other):
        other = self.coerce(other)
        return self.lo <= other.hi and other.lo <= self.hi

    def contains_zero(self):
        return self.lo <= 0 <= self.hi

    def is_exact(self):
        return self.lo == self.hi

    def lower(self):
        return Fraction(self.lo, 1 << self.bits)

    def upper(self):
        return Fraction(self.hi, 1 << self.bits)

    def midpoint(self):
        return Fraction(self.lo + self.hi, 1 << (self.bits + 1))

    def json(self):
        return {'lo_integer': str(self.lo), 'hi_integer': str(self.hi),
                'denominator_bits': self.bits}


def number_json(value):
    if isinstance(value, Interval):
        return value.json()
    value = Fraction(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
