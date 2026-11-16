"""Exact normalized polynomial features over rational square-root combinations.

Let R = D^-1 A with isolates assigned scaling degree one. Then
S_q(M)[u,v] = sqrt(d_u*d_v) S_q(R)[u,v] / d_v.
FLINT computes S_q(R) over Q. All remaining features are finite Q-linear
combinations of square roots of squarefree integers, with canonical equality.
There are no tolerances or floating-point hashes in this module.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

from flint import ctx, fmpq, fmpq_mat
import numpy as np

from .suite_data import adjacency


@lru_cache(maxsize=16384)
def squarefree(value):
    outside, inside, prime = 1, 1, 2
    while prime * prime <= value:
        exponent = 0
        while value % prime == 0:
            value //= prime
            exponent += 1
        outside *= prime**(exponent//2)
        if exponent % 2:
            inside *= prime
        prime += 1
    return outside, inside * value


@dataclass(frozen=True, slots=True)
class Radical:
    terms: tuple = ()

    @classmethod
    def make(cls, terms):
        result = {}
        for radicand, coefficient in terms:
            if radicand < 1:
                raise ValueError("Positive integer radicands are required")
            outside, inside = squarefree(int(radicand))
            result[inside] = result.get(inside, Fraction(0)) + Fraction(coefficient)*outside
        return cls(tuple(sorted((r, c) for r, c in result.items() if c)))

    @classmethod
    def rational(cls, value):
        value = Fraction(value)
        return cls(((1, value),)) if value else cls()

    def __add__(self, other):
        if not isinstance(other, Radical):
            other = Radical.rational(other)
        return Radical.make(self.terms + other.terms)

    __radd__ = __add__

    def scaled(self, scalar):
        return Radical(tuple((r, c*scalar) for r, c in self.terms if c*scalar))

    def __neg__(self):
        return self.scaled(-1)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if not isinstance(other, Radical):
            return self.scaled(Fraction(other))
        terms = []
        for a, x in self.terms:
            for b, y in other.terms:
                common = math.gcd(a, b)
                terms.append((a*b//common**2, x*y*common))
        return Radical.make(terms)

    __rmul__ = __mul__

    def __pow__(self, power):
        if type(power) is not int or power < 0:
            raise ValueError("A nonnegative integer power is required")
        value, result = self, ONE
        while power:
            if power & 1:
                result = result * value
            power //= 2
            if power:
                value = value * value
        return result

    def json(self):
        return [[r, str(c.numerator), str(c.denominator)] for r, c in self.terms]

    def __float__(self):
        return math.fsum(float(c)*math.sqrt(r) for r, c in self.terms)


ZERO, ONE = Radical(), Radical.rational(1)


def descriptors(q, order, variant):
    if order not in (2, 3) or q < 1 or variant not in ("poly", "endpoint", "joint", "equal19"):
        raise ValueError("Invalid dictionary")
    if order == 2 and variant in ("joint", "equal19"):
        raise ValueError("Joint dictionaries require triples")
    positions = [(0, 1)] if order == 2 else [(i, j) for i in range(3) for j in range(3) if i != j]
    # Match the manuscript's pair channel order I,J,A,S1,... .
    keys = [("pair", "I", 0, 1), ("constant",)] if order == 2 else [("constant",)]
    matrices = ["A", *range(1, q+1)] if order == 2 else ["I", "A", *range(1, q+1)]
    keys += [("pair", name, i, j) for name in matrices for i, j in positions]
    if variant != "poly":
        keys += [("endpoint", a, b, i, j) for i, j in positions
                 for a in range(q) for b in range(q-a) if a+b > 0]
    if variant == "joint":
        keys += [("joint", a, b, c) for a in range(q+1) for b in range(q+1) for c in range(q+1)]
    if variant == "equal19":
        if q != 1:
            raise ValueError("The equal-width dictionary is defined only at Q=1")
        keys[keys.index(("pair", "I", 1, 0))] = ("joint", 1, 1, 1)
    return keys


class Kernel:
    def __init__(self, graph, q, basis="monomial"):
        if basis not in ("monomial", "chebyshev", "shifted") or (basis == "shifted" and q != 1):
            raise ValueError("Unsupported basis or shifted-basis degree")
        self.a, self.q, self.basis = adjacency(graph), q, basis
        ctx.threads = 1
        self.n = len(self.a)
        self.d = [max(1, int(x)) for x in self.a.sum(1)]
        identity = fmpq_mat([[int(u == v) for v in range(self.n)] for u in range(self.n)])
        transition = fmpq_mat([[fmpq(int(self.a[u, v]), self.d[u])
                               for v in range(self.n)] for u in range(self.n)])
        rational = [identity, identity+transition if basis == "shifted" else transition]
        for _ in range(2, q+1):
            product = transition * rational[-1]
            rational.append(product if basis == "monomial" else 2*product-rational[-2])
        self.bank = []
        for matrix in rational:
            self.bank.append([[Fraction(int(matrix[u, v].numerator), int(matrix[u, v].denominator))
                               for v in range(self.n)] for u in range(self.n)])
        # S_q(R) has denominator dividing L^q, including the Chebyshev recurrence.
        # Integer products avoid repeated Fraction normalization inside contractions.
        self.scale = math.lcm(*self.d)
        self.scaled = []
        for degree, matrix in enumerate(self.bank):
            denominator = self.scale**degree
            values = [[value*denominator for value in row] for row in matrix]
            if any(value.denominator != 1 for row in values for value in row):
                raise AssertionError("Unexpected polynomial denominator")
            self.scaled.append(np.asarray([[int(value) for value in row] for row in values], dtype=object))
        self.degree_groups = {d: np.flatnonzero(np.asarray(self.d) == d) for d in sorted(set(self.d))}
        self.max_abs = [max(abs(int(value)) for value in matrix.flat) for matrix in self.scaled]
        self.int64_bank = {}
        self._pair, self._h = {}, {}

    def pair(self, q, u, v):
        u, v = int(u), int(v)
        key = q, u, v
        if key not in self._pair:
            self._pair[key] = Radical.make([(self.d[u]*self.d[v], self.bank[q][u][v]/self.d[v])])
        return self._pair[key]

    def h(self, q, u):
        key = q, int(u)
        if key not in self._h:
            self._h[key] = Radical.make([(self.d[int(u)]*d,
                Fraction(int(self.scaled[q][int(u), indices].sum()), d*self.scale**q))
                for d, indices in self.degree_groups.items()])
        return self._h[key]

    def value(self, row, key):
        if key[0] == "constant":
            return ONE
        if key[0] == "pair":
            _, name, i, j = key
            u, v = map(int, (row[i], row[j]))
            if name == "I":
                return ONE if u == v else ZERO
            if name == "A":
                return ONE if self.a[u, v] else ZERO
            return self.pair(name, u, v)
        if key[0] == "endpoint":
            _, a, b, i, j = key
            return self.h(a, row[i]) * self.h(b, row[j])
        degrees = key[1:]
        if 0 in degrees:
            pivot = int(row[degrees.index(0)])
            answer = ONE
            for degree, vertex in zip(degrees, row):
                if degree == 0:
                    if int(vertex) != pivot:
                        return ZERO
                else:
                    answer = answer * self.pair(degree, vertex, pivot)
            return answer
        u, v, w = map(int, row)
        a, b, c = degrees
        products = self.scaled[a][u]*self.scaled[b][v]*self.scaled[c][w]
        outer = self.d[u]*self.d[v]*self.d[w]
        denominator = self.scale**sum(degrees)
        return Radical.make([(outer*d, Fraction(int(products[indices].sum()), d*d*denominator))
                             for d, indices in self.degree_groups.items()])

    def column(self, rows, key):
        if key[0] == "joint" and 0 not in key[1:]:
            a, b, c = key[1:]
            bound = self.n*self.max_abs[a]*self.max_abs[b]*self.max_abs[c]
            if bound == 0:
                return [ZERO]*len(rows)
            if bound <= np.iinfo(np.int64).max:
                for degree in (a, b, c):
                    if degree not in self.int64_bank:
                        self.int64_bank[degree] = self.scaled[degree].astype(np.int64)
                matrices = self.int64_bank
            else:
                matrices = self.scaled
            answer, vocabulary = [], {}
            denominator = self.scale**(a+b+c)
            for start in range(0, len(rows), 256):
                block = np.asarray(rows[start:start+256], dtype=np.int64)
                products = matrices[a][block[:, 0]]*matrices[b][block[:, 1]]*matrices[c][block[:, 2]]
                totals = [products[:, indices].sum(1) for indices in self.degree_groups.values()]
                for index, row in enumerate(block):
                    outer = math.prod(self.d[int(v)] for v in row)
                    identity = outer, tuple(int(total[index]) for total in totals)
                    if identity not in vocabulary:
                        vocabulary[identity] = Radical.make([(outer*d, Fraction(total, d*d*denominator))
                            for d, total in zip(self.degree_groups, identity[1])])
                    answer.append(vocabulary[identity])
            return answer
        return [self.value(row, key) for row in rows]


def counts(values, weights):
    result = Counter()
    for value, weight in zip(values, weights, strict=True):
        result[value] += int(weight)
    return result


def moment(histogram, power, denominator):
    return sum(((value**power).scaled(Fraction(weight, denominator))
                for value, weight in histogram.items()), ZERO)


def _compare_prefixes(kernels, domains, q, order, keys_by_variant, powers):
    """Refine row colors jointly across graphs; never materialize the full table."""
    keys = list(keys_by_variant.values())[-1]
    if any(value != keys[:len(value)] for value in keys_by_variant.values()):
        raise ValueError("Shared comparison requires nested channel prefixes")
    rows, weights = zip(*domains)
    totals = [sum(map(int, w)) for w in weights]
    if totals[0] != totals[1]:
        raise ValueError("This paired protocol requires equal domain sizes")
    colors = [[0]*len(r) for r in rows]
    raw_witness, differences = None, {}
    snapshots = {}

    def snapshot(variant):
        result = {"multiset": {"status": "different" if raw_witness else "equal", "witness": raw_witness}}
        for power in powers:
            result[f"M{power}"] = {"status": "different" if power in differences else "equal",
                                    "witness": differences.get(power)}
        if differences and raw_witness is None:
            raise AssertionError("A moment distinguishes equal full row multisets")
        snapshots[variant] = {"q": q, "order": order, "variant": variant, "basis": kernels[0].basis,
            "width": len(keys_by_variant[variant]), "objects": totals, "orbit_rows": list(map(len, rows)),
            "decisions": result}
    for column, key in enumerate(keys):
        needed = [p for p in range(1, max(powers)+1)
                  if any(limit >= p and limit not in differences for limit in powers)]
        if raw_witness is not None and not needed:
            break
        values = [kernel.column(r, key) for kernel, r in zip(kernels, rows)]
        if raw_witness is None:
            vocabulary = {}
            colors = [[vocabulary.setdefault((old, value), len(vocabulary))
                       for old, value in zip(side, vals)] for side, vals in zip(colors, values)]
            histograms = [counts(c, w) for c, w in zip(colors, weights)]
            if histograms[0] != histograms[1]:
                witness = next(color for h in histograms for color in h
                               if histograms[0][color] != histograms[1][color])
                side = 0 if witness in histograms[0] else 1
                row = rows[side][colors[side].index(witness)]
                raw_witness = {"kind": "exact_row_projection", "through_channel": column,
                    "row": list(map(int, row)), "graph_side": side,
                    "values": [kernels[side].value(row, k).json() for k in keys[:column+1]],
                    "multiplicities": [h[witness] for h in histograms]}
        marginals = [counts(v, w) for v, w in zip(values, weights)] if needed else []
        if marginals and marginals[0] == marginals[1]:
            # Identical exact marginal distributions have identical moments at
            # every order. Avoid recomputing expensive radical powers for them.
            needed = []
        for power in needed:
            if not any(limit >= power and limit not in differences for limit in powers):
                continue
            a, b = [moment(h, power, total) for h, total in zip(marginals, totals)]
            if a != b:
                witness = {"kind": "exact_moment_difference", "descriptor": list(key),
                    "channel": column, "power": power, "left": a.json(), "right": b.json(),
                    "difference": (a-b).json(), "domain_size": totals[0]}
                for limit in powers:
                    if limit >= power and limit not in differences:
                        differences[limit] = witness
        for variant, prefix in keys_by_variant.items():
            if len(prefix) == column+1:
                snapshot(variant)
    # Early termination requires a witness for every readout; it remains valid
    # in each larger prefix because all those channels are retained.
    for variant in keys_by_variant:
        if variant not in snapshots:
            snapshot(variant)
    return snapshots


def compare(kernels, domains, q, order, variant, powers=(1, 2, 4), keys_override=None):
    keys = descriptors(q, order, variant) if keys_override is None else keys_override
    return _compare_prefixes(kernels, domains, q, order, {variant: keys}, powers)[variant]


def compare_nested(kernels, domains, q, order, variants, powers=(1, 2, 4)):
    """Evaluate polynomial/endpoint/joint prefixes in one streaming pass."""
    keys = {variant: descriptors(q, order, variant) for variant in variants}
    return list(_compare_prefixes(kernels, domains, q, order, keys, powers).values())
