"""Exact ordered-triple features evaluated on weighted automorphism orbits.

Orbit representatives compress evaluation only: neither orbit IDs nor orbit
sizes enter a feature vector. Weights restore all n**3 ordered tuples.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import pynauty

from .common import simple_adjacency
from .encodings import descriptors
from .polynomial_banks import scaled_polynomial_bank
from .runtime import report_phase


@lru_cache(maxsize=16384)
def square_part(value):
    square, radical, divisor = 1, 1, 2
    while divisor*divisor <= value:
        count = 0
        while value % divisor == 0:
            value //= divisor
            count += 1
        square *= divisor**(count//2)
        if count % 2:
            radical *= divisor
        divisor += 1
    return square, radical*value


@dataclass(frozen=True, slots=True)
class Radical:
    """A unique Q-linear combination of square roots of squarefree integers."""
    terms: tuple

    @classmethod
    def make(cls, terms=()):
        combined = {}
        for radical, coefficient in terms:
            if radical < 0:
                raise ValueError('Real square roots required')
            if coefficient and radical:
                square,radical = square_part(int(radical))
                combined[radical] = combined.get(radical, Fraction(0)) + Fraction(coefficient)*square
        return cls(tuple(sorted((radical,value) for radical,value in combined.items() if value)))

    @classmethod
    def integer(cls, value):
        return cls.make(((1,Fraction(int(value))),))

    def json(self):
        return [{'squarefree': radical, 'numerator': str(value.numerator),
                 'denominator': str(value.denominator)} for radical,value in self.terms]

    def __add__(self, other):
        if not isinstance(other, Radical):
            other = Radical.integer(other)
        return self.make(self.terms+other.terms)

    __radd__ = __add__

    def __neg__(self):
        return self.make((radical,-value) for radical,value in self.terms)

    def __sub__(self, other):
        return self + -other

    def scaled(self, weight):
        return self.make((radical,value*weight) for radical,value in self.terms)

    def __mul__(self, other):
        if not isinstance(other, Radical):
            return self.scaled(other)
        return multiply(self,other)

    __rmul__ = __mul__

    def __pow__(self, power):
        return powered(self,int(power))


ZERO, ONE = Radical.integer(0), Radical.integer(1)


@lru_cache(maxsize=32768)
def multiply(first, second):
    terms = []
    for a,x in first.terms:
        for b,y in second.terms:
            common = math.gcd(a,b)
            terms.append((a*b//(common*common),x*y*common))
    return Radical.make(terms)


@lru_cache(maxsize=32768)
def powered(value, exponent):
    if exponent < 0:
        raise ValueError('Nonnegative powers required')
    answer = ONE
    while exponent:
        if exponent & 1:
            answer = multiply(answer,value)
        exponent //= 2
        if exponent:
            value = multiply(value,value)
    return answer


def clear_caches():
    multiply.cache_clear()
    powered.cache_clear()


def ordered_triple_orbits(adjacency):
    adjacency = simple_adjacency(adjacency)
    n = len(adjacency)
    vertices = set(range(n))
    neighbors = {i:np.flatnonzero(adjacency[i]).tolist() for i in range(n)}

    def orbits(fixed):
        fixed = list(dict.fromkeys(fixed))
        rest = vertices-set(fixed)
        coloring = [{i} for i in fixed]+([rest] if rest else [])
        ids = pynauty.autgrp(pynauty.Graph(n, adjacency_dict=neighbors,
                                         vertex_coloring=coloring))[3]
        groups = {}
        for vertex, color in enumerate(ids):
            groups.setdefault(color,[]).append(vertex)
        return [(group[0],len(group)) for group in groups.values()]

    rows, weights = [], []
    for first, first_size in orbits([]):
        report_phase('triple_orbits', fixed_vertex=first, nodes=n)
        second_orbits = orbits([first])
        for second, second_size in second_orbits:
            third_orbits = second_orbits if first == second else orbits([first,second])
            for third, third_size in third_orbits:
                rows.append((first,second,third))
                weights.append(first_size*second_size*third_size)
    if sum(weights) != n**3:
        raise AssertionError('Orbit weights do not cover all ordered triples')
    return np.asarray(rows,dtype='int64'),np.asarray(weights,dtype='int64')


class TripleKernel:
    def __init__(self, adjacency, q, basis, budget=None, *, bank_cache=None, accelerator=None):
        self.adjacency = simple_adjacency(adjacency)
        self.n, self.q, self.basis, self.budget = len(adjacency),q,basis,budget
        if q < 1 or basis not in ('monomial','chebyshev'):
            raise ValueError('Positive Q and a supported basis required')
        # Replacing zero degrees by one only supplies the similarity scaling;
        # the isolated rows of A and every positive matrix power remain zero.
        self.degree = np.maximum(self.adjacency.sum(1),1).tolist()
        self.scale = math.lcm(*self.degree)
        self.groups = {d:np.flatnonzero(np.asarray(self.degree)==d) for d in set(self.degree)}
        self.banks = scaled_polynomial_bank(self.adjacency, self.degree, q, basis, budget,
                                            cache_directory=bank_cache, accelerator=accelerator)
        self.accelerator, self.cuda_joint = accelerator, None
        self.pair_cache, self.h_cache = {}, {}
        self.max_abs = [max(abs(int(value)) for value in bank.flat) for bank in self.banks]
        self.integer_banks = {}
        self.nonzero = [bank!=0 for bank in self.banks]
        self.degree_array = np.asarray(self.degree)

    def pair(self, degree, first, second):
        key = degree,int(first),int(second)
        if key not in self.pair_cache:
            coefficient = int(self.banks[degree][first,second])
            square,radical = square_part(self.degree[first]*self.degree[second])
            self.pair_cache[key] = Radical.make(((radical,Fraction(
                coefficient*square,self.degree[second]*self.scale**degree)),))
        return self.pair_cache[key]

    def h(self, degree, vertex):
        key = degree,int(vertex)
        if key not in self.h_cache:
            terms = []
            for d,indices in self.groups.items():
                count = int(self.banks[degree][vertex,indices].sum())
                square,radical = square_part(self.degree[vertex]*d)
                terms.append((radical,Fraction(count*square,d*self.scale**degree)))
            self.h_cache[key] = Radical.make(terms)
        return self.h_cache[key]

    def value(self, triple, key):
        if key[0]=='constant':
            return ONE
        if key[0]=='pair':
            _,matrix,i,j = key
            first,second = triple[i],triple[j]
            if matrix=='I':
                return ONE if first==second else ZERO
            if matrix=='A':
                return ONE if self.adjacency[first,second] else ZERO
            return self.pair(int(matrix[1:]),first,second)
        if key[0]=='endpoint':
            _,a,b,i,j = key
            return self.h(a,triple[i])*self.h(b,triple[j])
        degrees = key[1:]
        if 0 in degrees:
            pivot = triple[degrees.index(0)]
            result = ONE
            for vertex,degree in zip(triple,degrees):
                if degree==0:
                    if vertex!=pivot:
                        return ZERO
                else:
                    result = result*self.pair(degree,vertex,pivot)
            return result
        support = (self.nonzero[degrees[0]][triple[0]] & self.nonzero[degrees[1]][triple[1]]
                   & self.nonzero[degrees[2]][triple[2]])
        if not support.any():
            return ZERO
        # Omit only exact zeros. Large scaled integers remain Python integers;
        # sparse common supports avoid multiplying dense rows of mostly zeros.
        product = self.banks[degrees[0]][triple[0],support]*self.banks[degrees[1]][triple[1],support]
        product = product*self.banks[degrees[2]][triple[2],support]
        pivot_degrees = self.degree_array[support]
        outer = math.prod(self.degree[vertex] for vertex in triple)
        terms = []
        for d in self.groups:
            count = int(product[pivot_degrees==d].sum())
            if count:
                square,radical = square_part(outer*d)
                terms.append((radical,Fraction(count*square,d*d*self.scale**sum(degrees))))
        return Radical.make(terms)

    def int64_joint_blocks(self, triples, degrees):
        for degree in degrees:
            if degree not in self.integer_banks:
                self.integer_banks[degree] = self.banks[degree].astype('int64')
        rows = np.asarray(triples,dtype='int64')
        for start in range(0,len(rows),256):
            if self.budget:
                self.budget.check()
            block = rows[start:start+256]
            product = self.integer_banks[degrees[0]][block[:,0]]*self.integer_banks[degrees[1]][block[:,1]]
            product *= self.integer_banks[degrees[2]][block[:,2]]
            yield block, np.column_stack([product[:,indices].sum(1) for indices in self.groups.values()])

    def joint_radicals(self, blocks, degrees):
        answer, vocabulary = [], {}
        denominator = self.scale**sum(degrees)
        for rows, counts in blocks:
            for triple, row_counts in zip(rows, counts):
                outer = math.prod(self.degree[vertex] for vertex in triple)
                totals = tuple(map(int, row_counts))
                identity = outer,totals
                if identity not in vocabulary:
                    terms = []
                    for d,count in zip(self.groups,totals):
                        if count:
                            square,radical = square_part(outer*d)
                            terms.append((radical,Fraction(count*square,d*d*denominator)))
                    vocabulary[identity] = Radical.make(terms)
                answer.append(vocabulary[identity])
        return answer

    def cuda_joint_column(self, triples, degrees):
        from .brec_cuda import CudaJointSums
        with self.accelerator.allocation_guard():
            if self.cuda_joint is None:
                self.cuda_joint = CudaJointSums(self.accelerator, self.banks, self.max_abs,
                                                list(self.groups.values()))
            return self.joint_radicals(self.cuda_joint.blocks(triples, degrees, self.budget), degrees)

    def column(self, triples, key):
        if key[0]=='joint' and 0 not in key[1:]:
            degrees = key[1:]
            bound = math.prod(self.max_abs[degree] for degree in degrees)
            if not bound:
                return [ZERO]*len(triples)
            if self.accelerator is not None:
                return self.cuda_joint_column(triples, degrees)
            # All factors have nonzero integer bounds. This also bounds every
            # partial product and every signed degree-group sum, so int64 is
            # an exact arithmetic optimization rather than a numeric tolerance.
            if self.n*bound <= np.iinfo('int64').max:
                return self.joint_radicals(self.int64_joint_blocks(triples, degrees), degrees)
        return [self.value(triple,key) for triple in triples]

    def mean_sum(self, key):
        """Exact sum over all ordered triples, without expanding their table."""
        if key[0]=='constant':
            return Radical.integer(self.n**3)
        if key[0]=='pair':
            matrix = key[1]
            if matrix=='I':
                return Radical.integer(self.n**2)
            if matrix=='A':
                return Radical.integer(int(self.adjacency.sum())*self.n)
            return sum((self.h(int(matrix[1:]),vertex) for vertex in range(self.n)),ZERO).scaled(self.n)
        if key[0]=='endpoint':
            _,a,b,_,_ = key
            first = sum((self.h(a,vertex) for vertex in range(self.n)),ZERO)
            second = sum((self.h(b,vertex) for vertex in range(self.n)),ZERO)
            return (first*second).scaled(self.n)
        a,b,c = key[1:]
        return sum((self.h(a,vertex)*self.h(b,vertex)*self.h(c,vertex)
                    for vertex in range(self.n)),ZERO)


def weighted_counts(values, weights):
    counts = Counter()
    for value, weight in zip(values,weights):
        counts[value] += int(weight)
    return counts


def weighted_moment(values, weights, power):
    counts = weighted_counts(values,weights)
    return moment_from_counts(counts,power)


def moment_from_counts(counts, power):
    return sum((powered(value,power).scaled(weight) for value,weight in counts.items()),ZERO)
