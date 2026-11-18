"""CUDA products for BREC's exact integers and outward-rounded intervals."""
from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from functools import lru_cache

import numpy as np

from .common import ResourceLimit
from .intervals import Interval


_WORDS = r"""
using Word = unsigned int;
using Wide = unsigned long long;

__device__ void multiply_words(const Word* a, int a_words, const Word* b, int b_words,
                               Word* product)
{
    for (int k = 0; k < a_words + b_words; ++k)
        product[k] = 0;
    // The two carry terms and the 32-by-32 product fit in one 64-bit word.
    for (int u = 0; u < a_words; ++u) {
        Wide carry = 0;
        for (int v = 0; v < b_words; ++v) {
            const Wide digit = Wide(a[u]) * b[v] + product[u + v] + carry;
            product[u + v] = Word(digit);
            carry = digit >> 32;
        }
        product[u + b_words] = Word(carry);
    }
}
"""

# Each thread computes one endpoint of one output entry. A left factor is
# nonnegative, so the sign of the right endpoint determines which left
# endpoint to multiply. Round each product before summing, as Interval does.
_PRODUCT = _WORDS + r"""
extern "C" __global__ void sparse_product(
    const int* offsets, const int* neighbors, const Word* left,
    const Word* right, const unsigned char* negative, Word* result,
    int n, int bits, int left_words, int right_words, int output_words, int bounds)
{
    const Wide cell = Wide(blockIdx.x) * blockDim.x + threadIdx.x;
    const Wide plane = Wide(n) * n;
    if (cell >= bounds * plane)
        return;
    const int upper = cell / plane;
    const int i = (cell % plane) / n;
    const int j = cell % n;
    const int edges = offsets[n];
    const int product_words = left_words + right_words;
    const int skipped_words = bits / 32;
    const int skipped_bits = bits % 32;
    Word* sum = result + cell * output_words;
    for (int k = 0; k < output_words; ++k)
        sum[k] = 0;

    for (int edge = offsets[i]; edge < offsets[i + 1]; ++edge) {
        const Wide rhs = upper * plane + Wide(neighbors[edge]) * n + j;
        const bool minus = negative[rhs];
        const int left_bound = bounds == 2 && (upper != minus);
        const Word* a = left + (Wide(left_bound) * edges + edge) * left_words;
        const Word* b = right + rhs * right_words;
        Word product[PRODUCT_WORDS];
        multiply_words(a, left_words, b, right_words, product);

        bool round_up = false;
        if (upper != minus) {
            for (int k = 0; k < skipped_words && k < product_words; ++k)
                round_up |= product[k] != 0;
            if (skipped_bits && skipped_words < product_words)
                round_up |= (product[skipped_words] & ((Word(1) << skipped_bits) - 1)) != 0;
        }

        // Add in two's complement. The host has reserved a sign bit and
        // bounded the sum of absolute values, including outward rounding.
        Wide carry = minus;
        Wide rounding = round_up;
        for (int k = 0; k < output_words; ++k) {
            const int at = k + skipped_words;
            Wide digit = at < product_words ? product[at] : 0;
            if (at + 1 < product_words)
                digit |= Wide(product[at + 1]) << 32;
            rounding += Word(digit >> skipped_bits);
            const Word magnitude = Word(rounding);
            rounding >>= 32;
            const Wide total = Wide(sum[k]) + (minus ? ~magnitude : magnitude) + carry;
            sum[k] = Word(total);
            carry = total >> 32;
        }
    }
}
"""


_JOINT = _WORDS + r"""
extern "C" __global__ void joint_int64(
    const int* triples, const int* pivots, const int* offsets, const long long* a,
    const long long* b, const long long* c, long long* sums, int n, int groups, int count)
{
    const Wide index = Wide(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= Wide(count) * groups)
        return;
    const int* row = triples + 3 * (index / groups);
    const int group = index % groups;
    const long long* first = a + Wide(row[0]) * n;
    const long long* second = b + Wide(row[1]) * n;
    const long long* third = c + Wide(row[2]) * n;
    long long sum = 0;
    for (int k = offsets[group]; k < offsets[group + 1]; ++k) {
        const int t = pivots[k];
        sum += first[t] * second[t] * third[t];
    }
    sums[index] = sum;
}

extern "C" __global__ void joint_words(
    const int* triples, const int* pivots, const int* offsets, const Word* a,
    const Word* b, const Word* c, const signed char* sa, const signed char* sb,
    const signed char* sc, Word* sums, int n, int groups, int count, int wa, int wb,
    int wc, int output_words)
{
    const Wide index = Wide(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= Wide(count) * groups)
        return;
    const int* row = triples + 3 * (index / groups);
    const int group = index % groups;
    const Wide first = Wide(row[0]) * n;
    const Wide second = Wide(row[1]) * n;
    const Wide third = Wide(row[2]) * n;
    Word* sum = sums + index * output_words;
    for (int j = 0; j < output_words; ++j)
        sum[j] = 0;

    for (int k = offsets[group]; k < offsets[group + 1]; ++k) {
        const int t = pivots[k];
        const int sign = int(sa[first+t]) * sb[second+t] * sc[third+t];
        if (!sign)
            continue;
        Word pair[PRODUCT_WORDS], triple[PRODUCT_WORDS];
        multiply_words(a+(first+t)*wa, wa, b+(second+t)*wb, wb, pair);
        multiply_words(pair, wa+wb, c+(third+t)*wc, wc, triple);

        // Host bounds reserve a sign bit for the entire sum of magnitudes.
        Wide carry = sign < 0;
        for (int j = 0; j < output_words; ++j) {
            const Word digit = j < wa+wb+wc ? triple[j] : 0;
            const Wide total = Wide(sum[j]) + (sign < 0 ? ~digit : digit) + carry;
            sum[j] = Word(total);
            carry = total >> 32;
        }
    }
}
"""


def _word_count(bound):
    return max(1, (bound.bit_length() + 31) // 32)


def _pack(values, words):
    return np.frombuffer(
        b''.join(abs(int(value)).to_bytes(4 * words, 'little') for value in values),
        dtype='<u4')


def _decode_signed_words(words, shape):
    values = (int.from_bytes(row, 'little', signed=True) for row in words)
    return np.fromiter(values, dtype=object, count=math.prod(shape)).reshape(shape)


@lru_cache(maxsize=None)
def _kernel(capacity, name='sparse_product'):
    import cupy as cp
    source = _PRODUCT if name == 'sparse_product' else _JOINT
    return cp.RawKernel(source, name, options=(f'-DPRODUCT_WORDS={capacity}',))


class CudaProducts:
    def __init__(self, device):
        import cupy as cp
        self.cp = cp
        self.device = cp.cuda.Device(int(device.split(':')[1]))
        self.device.use()

    @contextmanager
    def allocation_guard(self):
        try:
            yield
        except self.cp.cuda.memory.OutOfMemoryError as exc:
            raise ResourceLimit('cuda_out_of_memory') from exc

    def _prepare(self, left, bits):
        cp = self.cp
        n = left.shape[1]
        bounds = len(left)
        rows, columns = np.nonzero(left[-1])
        offsets = np.concatenate(([0], np.bincount(rows, minlength=n).cumsum()))
        left_values = left[:, rows, columns].ravel()
        left_max = int(max(left_values, default=0))
        left_words = _word_count(left_max)
        max_degree = int(np.diff(offsets).max())
        with self.allocation_guard():
            csr_offsets = cp.asarray(offsets, dtype=cp.int32)
            csr_columns = cp.asarray(columns, dtype=cp.int32)
            weights = cp.asarray(_pack(left_values, left_words))
        # The factors are fixed throughout this polynomial bank.
        scale = 1 << bits

        def multiply(right):
            right_values = list(map(int, right.flat))
            right_max = max(map(abs, right_values))
            right_words = _word_count(right_max)
            # Bound even the absolute sum of the individually rounded terms.
            bound = max_degree * ((left_max * right_max + scale - 1) // scale)
            output_words = _word_count(2 * bound)
            capacity = 1 << (left_words + right_words - 1).bit_length()
            with self.allocation_guard():
                values = cp.asarray(_pack(right_values, right_words))
                signs = cp.asarray(np.fromiter((x < 0 for x in right_values), dtype=np.uint8))
                output = cp.empty((len(right_values), output_words), dtype=cp.uint32)
                kernel = _kernel(capacity)
                kernel(((len(right_values) + 127) // 128,), (128,),
                       (csr_offsets, csr_columns, weights, values, signs, output,
                        *map(np.int32, (n, bits, left_words, right_words, output_words, bounds))))
                # The blocking transfer surfaces asynchronous CUDA errors.
                result = output.get()
            return _decode_signed_words(result, right.shape)

        return multiply

    def integer_product(self, adjacency):
        product = self._prepare(adjacency[None], 0)
        return lambda right: product(right[None])[0]

    def interval_product(self, matrix, bits):
        def endpoints(values):
            return np.stack([np.fromiter((getattr(x, attr) for x in values.flat),
                                         dtype=object, count=values.size).reshape(values.shape)
                             for attr in ('lo', 'hi')])

        product = self._prepare(endpoints(matrix), bits)

        def multiply(right):
            lo, hi = product(endpoints(right))
            return np.fromiter((Interval(a, b, bits) for a, b in zip(lo.flat, hi.flat)),
                               dtype=object, count=right.size).reshape(right.shape)

        return multiply


class CudaJointSums:
    """Exact grouped products over the common pivot for streamed tuple rows."""

    def __init__(self, products, banks, max_abs, groups):
        self.cp = cp = products.cp
        self.banks, self.max_abs = banks, max_abs
        self.n = len(banks[0])
        self.group_count = len(groups)
        self.pivots = cp.asarray(np.concatenate(groups), dtype=cp.int32)
        lengths = [len(indices) for indices in groups]
        self.offsets = cp.asarray(np.cumsum([0, *lengths]), dtype=cp.int32)
        self.max_group = max(lengths)
        self.int64_banks = {}
        self.wide_banks = {}
        self.tuple_source = None

    def _int64_bank(self, degree):
        if degree not in self.int64_banks:
            self.int64_banks[degree] = self.cp.asarray(self.banks[degree].astype('int64'))
        return self.int64_banks[degree]

    def _wide_bank(self, degree):
        if degree not in self.wide_banks:
            values = list(map(int, self.banks[degree].flat))
            width = _word_count(self.max_abs[degree])
            signs = np.fromiter(((x > 0) - (x < 0) for x in values), dtype=np.int8)
            self.wide_banks[degree] = self.cp.asarray(_pack(values, width)), self.cp.asarray(signs), width
        return self.wide_banks[degree]

    def blocks(self, triples, degrees, budget=None):
        cp = self.cp
        if budget:
            budget.check()
        if self.tuple_source is not triples:
            self.rows = np.asarray(triples, dtype='int32').reshape(-1, 3)
            self.device_rows = cp.asarray(self.rows)
            self.tuple_source = triples
        # n bounds individual partial products even if a degree group is small.
        bound = math.prod(self.max_abs[degree] for degree in degrees)
        narrow = self.n * bound <= np.iinfo(np.int64).max
        if narrow:
            kernel = _kernel(1, 'joint_int64')
            inputs = tuple(self._int64_bank(degree) for degree in degrees)
        else:
            magnitudes, signs, widths = zip(*(self._wide_bank(degree) for degree in degrees))
            output_words = _word_count(2 * self.max_group * bound)
            capacity = 1 << (sum(widths) - 1).bit_length()
            kernel = _kernel(capacity, 'joint_words')
            inputs = (*magnitudes, *signs)
        for start in range(0, len(self.rows), 2048):
            if budget:
                budget.check()
            end = min(start + 2048, len(self.rows))
            shape = end - start, self.group_count
            parameters = tuple(map(np.int32, (self.n, self.group_count, shape[0])))
            if narrow:
                output = cp.empty(shape, dtype=cp.int64)
            else:
                output = cp.empty((math.prod(shape), output_words), dtype=cp.uint32)
                parameters += tuple(map(np.int32, (*widths, output_words)))
            kernel(((math.prod(shape) + 127) // 128,), (128,),
                   (self.device_rows[start:end], self.pivots, self.offsets, *inputs, output, *parameters))
            result = output.get()
            if not narrow:
                result = _decode_signed_words(result, shape)
            yield self.rows[start:end], result


def probe(device):
    """Initialize CUDA in a subprocess, keeping the Linux fork parent clean."""
    products = CudaProducts(device)
    cp = products.cp
    identity = np.eye(2, dtype=np.int64)
    rhs = np.array([[2**96, -2**97], [-1, 0]], dtype=object)
    if not np.array_equal(products.integer_product(identity)(rhs), rhs):
        raise RuntimeError('BREC CUDA integer self-check failed')
    from .encodings import polynomial_bank
    irregular = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    for bits in (128, 256, 512):
        expected = polynomial_bank(irregular, 4, 'chebyshev', 'interval', bits)
        actual = polynomial_bank(irregular, 4, 'chebyshev', 'interval', bits, accelerator=products)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f'BREC CUDA interval self-check failed at {bits} bits')
    small = np.array([[1, -2, 0], [-2, 1, 4], [0, 4, -3]], dtype=object)
    wide = small * 2**67
    banks = (small, wide)
    groups = ([0, 2], [1])
    triples = [(0, 0, 2), (1, 2, 1), (2, 2, 2)]
    joint = CudaJointSums(products, banks, [4, 4 * 2**67], groups)
    for degrees in ((0, 0, 0), (1, 1, 0), (1, 1, 1)):
        expected = [[sum(math.prod(banks[d][v, t] for d, v in zip(degrees, row)) for t in group)
                     for group in groups] for row in triples]
        actual = np.concatenate([block for _, block in joint.blocks(triples, degrees)])
        if actual.tolist() != expected:
            raise RuntimeError('BREC CUDA joint contraction self-check failed')
    properties = cp.cuda.runtime.getDeviceProperties(products.device.id)
    return {'cupy': cp.__version__, 'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'cuda_driver': cp.cuda.runtime.driverGetVersion(),
            'name': properties['name'].decode(),
            'compute_capability': products.device.compute_capability}


if __name__ == '__main__':
    print(json.dumps(probe(sys.argv[1])))
