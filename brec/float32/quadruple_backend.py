"""Run the dense quadruple kernel and yield per-round comparison gaps."""

from concurrent.futures import ThreadPoolExecutor
from ctypes import CDLL, c_int, c_void_p
from pathlib import Path
from subprocess import run

import numpy as np

from .numerics import adjacent_walk_average, gap_on_sorted, sorted_gap, walk_bank
from .protocol import ROUNDS


KERNELS = Path(__file__).with_name('kernels')
GIB = 2**30


def load_kernel(out):
    code = KERNELS/'quadruple.cpp'
    library = out/'quadruple_kernel.so'
    run(['g++', '-std=c++17', '-O3', '-march=native', '-fopenmp', '-ffp-contract=off',
         '-fno-math-errno', '-shared', '-fPIC', str(code), '-o', str(library)], check=True)
    kernel = CDLL(str(library.resolve()))
    for name, pointers in [('gates', 6), ('contract', 9), ('static_joint', 4)]:
        function = getattr(kernel, name)
        function.argtypes = [c_int, *([c_void_p]*pointers)]
        function.restype = None
    return kernel


def pointer(array):
    return None if array is None else array.ctypes.data


def pivot_lists(matrix):
    row_ids, pivots = np.nonzero(matrix)
    offsets = np.r_[0, np.cumsum(np.bincount(row_ids, minlength=len(matrix)))]
    return (offsets.astype(np.int32), pivots.astype(np.int32))


def cpu_gaps(kernel, graphs):
    n = len(graphs[0])
    banks = [walk_bank(graph) for graph in graphs]
    states = [np.empty(n**4, dtype=np.float32) for _ in graphs]
    for bank, state in zip(banks, states, strict=True):
        walk = bank[1]
        arguments = (*pivot_lists(walk), walk, state)
        kernel.static_joint(n, *map(pointer, arguments))
    yield sorted_gap(states)
    pairs = [bank[0] + bank[1] for bank in banks]
    wide = [adjacent_walk_average(bank) for bank in banks]
    pivots = [pivot_lists(matrix) for matrix in wide]
    gates = [np.empty(n**4, dtype=np.float32) for _ in range(4)]
    for depth in range(1, ROUNDS+1):
        for side, state in enumerate(states):
            previous = state if depth > 1 else None
            kernel.gates(n, *map(pointer, (pairs[side], previous, *gates)))
            arguments = (*pivots[side], wide[side], wide[side], *gates, state)
            kernel.contract(n, *map(pointer, arguments))
        yield sorted_gap(states)


class CudaState:
    def __init__(self, cp, code, device, bank):
        self.cp, self.device = cp, device
        self.n = len(bank[0])
        self.size = self.n**4
        self.host = np.empty(self.size, dtype=np.float32)
        self.gates = None
        with cp.cuda.Device(device):
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            free, _ = cp.cuda.runtime.memGetInfo()
            if free < 5*self.host.nbytes + 2*GIB:
                raise MemoryError(f'GPU {device} lacks space for the dense state and four branch buffers')
            self.module = cp.RawModule(code=code, options=('--std=c++17', '--fmad=false', '--prec-div=true'))
            self.functions = {name:self.module.get_function(name) for name in ('gates','contract','static_joint')}
            walk = bank[1]
            pair_base = bank[0]+walk
            wide = adjacent_walk_average(bank)
            self.static_inputs = [cp.asarray(array) for array in (*pivot_lists(walk), walk)]
            indices = [cp.asarray(array) for array in pivot_lists(wide)]
            carrier = cp.asarray(wide)
            self.inputs = [*indices, carrier, carrier]
            self.pair_base = cp.asarray(pair_base)
            self.state = cp.empty(self.size, dtype=cp.float32)

    def call(self, name, *args):
        self.functions[name]((4096,), (256,), args)

    def update(self, depth):
        n, size = np.int32(self.n), np.uint64(self.size)
        with self.cp.cuda.Device(self.device):
            if depth == 0:
                self.call('static_joint', *self.static_inputs, n, size, self.state)
            else:
                if self.gates is None:
                    self.gates = [self.cp.empty_like(self.state) for _ in range(4)]
                self.call('gates', self.pair_base, self.state, n, size, np.int32(depth > 1), *self.gates)
                self.call('contract', *self.inputs, *self.gates, n, size, self.state)
            self.state.get(out=self.host)
        self.host.sort()
        return self.host


def cuda_gaps(device_ids, graphs):
    import cupy as cp

    code = (KERNELS/'quadruple.cu').read_text()
    banks = [walk_bank(graph) for graph in graphs]
    with ThreadPoolExecutor(2) as pool:
        pending = [pool.submit(CudaState, cp, code, device, bank)
                   for device, bank in zip(device_ids, banks, strict=True)]
        states = [future.result() for future in pending]
        for depth in range(ROUNDS+1):
            pending = [pool.submit(state.update, depth) for state in states]
            yield gap_on_sorted([future.result() for future in pending])
