"""Memory-bounded scheduling and results for ordered 4-FWL quadruples."""

from contextlib import closing
import resource
from time import perf_counter

import numpy as np

from brec.suite_resources import Resources
from rebuttal_experiments.brec_data import main_graph_pair
from rebuttal_experiments.common import atomic_json

from .protocol import BRANCH_DEGREES, PIVOT_FRACTION_BITS, TEST_PAIRS, TOLERANCE, describe_protocol
from .quadruple_backend import GIB, cpu_gaps, cuda_gaps, load_kernel
from .runtime import experiment_directory, pair_ids_by_size, read_json, verify_lower_order


CARRIER = '(S^23+S^24)/2'
PROTOCOL = describe_protocol(
    ordered_quadruples='all tuples evaluated directly',
    static_joint_degree=1,
    branch_degrees=list(BRANCH_DEGREES),
    first_pivot_carrier=CARRIER,
    second_pivot_carrier=CARRIER,
    pivot_sum_fraction_bits=PIVOT_FRACTION_BITS,
    readout='maximum absolute gap between sorted scalar channels',
)


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def evaluate_pair(pair_id, vertices, stream, backend):
    start = perf_counter()
    gaps = []
    first = None
    with closing(stream):
        for depth, gap in enumerate(stream):
            gaps.append(gap)
            rss = peak_rss_bytes()/GIB
            elapsed = perf_counter()-start
            print(f'  pair {pair_id}, round {depth}: {gap:.9g}, {elapsed:.1f}s, peak {rss:.2f} GiB', flush=True)
            if gap > TOLERANCE:
                first = depth
                break
    return {
        'pair_id': pair_id,
        'vertices': vertices,
        'gaps': gaps,
        'first_different_round': first,
        'seconds': perf_counter()-start,
        'backend': backend,
    }


def summarize(records, ids):
    distinguished = {row['pair_id'] for row in records if row['first_different_round'] is not None}
    cohort = TEST_PAIRS[4]
    return {
        **PROTOCOL,
        'target_ids': sorted(cohort),
        'tested': len(records),
        'distinguished': len(distinguished),
        'distinguished_ids': sorted(distinguished),
        'unresolved_ids': sorted(ids - distinguished),
        'not_tested_ids': sorted(cohort - ids),
        'peak_rss_bytes': peak_rss_bytes(),
    }


def run(data, out, ids, memory_gib, devices=None, lower_order_out=None):
    if lower_order_out:
        verify_lower_order(lower_order_out, ids, ('static_gaps', 'recursive_gaps'))
    with experiment_directory(out, PROTOCOL):
        run_pairs(data, out, ids, memory_gib, devices)


def run_pairs(data, out, ids, memory_gib, devices):
    records = []
    kernel = None
    order = pair_ids_by_size(data, ids, largest_first=False)
    overall_start = perf_counter()
    for pid in order:
        path = out/'records'/f'pair-{pid:03d}.json'
        if path.exists():
            records.append(read_json(path))
            continue
        graphs = main_graph_pair(data, pid)
        vertices = len(graphs[0])
        if vertices != len(graphs[1]):
            raise ValueError('Expected equal-size graphs for sorted readout')
        host_buffers = 2 if devices else 8
        planned = host_buffers*vertices**4*np.dtype('float32').itemsize + GIB
        available = min(memory_gib*GIB, Resources().memory_available())
        if planned > available:
            raise MemoryError(f'Pair {pid} needs {planned/GIB:.2f} GiB, {available/GIB:.2f} GiB available')
        print(f'Main [{len(records)}/{len(order)}] pair {pid}, {vertices} vertices, {planned/GIB:.2f} GiB reserved',
              flush=True)
        atomic_json(out/'progress-main.json', {'completed': len(records), 'total': len(order), 'pair': pid})
        if devices:
            stream = cuda_gaps(devices, graphs)
        else:
            if kernel is None:
                kernel = load_kernel(out)
            stream = cpu_gaps(kernel, graphs)
        record = evaluate_pair(pid, vertices, stream, 'cuda' if devices else 'cpu')
        atomic_json(path, record)
        records.append(record)
    summary = summarize(records, ids)
    summary['seconds_this_run'] = perf_counter()-overall_start
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'progress-main.json', {'completed': len(records), 'total': len(order)})
    print(f'Completed {len(records)}/{len(order)}, distinguished {summary["distinguished"]}', flush=True)
