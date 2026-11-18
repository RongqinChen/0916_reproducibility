"""Experiment 3: exact/interval BREC pair decisions and relabeling controls."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import traceback
from collections import Counter
from functools import partial
from pathlib import Path

from brec.determinism import prepare_environment, start_brec
prepare_environment()
import numpy as np

from .brec_data import graph_at as graph_at
from .brec_data import (OFFICIAL_RELIABILITY, PAIR_COUNT, load_brec_data,
                        main_graph_pair, official_reliability_pairs, reliability_mapping,
                        validate_reliability_configuration)
from .readouts import (DIFFERENT, EQUAL, UNRESOLVED, exact_multiset,
                       interval_multiset, mean_decision)
from .common import (Budget, ResourceLimit, atomic_json, configuration, hard_wall_limit,
                     permutations, run_directory)
from .encodings import descriptors, feature_matrix, moment_blocks
from .runtime import report_phase
from .scheduling import (add_device_arguments, device_configuration, estimate_bytes,
                         ordered_cpus, run_tasks, selected_devices, worker_count)


def validate_configuration(config):
    """Reject invalid protocols before creating output or starting workers."""
    choices = {'bases': {'monomial', 'chebyshev'},
               'variants': {'polynomial', 'endpoint'},
               'readouts': {'multiset', 'mean'}}
    for key, allowed in choices.items():
        if key == 'readouts' and key not in config:
            continue
        values = config.get(key)
        if (not isinstance(values, list) or not values
                or any(not isinstance(x, str) or x not in allowed for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key} must be a nonempty list of distinct supported values')
    for key in ('q_values', 'mean_p_values', 'precision_bits'):
        values = config.get(key)
        minimum = 2 if key == 'precision_bits' else 1
        if (not isinstance(values, list) or not values
                or any(type(x) is not int or x < minimum for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain distinct integers >= {minimum}')
    if config['precision_bits'] != sorted(config['precision_bits']):
        raise ValueError('precision_bits must increase')
    scale = config.get('memory_reservation_scale', 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('memory_reservation_scale must be in (0,1]')
    validate_reliability_configuration(config)
    for key, minimum in (('seed', 0), ('max_candidate_edges', 1)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    for key in ('memory_gib', 'seconds_per_condition'):
        value = config.get(key)
        if key == 'seconds_per_condition' and key in config and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite; only seconds_per_condition may be null')


def is_regular(graph):
    degrees = graph.sum(1)
    return bool(np.all(degrees == degrees[0]))


def aligned_control(original, changed, new_to_old, mode):
    n = len(new_to_old)
    expected = original.reshape(n, n, -1)[np.ix_(new_to_old, new_to_old)].reshape(changed.shape)
    if mode == 'exact':
        if not np.array_equal(expected, changed):
            raise AssertionError('Exact same-graph aligned features changed')
        return {'aligned_exact_equal': True}
    if mode == 'interval':
        if not all(a.overlaps(b) for a, b in zip(expected.flat, changed.flat)):
            raise AssertionError('Same-graph valid enclosures are disjoint')
        return {'aligned_enclosures_overlap': True,
                'scope': 'Necessary enclosure consistency, not an equality certificate.'}
    return {'aligned_max_absolute_error': float(np.max(np.abs(expected - changed)))}


def relabel_controls(graphs, originals, q, basis, variant, mode, bits, config, pid, budget,
                     *, features=feature_matrix):
    result = []
    for graph_index, (graph, original) in enumerate(zip(graphs, originals)):
        for index, permutation in enumerate(permutations(
                len(graph), config['relabels'], config['seed'] + 1009 * pid + graph_index)):
            report_phase('main_relabel', graph=graph_index, relabel=index + 1,
                         total=config['relabels'], bits=bits)
            changed = features(graph[np.ix_(permutation, permutation)], q, 2, variant,
                               basis, mode, bits, budget)
            result.append({'graph': graph_index, 'new_index_to_old': permutation.tolist(),
                           **aligned_control(original, changed, permutation, mode)})
    return result


def reliability_control(data, pid, q, basis, variant, mode, bits, config, budget,
                        *, features=feature_matrix):
    result = []
    count = config.get('reliability_relabels', 0)
    for indices, graphs in official_reliability_pairs(data, pid, config):
        arithmetic = ('interval' if mode == 'exact' and not all(is_regular(g) for g in graphs)
                      else mode)
        if budget:
            budget.check()
        report_phase('reliability_mapping', nodes=len(graphs[0]), dataset_indices=indices)
        mapping = reliability_mapping(*graphs)
        if budget:
            budget.check()
        # Use the verified isomorphism to align features on the control graphs.
        report_phase('reliability_features', graph=0, bits=bits)
        original = features(graphs[0], q, 2, variant, basis, arithmetic, bits, budget)
        report_phase('reliability_features', graph=1, bits=bits)
        changed = features(graphs[1], q, 2, variant, basis, arithmetic, bits, budget)
        report_phase('reliability_alignment')
        control = {'official_representatives': aligned_control(
                       original, changed, np.argsort(mapping), arithmetic),
                   'control_old_to_new': mapping.tolist()}
        if config.get('reliability_protocol') == OFFICIAL_RELIABILITY:
            control['dataset_indices'] = list(indices)
        result.append(control)
        del changed
        relabels = permutations(len(graphs[0]), count, config['seed'] + 7919 * pid) if count else ()
        for index, permutation in enumerate(relabels):
            report_phase('reliability_relabel', relabel=index + 1, total=count, bits=bits)
            relabelled = features(graphs[0][np.ix_(permutation, permutation)], q, 2,
                                  variant, basis, arithmetic, bits, budget)
            result.append({'new_index_to_old': permutation.tolist(),
                           **aligned_control(original, relabelled, permutation, arithmetic)})
        del original
    return result


def evaluate_condition(data, pid, basis, variant, q, readout, p, config):
    return next(evaluate_budgets(data, pid, basis, variant, [(q, readout, p)], config))[1]


def _readout_decisions(rows, settings, columns, mode, bits, budget, config):
    """Reuse cumulative moment blocks for all requested channel projections."""
    blocks = zip(*(moment_blocks(row, budget) for row in rows))
    moments = []
    moment_error = None
    for setting in sorted(settings, key=lambda setting: (setting[1], setting[2], setting[0])):
        q, readout, p = setting
        report_phase('main_readout', readout=readout, p=p, bits=bits, q=q)
        if readout == 'mean' and moment_error is not None:
            yield setting, moment_error
            continue
        try:
            if readout == 'mean':
                while len(moments) < p:
                    moments.append(next(blocks))
                vectors = ([value for block in side for value in block[columns[q]]]
                           for side in zip(*moments[:p]))
                decision = mean_decision(*vectors)
            else:
                projected = (row[:, columns[q]] for row in rows)
                decision = (exact_multiset(*projected) if mode == 'exact' else
                            interval_multiset(*projected, budget, config['max_candidate_edges']))
        except (ResourceLimit, MemoryError) as exc:
            decision = {'status': 'resource_limited', 'reason': str(exc)}
            if readout == 'mean' and len(moments) < p:
                # Larger prefixes also depend on the failed block.
                moment_error = decision
        yield setting, decision


def evaluate_budgets(data, pid, basis, variant, settings, config):
    budget = Budget(config['seconds_per_condition'], config['memory_gib'])
    features = feature_matrix
    if config.get('bank_cache') is not None:
        features = partial(features, bank_cache=config['bank_cache'])
    if config.get('device', 'cpu') != 'cpu':
        from .brec_cuda import CudaProducts
        features = partial(features, accelerator=CudaProducts(config['device']))
    report_phase('load_graphs')
    graphs = main_graph_pair(data, pid)
    mode = 'exact' if all(is_regular(g) for g in graphs) else 'interval'
    precisions = [None] if mode == 'exact' else config['precision_bits']
    attempts = {setting: [] for setting in settings}
    for bits in precisions:
        precision = bits or 128
        degrees = {q for q, _, _ in attempts}
        max_q = max(degrees)
        keys = {q: descriptors(q, 2, variant) for q in degrees}
        positions = {key: index for index, key in enumerate(keys[max_q])}
        # Endpoint channels can move when Q changes. Select them by descriptor
        # and preserve the protocol's original order for each readout.
        columns = {q: (slice(None) if q == max_q else [positions[key] for key in keys[q]])
                   for q in degrees}
        rows = []
        for index, graph in enumerate(graphs):
            report_phase('main_features', graph=index, arithmetic=mode, bits=bits, q=max_q)
            rows.append(features(graph, max_q, 2, variant, basis, mode, precision, budget))
        finalized = []
        for setting, decision in _readout_decisions(rows, attempts, columns, mode, bits, budget, config):
            if decision['status'] == 'resource_limited':
                del attempts[setting]
                yield setting, decision
                continue
            attempts[setting].append({'bits': bits, 'status': decision['status'],
                                      'reason': decision.get('reason')})
            if decision['status'] != UNRESOLVED or bits == precisions[-1]:
                finalized.append((setting, decision))
        if finalized:
            try:
                # Run real recomputations for the same-graph controls once
                # on the union of the channels that stop at this precision.
                control_q = max(q for (q, _, _), _ in finalized)
                relabels = []
                if config.get('relabels', 0):
                    relabels = relabel_controls(
                        graphs, [row[:, columns[control_q]] for row in rows], control_q,
                        basis, variant, mode, precision, config, pid, budget, features=features)
                reliability = reliability_control(data, pid, control_q, basis, variant, mode,
                                                  precision, config, budget, features=features)
                controls = {'relabel_controls': relabels,
                            'official_reliability_controls': reliability}
            except (ResourceLimit, MemoryError) as exc:
                for setting, _ in finalized:
                    yield setting, {'status': 'resource_limited', 'reason': str(exc)}
                    del attempts[setting]
            else:
                for setting, decision in finalized:
                    yield setting, {**decision, **controls, 'arithmetic': mode,
                                    'precision_attempts': attempts.pop(setting),
                                    'width': len(keys[setting[0]]),
                                    'elapsed_seconds': time.monotonic() - budget.started}
        if not attempts:
            return
        del rows


def experiment_settings(config):
    readouts = config.get('readouts', ['multiset', 'mean'])
    for basis, variant, q in itertools.product(config['bases'], config['variants'], config['q_values']):
        if 'multiset' in readouts:
            yield basis, variant, q, 'multiset', 1
        if 'mean' in readouts:
            for p in config['mean_p_values']:
                yield basis, variant, q, 'mean', p


def condition_record(data, config, task):
    row = dict(task['record'])
    try:
        with hard_wall_limit(config['seconds_per_condition']):
            result = evaluate_condition(data, row['pair_id'], row['basis'], row['variant'],
                                        row['q'], row['readout'], row['p'], config)
        row.update(result)
    except (ResourceLimit, MemoryError) as exc:
        row.update(status='resource_limited', reason=str(exc))
    except Exception as exc:
        row.update(status='implementation_error', reason=repr(exc), traceback=traceback.format_exc())
    return row


def group_conditions(tasks, config):
    # Sharing wall time would change the semantics of a finite per-condition limit.
    if config['seconds_per_condition'] is not None:
        return tasks
    groups = {}
    for task in tasks:
        row = task['record']
        key = tuple(row[name] for name in ('pair_id', 'basis', 'variant'))
        groups.setdefault(key, []).append(task)
    return [{**group[0], 'conditions': group,
             'estimated_bytes': max(task['estimated_bytes'] for task in group),
             'work': max(task['work'] for task in group)}
            if len(group) > 1 else group[0] for group in groups.values()]


def task_records(data, config, task):
    if 'device' in task:
        config = {**config, 'device': task['device']}
    if 'conditions' not in task:
        return condition_record(data, config, task)
    return grouped_records(data, config, task['conditions'])


def grouped_records(data, config, tasks):
    pending = {tuple(task['record'][name] for name in ('q', 'readout', 'p')): task
               for task in tasks}
    first = tasks[0]['record']
    try:
        for key, result in evaluate_budgets(data, first['pair_id'], first['basis'],
                                             first['variant'], pending, config):
            task = pending.pop(key)
            yield task['stem'], {**task['record'], **result}
    except (ResourceLimit, MemoryError) as exc:
        error = {'status': 'resource_limited', 'reason': str(exc)}
    except Exception as exc:
        error = {'status': 'implementation_error', 'reason': repr(exc),
                 'traceback': traceback.format_exc()}
    else:
        return
    for task in pending.values():
        yield task['stem'], {**task['record'], **error}


def pending_conditions(data, ids, config, out, identity):
    sizes, regular = {}, {}
    started = time.monotonic()
    for pid in ids:
        main = main_graph_pair(data, pid)
        _, controls = next(official_reliability_pairs(data, pid, config))
        # Verify the control path on every requested pair before any long task.
        # Main pairs may be nonisomorphic and are never passed to this check.
        try:
            reliability_mapping(*controls)
        except ValueError as exc:
            raise ValueError(f'Reliability preflight failed for pair{pid:03d}: {exc}') from exc
        sizes[pid] = max(len(graph) for graph in main + controls)
        regular[pid] = all(is_regular(graph) for graph in main)
    print(f'BREC preflight: {len(ids)} reliability pairs aligned and integer-verified '
          f'in {time.monotonic() - started:.2f}s', flush=True)
    for basis, variant, q, readout, p in experiment_settings(config):
        width = q + 3 + (q * (q + 1) // 2 - 1 if variant == 'endpoint' else 0)
        for pid in ids:
            stem = f'{basis}_{variant}_q{q}_{readout}_p{p}_pair{pid:03d}'
            row = {'pair_id': pid, 'basis': basis, 'variant': variant,
                   'q': q, 'order': 2, 'readout': readout, 'p': p, 'identity_sha256': identity}
            target = out / 'records' / (stem + '.json')
            if target.exists():
                old = json.loads(target.read_text())
                if any(old.get(key) != value for key, value in row.items()):
                    raise ValueError('Record identity mismatch')
                if old.get('status') not in (DIFFERENT, EQUAL, UNRESOLVED,
                                              'resource_limited', 'implementation_error'):
                    raise ValueError(f'Invalid saved condition status: {target}')
                if old['status'] == 'implementation_error':
                    raise ValueError(f'Inspect the saved implementation error before resuming: {target}')
                continue
            matching = config['max_candidate_edges'] if readout == 'multiset' and not regular[pid] else 0
            n = sizes[pid]
            yield {'stem': stem, 'record': row,
                   'estimated_bytes': estimate_bytes(n, width, q, matching),
                   'work': n**3 * q + n**2 * width * p}


def summarize(out, config):
    summaries = []
    for basis, variant, q, readout, p in experiment_settings(config):
        stem = f'{basis}_{variant}_q{q}_{readout}_p{p}'
        counts = Counter()
        for pid in range(PAIR_COUNT):
            path = out / 'records' / f'{stem}_pair{pid:03d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                counts[record['status']] += 1
            else:
                counts['not_executed'] += 1
        assert sum(counts.values()) == PAIR_COUNT
        summaries.append({'setting': stem, 'denominator': PAIR_COUNT, 'counts': dict(counts),
                          'certified_detection_lower_bound': counts[DIFFERENT],
                          'detection_upper_bound': PAIR_COUNT - counts[EQUAL]})
    atomic_json(out / 'summary.json', {
        'settings': summaries,
        'scope': 'New order-2 protocol; unresolved/resource-limited items remain in denominator. Lower-bound rankings do not establish true expressivity rankings.'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rebuttal/brec.json')
    parser.add_argument('--brec-file', type=Path, required=True)
    parser.add_argument('--pair-ids', nargs='+', type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    add_device_arguments(parser)
    parser.add_argument('--bank-cache', type=Path, help='Share exact polynomial banks across encoding runs')
    parser.add_argument('--workers', type=int,
                        help='Maximum parallel jobs; default: up to two per CUDA device or all eligible CPUs')
    parser.add_argument('--memory-gib', type=float,
                        help='Override the configuration total scheduler+worker RSS budget in GiB')
    parser.add_argument('--reserve-gib', type=float, default=8,
                        help='Keep this much system RAM available; default: 8 GiB')
    args = parser.parse_args(argv)
    devices = selected_devices(args, parser)
    cpus = ordered_cpus()
    workers = worker_count(args.workers, cpus, devices)
    if workers < 1 or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error('Positive --workers and finite nonnegative --reserve-gib required')
    config = configuration(args.config)
    if args.memory_gib is not None:
        config['memory_gib'] = args.memory_gib
    try:
        validate_configuration(config)
    except ValueError as exc:
        parser.error(str(exc))
    config = {**config, 'determinism': start_brec(config['seed'])}
    config.update(device_configuration(devices, parser))
    if args.bank_cache is not None:
        config['bank_cache'] = str(args.bank_cache.resolve())
    config = {**config, 'scheduling': {'workers': workers, 'reserve_gib': args.reserve_gib,
                                     'memory_scope': 'parent_and_workers_rss',
                                     'cpu_order': cpus}}
    ids = args.pair_ids if args.pair_ids is not None else list(range(PAIR_COUNT))
    if not ids or len(set(ids)) != len(ids) or any(x < 0 or x >= PAIR_COUNT for x in ids):
        parser.error('Distinct pair IDs in 0..399 required')
    config = {**config, 'executed_pair_ids': ids}
    data = load_brec_data(args.brec_file, config['expected_brec_sha256'])
    with run_directory(args.out, config, [args.config, args.brec_file], args.resume) as (out, identity):
        tasks = group_conditions(list(pending_conditions(data, ids, config, out, identity)), config)
        try:
            run_tasks(tasks, partial(task_records, data, config), out,
                      workers=workers, memory_gib=config['memory_gib'],
                      reserve_gib=args.reserve_gib, seed=config['seed'],
                      reservation_scale=config.get('memory_reservation_scale', 1.0),
                      devices=devices)
        finally:
            summarize(out, config)


if __name__ == '__main__':
    main()
