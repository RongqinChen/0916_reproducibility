"""Standalone raw-moment/normalization audit; does not change active BREC runs.

Both K and K/n moments are accumulated BEFORE rounding from the same streamed
features. Scale-restored comparisons separate loss due to rounding from changes
in accumulation. Includes actual computation on specified negative IDs; never
uses the active runner's theoretical skips. This remains a float32 diagnostic.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .budget_sweep import polynomial_bank
from .encodings import channel_count, feature_chunks
from .theory import group_for_pair
from supplementary.common import (atomic_json, begin_run, finish_run, load_brec,
                                  pair_graphs, validate_pair_ids, checksum,
                                  read_pair_ids)


def moments(adj, q, basis, max_p, chunk_size):
    width = channel_count(3, q, 'joint')
    offset = channel_count(3, q, 'endpoint')
    total = np.zeros((2, max_p, width), dtype=np.float32)
    correction = np.zeros_like(total)
    bank = polynomial_bank(adj, q, basis)
    for chunk in feature_chunks(adj, bank, q, 3, 'joint', chunk_size, None):
        scaled = chunk.copy()
        scaled[:, offset:] /= len(adj)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for mode, values in enumerate((chunk, scaled)):
                powered = np.ones_like(values)
                for power in range(max_p):
                    powered *= values
                    increment = powered.sum(0) - correction[mode, power]
                    updated = total[mode, power] + increment
                    correction[mode, power] = (updated - total[mode, power]) - increment
                    total[mode, power] = updated
    result = total / len(adj)**3
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite raw moments')
    return result


def compare(groups, atol, rtol):
    left, right, reliability = groups
    def close(a, b):
        return bool(np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))))
    same_good, max_same = True, 0.0
    for group in groups:
        for i in range(len(group)):
            for j in range(i):
                same_good = close(group[i], group[j]) and same_good
                max_same = max(max_same, float(np.max(np.abs(group[i] - group[j]))))
    all_cross, min_cross = True, float('inf')
    for a in left:
        for b in right:
            all_cross = (not close(a, b)) and all_cross
            min_cross = min(min_cross, float(np.max(np.abs(a - b))))
    return {'all_sampled_cross_relabels_separated': all_cross,
            'sampled_same_graph_consistency': same_good,
            'numerically_reliable_separation': all_cross and same_good,
            'max_same_graph_linf': max_same, 'min_cross_graph_linf': min_cross,
            'same_graph_comparisons': sum(len(g) * (len(g) - 1) // 2 for g in groups),
            'cross_graph_comparisons': len(left) * len(right),
            'reliability_graph_samples': len(reliability)}


def decisions(values, sizes, args, q, pid):
    offset = channel_count(3, q, 'endpoint')
    poly = channel_count(3, q, 'polynomial')
    blocks = {'polynomial': slice(0, poly), 'endpoint': slice(0, offset),
              'joint': slice(None), 'joint_only': slice(offset, None)}
    count = 2 * args.relabels
    rows = []
    for scale, name in enumerate(('none', 'n')):
        for readout_p in args.p_values:
            raw = values[:, scale, :readout_p, :]
            for block, columns in blocks.items():
                def assess(vector, atol, rtol):
                    vector = vector[:, :, columns].reshape(len(vector), -1)
                    result = compare((vector[:count:2], vector[1:count:2], vector[count:]), atol, rtol)
                    result['reference_upper_bound_violation'] = (
                        group_for_pair(pid) == '3-FWL-indistinguishable'
                        and result['numerically_reliable_separation'])
                    result['confirmed_separated'] = (result['numerically_reliable_separation']
                                                     and not result['reference_upper_bound_violation'])
                    return result
                for decimals in args.decimals:
                    rounded = np.round(raw, decimals)
                    for atol in args.atols:
                        rows.append({'scale': name, 'p': readout_p, 'block': block,
                                     'comparison': 'rounded_fixed_absolute_units',
                                     'decimals': decimals, 'atol': atol, 'rtol': 0.0,
                                     **assess(rounded, atol, 0.0)})
                restored = raw.copy()
                if name == 'n':
                    for exponent in range(readout_p):
                        restored[:, exponent, offset:] *= sizes[:, None]**(exponent + 1)
                for atol in args.atols:
                    rows.append({'scale': name, 'p': readout_p, 'block': block,
                                 'comparison': 'unrounded_restored_K_units',
                                 'decimals': None, 'atol': atol, 'rtol': args.rtol,
                                 **assess(restored, atol, args.rtol)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--brec-file', type=Path, required=True)
    p.add_argument('--trust-pickle', action='store_true')
    pairs = p.add_mutually_exclusive_group(required=True)
    pairs.add_argument('--pair-ids', nargs='+', type=int)
    pairs.add_argument('--pair-list', type=Path,
                       help='whitespace/comma separated pair IDs')
    p.add_argument('--q-values', nargs='+', type=int, default=[4])
    p.add_argument('--p-values', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--bases', nargs='+', choices=['monomial', 'chebyshev'], default=['monomial', 'chebyshev'])
    p.add_argument('--decimals', nargs='+', type=int, default=[8])
    p.add_argument('--atols', nargs='+', type=float, default=[1e-8])
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--relabels', type=int, default=32)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--max-joint-work', type=int, default=2_000_000_000,
                   help='Per-graph n^4*(Q+1)^3 guard; 0 disables it explicitly')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.pair_ids = (args.pair_ids if args.pair_ids is not None
                     else read_pair_ids(args.pair_list))
    validate_pair_ids(args.pair_ids)
    if not 2 <= args.relabels <= 32 or min(args.q_values + args.p_values + [args.chunk_size]) < 1:
        p.error('Invalid budgets or relabel count (at least two needed for consistency)')
    if args.max_joint_work < 0 or any(d < 0 or d > 15 for d in args.decimals):
        p.error('Invalid resource guard or rounding precision')
    if any(not np.isfinite(t) or t < 0 for t in args.atols + [args.rtol]):
        p.error('Tolerances must be finite and nonnegative')
    for name in ('q_values', 'p_values', 'bases', 'decimals', 'atols'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error('Duplicate ' + name)
    root = Path(__file__).resolve().parent
    identity = begin_run(args, 'brec_raw_moment_audit_v1',
                         [root / f for f in ('numerical_audit.py', 'encodings.py',
                                             'budget_sweep.py', 'theory.py')])
    data = load_brec(args.brec_file, args.trust_pickle)
    statuses = []
    for pid in args.pair_ids:
        main_graphs, reliability = pair_graphs(data, pid, args.relabels)
        graphs = main_graphs + reliability
        sizes = np.asarray([len(g) for g in graphs], dtype=np.float32)
        for basis in args.bases:
            for q in args.q_values:
                stem = f'pair-{pid:03d}_{basis}_q-{q}'
                target = args.out_dir / (stem + '.json')
                if target.exists():
                    old = json.loads(target.read_text())
                    if old['status'] == 'complete':
                        if checksum(args.out_dir / old['raw_file']) != old['raw_sha256']:
                            raise ValueError('Raw-moment archive checksum mismatch')
                        statuses.append('complete')
                        continue
                tick = time.perf_counter()
                record = {'pair_id': pid, 'basis': basis, 'q': q,
                          'reference_group': group_for_pair(pid), 'relabels': args.relabels,
                          'group_source': 'repository fixed BREC ID table, not recomputed here'}
                work = int(max(sizes)**4 * (q + 1)**3)
                if args.max_joint_work and work > args.max_joint_work:
                    record.update(status='resource_guard', estimated_joint_work_per_graph=work)
                else:
                    try:
                        values = np.stack([moments(g, q, basis, max(args.p_values), args.chunk_size)
                                           for g in graphs])
                        raw_path = args.out_dir / (stem + '.raw.npz')
                        np.savez_compressed(raw_path, moments=values, sizes=sizes,
                                            group=np.asarray(['main'] * len(main_graphs)
                                                             + ['reliability'] * len(reliability)))
                        offset = channel_count(3, q, 'endpoint')
                        restoration = []
                        for exponent in range(max(args.p_values)):
                            unscaled = values[:, 0, exponent, offset:]
                            restored = values[:, 1, exponent, offset:] * sizes[:, None]**(exponent + 1)
                            error = np.abs(unscaled - restored)
                            restoration.append({'power': exponent + 1,
                                                'max_absolute_discrepancy': float(error.max()),
                                                'unscaled_max_absolute_value': float(np.abs(unscaled).max())})
                        zeros = []
                        for mode, scale in enumerate(('none', 'n')):
                            joint = values[:, mode, :, offset:]
                            for d in args.decimals:
                                zeros.append({'scale': scale, 'decimals': d,
                                              'nonzero_before_rounding': int(np.count_nonzero(joint)),
                                              'nonzero_rounded_to_zero': int(np.count_nonzero(
                                                  (joint != 0) & (np.round(joint, d) == 0)))})
                        record.update(status='complete', raw_file=raw_path.name,
                                      raw_sha256=checksum(raw_path), rounding_diagnostics=zeros,
                                      scale_restoration_diagnostics=restoration,
                                      comparisons=decisions(values, sizes, args, q, pid))
                    except (FloatingPointError, MemoryError) as exc:
                        record.update(status='execution_error', error=str(exc))
                record['seconds'] = time.perf_counter() - tick
                atomic_json(target, record)
                statuses.append(record['status'])
                print(stem, record['status'], flush=True)
    finish_run(args, identity, statuses)


if __name__ == '__main__':
    main()
