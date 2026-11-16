"""Exact experiments and independently recomputed reliability controls."""
import itertools
import random

import numpy as np

from .suite_data import (graphs, isomorphism, mechanism_graphs, reference_group,
                   tuple_orbits, two_fwl)
from .suite_exact import Kernel, compare, compare_nested, descriptors, counts, moment


def statuses(records):
    return {(r["basis"], r["variant"], readout): value["status"]
            for r in records for readout, value in r["decisions"].items()}


def evaluate(graph_pair, q, order, bases, variants, powers, domains=None):
    domains = domains or [tuple_orbits(g, order) for g in graph_pair]
    kernels = {basis: [Kernel(g, q, basis) for g in graph_pair] for basis in bases}
    nested = [v for v in ("poly", "endpoint", "joint") if v in variants]
    results = []
    for basis in bases:
        results.extend(compare_nested(kernels[basis], domains, q, order, nested, powers))
        if "equal19" in variants:
            results.append(compare(kernels[basis], domains, q, order, "equal19", powers))
    by = {(r["basis"], r["variant"]): r["decisions"] for r in results}
    for variant in variants:
        if len({by[b, variant]["multiset"]["status"] for b in bases}) != 1:
            raise AssertionError("Complete-basis multiset equivalence failed")
    for basis in bases:
        previous = None
        for variant in ("poly", "endpoint", "joint"):
            if variant not in variants:
                continue
            current = by[basis, variant]
            if previous is not None:
                for readout in current:
                    if previous[readout]["status"] == "different" and current[readout]["status"] != "different":
                        raise AssertionError("Retained-channel augmentation lost a distinction")
            previous = current
        if "equal19" in variants and "joint" in variants:
            if by[basis, "equal19"]["multiset"]["status"] != by[basis, "joint"]["multiset"]["status"]:
                raise AssertionError("The 19-channel and 27-channel raw dictionaries disagree")
    return results, kernels


def check_alignment(original, changed, new_to_old, order, keys, seed, sample_rows=16):
    """All rational bank entries, plus explicitly sampled full derived rows."""
    p = np.asarray(new_to_old)
    if not np.array_equal(changed.a, original.a[np.ix_(p, p)]):
        raise AssertionError("Relabeling adjacency alignment failed")
    for before, after in zip(original.bank, changed.bank):
        for u in range(original.n):
            for v in range(original.n):
                if after[u][v] != before[int(p[u])][int(p[v])]:
                    raise AssertionError("Exact bank alignment failed")
    random_ids = random.Random(seed).sample(range(original.n**order), min(sample_rows, original.n**order))
    for index in random_ids:
        new_row = np.unravel_index(index, (original.n,)*order)
        old_row = tuple(int(p[v]) for v in new_row)
        for key in keys:
            if original.value(old_row, key) != changed.value(new_row, key):
                raise AssertionError("Sampled full feature row alignment failed")
    return {"all_bank_entries_equal": True, "sampled_full_rows_equal": True,
            "sample_rows": len(random_ids), "channels_per_row": len(keys)}


def check_graph_aggregates(original, changed, original_domain, changed_domain, q, order, powers):
    """Compare each graph with itself after relabeling, using its entire dictionary.

    Full multiset and all channel moments must agree exactly. Testing only the
    cross-graph decision would miss a common label-dependent drift on both sides.
    Every smaller dictionary and the equal19 dictionary is a coordinate projection.
    """
    variant = "joint" if order == 3 else "endpoint"
    result = compare([original, changed], [original_domain, changed_domain], q, order, variant, powers)
    if any(value["status"] != "equal" for value in result["decisions"].values()):
        raise AssertionError("Relabeling changed a graph's exact full multiset or moment aggregate")
    return {"full_graph_multiset_equal": True, "all_channel_moments_equal": True,
            "width": result["width"], "moments_through": max(powers),
            "objects": result["objects"], "orbit_rows": result["orbit_rows"]}


def run_exact(data, job, config, checkpoint=None):
    pid, q, order = job["pair_id"], job["q"], job["order"]
    variants = ["poly", "endpoint"] + (["joint"] if order == 3 else [])
    if q == 1 and order == 3:
        variants.append("equal19")
    bases, powers = config["bases"], config["powers"]
    main = graphs(data, pid)
    main_domains = [tuple_orbits(g, order) for g in main]
    results, kernels = evaluate(main, q, order, bases, variants, powers, main_domains)
    expected = statuses(results)
    controls = []
    if checkpoint:
        checkpoint({"phase": "main_decisions_computed", "records": results, "controls": controls})
    # A separate construction and comparison, not a copy of the first output.
    repeated, _ = evaluate(main, q, order, bases, variants, powers)
    if results != repeated:
        raise AssertionError("Repeated execution changed decisions or exact evidence")
    controls.append({"kind": "repeat", "decisions_and_evidence_equal": True})
    rng = np.random.default_rng(config["seed"] + pid*1009 + order*100003)
    full_keys = descriptors(q, order, "joint" if order == 3 else "endpoint")
    for index in range(config["relabels"]):
        perms = [rng.permutation(len(g)) for g in main]
        permuted = [g[np.ix_(p, p)] for g, p in zip(main, perms)]
        changed_domains = [tuple_orbits(g, order) for g in permuted]
        changed_results, changed_kernels = evaluate(permuted, q, order, bases, variants, powers, changed_domains)
        if statuses(changed_results) != expected:
            raise AssertionError("Relabeling changed a cross-graph decision")
        alignments = [check_alignment(kernels[b][side], changed_kernels[b][side], perms[side],
                         order, full_keys, config["seed"]+index)
                      for b in bases for side in (0, 1)]
        aggregates = [check_graph_aggregates(kernels[b][side], changed_kernels[b][side],
                         main_domains[side], changed_domains[side], q, order, powers)
                      for b in bases for side in (0, 1)]
        controls.append({"kind": "main_relabel", "index": index,
                         "new_to_old": [p.tolist() for p in perms],
                         "all_decisions_equal": True, "alignment": alignments, "aggregates": aggregates})
        if checkpoint:
            checkpoint({"phase": "main_relabel_controls", "records": results, "controls": controls})
    reliability = graphs(data, pid, reliability=True)
    mapping = isomorphism(*reliability)
    reliability_domains = [tuple_orbits(g, order) for g in reliability]
    reliability_results, reliability_kernels = evaluate(reliability, q, order, bases, variants, powers, reliability_domains)
    if set(statuses(reliability_results).values()) != {"equal"}:
        raise AssertionError("Official isomorphic controls were distinguished")
    controls.append({"kind": "official_reliability", "old_to_new": mapping.tolist(),
                     "all_decisions_equal": True,
                     "alignment": [check_alignment(ks[0], ks[1], np.argsort(mapping), order,
                                    full_keys, config["seed"]) for ks in reliability_kernels.values()]})
    for index in range(config["reliability_relabels"]):
        perms = [rng.permutation(len(g)) for g in reliability]
        permuted = [g[np.ix_(p, p)] for g, p in zip(reliability, perms)]
        changed_domains = [tuple_orbits(g, order) for g in permuted]
        changed, changed_kernels = evaluate(permuted, q, order, bases, variants, powers, changed_domains)
        if set(statuses(changed).values()) != {"equal"}:
            raise AssertionError("Relabeled official reliability controls were distinguished")
        controls.append({"kind": "official_reliability_relabel", "index": index,
                         "all_decisions_equal": True, "new_to_old": [p.tolist() for p in perms],
                         "aggregates": [check_graph_aggregates(reliability_kernels[b][side],
                             changed_kernels[b][side], reliability_domains[side], changed_domains[side],
                             q, order, powers) for b in bases for side in (0, 1)],
                         "alignment": [check_alignment(reliability_kernels[b][side],
                             changed_kernels[b][side], perms[side], order, full_keys, config["seed"]+index)
                             for b in bases for side in (0, 1)]})
    return {"records": results, "controls": controls, "arithmetic": "exact_radical",
            "full_graph_aggregate_relabel_checks_passed": True,
            "reference_group": reference_group(pid)}


def run_reference(data, job, config):
    pid = job["pair_id"]
    main, reliability = graphs(data, pid, count=64), graphs(data, pid, True, count=64)
    for i in range(64):
        isomorphism(main[i % 2], main[i])
        isomorphism(reliability[0], reliability[i])
    # Nonisomorphism is checked independently of the weaker FWL test.
    try:
        isomorphism(main[0], main[1])
    except ValueError:
        pass
    else:
        raise AssertionError("The main BREC graph pair is isomorphic")
    return {"official_graphs_validated": 128, "two_fwl": two_fwl(*main[:2]),
            "historical_reference_group": reference_group(pid),
            "historical_group_is_independently_computed": False}


def run_mechanism(config):
    graphs_pair = mechanism_graphs()
    # This check deliberately enumerates all 4,096 triples independently of orbit compression.
    domains = [tuple_orbits(g, 3, compress=False) for g in graphs_pair]
    records, base_kernels = evaluate(graphs_pair, 1, 3, ["monomial", "chebyshev"],
                         ["poly", "endpoint", "joint", "equal19"], [1, 2, 3, 4], domains)
    shifted = [Kernel(g, 1, "shifted") for g in graphs_pair]
    mixed = compare(shifted, domains, 1, 3, "joint", [1, 2, 3, 4])
    original = [Kernel(g, 1) for g in graphs_pair]
    for k, changed, (rows, _) in zip(original, shifted, domains):
        for row in rows:
            for u, v in itertools.combinations_with_replacement(map(int, row), 2):
                identity = int(u == v)
                if changed.pair(1, u, v)-identity != k.pair(1, u, v):
                    raise AssertionError("Shifted-basis inverse check failed")
    for r in records:
        expected_raw = "different" if r["variant"] in ("joint", "equal19") else "equal"
        if r["decisions"]["multiset"]["status"] != expected_raw:
            raise AssertionError("Mechanism multiset result disagrees with its prediction")
        if r["variant"] == "joint" and any(r["decisions"][f"M{p}"]["status"] != "equal" for p in range(1, 5)):
            raise AssertionError("Original-basis marginal moments should agree")
    if mixed["decisions"]["M3"]["status"] != "different":
        raise AssertionError("The shifted third moment should distinguish the pair")
    # Exact value of the stated shifted K111 third-moment witness.
    values = [moment(counts(k.column(rows, ("joint", 1, 1, 1)), weights), 3, 16**3)
              for k, (rows, weights) in zip(shifted, domains)]
    from fractions import Fraction
    from .suite_exact import Radical
    if values[0]-values[1] != Radical.rational(Fraction(1, 331776)):
        raise AssertionError("The shifted third-moment witness has the wrong exact value")
    # Full decision checks on eight independent permutations, for both bases and shifted basis.
    rng = np.random.default_rng(config["seed"])
    aggregate_controls = []
    for _ in range(config["relabels"]):
        relabeled = []
        for graph in graphs_pair:
            p = rng.permutation(len(graph))
            relabeled.append(graph[np.ix_(p, p)])
        changed_domains = [tuple_orbits(g, 3) for g in relabeled]
        changed, changed_kernels = evaluate(relabeled, 1, 3, ["monomial", "chebyshev"],
                              ["poly", "endpoint", "joint", "equal19"], [1, 2, 3, 4], changed_domains)
        if statuses(changed) != statuses(records):
            raise AssertionError("Mechanism relabeling changed decisions")
        shifted_result, shifted_kernels = evaluate(relabeled, 1, 3, ["shifted"], ["joint"], [1, 2, 3, 4], changed_domains)
        if statuses(shifted_result) != statuses([mixed]):
            raise AssertionError("Shifted-basis relabeling changed decisions")
        aggregate_controls.extend(check_graph_aggregates(base_kernels[b][side], changed_kernels[b][side],
            domains[side], changed_domains[side], 1, 3, [1, 2, 3, 4])
            for b in ("monomial", "chebyshev") for side in (0, 1))
        aggregate_controls.extend(check_graph_aggregates(shifted[side], shifted_kernels["shifted"][side],
            domains[side], changed_domains[side], 1, 3, [1, 2, 3, 4]) for side in (0, 1))
    return {"records": records+[mixed], "full_ordered_triples": 4096,
            "forward_inverse_checks": True, "relabelings": config["relabels"],
            "third_moment_difference": (values[0]-values[1]).json(),
            "aggregate_controls": aggregate_controls, "full_graph_aggregate_relabel_checks_passed": True}
