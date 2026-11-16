"""Coverage, paired logical checks, numerical discrepancies, and neural summaries."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import tempfile

from .suite_data import category, reference_group, sha256
from .suite_runtime import atomic_json, digest, validate_record


def write_csv_stream(path, rows, fields):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+".")
    count = 0
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return write_csv_stream(path, rows, fields)


def diagnostic_discrepancies(jobs, out, lookup):
    for job in jobs.values():
        spec = job["job"]
        if spec["stage"] != "diagnostic" or job["status"] != "complete":
            continue
        pid = spec["pair_id"]
        result = json.loads((out/"records"/(spec["id"]+".json")).read_text())["result"]
        for sample in result["diagnostics"]:
            self_pair = sample["sample"] == "official_reliability" or sample["sample"].startswith("self_relabel")
            for readout, observations in (("multiset", sample["multiset"]), ("moments", sample["moments"])):
                for observation in observations:
                    label = "multiset" if readout == "multiset" else f"M{observation['p']}"
                    exact = "equal" if self_pair else lookup.get((pid, spec["order"], spec["q"], spec["basis"], spec["variant"], label))
                    if exact in ("equal", "different") and observation["different"] != (exact == "different"):
                        yield {"job_id": spec["id"], "pair_id": pid, "sample": sample["sample"],
                               "readout": label, "exact_status": exact, **observation}


def summarize(out, verify_artifacts=False):
    out = Path(out)
    manifest = json.loads((out/"manifest.json").read_text())
    token = digest(manifest["identity"])
    if token != manifest["identity_sha256"]:
        raise ValueError("Manifest identity checksum mismatch")
    config = manifest["identity"]["config"]
    requested = json.loads((out/"requested.json").read_text())
    jobs, issues = {}, []
    for jid, job in requested.items():
        path = out/"records"/(jid+".json")
        if not path.exists():
            jobs[jid] = {"job": job, "status": "not_executed"}
            continue
        row = json.loads(path.read_text())
        validate_record(row, job, token, out, config, check_artifacts=verify_artifacts)
        # Keep only small summaries resident. Large diagnostic payloads and exact
        # witnesses are read one job at a time, never accumulated for all jobs.
        if row["status"] == "complete" and job["stage"] == "diagnostic":
            row.pop("result")
        elif row["status"] == "complete" and job["stage"] == "exact":
            row["result"] = {"records": [{**r, "decisions": {k: {"status": v["status"]}
                for k, v in r["decisions"].items()}} for r in row["result"]["records"]]}
        elif row["status"] == "complete" and job["stage"] == "mechanism":
            row.pop("result")
        jobs[jid] = row
    references = {r["job"]["pair_id"]: r["result"] for r in jobs.values()
                  if r["job"]["stage"] == "reference" and r["status"] == "complete"}
    for pid, ref in references.items():
        observed = not ref["two_fwl"]["equal"]
        expected = reference_group(pid) == "2fwl"
        if observed != expected:
            issues.append({"kind": "historical_reference_disagreement", "pair_id": pid,
                           "observed_2fwl_distinction": observed, "historical_group": reference_group(pid)})
    exact_rows, lookup = [], {}
    for order in (2, 3):
        for q in sorted(set(config["q_values"]+([1] if order == 3 else []))):
            variants = ["poly", "endpoint"]+(["joint"] if order == 3 else [])
            if order == 3 and q == 1:
                variants.append("equal19")
            for pid in range(400):
                jid = f"exact_{pid:03d}_o{order}_q{q}"
                job = jobs.get(jid, {"status": "not_requested"})
                records = job.get("result", {}).get("records", [])
                mapping = {(r["basis"], r["variant"]): r for r in records}
                for basis in config["bases"]:
                    for variant in variants:
                        record = mapping.get((basis, variant))
                        for readout in ["multiset", *(f"M{p}" for p in config["powers"])]:
                            status = record["decisions"][readout]["status"] if record else job["status"]
                            if status == "complete":
                                raise ValueError(f"Complete job lacks required decision: {jid}")
                            row = {"pair_id": pid, "category": category(pid), "reference_group": reference_group(pid),
                                   "order": order, "q": q, "basis": basis, "variant": variant,
                                   "readout": readout, "status": status,
                                   "controls_passed": job["status"] == "complete"}
                            exact_rows.append(row)
                            lookup[pid, order, q, basis, variant, readout] = status
                            if status != "different":
                                continue
                            if variant in ("poly", "endpoint") and pid in references and references[pid]["two_fwl"]["equal"]:
                                issues.append({"kind": "computed_2fwl_bound_violation", **row})
                            if (order == 2 or variant in ("poly", "endpoint")) and reference_group(pid) != "2fwl":
                                issues.append({"kind": "historical_2fwl_bound_flag", **row})
                            if order == 3 and reference_group(pid) == "not_3fwl":
                                issues.append({"kind": "historical_3fwl_bound_flag", **row})
    for row in exact_rows:
        if row["status"] != "different":
            continue
        pid, order, q, basis, variant, readout = [row[k] for k in ("pair_id", "order", "q", "basis", "variant", "readout")]
        if readout != "multiset" and lookup[pid, order, q, basis, variant, "multiset"] == "equal":
            issues.append({"kind": "moment_exceeds_multiset", **row})
        for larger in config["q_values"]:
            if larger > q and lookup.get((pid, order, larger, basis, variant, readout)) == "equal":
                issues.append({"kind": "degree_monotonicity_failure", **row, "larger_q": larger})
    aggregates = defaultdict(Counter)
    for r in exact_rows:
        condition = tuple(r[k] for k in ("order", "q", "basis", "variant", "readout"))
        for group in ("all", "category:"+r["category"], "reference:"+r["reference_group"]):
            aggregates[condition+(group,)][r["status"]] += 1
    summary_rows = []
    for key, values in sorted(aggregates.items()):
        denominator = sum(values.values())
        summary_rows.append({**dict(zip(("order", "q", "basis", "variant", "readout", "group"), key)),
            "denominator": denominator, "different": values["different"], "equal": values["equal"],
            "resource_limited": values["resource_limited"], "failed": values["failed"],
            "interrupted": values["interrupted"], "not_executed": values["not_executed"],
            "not_requested": values["not_requested"],
            "distinction_lower_bound": values["different"], "distinction_upper_bound": denominator-values["equal"]})
    neural_rows = []
    for job in jobs.values():
        if job["status"] != "complete":
            continue
        spec, result = job["job"], job.get("result", {})
        pid = spec.get("pair_id")
        if spec["stage"] == "neural":
            aggregate_passed = result["aggregate_relabel_consistency"]["passed"]
            if not aggregate_passed:
                issues.append({"kind": "neural_aggregate_invariance_failure", "job_id": spec["id"]})
            historical_violation = result["numerical_decision"] == "different" and (
                reference_group(pid) == "not_3fwl" or (spec["order"] == 2 and reference_group(pid) != "2fwl"))
            computed_violation = result["numerical_decision"] == "different" and spec["order"] == 2 and pid in references and references[pid]["two_fwl"]["equal"]
            original = result["input_baseline"]["decisions"]["multiset"]["status"]
            neural_rows.append({"pair_id": pid, "order": spec["order"], "seed": spec["seed"],
                "reference_group": reference_group(pid), "input_status": original,
                "numerical_decision": result["numerical_decision"],
                "historical_bound_flag": historical_violation, "computed_bound_flag": computed_violation,
                "physical_batch": result["physical_batch"], "parameters": result["parameters"]})
            neural_rows[-1]["aggregate_invariance_passed"] = aggregate_passed
            if historical_violation or computed_violation:
                issues.append({"kind": "neural_reference_bound_flag", "job_id": spec["id"],
                               "computed_2fwl_violation": computed_violation})
    neural_summary = []
    for order in (2, 3):
        for seed in config["neural_seeds"]:
            for group in ("all", "2fwl", "only_3fwl", "not_3fwl"):
                selected = [r for r in neural_rows if r["order"] == order and r["seed"] == seed
                            and (group == "all" or r["reference_group"] == group)]
                valid = [r for r in selected if r["numerical_decision"] != "unresolved"
                         and not r["historical_bound_flag"] and not r["computed_bound_flag"]]
                neural_summary.append({"order": order, "seed": seed, "group": group,
                    "denominator": sum(group == "all" or reference_group(pid) == group for pid in range(400)),
                    "executed": len(selected), "common_valid": len(valid),
                    "raw_numerical_detections": sum(r["numerical_decision"] == "different" for r in selected),
                    "confirmed_numerical_detections": sum(r["numerical_decision"] == "different" for r in valid),
                    "unresolved": sum(r["numerical_decision"] == "unresolved" for r in selected),
                    "added": sum(r["input_status"] == "equal" and r["numerical_decision"] == "different" for r in valid),
                    "lost": sum(r["input_status"] == "different" and r["numerical_decision"] == "not_detected" for r in valid)})
    folder = out/"reports"
    folder.mkdir(exist_ok=True)
    write_csv(folder/"exact_per_pair.csv", exact_rows)
    write_csv(folder/"exact_summary.csv", summary_rows)
    discrepancy_count = write_csv_stream(folder/"numerical_discrepancies.csv",
        diagnostic_discrepancies(jobs, out, lookup),
        ["job_id", "pair_id", "sample", "readout", "exact_status", "scale", "decimals", "different",
         "p", "atol", "rtol", "moment_decimals"])
    write_csv(folder/"neural_per_pair.csv", neural_rows)
    write_csv(folder/"neural_summary.csv", neural_summary)
    coverage = {stage: dict(Counter(row["status"] for row in jobs.values() if row["job"]["stage"] == stage))
                for stage in ("reference", "mechanism", "exact", "diagnostic", "neural")}
    summary = {"identity_sha256": token, "coverage": coverage, "issues": issues,
               "numerical_discrepancies": discrepancy_count,
               "all_requested_jobs_complete": all(r["status"] == "complete" for r in jobs.values()),
               "safe_to_use_as_complete_results": not issues and all(r["status"] == "complete" for r in jobs.values()),
               "warning": "Completion applies only to requested jobs. Consult each 400-pair denominator and missing counts."}
    atomic_json(folder/"summary.json", summary)
    atomic_json(folder/"issues.json", issues)
    return summary
