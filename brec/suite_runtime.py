"""Isolated jobs, bounded aggregate RSS, atomic records, and strict resume identity."""
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil

from .suite_data import graphs, load_data, sha256

ROOT = Path(__file__).resolve().parent.parent
GIB = 2**30
STAGES = ("reference", "mechanism", "exact", "diagnostic", "neural")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_record(path, value):
    atomic_json(path, {**value, "record_sha256": digest(value)})


def file_stamp(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def validate_record(record, job, token, out, config, check_artifacts=True):
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    if record.get("record_sha256") != digest(payload):
        raise ValueError(f"Record checksum mismatch: {job['id']}")
    if record.get("identity_sha256") != token or any(record.get("job", {}).get(k) != v for k, v in job.items()):
        raise ValueError(f"Record condition or identity mismatch: {job['id']}")
    if record.get("status") not in {"complete", "failed", "resource_limited", "interrupted"}:
        raise ValueError("Unknown job status")
    if record["status"] != "complete":
        return
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("Complete record has no result")
    if job["stage"] == "exact":
        from collections import Counter
        expected = {"repeat": 1, "main_relabel": config["relabels"], "official_reliability": 1,
                    "official_reliability_relabel": config["reliability_relabels"]}
        if Counter(c["kind"] for c in result.get("controls", [])) != Counter(expected):
            raise ValueError("Exact record lacks required reliability controls")
        if result.get("full_graph_aggregate_relabel_checks_passed") is not True:
            raise ValueError("Exact record lacks graph aggregate invariance checks")
        for control in result["controls"]:
            if "relabel" in control["kind"]:
                aggregates = control.get("aggregates", [])
                if len(aggregates) != 2*len(config["bases"]) or not all(
                    a.get("full_graph_multiset_equal") is True and a.get("all_channel_moments_equal") is True
                    for a in aggregates):
                    raise ValueError("Exact aggregate invariance evidence is incomplete")
    if job["stage"] == "neural":
        expected = {str(Path("artifacts")/job["id"]/name) for name in ("model.pt", "embeddings.npz")}
        if not expected.issubset(record.get("artifacts", {})):
            raise ValueError("Neural record lacks model or embedding artifacts")
        if "aggregate_relabel_consistency" not in result:
            raise ValueError("Neural record lacks aggregate consistency checks")
    if check_artifacts:
        for name, expected in record.get("artifacts", {}).items():
            target = (Path(out)/name).resolve()
            if not target.is_relative_to(Path(out).resolve()) or not target.is_file() or sha256(target) != expected:
                raise ValueError(f"Artifact missing or checksum mismatch: {name}")


def protocol(config):
    integers = ("q_values", "powers", "diagnostic_q_values", "neural_seeds")
    for name in integers:
        value = config.get(name)
        if not isinstance(value, list) or not value or any(type(x) is not int or x < 1 for x in value) or len(set(value)) != len(value):
            raise ValueError(f"{name} must contain distinct positive integers")
    if config["q_values"] != sorted(config["q_values"]) or config["powers"] != sorted(config["powers"]):
        raise ValueError("Degree and moment budgets must be increasing")
    if config.get("bases") != ["monomial", "chebyshev"]:
        raise ValueError("Both complete bases are required in the fixed protocol")
    for name in ("relabels", "reliability_relabels", "diagnostic_relabels", "seed"):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if min(config["relabels"], config["reliability_relabels"]) < 1:
        raise ValueError("At least one main and official reliability relabeling is required")
    for name in ("neural_invariance_atol", "neural_invariance_rtol"):
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not config["atols"] or any(not isinstance(v, (int, float)) or not 0 <= v < float("inf") for v in config["atols"]):
        raise ValueError("Invalid numerical tolerances")
    if not config["decimals"] or any(v is not None and (type(v) is not int or not 0 <= v <= 15) for v in config["decimals"]):
        raise ValueError("Invalid rounding settings")
    if not isinstance(config.get("expected_sha256"), str) or len(config["expected_sha256"]) != 64:
        raise ValueError("A frozen dataset SHA256 is required")
    return config


def make_jobs(config, pair_ids, stages):
    jobs = []
    for stage in STAGES:
        if stage not in stages:
            continue
        if stage == "mechanism":
            jobs.append({"stage": stage, "id": "mechanism"})
            continue
        for pid in pair_ids:
            base = {"stage": stage, "pair_id": pid}
            if stage == "reference":
                jobs.append({**base, "id": f"reference_{pid:03d}"})
            elif stage == "exact":
                for order in (2, 3):
                    for q in sorted(set(config["q_values"]+([1] if order == 3 else []))):
                        jobs.append({**base, "order": order, "q": q, "id": f"exact_{pid:03d}_o{order}_q{q}"})
            elif stage == "diagnostic":
                for order, variants in ((2, ["poly", "endpoint"]), (3, ["poly", "endpoint", "joint"])):
                    for q in config["diagnostic_q_values"]:
                        for basis in config["bases"]:
                            for variant in variants:
                                for dtype in ("float32", "float64"):
                                    jobs.append({**base, "q": q, "order": order, "basis": basis,
                                        "variant": variant, "dtype": dtype,
                                        "id": f"diagnostic_{pid:03d}_o{order}_q{q}_{basis}_{variant}_{dtype}"})
            elif stage == "neural":
                for order in (2, 3):
                    for seed in config["neural_seeds"]:
                        jobs.append({**base, "order": order, "seed": seed,
                                     "id": f"neural_{pid:03d}_o{order}_s{seed}"})
    return jobs


def identity(config, dataset):
    files = sorted((ROOT/"brec").glob("suite_*.py"))
    files += [ROOT/"brec"/"__init__.py", ROOT/"brec"/"determinism.py", ROOT/"reproduce.py",
              ROOT/"configs"/"brec_suite.json", ROOT/"requirements"/"brec_suite.txt"]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    versions = {}
    for package in ("numpy", "networkx", "pynauty", "python-flint", "psutil", "threadpoolctl", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"config": config, "dataset_sha256": sha256(dataset), "sources": sources,
            "python": platform.python_version(), "packages": versions,
            "platform": platform.platform(), "machine": platform.machine()}


def reservation(job, nodes):
    stage, order = job["stage"], job.get("order", 2)
    if stage == "mechanism":
        return 2.
    if stage == "reference":
        return max(1., nodes**3*12/GIB+1.)
    if stage == "neural":
        return max(3., min(84., nodes**order*27*4*20*2/GIB+2.))
    if stage == "diagnostic":
        return max(2., nodes**order*800/GIB+1.)
    # Python exact numbers and row-color dictionaries can dominate storage.
    # Orbit compression often lowers actual use. RSS remains actively monitored.
    return max(2., (nodes**order*600 + (job["q"]+1)*nodes**2*400)/GIB+1.)


def tree_rss(pid):
    # Workers do not launch child processes. Avoid a system-wide PID scan:
    # it is unnecessary and is prohibited in some desktop sandboxes.
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run(args, config):
    from .suite_resources import Resources, enforce_address_space_limit
    resources = Resources()
    available = min(args.memory_gib, resources.memory_total()/GIB)-args.reserve_gib
    if available <= args.parent_memory_gib:
        raise ValueError("No worker memory remains after parent and operating-system reservations")
    args.parent_address_space_limit = enforce_address_space_limit(args.parent_memory_gib, coordinator=True)
    args.dataset_stamp = file_stamp(args.data)
    data = load_data(args.data, config["expected_sha256"])
    if file_stamp(args.data) != args.dataset_stamp:
        raise ValueError("Dataset changed during checksum verification")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/".writer.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError("Another runner owns this output directory") from exc
    current = identity(config, args.data)
    manifest = out/"manifest.json"
    if manifest.exists():
        if not args.resume:
            raise ValueError("Output already contains a run; use --resume or a new directory")
        if json.loads(manifest.read_text())["identity"] != current:
            raise ValueError("Resume refused: code, configuration, data, or environment changed")
    else:
        atomic_json(manifest, {"identity": current, "identity_sha256": digest(current)})
        for name in current["sources"]:
            target = out/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        atomic_json(out/"source"/"effective_config.json", config)
    token = digest(current)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    jobs = make_jobs(config, args.pairs, stages)
    sizes = {pid: len(graphs(data, pid)[0]) for pid in args.pairs}
    for job in jobs:
        if job["stage"] == "neural":
            n, order = sizes[job["pair_id"]], job["order"]
            per_graph = n**order*max(27 if order == 3 else 9, 16)*4
            # Freeze batch policy independently of concurrent jobs and retry resource limits.
            base_memory = reservation(job, n)
            job["physical_batch"] = next((b for b in (32, 16, 8, 4, 2)
                if per_graph*b*20 < base_memory*GIB*.7), 2)
    # Persist the union of requested jobs. Reports never call unrequested cases equal.
    requested_path = out/"requested.json"
    requested = json.loads(requested_path.read_text()) if requested_path.exists() else {}
    requested.update({j["id"]: j for j in jobs})
    atomic_json(requested_path, requested)
    cpu_limit = resources.cpus()
    workers = min(args.workers, cpu_limit)
    pending = []
    for job in jobs:
        path = out/"records"/(job["id"]+".json")
        if path.exists():
            record = json.loads(path.read_text())
            validate_record(record, job, token, out, config)
            if record["status"] == "complete" or not args.retry_incomplete:
                continue
            archive = out/"previous_attempts"/(job["id"]+f"_{time.time_ns()}.json")
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, archive)
            path.unlink()
        job = {**job, "memory_gib": min(available-args.parent_memory_gib,
                    reservation(job, sizes.get(job.get("pair_id"), 16))*args.job_memory_scale)}
        pending.append(job)
    from .suite_scheduler import supervise
    try:
        supervise(pending, out, token, config, args, available, workers,
                  free_memory=lambda: resources.memory_available()/GIB)
    finally:
        lock.close()
    return out


def worker(path):
    import traceback
    spec = json.loads(Path(path).read_text())
    job, config = spec["job"], spec["config"]
    out = Path(spec["out"])
    record = {"job": job, "identity_sha256": spec["identity_sha256"]}
    started = time.monotonic()
    try:
        from .suite_resources import enforce_address_space_limit
        record["address_space_limit"] = enforce_address_space_limit(job["memory_gib"],
            original_soft_limit=spec.get("parent_address_space_limit", {}).get("original_soft_limit"))
        manifest = json.loads((out/"manifest.json").read_text())
        frozen = manifest["identity"]
        if digest(frozen) != spec["identity_sha256"] or frozen["config"] != config:
            raise ValueError("Worker protocol identity mismatch")
        requested = json.loads((out/"requested.json").read_text()).get(job["id"])
        if requested is None or any(job.get(key) != value for key, value in requested.items()):
            raise ValueError("Worker job does not match the requested experiment")
        if file_stamp(spec["data"]) != spec["dataset_stamp"]:
            raise ValueError("Dataset changed after admission")
        for name, checksum in frozen["sources"].items():
            if sha256(ROOT/name) != checksum:
                raise ValueError("Frozen worker source changed")
        for name, version in frozen["packages"].items():
            try:
                current = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                current = None
            if current != version:
                raise ValueError("Worker package environment changed")
        # Parent verified the whole file; workers read only required fixed-width records.
        from .suite_data import Graph6Store
        data = Graph6Store(spec["data"])
        stage = job["stage"]
        from .suite_structural import run_exact, run_reference, run_mechanism
        if stage == "exact":
            result = run_exact(data, job, config, lambda value: atomic_json(
                out/"progress"/(job["id"]+".json"), {"identity_sha256": spec["identity_sha256"],
                    "warning": "Partial output; reliability controls may be unfinished", **value}))
        elif stage == "reference":
            result = run_reference(data, job, config)
        elif stage == "mechanism":
            result = run_mechanism(config)
        elif stage == "diagnostic":
            from .suite_numerical import run_diagnostic
            result = run_diagnostic(data, job, config)
        elif stage == "neural":
            from .suite_neural import run_neural
            result = run_neural(data, job, config, out/"artifacts"/job["id"])
        else:
            raise ValueError("Unknown worker stage")
        if file_stamp(spec["data"]) != spec["dataset_stamp"]:
            raise ValueError("Dataset changed while the job was running")
        record.update(status="complete", result=result)
        artifacts = out/"artifacts"/job["id"]
        if artifacts.exists():
            record["artifacts"] = {str(p.relative_to(out)): sha256(p) for p in artifacts.iterdir() if p.is_file()}
    except MemoryError as exc:
        record.update(status="resource_limited", reason=str(exc))
    except Exception as exc:
        record.update(status="failed", reason=str(exc), traceback=traceback.format_exc())
        print(record["traceback"], flush=True)
    record["elapsed_seconds"] = time.monotonic()-started
    record["final_rss_bytes"] = psutil.Process().memory_info().rss
    write_record(out/"records"/(job["id"]+".json"), record)
