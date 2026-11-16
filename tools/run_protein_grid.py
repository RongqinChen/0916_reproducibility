#!/usr/bin/env python3
"""Run the GNNPlus GINE TU grid as resumable encoding/width shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/rebuttal/protein.json"

state_lock = threading.Lock()


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable is unavailable")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("TU_DATA_ROOT", "data/TUD")))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable,
                        help="Python from the original Co-Sparsify environment")
    parser.add_argument("--datasets", nargs="+", choices=("PROTEINS_full", "ENZYMES"))
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse an existing run root and completed shard records")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the selected matrix and print task planning only")
    return parser.parse_args(argv)


def load_plan(args):
    config = json.loads(args.config.resolve().read_text())
    if config.get("protocol") != "gnnplus-gine-tu-structural-edge-v1":
        raise ValueError("Unexpected protein experiment protocol")
    datasets = args.datasets or config["datasets"]
    if len(datasets) != len(set(datasets)) or not set(datasets) <= set(config["datasets"]):
        raise ValueError("Invalid dataset selection")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    if args.reserve_gib <= 0:
        raise ValueError("--reserve-gib must be positive")
    reference = (args.reference_root or Path(config["reference_root"])).resolve()
    tasks = [
        (dataset, encoding, width)
        for dataset in datasets
        for width in config["poly_dims"]
        for encoding in config["encodings"]
    ]
    return config, datasets, reference, tasks


def validate_reference(reference, config):
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["reference_commit"]:
        raise ValueError(
            f"Reference commit is {commit}, expected {config['reference_commit']}"
        )
    dirty = subprocess.check_output(
        [
            "git", "-C", str(reference), "status", "--porcelain",
            "--untracked-files=all", "--", "*.py", "*.yaml",
        ],
        text=True,
    )
    if dirty.strip():
        raise ValueError("The Co-Sparsify Python/YAML tree must be unmodified")
    reference_config = reference / config["reference_config"]
    if not reference_config.is_file():
        raise FileNotFoundError(reference_config)


def discover_gpus(requested):
    if requested:
        if len(requested) != len(set(requested)) or any(gpu < 0 for gpu in requested):
            raise ValueError("GPU indices must be unique and nonnegative")
        return requested
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def validate_python(python, reference):
    code = r"""
import importlib.metadata
import importlib
import sys
sys.path.insert(0, sys.argv[1])
modules = ('torch', 'torch_geometric', 'torch_scatter', 'torch_sparse',
           'lightning', 'comet_ml', 'numpy', 'sklearn', 'sage')
for module in modules:
    importlib.import_module(module)
    try:
        version = importlib.metadata.version(module.replace('_', '-'))
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown'
    print(f'{module}={version}')
import torch
print(f'python={sys.version.split()[0]}')
print(f'cuda_available={torch.cuda.is_available()}')
"""
    subprocess.run([python, "-c", code, str(reference)], cwd=reference, check=True)


def prewarm_command(python, reference, data_root, dataset, folds):
    code = r"""
import sys
reference, data_root, dataset, folds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, reference)
from datasets import TUDataset
data = TUDataset(dataset, data_root, None, None, None)
for fold in map(int, folds):
    data.get_split(fold)
print(f'prepared dataset={dataset} graphs={len(data)} folds={folds}')
"""
    return [
        python, "-c", code, str(reference), str(data_root), dataset,
        *map(str, folds),
    ]


def shard_command(python, config_path, reference, data_root, dataset, encoding, width, out):
    return [
        python, "-m", "rebuttal_experiments", "protein", "run",
        "--config", str(config_path),
        "--reference-root", str(reference),
        "--dataset", dataset,
        "--data-root", str(data_root),
        "--out", str(out),
        "--encodings", encoding,
        "--poly-dims", str(width),
        "--resume",
    ]


def report_command(python, dataset_root, shards):
    return [
        python, "-m", "rebuttal_experiments", "protein", "report",
        "--out", str(dataset_root),
        "--shards", *map(str, shards),
    ]


def run_logged(label, command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(f"{now()} START {label}\n")
        stream.write(" ".join(command) + "\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
        stream.write(f"\n{now()} EXIT {returncode} {label}\n")
    if returncode:
        raise RuntimeError(f"{label} exited with status {returncode}; see {log_path}")


def wait_for_memory(label, reserve_gib):
    while available_gib() < reserve_gib:
        print(
            f"{now()} waiting for {label}: MemAvailable={available_gib():.2f} GiB",
            flush=True,
        )
        time.sleep(15)


def main(argv=None):
    args = parse_args(argv)
    config_path = args.config.resolve()
    config, datasets, reference, tasks = load_plan(args)
    args.data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    if args.dry_run:
        print(json.dumps({
            "tasks": len(tasks),
            "native_training_runs": len(tasks) * len(config["folds"]),
            "datasets": datasets,
            "reference_root": str(reference),
            "data_root": str(args.data_root),
            "output_root": str(output_root),
            "task_list": tasks,
        }, indent=2))
        return

    validate_reference(reference, config)
    gpus = discover_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No CUDA GPU was detected")
    validate_python(args.python, reference)
    if (output_root / "run_status.json").exists() and not args.resume:
        raise FileExistsError(f"{output_root} exists; pass --resume to continue it")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    python_bin = str(Path(args.python).resolve().parent)
    base_env["PATH"] = python_bin + os.pathsep + base_env.get("PATH", "")
    status_path = output_root / "run_status.json"
    status = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "reference_commit": config["reference_commit"],
        "profile_count": len(tasks),
        "native_training_runs": len(tasks) * len(config["folds"]),
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "datasets": datasets,
        "preparation": {},
        "profiles": {},
        "reports": {},
        "errors": [],
    }
    atomic_json(status_path, status)

    def update(section, key, value):
        with state_lock:
            if key is None:
                status[section] = value
            else:
                status[section][key] = value
            status["updated_at"] = now()
            atomic_json(status_path, status)

    args.data_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        update("preparation", dataset, "running")
        try:
            run_logged(
                f"prewarm-{dataset}",
                prewarm_command(
                    args.python, reference, args.data_root, dataset, config["folds"]
                ),
                log_root / f"prewarm_{dataset}.log",
                base_env,
            )
            update("preparation", dataset, "complete")
        except BaseException as exc:
            message = f"prewarm {dataset}: {type(exc).__name__}: {exc}"
            update("preparation", dataset, "failed")
            with state_lock:
                status["errors"].append(message)
            update("status", None, "failed")
            raise SystemExit(1) from exc

    shard_roots = {dataset: [] for dataset in datasets}
    task_by_key = {}
    for dataset, encoding, width in tasks:
        shard = output_root / "datasets" / dataset / "shards" / f"{encoding}_k{width}"
        key = f"{dataset}/{encoding}/k{width}"
        shard_roots[dataset].append(shard)
        task_by_key[key] = (dataset, encoding, width)
        update("profiles", key, "queued")

    pending = deque(tasks)
    gpu_slots = list(gpus) * args.workers_per_gpu
    available = deque(gpu_slots)
    running = {}
    failures = []

    def execute(task, gpu):
        dataset, encoding, width = task
        key = f"{dataset}/{encoding}/k{width}"
        wait_for_memory(key, args.reserve_gib)
        update("profiles", key, f"running_cuda_{gpu}")
        shard = output_root / "datasets" / dataset / "shards" / f"{encoding}_k{width}"
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        run_logged(
            key,
            shard_command(
                args.python, config_path, reference, args.data_root,
                dataset, encoding, width, shard,
            ),
            log_root / f"{dataset}_{encoding}_k{width}.log",
            env,
        )
        update("profiles", key, "complete")
        return task

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_slots)) as executor:
        while pending or running:
            while pending and available and not failures:
                task = pending.popleft()
                gpu = available.popleft()
                future = executor.submit(execute, task, gpu)
                running[future] = (task, gpu)
            if not running:
                break
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task, gpu = running.pop(future)
                available.append(gpu)
                try:
                    future.result()
                except BaseException as exc:
                    message = f"{task}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    with state_lock:
                        status["errors"].append(message)
                    update("profiles", f"{task[0]}/{task[1]}/k{task[2]}", "failed")
                    while pending:
                        queued = pending.popleft()
                        update("profiles", f"{queued[0]}/{queued[1]}/k{queued[2]}", "skipped")

    if failures:
        update("status", None, "failed")
        raise SystemExit(1)

    for dataset in datasets:
        dataset_root = output_root / "datasets" / dataset
        update("reports", dataset, "running")
        try:
            run_logged(
                f"report-{dataset}",
                report_command(args.python, dataset_root, shard_roots[dataset]),
                log_root / f"report_{dataset}.log",
                base_env,
            )
            update("reports", dataset, "complete")
        except BaseException as exc:
            message = f"report {dataset}: {type(exc).__name__}: {exc}"
            update("reports", dataset, "failed")
            with state_lock:
                status["errors"].append(message)
            update("status", None, "failed")
            raise SystemExit(1) from exc
    update("status", None, "complete")
    print(f"{now()} complete: {output_root}")


if __name__ == "__main__":
    main()
