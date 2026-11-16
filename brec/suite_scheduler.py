"""Memory admission, process supervision, interruption cleanup, and telemetry."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def supervise(jobs, out, token, config, args, process_budget_gib, workers,
              launch_factory=None, free_memory=None, rss=None, poll_seconds=.2):
    from .suite_runtime import GIB, STAGES, atomic_json, stop, tree_rss, write_record, file_stamp
    pending = sorted(jobs, key=lambda j: (STAGES.index(j["stage"]), -j["memory_gib"], j["id"]))
    free_memory = free_memory or (lambda: psutil.virtual_memory().available/GIB)
    rss = rss or tree_rss
    active, completed, peak_workers, peak_rss = {}, 0, 0, 0
    began = time.monotonic()
    last_heartbeat, blocked_since = began, None
    interruption = None
    admission_timeout = args.admission_timeout
    telemetry = {"workers_limit": workers, "process_budget_gib": process_budget_gib,
                 "system_reserve_gib": args.reserve_gib,
                 "parent_address_space_limit": getattr(args, "parent_address_space_limit", None),
                 "parent_memory_reservation_gib": getattr(args, "parent_memory_gib", 0)}
    invocation = out/"scheduler"/f"{time.time_ns()}.json"

    def heartbeat(final=False):
        value = {**telemetry, "completed": completed, "pending": len(pending),
                 "active": [{"job_id": j["id"], "pid": p.pid,
                             "elapsed_seconds": time.monotonic()-started,
                             "reservation_gib": j["memory_gib"], "rss_bytes": rss(p.pid)}
                            for p, j, started, _ in active.values()],
                 "peak_active_workers": peak_workers, "peak_observed_worker_rss_bytes": peak_rss,
                 "elapsed_seconds": time.monotonic()-began, "finished": final}
        atomic_json(out/"status.json", value)
        if final:
            atomic_json(invocation, value)
        return value

    def failure(job, reason, status="resource_limited"):
        target = out/"records"/(job["id"]+".json")
        if not target.exists():
            write_record(target, {"identity_sha256": token, "job": job, "status": status,
                "reason": reason, "partial_progress": str(Path("progress")/(job["id"]+".json"))})

    def launch(job):
        if launch_factory:
            return launch_factory(job)
        spec = {"job": job, "data": str(args.data.resolve()), "out": str(out), "config": config,
                "identity_sha256": token, "dataset_stamp": args.dataset_stamp,
                "parent_address_space_limit": args.parent_address_space_limit}
        job_path = out/"jobs"/(job["id"]+".json")
        atomic_json(job_path, spec)
        log_path = out/"logs"/(job["id"]+".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a")
        snapshot = out/"source"
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(snapshot))
        try:
            process = subprocess.Popen([sys.executable, str(snapshot/"reproduce.py"), "brec-worker", str(job_path)],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=snapshot, start_new_session=True)
        except BaseException:
            log.close()
            raise
        return process, log

    def interrupted(signum, frame):
        # Defer raising until the child has been registered. Raising between
        # Popen() and insertion into active could otherwise orphan that child.
        nonlocal interruption
        interruption = signum

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, interrupted)
    print(f"Queued {len(pending)} jobs; at most {workers} single-thread workers; "
          f"{process_budget_gib:.1f} GiB process budget", flush=True)
    try:
        while pending or active:
            if interruption is not None:
                raise KeyboardInterrupt(f"Scheduler received signal {interruption}")
            observed = {}
            for key, (process, job, started, log) in list(active.items()):
                elapsed, used = time.monotonic()-started, rss(process.pid)
                observed[key] = used
                if process.poll() is None and (used > job["memory_gib"]*GIB or
                                               (args.job_timeout and elapsed > args.job_timeout)):
                    reason = "worker RSS limit" if used > job["memory_gib"]*GIB else "worker wall-time limit"
                    stop(process)
                    failure(job, reason)
                if process.poll() is not None:
                    log.close()
                    if not (out/"records"/(key+".json")).exists():
                        failure(job, f"Worker exited with code {process.returncode}; see its log", "failed")
                    del active[key]
                    completed += 1
                    print(f"Finished {key}; {completed} jobs finished in this invocation", flush=True)
            parent_rss = psutil.Process().memory_info().rss
            used_workers = sum(observed.get(key, 0) for key in active)
            peak_rss = max(peak_rss, used_workers)
            if parent_rss+used_workers > process_budget_gib*GIB and active:
                key = max(active, key=lambda name: observed.get(name, 0))
                process, job, _, _ = active[key]
                stop(process)
                failure(job, "aggregate process RSS limit")
                continue
            parent_reservation = max(parent_rss/GIB, getattr(args, "parent_memory_gib", 0))
            reserved = parent_reservation+sum(j["memory_gib"] for _, j, _, _ in active.values())
            free = free_memory()
            launched = False
            # One memory snapshot per scheduling pass, not one system call per pending job.
            for index, job in enumerate(pending):
                if len(active) >= workers:
                    break
                if reserved+job["memory_gib"] > process_budget_gib or free-job["memory_gib"] < args.reserve_gib:
                    continue
                process, log = launch(job)
                active[job["id"]] = process, job, time.monotonic(), log
                peak_workers = max(peak_workers, len(active))
                pending.pop(index)
                launched = True
                if completed == 0 and len(active) == 1:
                    heartbeat()
                break
            if pending and not active and not launched:
                blocked_since = blocked_since or time.monotonic()
                if time.monotonic()-blocked_since >= admission_timeout:
                    for job in pending:
                        failure(job, "Memory admission timed out; no experiment was executed")
                    pending.clear()
            else:
                blocked_since = None
            if time.monotonic()-last_heartbeat >= 5:
                heartbeat()
                last_heartbeat = time.monotonic()
            if not launched:
                time.sleep(poll_seconds)
    except BaseException:
        telemetry["interrupted"] = True
        # Ignore further interrupts while reaping all owned process groups.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        for process, job, _, log in active.values():
            stop(process)
            log.close()
            failure(job, "Runner interrupted", "interrupted")
        active.clear()
        raise
    finally:
        try:
            heartbeat(final=True)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return telemetry
