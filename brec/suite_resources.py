"""Read host, CPU-affinity, and standard Linux cgroup-v2 limits without changing them."""
import os
from pathlib import Path
import sys

import psutil


def enforce_address_space_limit(gib, coordinator=False, original_soft_limit=None):
    """Linux allocator limit, enforced by the kernel rather than a polling loop.

    A process's resident user pages are bounded by its mapped address space.
    Counting full address spaces is conservative because shared mappings are
    counted separately. Native allocation failure may terminate a worker; the
    supervisor retains that job as failed/resource-limited, never as a negative.
    """
    requested = int(gib*2**30)
    if not sys.platform.startswith("linux"):
        return {"enforced": False, "requested_bytes": requested,
                "reason": "Hard address-space enforcement is enabled on Linux only"}
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    external_soft = soft if original_soft_limit is None else original_soft_limit
    finite = [requested]+[v for v in (external_soft, hard) if v != resource.RLIM_INFINITY]
    limit = min(finite)
    # Keep the coordinator's original hard ceiling so children can receive a
    # larger reservation. Both soft and hard limits are fixed in each worker.
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard if coordinator else limit))
    return {"enforced": True, "kind": "RLIMIT_AS", "limit_bytes": limit,
            "original_soft_limit": soft, "coordinator": coordinator}


def cgroup_paths(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    if not root.is_dir() or not membership.is_file():
        return ()
    root = root.resolve()
    relative = next((line.split("::", 1)[1] for line in membership.read_text().splitlines()
                     if line.startswith("0::")), None)
    if relative is None:
        return ()
    leaf = (root/relative.lstrip("/")).resolve()
    if not leaf.is_relative_to(root) or not leaf.is_dir():
        leaf = root  # A container may expose its own cgroup as the mount root.
    return tuple(p for p in (leaf, *leaf.parents) if p.is_relative_to(root))


def _integer(path):
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except FileNotFoundError:
        return None


class Resources:
    def __init__(self, paths=None):
        self.paths = cgroup_paths() if paths is None else paths

    def memory_total(self):
        limits = [psutil.virtual_memory().total]
        for path in self.paths:
            maximum = _integer(path/"memory.max")
            if maximum is not None:
                limits.append(maximum)
        return min(limits)

    def memory_available(self):
        limits = [psutil.virtual_memory().available]
        for path in self.paths:
            maximum, current = _integer(path/"memory.max"), _integer(path/"memory.current")
            if maximum is not None and current is not None:
                limits.append(max(0, maximum-current))
        return min(limits)

    def cpus(self):
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        for path in self.paths:
            try:
                quota, period = (path/"cpu.max").read_text().split()
            except FileNotFoundError:
                continue
            if quota != "max":
                allowed = min(allowed, max(1, int(quota)//int(period)))
        return allowed
