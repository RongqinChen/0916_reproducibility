"""Strict process policy for BREC entry points only (stdlib on import).

NumPy has no use_deterministic_algorithms switch. BREC therefore fixes the
process environment, BLAS/OpenMP threads, RNGs and supported arithmetic path.
This does not equate reproducibility with permutation invariance or accuracy.
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
from pathlib import Path

POLICY_VERSION = 'brec-determinism-v1'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
PROCESS_ENVIRONMENT = {**{key: '1' for key in THREAD_VARIABLES},
                       'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'}
_STARTUP_HASHSEED = os.environ.get('PYTHONHASHSEED')
_LIMITER = None
_LIMITERS = []


def child_environment(environment=None):
    result = dict(os.environ if environment is None else environment)
    result.update(PROCESS_ENVIRONMENT)
    return result


def prepare_environment():
    """Call before importing NumPy/PyTorch in a BREC process."""
    torch = sys.modules.get('torch')
    if (torch is not None and torch.cuda.is_initialized()
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise RuntimeError('Start BREC in a fresh process: CUDA was initialized before its deterministic workspace was set')
    for key, value in PROCESS_ENVIRONMENT.items():
        if key != 'PYTHONHASHSEED':
            os.environ[key] = value


def ensure_startup():
    """CLI-only: restart once if Python's hash seed was not fixed at startup."""
    if sys.flags.ignore_environment:
        raise RuntimeError('BREC requires environment settings; do not use Python -E/-I')
    if _STARTUP_HASHSEED != '0':
        arguments = getattr(sys, 'orig_argv', None)
        if not arguments:
            raise RuntimeError('Start BREC with PYTHONHASHSEED=0 in a fresh process')
        os.execve(sys.executable, [sys.executable, *arguments[1:]], child_environment())
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('BREC hash-seed environment changed during this process')
    prepare_environment()


def numpy_settings():
    import numpy as np
    from threadpoolctl import threadpool_info
    pools = [{key: item.get(key) for key in ('user_api', 'internal_api', 'prefix',
                                            'version', 'num_threads', 'architecture')}
             for item in threadpool_info()]
    pools.sort(key=lambda item: (str(item['user_api']), str(item['prefix']), str(item['version'])))
    return {'numpy_version': np.__version__, 'threadpools': pools,
            'rng_policy': 'explicit PCG64 generators; global NumPy and Python RNGs seeded',
            'numpy_global_deterministic_switch': False}


def configure_numpy(seed=0):
    global _LIMITER
    prepare_environment()
    import numpy as np
    from threadpoolctl import threadpool_limits
    # Keep the limiter alive, including when NumPy was loaded by an embedding
    # application before this BREC entry point.
    _LIMITER = threadpool_limits(limits=1)
    _LIMITERS.append(_LIMITER)
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    settings = numpy_settings()
    if any(pool['num_threads'] != 1 for pool in settings['threadpools']):
        raise RuntimeError('Cannot enforce single-threaded numeric libraries for BREC')
    return settings


def assert_torch_determinism(torch):
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or not torch.backends.cudnn.deterministic
            or torch.backends.cudnn.benchmark
            or torch.get_num_threads() != 1
            or torch.get_num_interop_threads() != 1
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise RuntimeError('BREC deterministic policy was disabled or changed')


def configure_torch(seed=None):
    prepare_environment()
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    assert_torch_determinism(torch)
    return torch


def runtime_identity(use_torch=False):
    result = {'policy': POLICY_VERSION, 'environment': {key: os.environ.get(key) for key in PROCESS_ENVIRONMENT},
              'python': platform.python_version(), 'platform': platform.platform(),
              'machine': platform.machine(), 'processor': platform.processor(),
              'policy_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy': numpy_settings(),
              'scope': 'same ordered inputs, seed, scientific configuration, software and hardware; replay comparison required'}
    if use_torch:
        import torch
        assert_torch_determinism(torch)
        result['torch'] = {'version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                           'cudnn_version': torch.backends.cudnn.version(),
                           'deterministic_algorithms': True, 'warn_only': False,
                           'cudnn_deterministic': True, 'cudnn_benchmark': False,
                           'threads': torch.get_num_threads(),
                           'interop_threads': torch.get_num_interop_threads()}
    return result


def start_brec(seed=0, use_torch=False):
    ensure_startup()
    configure_numpy(seed)
    if use_torch:
        configure_torch(seed)
        # Torch may load another OpenMP runtime; constrain/check it as well.
        configure_numpy(seed)
    return runtime_identity(use_torch)


def is_determinism_error(error):
    message = str(error).lower()
    return 'deterministic' in message or 'cublas_workspace_config' in message
