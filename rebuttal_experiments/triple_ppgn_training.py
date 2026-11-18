"""Fixed-float32 paired BREC training with the borrowed triple PPGN."""
from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

from brec.determinism import prepare_environment, configure_numpy, configure_torch
prepare_environment()
import numpy as np
import torch
from torch import nn

from brec.ppgn_realization import rpc_score
from brec.theory import group_for_pair
from .brec_data import graph_at, load_brec_data
from .common import atomic_json, atomic_npz, atomic_output, sha256, source_identity
from .legacy_triple_ppgn import RealizationPPGN
from .runtime import report_phase, report_to
from .triple_ppgn_inputs import TripleInputs


def embedding_cached_step(model, indices, get_input, optimizer, clip=1.):
    """Exact chain-rule gradient; no batch-dependent or stochastic modules allowed.

    The borrowed network has neither batch normalization nor dropout. Its
    per-graph embeddings can therefore be recomputed separately with cached
    d(loss)/d(embedding), retaining only one graph's activation tape at a time.
    """
    if len(indices) % 2:
        raise ValueError('Alternating graph pairs required')
    if any(isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))
           for module in model.modules()):
        raise ValueError('Embedding gradient caching requires a stateless deterministic model')
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        embeddings = torch.cat([model(get_input(index)).detach() for index in indices])
    embeddings.requires_grad_()
    targets = -torch.ones(len(indices)//2, device=embeddings.device, dtype=embeddings.dtype)
    loss = torch.nn.functional.cosine_embedding_loss(embeddings[::2], embeddings[1::2], targets)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    gradients, = torch.autograd.grad(loss, embeddings)
    for offset, index in enumerate(indices):
        output = model(get_input(index))
        output.backward(gradients[offset:offset+1])
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
        raise FloatingPointError('Nonfinite or missing model gradient')
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return float(loss.detach())


def train(model, inputs, variant, config, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    history = []
    logical = config['effective_batch']
    for epoch in range(config['epochs']):
        losses = []
        for start in range(0, 64, logical):
            report_phase('training', variant=variant, epoch=epoch+1, epochs=config['epochs'],
                         logical_batch=start//logical+1)
            loss = embedding_cached_step(model, list(range(start, start+logical)),
                                         lambda index: inputs.tensor(index, variant, device), optimizer)
            losses.append(loss)
        mean = float(np.mean(losses))
        history.append({'epoch': epoch+1, 'loss': mean, 'optimizer_steps': len(losses)})
        schedule.step(mean)
        print(f'{variant} epoch={epoch+1} loss={mean:.7g}', flush=True)
        if config['early_stop_loss'] > 0 and mean < config['early_stop_loss']:
            break
    return history


@torch.no_grad()
def embed(model, inputs, variant, indices, device, role):
    model.eval()
    values = []
    for offset, index in enumerate(indices):
        report_phase('evaluation', variant=variant, role=role, graph=offset+1, total=len(indices))
        value = model(inputs.tensor(index, variant, device))
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite model output')
        values.append(value.cpu().numpy())
    return np.concatenate(values)


def score(main, reliability, threshold):
    first, control = rpc_score(main), rpc_score(reliability)
    valid = first['numerically_valid'] and control['numerically_valid']
    reliable = valid and control['score'] < threshold
    separated = first['score'] > threshold and not np.isclose(
        first['score'], control['score'], atol=1e-6, rtol=1e-5)
    return {'main': first, 'reliability': control, 'reliability_passed': bool(reliable),
            'raw_separated': bool(separated),
            'outcome': 'separated' if reliable and separated else 'not_separated' if reliable else 'unresolved'}


def fit_variant(inputs, variant, config, device, seed):
    configure_numpy(seed)
    configure_torch(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = RealizationPPGN(3, config['seed_q'], config['hidden_dim'], config['layers'],
                            config['output_dim'], config['checkpoint_blocks'], config['input_width']).to(device)
    started = time.monotonic()
    history = train(model, inputs, variant, config, device)
    main = embed(model, inputs, variant, list(range(64)), device, 'main')
    reliability = embed(model, inputs, variant, list(range(64, 128)), device, 'reliability')
    result = score(main, reliability, config['rpc_threshold'])
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    metadata = {'history': history, 'evaluation': result, 'elapsed_seconds': time.monotonic()-started,
                'parameters': sum(value.numel() for value in model.parameters()),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device)
                    if str(device).startswith('cuda') else 0}
    del model
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return metadata, main, reliability, state


def worker(out, pair_id, seed, device):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    if source_identity() != manifest['identity']['sources']:
        raise ValueError('Worker source differs from the frozen run')
    config = manifest['identity']['configuration']
    if pair_id not in config['executed_pair_ids'] or seed not in config['seeds']:
        raise ValueError('Worker condition outside the frozen run')
    data = load_brec_data(config['brec_file'], None)
    indices = list(range(64*pair_id, 64*(pair_id+1))) + list(range(64*(pair_id+400), 64*(pair_id+401)))
    graphs = [graph_at(data, index) for index in indices]
    if len({len(graph) for graph in graphs}) != 1:
        raise ValueError('Expected equal vertex counts within a BREC condition')
    inputs = TripleInputs(graphs, config['seed_q'], config['input_cache_gib'])
    identity = manifest['identity_sha256']
    stem = f'pair{pair_id:03d}_seed{seed}'
    with report_to(out/'workers'/f'{stem}.json'):
        for variant in config['variants']:
            target = out/'records'/f'{stem}_{variant}.json'
            if target.exists():
                old = json.loads(target.read_text())
                if old.get('identity_sha256') != identity or old.get('status') != 'complete':
                    raise ValueError(f'Inspect incomplete or mismatched record: {target}')
                continue
            row = {'pair_id': pair_id, 'seed': seed, 'variant': variant, 'order': 3,
                   'identity_sha256': identity, 'reference_group': group_for_pair(pair_id),
                   'device': str(device), 'nodes': len(graphs[0])}
            report_phase('initializing', variant=variant)
            try:
                metadata, main, reliability, state = fit_variant(inputs, variant, config, device, seed)
                arrays = out/'artifacts'/f'{stem}_{variant}.npz'
                weights = out/'artifacts'/f'{stem}_{variant}.pt'
                report_phase('saving', variant=variant)
                atomic_npz(arrays, main=main, reliability=reliability)
                with atomic_output(weights) as temporary:
                    torch.save(state, temporary)
                row.update(status='complete', **metadata,
                           artifacts={str(path.relative_to(out)): sha256(path) for path in (arrays, weights)})
                row['reference_upper_bound_violation'] = (
                    row['reference_group'] == '3-FWL-indistinguishable'
                    and row['evaluation']['outcome'] == 'separated')
            except Exception as exc:
                row.update(status='resource_limited' if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))
                           else 'implementation_error', reason=repr(exc), traceback=traceback.format_exc())
                atomic_json(target, row)
                raise
            atomic_json(target, row)
            print(f'completed {stem}_{variant}: {row["evaluation"]["outcome"]}', flush=True)
