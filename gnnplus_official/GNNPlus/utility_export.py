"""Optional per-graph losses for prespecified structural/task association analysis."""
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import RandomSampler
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss


def attach_utility_graph_ids(dataset):
    """Attach stable dataset-order IDs before GraphGym creates split loaders."""
    size = len(dataset)
    data = getattr(dataset, '_data', None)
    if data is None:
        data = getattr(dataset, 'data', None)
    slices = getattr(dataset, 'slices', None)
    if data is None or not isinstance(slices, dict):
        raise RuntimeError(
            'Stable per-graph utility export requires an in-memory dataset with slices')
    if 'utility_graph_id' in slices or 'utility_graph_id' in data:
        raise RuntimeError('Dataset already contains utility_graph_id')
    data.utility_graph_id = torch.arange(size, dtype=torch.long)
    slices['utility_graph_id'] = torch.arange(size + 1, dtype=torch.long)
    if hasattr(dataset, '_data_list'):
        dataset._data_list = None


@torch.no_grad()
def export_per_graph_losses(model, loader, output, split='test'):
    """Write losses in sequential split-local dataset order for graph tasks."""
    if isinstance(getattr(loader, 'sampler', None), RandomSampler):
        raise RuntimeError('Per-graph export requires a sequential evaluation loader')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    model.eval()
    written = 0
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('split', 'graph_index', 'loss'))
        writer.writeheader()
        for batch in loader:
            batch.split = split
            batch.to(torch.device(cfg.accelerator))
            result = model(batch)
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError('Per-graph export supports models returning (prediction, target)')
            prediction, truth = result[:2]
            if prediction.ndim == 0 or truth.ndim == 0 or prediction.shape[0] != truth.shape[0]:
                raise RuntimeError('Per-graph export requires one prediction and target per graph')
            graph_ids = getattr(batch, 'utility_graph_id', None)
            if graph_ids is None or graph_ids.reshape(-1).numel() != prediction.shape[0]:
                raise RuntimeError('Stable utility_graph_id values are missing from the evaluation batch')
            graph_ids = graph_ids.reshape(-1).detach().cpu().tolist()
            if len(set(map(int, graph_ids))) != len(graph_ids) or min(map(int, graph_ids)) < 0:
                raise RuntimeError('Invalid or repeated utility_graph_id in evaluation batch')
            for offset in range(prediction.shape[0]):
                loss, _ = compute_loss(prediction[offset:offset + 1],
                                       truth[offset:offset + 1])
                writer.writerow({'split': split, 'graph_index': int(graph_ids[offset]),
                                 'loss': float(loss.detach().cpu())})
                written += 1
    os.replace(temporary, target)
    return written
