import logging
import time
import json
from pathlib import Path

import numpy as np
import torch

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from GNNPlus.loss.subtoken_prediction_loss import subtoken_cross_entropy
from GNNPlus.utils import cfg_to_dict, make_wandb_name, dirichlet_energy, mean_average_distance, mean_norm
from GNNPlus.utility_export import export_per_graph_losses


def _cpu_smoothing_metrics(batch):
    """Optional diagnostics run on detached CPU tensors only."""
    x = batch.x.detach().cpu()
    edge_index = batch.edge_index.detach().cpu()
    graph_index = batch.batch.detach().cpu()
    return {
        'dirichlet': dirichlet_energy(x, edge_index, graph_index),
        'mad': mean_average_distance(x, edge_index, graph_index),
        'emb_norm': mean_norm(x),
    }


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach()
            _pred = pred_score.detach()
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
            
        extra_stats = {}
        if cfg.train.eval_smoothing_metrics:
            extra_stats.update(_cpu_smoothing_metrics(batch))
            
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach()
            _pred = pred_score.detach()
        
        if cfg.train.eval_smoothing_metrics:
            extra_stats.update(_cpu_smoothing_metrics(batch))
            
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


def _resume_epoch_state(start_epoch):
    """Only accept metadata from the exact epoch restored by load_ckpt."""
    if start_epoch == 0:
        return None
    state_path = Path(cfg.run_dir) / 'ckpt' / f'{start_epoch - 1}.state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['epoch'] != start_epoch - 1:
            raise RuntimeError('Checkpoint state epoch mismatch')
        return state
    # Backward compatibility with checkpoints from the previous log format.
    path=Path(cfg.run_dir)/"logging.log"
    found=None
    if path.exists():
        with path.open() as handle:
            for line in handle:
                marker="EPOCH_STATE "
                if marker not in line:
                    continue
                try:
                    state=json.loads(line.split(marker,1)[1])
                except (ValueError,TypeError):
                    continue
                if state.get("epoch")==start_epoch-1:
                    found=state
    if found is None:
        raise RuntimeError(
            "Checkpoint has no matching state file or legacy EPOCH_STATE. "
            "Cannot recover historical best-validation selection safely. "
            "Use the matching log or start a new run; legacy checkpoints "
            "require an explicit metadata migration.")
    return found


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """Validation-selected best epoch with checkpoint-aligned log metadata."""
    run_started = time.perf_counter()
    tracking_device = None
    if torch.cuda.is_available() and str(cfg.accelerator) != 'cpu':
        tracking_device = torch.device(cfg.accelerator)
        if tracking_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(tracking_device)
        else:
            tracking_device = None
    start_epoch=0
    if cfg.train.auto_resume:
        start_epoch=load_ckpt(model,optimizer,scheduler,cfg.train.epoch_resume)
    restored=_resume_epoch_state(start_epoch)
    metric="loss" if cfg.metric_best=="auto" else cfg.metric_best
    direction="argmin" if cfg.metric_best=="auto" else cfg.metric_agg
    if direction not in ("argmin","argmax"):
        raise ValueError("metric_agg must be argmin or argmax")
    if restored and (restored["metric"]!=metric or restored["direction"]!=direction):
        raise RuntimeError("Checkpoint metric selection settings changed")
    best=restored["best"] if restored else None
    latest=restored["latest"] if restored else None
    if start_epoch>=cfg.optim.max_epoch:
        if cfg.train.export_per_graph:
            raise RuntimeError('Per-graph export needs a fresh training run so the selected model state is available')
        logging.info("Training already complete; best=%s",best)
        for logger in loggers:
            logger.close()
        return
    if len(loggers)!=3 or len(loaders)!=3:
        raise ValueError("Custom training expects train/val/test loaders")
    run=None
    if cfg.wandb.use:
        import wandb
        run=wandb.init(entity=cfg.wandb.entity,project=cfg.wandb.project,
                      name=cfg.wandb.name or make_wandb_name(cfg))
        run.config.update(cfg_to_dict(cfg))
    durations=[]
    best_model_state=None
    for epoch in range(start_epoch,cfg.optim.max_epoch):
        tick=time.perf_counter()
        train_epoch(loggers[0],loaders[0],model,optimizer,scheduler,
                    cfg.optim.batch_accumulation)
        train_stats=loggers[0].write_epoch(epoch)
        evaluate=is_eval_epoch(epoch) or latest is None or epoch==start_epoch
        improved=False
        if evaluate:
            scores=[]
            for i,split in ((1,"val"),(2,"test")):
                eval_epoch(loggers[i],loaders[i],model,split)
                scores.append(loggers[i].write_epoch(epoch))
            latest=[train_stats,*scores]
            value=float(scores[0][metric])
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation metric {metric}")
            improved=best is None or (
                value<best["value"] if direction=="argmin" else value>best["value"])
            if improved:
                best=dict(epoch=epoch,value=value,stats=latest)
                if cfg.train.export_per_graph:
                    best_model_state={name: value.detach().cpu().clone()
                                      for name,value in model.state_dict().items()}
        else:
            latest=[train_stats,*latest[1:]]
        if cfg.optim.scheduler=="reduce_on_plateau":
            if evaluate:
                scheduler.step(latest[1]["loss"])
        else:
            scheduler.step()
        durations.append(time.perf_counter()-tick)
        evaluation = (
            f"Val {metric}: {latest[1][metric]:10.6f} | "
            f"Test {metric}: {latest[2][metric]:10.6f}"
            if evaluate else "Validation/test: skipped")
        logging.info(
            '\n%s\nCurrent epoch %4d | Train loss: %.6f | %s | Time: %.2fs\n'
            'Best    epoch %4d | Train loss: %.6f | Val %s: %10.6f | Test %s: %10.6f\n%s',
            '-' * 96, epoch, train_stats['loss'], evaluation, durations[-1],
            best['epoch'], best['stats'][0]['loss'], metric, best['stats'][1][metric],
            metric, best['stats'][2][metric], '-' * 96)
        if cfg.train.enable_ckpt and (
                (cfg.train.ckpt_best and improved) or
                (not cfg.train.ckpt_best and is_ckpt_epoch(epoch))):
            # Only saved checkpoints need resume metadata. Preserve all task
            # metrics, but omit repeated timing and configuration fields.
            omitted = {'params', 'lr', 'time_iter', 'time_epoch', 'eta',
                       'eta_hours', 'gpu_memory'}
            compact = lambda rows: [{k: v for k, v in row.items()
                                     if k not in omitted} for row in rows]
            state = dict(epoch=epoch, metric=metric, direction=direction,
                         best={**best, 'stats': compact(best['stats'])},
                         latest=compact(latest))
            state_path = Path(cfg.run_dir) / 'ckpt' / f'{epoch}.state.json'
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, separators=(',', ':')))
            temporary.replace(state_path)
            save_ckpt(model,optimizer,scheduler,epoch)
            if cfg.train.ckpt_clean:
                clean_ckpt()
        if run is not None:
            # Training intentionally reports loss only, not a fabricated metric.
            report={f"{split}/{key}":value
                    for split,stats in zip(("train","val","test"),latest)
                    for key,value in stats.items()}
            report["best/epoch"]=best["epoch"]
            report[f"best/val_{metric}"]=best["value"]
            run.log(report,step=epoch)
    val_stats, test_stats = best['stats'][1:]
    rows = [f"{'Metric':<20} {'Validation':>14} {'Test':>14}"]
    for key in dict.fromkeys(('loss', metric)):
        rows.append(f"{key:<20} {val_stats[key]:>14.6f} {test_stats[key]:>14.6f}")
    logging.info("FINAL_RESULT | seed=%s | split=%s | best epoch=%s\n%s\n%s\n%s",
                 cfg.seed, cfg.dataset.split_index, best['epoch'], '-' * 52,
                 '\n'.join(rows), '-' * 52)
    logging.info("Mean epoch seconds %.3f; total loop hours %.3f",
                 np.mean(durations),np.sum(durations)/3600)
    train_seconds = time.perf_counter() - run_started
    if tracking_device is not None:
        torch.cuda.synchronize(tracking_device)
        peak_cuda_bytes = torch.cuda.max_memory_allocated(tracking_device)
    else:
        peak_cuda_bytes = 0
    if cfg.train.export_per_graph:
        if best_model_state is None:
            raise RuntimeError('No best model state was captured for per-graph export')
        model.load_state_dict(best_model_state)
        count=export_per_graph_losses(
            model,loaders[2],Path(cfg.run_dir)/'per_graph_losses.csv','test')
        logging.info('Exported %s test graph losses to %s',count,
                     Path(cfg.run_dir)/'per_graph_losses.csv')
    logging.info('RUN_TRAIN_SECONDS=%.6f', train_seconds)
    logging.info('RUN_PEAK_CUDA_ALLOCATED_BYTES=%s', peak_cuda_bytes)
    for logger in loggers:
        logger.close()
    if run is not None:
        run.finish()
    logging.info("Task done, results saved in %s",cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from GNNPlus.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=1)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
