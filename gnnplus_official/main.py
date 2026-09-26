import datetime
import torch
import logging
import time

import GNNPlus  # noqa, register custom modules
from GNNPlus.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from GNNPlus.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from GNNPlus.logger import create_logger
from GNNPlus.setting_output import setting_directory, set_run_directory


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg."""
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    cfg.num_workers = 1
    cfg.tensorboard_each_run = False
    cfg.tensorboard_agg = False

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    setting_dir = setting_directory(cfg)

    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        set_run_directory(cfg, setting_dir, seed, split_index)
        cfg.num_workers = 1
        cfg.tensorboard_each_run = False
        cfg.tensorboard_agg = False
        # One append-only file per seed/split; preserve history for resume.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(f"{cfg.run_dir}/logging.log",mode="a"),
                      logging.StreamHandler()],force=True)
        with open(f"{cfg.run_dir}/config.yaml", "w") as handle:
            handle.write(cfg.dump())
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        if torch.cuda.is_available() and str(cfg.accelerator).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(cfg.accelerator))
        train_started = time.perf_counter()
        # Print model info
        logging.debug(cfg)
        cfg.params = params_count(model)
        logging.info('MODEL\n%s\nParameters: %s\n%s',
                     model, format(cfg.params, ','), '-' * 72)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
        logging.info('RUN_TRAIN_SECONDS=%.6f', time.perf_counter() - train_started)
        if torch.cuda.is_available() and str(cfg.accelerator).startswith('cuda'):
            logging.info('RUN_PEAK_CUDA_ALLOCATED_BYTES=%s',
                         torch.cuda.max_memory_allocated(torch.device(cfg.accelerator)))
    logging.info(f"[*] All done: {datetime.datetime.now()}")
