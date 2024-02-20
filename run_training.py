import logging
import os
os.environ['USE_PYGEOS'] = '0'
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import TrainingEngine, build_training_engine

from pathlib import Path
import tempfile

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == 'train':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info('Starting training...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        logger.info('Starting caching...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':

    # Location of path with all training configs
    CONFIG_PATH = 'nuplan/planning/script/config/training'
    CONFIG_NAME = 'default_training'

    # Create a temporary directory to store the cache and experiment artifacts
    SAVE_DIR = Path('nuplan_output_test/')  # optionally replace with persistent dir
    EXPERIMENT = 'training_multi_model'
    JOB_NAME = 'train_multi_checkfeatures'
    LOG_DIR = str(SAVE_DIR / EXPERIMENT / JOB_NAME)
    MODEL_PATH = "urban_multi_model.ckpt"

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}',
        f'cache.cache_path={str(SAVE_DIR)}/cache',
        f'experiment_name={EXPERIMENT}',
        f'job_name={JOB_NAME}',
        'py_func=train',
        'optimizer=adamw',
        'optimizer.lr=1.25e-5',
        'lr_scheduler=one_cycle_lr',
        'lr_scheduler.pct_start=0.45',
        'model=urban_multi_model',
        '+training=training_urban_multi_model',  
        'scenario_builder=nuplan', 
        'scenario_filter.limit_total_scenarios=0.04',  
        #'lightning.trainer.params.accelerator=ddp_spawn',  
        'lightning.trainer.params.max_epochs=50',
        'data_loader.params.batch_size=10',
        'data_loader.params.num_workers=20',
    ])

    main(cfg)
