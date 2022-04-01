import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ezflow.engine.registry import loss_functions as LOSS_REGISTRY, optimizers as OPTIM_REGISTRY, schedulers as SCHED_REGISTRY

def load_cfg_trainer_params(cfg):
    """
    """

    callbacks = []
    # Checkpoint callback loading
    if cfg.MODEL_CHECKPOINT_CALLBACK.USE:
        if cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS is not None:
            os.makedirs(cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS.dirpath, exist_ok=True)
            save_params = cfg.MODEL_CHECKPOINT_CALLBACK.PARAMS.to_dict()
            callbacks.append(
                ModelCheckpoint(**save_params)
            )
        else:
            raise Exception("ModelCheckpoint parameters missing!")   
    
    # EarlyStopping callback loading
    if cfg.EARLY_STOPPING.USE:
        if cfg.EARLY_STOPPING.PARAMS is not None:
            stopping_params = cfg.EARLY_STOPPING.PARAMS.to_dict()
            callbacks.append(
                EarlyStopping(**stopping_params)
            )
        else:
            raise Exception("EarlyStopping parameters missing!")   

    # Logger loading
    logger=None
    if cfg.TENSORBOARD_LOGGER.USE:
        os.makedirs(cfg.TENSORBOARD_LOGGER.LOG_PATH, exist_ok=True)
        logger = TensorBoardLogger(cfg.TENSORBOARD_LOGGER.LOG_PATH, cfg.TENSORBOARD_LOGGER.NAME)

    lightning_params = cfg.LIGHTNING_PARAMS.to_dict()

    lightning_params.update(
        {
            "callbacks":callbacks,
            "logger":logger
        }
    )

    return lightning_params
        



