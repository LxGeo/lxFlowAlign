# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:58:41 2022

@author: cherif
"""

import os
import click
import torch
from lxFlowAlign.training.ptl.optflow_model import lightningOptFlowModel
import ezflow
from ezflow.engine import get_training_cfg
from ezflow.models import get_default_model_cfg
from ezflow.engine import Trainer
from LxGeoPyLibs.dataset.multi_dataset import MultiDatasets
from lxFlowAlign.dataset.ptl.optical_flow_dataset import OptFlowRasterDataset


@click.command()
@click.argument('arch', type=click.Choice(list(ezflow.model_zoo._ModelZooConfigs.MODEL_NAME_TO_CONFIG.keys()), False))
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('val_data_dir', type=click.Path(exists=True))
@click.argument('ckpt_dir', type=click.Path(exists=False))
@click.argument('log_dir', type=click.Path(exists=False))
@click.option('--custom_model_cfg', required=False, type=click.Path(exists=True))
@click.option('--custom_training_cfg', required=False, type=click.Path(exists=True))
def main(arch, train_data_dir, val_data_dir, ckpt_dir, log_dir, custom_model_cfg, custom_training_cfg):
    
        
    model_cfg = get_default_model_cfg(arch) if not custom_training_cfg \
        else ezflow.config.get_cfg(cfg_path=custom_model_cfg, custom=True)
    model_cfg.ENCODER.FEATURE.IN_CHANNELS=1
    model_cfg.ENCODER.CONTEXT.IN_CHANNELS=1
    light_model = lightningOptFlowModel(arch, model_cfg)
    
    training_cfg = get_training_cfg(cfg_path="raft_default.yaml", custom=False) if not custom_training_cfg \
        else get_training_cfg(cfg_path=custom_training_cfg, custom=True)
    
    training_cfg.CKPT_DIR = ckpt_dir
    training_cfg.LOG_DIR = log_dir
    
    ## data loaders
    train_dataset = MultiDatasets( (OptFlowRasterDataset(
        os.path.join(train_data_dir, sub_folder)) for sub_folder in os.listdir(train_data_dir)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=0, shuffle=True)
    
    valid_dataset = MultiDatasets( (OptFlowRasterDataset(
        os.path.join(val_data_dir, sub_folder)) for sub_folder in os.listdir(val_data_dir)))    
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=0, shuffle=True)
    
    trainer = Trainer(
        cfg=training_cfg,
        model=light_model,
        train_loader=train_dataloader,
        val_loader=valid_dataloader
    )
    
    trainer.train(n_epochs=10)
    



if __name__ == "__main__":
    main()