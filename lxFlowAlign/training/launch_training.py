# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:58:41 2022

@author: cherif
"""

import os
import click
import torch
from lxFlowAlign.training.ptl.optflow_model import lightningOptFlowModel
from lxFlowAlign.training.ptl.trainer_parameter_loader import load_cfg_trainer_params
import ezflow
from ezflow.engine import get_training_cfg
from ezflow.models import get_default_model_cfg
from ezflow.data.dataloader.device_dataloader import DeviceDataLoader
from LxGeoPyLibs.dataset.multi_dataset import MultiDatasets
from lxFlowAlign.dataset.ptl.optical_flow_dataset import OptFlowRasterDataset

import pytorch_lightning as pl




@click.command()
@click.argument('arch', type=click.Choice(list(ezflow.model_zoo._ModelZooConfigs.MODEL_NAME_TO_CONFIG.keys()), False))
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('val_data_dir', type=click.Path(exists=True))
@click.argument('ckpt_dir', type=click.Path(exists=False))
@click.argument('log_dir', type=click.Path(exists=False))
@click.option('--custom_model_cfg', required=True, type=click.Path(exists=True))
@click.option('--custom_training_cfg', required=True, type=click.Path(exists=True))
def main(arch, train_data_dir, val_data_dir, ckpt_dir, log_dir, custom_model_cfg, custom_training_cfg):
    
        
    model_cfg = ezflow.config.get_cfg(cfg_path=custom_model_cfg, custom=True)
    
    light_model = lightningOptFlowModel(arch, model_cfg)
    
    training_cfg = get_training_cfg(cfg_path=custom_training_cfg, custom=True)

    training_cfg["MODEL_CHECKPOINT_CALLBACK"]["PARAMS"]["dirpath"] = ckpt_dir
    training_cfg["TENSORBOARD_LOGGER"]["LOG_PATH"] = log_dir

    training_params = load_cfg_trainer_params(training_cfg)    
        
    ## data loaders
    train_dataset = MultiDatasets( (OptFlowRasterDataset(
        os.path.join(train_data_dir, sub_folder)) for sub_folder in os.listdir(train_data_dir)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=3, shuffle=True, drop_last=True)
    train_dataloader = train_dataloader

    valid_dataset = MultiDatasets( (OptFlowRasterDataset(
        os.path.join(val_data_dir, sub_folder)) for sub_folder in os.listdir(val_data_dir)))    
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_cfg.DATA.BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    valid_dataloader = valid_dataloader

    light_model = light_model.cuda()
    light_model.sample_val = next(iter(valid_dataloader))

    trainer = pl.Trainer(**training_params)

    if (training_params["auto_scale_batch_size"] or training_params["auto_lr_find"]):
        trainer.tune(light_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    else:
        trainer.fit(light_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


    



if __name__ == "__main__":
    main()