# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:49:21 2022

@author: cherif
"""

from matplotlib import pyplot as plt
import torch
import numpy as np
import os

import pytorch_lightning as pl
from ezflow.models import build_model
from ezflow.utils.registry import Registry
from ezflow.functional import FUNCTIONAL_REGISTRY
from ezflow.engine.registry import loss_functions as loss_registry, optimizers as optimizers_registery, schedulers as schedulers_registry


class lightningOptFlowModel(pl.LightningModule):
    
    def __init__(self, arch, cfg, **kwargs):
        
        super(lightningOptFlowModel, self).__init__()
        
        self.arch=arch
        self.cfg = cfg
        
        self.model = build_model(self.arch, cfg=self.cfg)

        if self.cfg.CRITERION.CUSTOM:
            criterion_params={}
            if "PARAMS" in self.cfg.CRITERION: criterion_params = self.cfg.CRITERION.PARAMS.to_dict()
            self.loss_fn = FUNCTIONAL_REGISTRY.get(self.cfg.CRITERION.NAME)(**criterion_params)
        else:
            self.loss_fn = loss_registry.get(self.cfg.CRITERION.NAME)()
                
        self.save_hyperparameters()
        
    
    def forward(self, image_pair):
        return self.model(*image_pair)    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # remove valid band if needed
        if not self.cfg.CRITERION.CUSTOM:
            y=y[:, :2, :, :]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        # remove valid band if needed
        if not self.cfg.CRITERION.CUSTOM:
            y=y[:, :2, :, :]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True,)
        #return loss

    def on_after_backward(self):
        """"""
        return
        for name, param in self.model.named_parameters():
            print(name, end="\t")
            print(param.grad.abs().sum())
        

    def configure_optimizers(self):

        optimizer = optimizers_registery.get(self.cfg.OPTIMIZER.NAME)(
            self.model.parameters(), lr=self.cfg.OPTIMIZER.LR,
            betas = self.cfg.OPTIMIZER.PARAMS.betas,
            eps = self.cfg.OPTIMIZER.PARAMS.eps,
            weight_decay = self.cfg.OPTIMIZER.PARAMS.weight_decay,
        )

        schedulers = []
        if self.cfg.SCHEDULER.USE:
            sched = schedulers_registry.get(self.cfg.SCHEDULER.NAME)
            if self.cfg.SCHEDULER.PARAMS is not None:
                scheduler_params = self.cfg.SCHEDULER.PARAMS.to_dict()
                scheduler = sched(optimizer, **scheduler_params)
            else:
                scheduler = sched(optimizer)
            
            schedulers.append(scheduler)
        
        return [optimizer], schedulers
    
    def image_grid(self, image_pair, gt_flow, preds):  
        
        def norm(x):
            return (x-x.min()) / (x.max()-x.min())
        
        bs = self.sample_val[1].shape[0]
        img1, img2 = image_pair
        preds = torch.cat((preds,img1), dim=1)
                
        for i in range(bs):
            self.logger.experiment.add_image("img1_{}".format(i),norm(img1[i]),self.current_epoch)
            self.logger.experiment.add_image("img2_{}".format(i),norm(img2[i]),self.current_epoch)
            
            self.logger.experiment.add_image("flow_gt_{}".format(i),norm(gt_flow[i,:2]),self.current_epoch)
            self.logger.experiment.add_image("flow_preds_{}".format(i),norm(preds[i,:2]),self.current_epoch)
    
    def validation_epoch_end(self, outputs):
        
        if hasattr(self, "sample_val"):
            image_pair,gt_flow = self.sample_val
            image_pair = [im.to(self.device) for im in image_pair]
            logits = self.forward(image_pair)            
            self.image_grid(image_pair, gt_flow, logits)
    
    
    
    
    
    