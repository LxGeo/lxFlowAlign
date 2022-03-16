# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:49:21 2022

@author: cherif
"""

import pytorch_lightning as pl
from ezflow.models import build_model

from matplotlib import pyplot as plt
import torch
import numpy as np
import os


class lightningOptFlowModel(pl.LightningModule):
    
    def __init__(self, arch, model_cfg, **kwargs):
        
        super(lightningOptFlowModel, self).__init__()
        
        self.arch=arch
        self.model_cfg = model_cfg
        
        self.model = build_model(self.arch, cfg=self.model_cfg)
                
        self.save_hyperparameters()
    
    def forward(self, *image_pair):
        return self.model(*image_pair)    
    
    
    def image_grid(self, image_pair, gt_flow, preds):  
        
        def norm(x):
            return (x-x.min()) / (x.max()-x.min())
        
        bs = self.batch_size
        img1, img2 = image_pair
                
        for i in range(bs):
            self.logger.experiment.add_image("img1_{}".format(i),norm(img1[i]),self.current_epoch)
            self.logger.experiment.add_image("img2_{}".format(i),norm(img2[i]),self.current_epoch)
            
            self.logger.experiment.add_image("flow_gt_{}".format(i),norm(gt_flow[i]),self.current_epoch)
            self.logger.experiment.add_image("flow_preds_{}".format(i),norm(preds[i]),self.current_epoch)
            
        
    def validation_epoch_end(self, outputs):
        
        if hasattr(self, "sample_val"):
            image_pair,gt_flow = self.sample_val
            image_pair = (im.to(self.device) for im in image_pair)
            logits = self.forward(*image_pair)            
            self.image_grid(image_pair, gt_flow, logits)
    
    
    
    
    
    