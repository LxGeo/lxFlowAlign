# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:16:47 2022

@author: cherif
"""

import os
import rasterio as rio
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity


class OptFlowRasterDataset(Dataset):
    
    def __init__(self, image1_path=None, image2_path=None, optflow_path=None,
                 augmentation_transforms=None,preprocessing=None, patch_size=(256,256), patch_overlap=(100,100)):
        
        if os.path.isdir(image1_path):
            in_dir=image1_path
            self.image1_path = os.path.join(in_dir, "image_1.tif")
            self.image2_path = os.path.join(in_dir, "image_2.tif")
            self.optflow_path = os.path.join(in_dir, "flow.tif")
        
        assert all([os.path.isfile(f) for f in (self.image1_path, self.image2_path, self.optflow_path)]), "One of optFlow rasters are missing!"
        
        if augmentation_transforms is None:
            self.augmentation_transforms=[Trans_Identity()]
        else:
            self.augmentation_transforms = augmentation_transforms
        
        self.rasters_map={
            "image_1": rio.open(self.image1_path),
            "image_2": rio.open(self.image2_path),
            "flow": rio.open(self.optflow_path)
            }
        
        raster_shapes = [dst.shape for dst in self.rasters_map.values()]
        assert len(set(raster_shapes)) == 1, "Rasters have different shapes!" 
        
        self.Y_size, self.X_size = raster_shapes[0]                
        self.preprocessing=preprocessing
        self.setup(patch_size, patch_overlap)
            
    
    def __del__(self):        
        for v in self.rasters_map.values():
            v.close()
    
    def setup(self, patch_size, patch_overlap):
        """
        Setup patch loading settings
        """
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        self.window_x_starts = list(range(0, self.X_size-self.patch_size[0], self.patch_overlap[0]))
        if (self.X_size-1)%self.patch_size[0] != 0: self.window_x_starts.append(self.X_size-1-self.patch_size[0])
        self.window_y_starts = list(range(0, self.Y_size-self.patch_size[1], self.patch_overlap[1]))
        if (self.Y_size-1)%self.patch_size[1] != 0: self.window_y_starts.append(self.Y_size-1-self.patch_size[1])
        self.is_setup=True

    def __len__(self):
        assert self.is_setup, "Dataset is not set up!"
        x_count = len(self.window_x_starts)
        y_count = len(self.window_y_starts)
        return x_count*y_count*len(self.augmentation_transforms)
    
    def __getitem__(self, idx):
        
        assert self.is_setup, "Dataset is not set up!"
        window_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        window_x_start = self.window_x_starts[window_idx//len(self.window_y_starts)]
        window_y_start = self.window_y_starts[window_idx%len(self.window_y_starts)]
        c_window = rio.windows.Window(window_x_start, window_y_start, *self.patch_size )
        
        img1 = self.rasters_map["image_1"].read(window=c_window) 
        img2 = self.rasters_map["image_2"].read(window=c_window)
        flow = self.rasters_map["flow"].read(window=c_window) 
        
        c_trans = self.augmentation_transforms[transform_idx]
        img1, flow = c_trans(img1, flow)
        img2, _ = c_trans(img2, flow)
        
        if self.preprocessing:
            img1 = self.preprocessing(img1)
            img2 = self.preprocessing(img2)
        
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        flow = torch.from_numpy(flow).float()
        
        # valid_mask to flow
        flow = torch.cat((flow, img1), dim=0)

        return (img1, img2), flow
        