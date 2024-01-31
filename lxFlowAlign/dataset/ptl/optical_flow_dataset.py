# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:16:47 2022

@author: cherif
"""
from lxFlowAlign import _logger
import os
import rasterio as rio
import torch
from torch.utils.data import Dataset
from LxGeoPyLibs.vision.image_transformation import Trans_Identity
import multiprocessing
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.patchified_dataset import PatchifiedDataset, PixelPatchifiedDataset
from LxGeoPyLibs.dataset.common_interfaces import BoundedDataset, Pixelized2DDataset
import pygeos
import numpy as np

class RasterRegister(dict):

    def __init__(self):
        super(RasterRegister, self).__init__()
    
    def __del__(self):
        for k,v in self.items():
            print("Closing raster at {}".format(k))
            v.close()

rasters_map=RasterRegister()
lock = multiprocessing.Lock()

class OptFlowRasterDataset(PixelPatchifiedDataset):
    
    READ_RETRY_COUNT = 4

    def __init__(self, image1_path=None, image2_path=None, optflow_path=None,
                 pixel_patch_size=(512,512), pixel_patch_overlap=0,
                 weighted_flow=True):
        
        self.weighted_flow=weighted_flow
        self.images_path_list=[]
        if os.path.isdir(image1_path):
            in_dir=image1_path
            self.image1_path = os.path.join(in_dir, "image_1.tif"); self.images_path_list.append(self.image1_path)
            self.image2_path = os.path.join(in_dir, "image_2.tif"); self.images_path_list.append(self.image2_path)
            self.optflow_path = os.path.join(in_dir, "flow.tif"); self.images_path_list.append(self.optflow_path)
            self.bound_geom_path = os.path.join(in_dir, "bound_geom.txt")
        else:
            self.images_path_list = [image1_path, image2_path, optflow_path]
        
        assert all([os.path.isfile(f) for f in self.images_path_list]), "One of optFlow rasters are missing!"
                
        self.datasets_map = dict()
        self.datasets_map.update({
            "image1": RasterDataset(self.image1_path),
            "image2": RasterDataset(self.image2_path),
            "optflow": RasterDataset(self.optflow_path),
            })
        
        common_aoi = pygeos.set_operations.intersection_all([c_raster_dst.bounds_geom for c_raster_dst in self.datasets_map.values()])
        if os.path.isfile(self.bound_geom_path):
            with open(self.bound_geom_path) as bound_f:
                bound_geom = pygeos.from_wkt(bound_f.read())
                common_aoi = pygeos.intersection(common_aoi, bound_geom)
        PixelPatchifiedDataset.__init__(self, 
            self.datasets_map["image1"].rio_dataset().transform[0],
            -self.datasets_map["image1"].rio_dataset().transform[4],
            pixel_patch_size,
            pixel_patch_overlap, 
            common_aoi
            )
        
    
    def __len__(self):
        return PatchifiedDataset.__len__(self)
    
    def __getitem__(self, idx):
        
        c_window = PatchifiedDataset.__getitem__(self, idx)
        
        def fix_out_of_bound_image(img):
            img[:,(img[0]==0) & (img[1]==0) & (img[2]==0)] = np.array([1,0,0])[:,np.newaxis]

        lock.acquire()
        for _ in range(self.READ_RETRY_COUNT):
            try:
                img1 = self.datasets_map["image1"]._load_padded_raster_window(window_geom=c_window, patch_size=self.pixel_patch_size) 
                fix_out_of_bound_image(img1)
                img2 = self.datasets_map["image2"]._load_padded_raster_window(window_geom=c_window, patch_size=self.pixel_patch_size)
                fix_out_of_bound_image(img2)
                flow = self.datasets_map["optflow"]._load_padded_raster_window(window_geom=c_window, patch_size=self.pixel_patch_size)
                flow=-flow
                break
            except rio.errors.RasterioIOError as e:
                _logger.error(f"Unable to read window {c_window}")
                raise Exception("Dataset loading error")
        lock.release()
        
        if self.weighted_flow and flow.shape[0]<3:
            weight_map = np.zeros_like(flow[0]) + (flow[0]!=0)
            flow = np.stack([flow[0], flow[1], weight_map])
                
        img1 = torch.from_numpy(img1).float() 
        img2 = torch.from_numpy(img2).float()
        flow = torch.from_numpy(flow).float()
                
        # valid_mask to flow
        flow = torch.cat((flow, img1), dim=0)

        return (img2, img1), flow


def worker_init_fn(worker_id):
    from LxGeoPyLibs.dataset.raster_dataset import rasters_map
    w_multi_dst = torch.utils.data.get_worker_info().dataset.datasets
    for w_dst in w_multi_dst:
        for component_raster in w_dst.datasets_map.values():
            rasters_map.update({component_raster.image_path: rio.open(component_raster.image_path)})
        
if __name__ == "__main__":
    a=OptFlowRasterDataset("C:/DATA_SANDBOX/got_backup/mcherif/Documents/DATA_SANDBOX/lxFlowAlign/data/faults/train_data/example5_TG12072023/")
    item = a[22]
    a.patches_gdf()