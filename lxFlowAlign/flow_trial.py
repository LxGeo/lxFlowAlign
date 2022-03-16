# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:25:10 2022

@author: cherif
"""

#%% Path and modules addition ###
import os, sys
LxGeoPyLibs_path=os.path.join(os.environ["LX_GEO_REPOS_ROOT"], "LxGeoPyLibs")
sys.path.append(LxGeoPyLibs_path)

#%% Imports ###
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import concurrent.futures
from rasterio import windows
from ezflow.models import build_model
from ezflow.models import get_default_model_cfg
from matplotlib import pyplot as plt
import torch 
######

from lxFlowAlign.dataset.fake_disalignment import disalign_dataset
from lxFlowAlign.utils.rasterization_utils import rasterize_from_profile
from lxFlowAlign.utils.spatial_utils import extents_to_profile

#%% FN Defs ###


#%% Pipe step (rasterization)
### Inputs def
input_shapefile_1 = "C:/DATA_SANDBOX/Alignment2/MOURMELON/OSM_data/MOURMELON_osm_polys_polygons.shp"
input_shapefile_2 = "C:/DATA_SANDBOX/Alignment2/MOURMELON/17MAY17110802/17MAY17110802_bativec.shp"

# Load shapefile
in_gdf_1 = gpd.read_file(input_shapefile_1)
in_gdf_2 = gpd.read_file(input_shapefile_2)
assert in_gdf_1.crs == in_gdf_2.crs, "Input shapefiles must share the same crs"

# compute common extents
gdf_1_box = box(*in_gdf_1.total_bounds)
gdf_2_box = box(*in_gdf_2.total_bounds)
common_extent = (gdf_1_box.union(gdf_2_box)).bounds

rasterized_profile = extents_to_profile(common_extent, crs=in_gdf_1.crs)

rasterized_shp1 = rasterize_from_profile(in_gdf_1.geometry, rasterized_profile, 1)
rasterized_shp2 = rasterize_from_profile(in_gdf_2.geometry, rasterized_profile, 1)

#%% Pipe step (disalgnment rasterization)

flow_profile = extents_to_profile(common_extent, crs=in_gdf_1.crs, count=2, dtype=np.float32)
gdf1_disaligned = disalign_dataset(in_gdf_1)
dispx_shp1_disaligned = rasterize_from_profile(gdf1_disaligned.geometry, flow_profile, gdf1_disaligned.disp_x.values)
dispy_shp1_disaligned = rasterize_from_profile(gdf1_disaligned.geometry, flow_profile, gdf1_disaligned.disp_y.values)



#%% Pipe step 2
offset_x=1200
offset_y=1200
patch_size=256
s1 = rasterized_shp1[offset_x:offset_x+patch_size, offset_y:offset_y+patch_size]
s2 = rasterized_shp2[offset_x:offset_x+patch_size, offset_y:offset_y+patch_size]

s1 = torch.Tensor(s1)
s2 = torch.Tensor(s2)

#s1=s1.repeat(3,1,1)
#s2=s2.repeat(3,1,1)

s1 = torch.unsqueeze(s1, 0)
s2 = torch.unsqueeze(s2, 0)
s1 = torch.unsqueeze(s1, 0)
s2 = torch.unsqueeze(s2, 0)

#%% Pipe step 3
from ezflow.models import Predictor
from torchvision.transforms import Resize

from ezflow.models import get_default_model_cfg

raft_cfg = get_default_model_cfg("RAFT")
raft_cfg["ENCODER"]["FEATURE"]["IN_CHANNELS"]=1
raft_cfg["ENCODER"]["CONTEXT"]["IN_CHANNELS"]=1

predictor = Predictor("RAFT", model_cfg=raft_cfg)
flow = predictor(s1, s2)


























