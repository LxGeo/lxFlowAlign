# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:44:30 2022

@author: cherif
"""

import rasterio as rio
from shapely.geometry import box

def extents_to_profile(extent, gsd=0.5, **kwargs):
    
    in_width = round((extent[2]-extent[0])/gsd)
    in_height = round((extent[3]-extent[1])/gsd)
    in_transform = rio.transform.from_origin(extent[0], extent[-1], gsd,gsd)
    rasterization_profile = {
    "driver": "GTiff", "count":1, "height": in_height, "width":in_width,
    "dtype":rio.uint8, "transform":in_transform
    }
    rasterization_profile.update(kwargs)
    return rasterization_profile

def get_common_extents(*extents):
    """
    Return common extents of all arguments
    """
    assert len(extents)>0, "Input extents arguments missing!"
    
    minx, miny, maxx, maxy = extents[0]
    
    for ext in extents:
        minx = min(minx, ext[0])
        miny = min(miny, ext[1])
        maxx = min(maxx, ext[2])
        maxy = min(maxy, ext[3])
    
    return (minx, miny, maxx, maxy)