# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:39:13 2022

@author: cherif
"""

import rasterio as rio
from rasterio.features import rasterize
from rasterio.windows import Window
import numpy as np
import pandas as pd

def rasterize_from_profile(geometry_iter, c_profile, burn_value):
    """
    rasterize shapes of geometry iterator using the background profile and the burn value.
    returns a numpy array of raster
    """
    def geom_burn_iter(geometry_iter,burn_value):
        if isinstance(burn_value,(list,pd.core.series.Series,np.ndarray)):
            for g,b in zip(geometry_iter, burn_value):
                yield (g,b)
        else:
            for g in geometry_iter:
                yield (g,burn_value)
    
    #out_dtype = burn_value.dtype if type(burn_value) is np.ndarray else type(burn_value)
    #out_dtype = type(burn_value) if type(burn_value) == int or float else burn_value.dtype
    out_dtype = c_profile["dtype"]
    out_raster = rasterize(geom_burn_iter(geometry_iter, burn_value),
                     (c_profile["height"], c_profile["width"]),
                     fill=0,
                     transform=c_profile["transform"],
                     all_touched=True,
                     #default_value=None,
                     dtype=out_dtype)
    return out_raster

def rasterize_gdf(gdf, input_profile, burn_column="b_val"):
    """
    Inputs:
        gdf: geodataframe of geometries with burn value at column burn_column
        rio_dataset: rasterio dataset of refrence (background) image
        burn_column: column name in shapefile representing burning values
    """
            
    new_profile = input_profile
            
    rasterization_images=[]
    
    if (not (burn_column in gdf.columns)):
        single_rasterization= rasterize_from_profile(gdf.geometry, new_profile, 1)
        rasterization_images.append(single_rasterization)
        
    else:
        burning_values=np.sort(gdf[burn_column].unique())
        for c_burn_value in burning_values:
            c_burn_gdf_view = gdf[gdf[burn_column]==c_burn_value]
            single_rasterization = rasterize_from_profile(c_burn_gdf_view.geometry, new_profile, c_burn_value)
            rasterization_images.append(single_rasterization)
    return np.max(rasterization_images, axis=0)

def rasterize_view(args):
    """
    Runs rasterization on a windowed view
    """
    geometry_view, shapes_gdf, profile, burn_column = args
    view_gdf = gpd.GeoDataFrame(geometry=[geometry_view], crs=shapes_gdf.crs)
    gdf_view = gpd.overlay(shapes_gdf, view_gdf, "intersection")
    if (len(gdf_view)>0):
        return rasterize_gdf(gdf_view, profile, burn_column)
    else:
        return np.zeros((profile["height"], profile["width"]), dtype=profile["dtype"])

def parralel_rasterization(shapes_gdf, output_dst, burn_column):
    """
    Split rasterization process
    Inputs:
        shapes_gdf: geodataframe of all features
        output_dst: output rio dataset to fill
        burn_column: str defining burn column in shapes_gdf
    """
    
    windows_list = [window for ij, window in output_dst.block_windows()]
    window_geometries = [ box(*windows.bounds(c_window,output_dst.transform)) for c_window in windows_list]
            
    # windows.transform(c_window,output_dst.transform)
    concurrent_args = []
    for c_window, c_window_geometrie in zip(windows_list, window_geometries):
        c_profile = output_dst.profile.copy()
        c_profile.update( 
            height=c_window.height,
            width=c_window.width,
            transform = windows.transform(c_window,output_dst.transform)
            )
        concurrent_args.append( (c_window_geometrie, shapes_gdf, c_profile, burn_column) )
    
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=8
        ) as executor:
        futures = executor.map(rasterize_view, concurrent_args)
        for window, result in zip(windows_list, futures):
            output_dst.write(result,1, window=window)
