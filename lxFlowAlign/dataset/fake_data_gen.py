# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:49:56 2022

@author: cherif
"""

import click
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from lxFlowAlign.utils.spatial_utils import extents_to_profile, get_common_extents
from lxFlowAlign.dataset.fake_disalignment import disalign_dataset
from lxFlowAlign.utils.rasterization_utils import rasterize_from_profile

def generate_fake_data(input_gdf, raster_extents=None):
    """
    Given input geodataframe apply fake noise to disalign it.
    Results in (
        * binary rasterization of input_gdf
        * binary rasterization of disalgined gdf
        * Two band optical flow of aligned on input_gdf geometries
        ) each with its respective raster profile
    """
        
    gdf_disaligned = disalign_dataset(input_gdf)
    
    if not raster_extents:
        raster_extents = get_common_extents(input_gdf.total_bounds, gdf_disaligned.total_bounds)
    
    rasterized_profile = extents_to_profile(raster_extents, crs=input_gdf.crs, dtype=rio.uint8)
    flow_profile = extents_to_profile(raster_extents, crs=input_gdf.crs, count=2, dtype=np.float32)
    
    bin_raster_1 = rasterize_from_profile(input_gdf.geometry, rasterized_profile, 1)
    bin_raster_2 = rasterize_from_profile(gdf_disaligned.geometry, rasterized_profile, 1)
    
    dispx = rasterize_from_profile(input_gdf.geometry, flow_profile, gdf_disaligned.disp_x.values)
    dispy = rasterize_from_profile(input_gdf.geometry, flow_profile, gdf_disaligned.disp_y.values)
    
    flow = np.stack([dispx, dispy])
    bin_raster_1 = np.expand_dims(bin_raster_1,0)
    bin_raster_2 = np.expand_dims(bin_raster_2,0)
    
    return (bin_raster_1, rasterized_profile), (bin_raster_2, rasterized_profile), (flow, flow_profile)


@click.command()
@click.argument('input_shp_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option("--extents", type=click.STRING, help="Optional extents. Could be dataset path or comma separeted bounds!")
def main(input_shp_path, output_dir, extents):
    """
    """
    input_gdf = gpd.read_file(input_shp_path)
    
    raster_extents=None
    if extents:
        if os.path.isfile(extents):
            raster_extents = gpd.read_file(extents).total_bounds
        else:
            try:
                raster_extents = list(map(lambda x: float(x), extents.split(",")))
            except:
                print("Error parsing extents string! Should be comma separted floats!")
                return
            if not box(*raster_extents).intersects(box(*input_gdf.total_bounds)):
                print("Extents are out of shapefile bounds!")
                return
    
    bin_raster_1, bin_raster_2, flow = generate_fake_data(input_gdf, raster_extents)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    rasters_paths = [os.path.join(output_dir, x) for x in ("image_1.tif", "image_2.tif", "flow.tif")]
    for (raster_data, c_profile), out_path in zip((bin_raster_1, bin_raster_2, flow),rasters_paths ):
        with rio.open(out_path, "w", **c_profile) as tar:
            tar.write(raster_data.astype(c_profile["dtype"]))

if __name__ == "__main__":
    main()
    