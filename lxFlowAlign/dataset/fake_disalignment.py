# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:44:32 2022

@author: cherif
"""

#from pysal.lib.weights import DistanceBand
from libpysal.weights import DistanceBand
import geopandas as gpd
import numpy as np
from shapely.affinity import translate

def disalign_dataset(in_gdf, transform_geoms=True):
    
    gdf = in_gdf.copy()
    
    gdf["disp_x"]=-0.5
    gdf["disp_y"]=-0.5
    
    distance_noise_map = { 50: 10, 20:5, 15:3, 1:1}

    for dist_th, max_noise in distance_noise_map.items():
        
        w = DistanceBand.from_dataframe(gdf,dist_th, silence_warnings=True)
        gdf["comp"]=w.component_labels
        
        comp_disp_x = np.random.uniform(-max_noise, max_noise, (w.n_components))
        comp_disp_y = np.random.uniform(-max_noise, max_noise, (w.n_components))
        
        gdf["disp_x"] += comp_disp_x[w.component_labels]
        gdf["disp_y"] += comp_disp_y[w.component_labels]
    
    if transform_geoms:
        gdf.geometry = gdf.apply(lambda row: translate(row.geometry, row.disp_x, row.disp_y), axis=1)
    return gdf

from numpy.random import default_rng

def non_uniform_disalign_dataset(in_gdf, percentage=0.1):

    gdf = disalign_dataset(in_gdf, transform_geoms=False)
    rng = default_rng()
    random_indices = rng.choice(len(gdf), size=int(len(gdf)*percentage), replace=False)
    random_magnitudes = abs(np.random.normal(0, 10, len(random_indices)))    
    
    norm_vector_mag = np.sqrt(gdf["disp_x"] **2 + gdf["disp_y"]**2)

    gdf.loc[random_indices, "disp_x"] /= norm_vector_mag.loc[random_indices] / random_magnitudes
    gdf.loc[random_indices, "disp_y"] /= norm_vector_mag.loc[random_indices] / random_magnitudes
    gdf.geometry = gdf.apply(lambda row: translate(row.geometry, row.disp_x, row.disp_y), axis=1)
    return gdf


if __name__ == "__main__":
    input_shapefile_1 = "C:/DATA_SANDBOX/lxFlowAlign/data/MOURMELON/OSM_data/MOURMELON_osm_polys_polygons.shp"
    output_shapefile_1 = "C:/DATA_SANDBOX/lxFlowAlign/data/MOURMELON/OSM_data/disaligned_MOURMELON_osm_polys_polygons.shp"
    in_gdf_1 = gpd.read_file(input_shapefile_1)
    
    out_gdf = disalign_dataset(in_gdf_1)
    
    #in_gdf_1.to_file(output_shapefile_1)
        
