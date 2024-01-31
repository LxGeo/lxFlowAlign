
import click
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from functools import partial
from LxGeoPyLibs.synthetic.image.gauss_height import generate_gauss_height_map
from LxGeoPyLibs.geometry.rasterizers.linestrings_rasterizer import linestring_to_multiclass
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from matplotlib import pyplot as plt
import torch

def warp(x, flow):
    """
    """

    _, H, W = x.shape

    xx = np.arange(0, W).reshape(1, -1).repeat(H, 0)
    yy = np.arange(0, H).reshape(-1, 1).repeat(W, 1)

    grid = np.stack((xx, yy), 0)
    vgrid = flow*1 + grid
    vgrid[0, :, :] = 2.0 * vgrid[0, :, :] / max(W - 1, 1) - 1.0
    vgrid[1, :, :] = 2.0 * vgrid[1, :, :] / max(H - 1, 1) - 1.0

    output = torch.nn.functional.grid_sample(torch.from_numpy(x)[None,:].float(),torch.from_numpy(vgrid).permute(1,2,0)[None,:].float(), mode="nearest")
    return output.numpy()[0]



def generate_fake_data(input_gdf, raster_extents, resolution=0.5):
    """
    """
    segmap = linestring_to_multiclass(input_gdf.geometry, raster_extents, input_gdf.crs)
    x_disp_map = generate_gauss_height_map( 
        (segmap.shape[-2], segmap.shape[-1]),
         initial_value=0, dims_blobs_scaler=(30,30), 
         blobs_size_normal_dist=partial(np.random.normal, loc=segmap.shape[-2]/10, scale=1) 
         ) * 50
    y_disp_map = generate_gauss_height_map( 
        (segmap.shape[-2], segmap.shape[-1]), 
        initial_value=0, dims_blobs_scaler=(30,30) , 
         blobs_size_normal_dist=partial(np.random.normal, loc=segmap.shape[-1]/10, scale=1) 
        ) * 50
    flow_map = np.stack([x_disp_map, y_disp_map])
    
    rasterized_profile = extents_to_profile(raster_extents, gsd=resolution, crs=input_gdf.crs,  count=3, dtype=rio.uint8)
    flow_profile = extents_to_profile(raster_extents, gsd=resolution, crs=input_gdf.crs, count=2, dtype=np.float32)

    transformed_segmap = warp(segmap, flow_map)
    # fix missing black values
    transformed_segmap[:,(transformed_segmap[0]==0) & (transformed_segmap[1]==0) & (transformed_segmap[2]==0)] = np.array([1,0,0])[:,np.newaxis]
    # Enforcing flow only on image 1 features
    flow_map[:, segmap[0]==1]=0
    
    return (transformed_segmap, rasterized_profile), (segmap, rasterized_profile), (flow_map, flow_profile)

def main(input_shp_path, output_dir, extents):
    """
    """
    input_gdf = gpd.read_file(input_shp_path)
    
    raster_extents=input_gdf.total_bounds
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

@click.command()
@click.argument('input_shp_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option("--extents", type=click.STRING, help="Optional extents. Could be dataset path or comma separeted bounds!")
def main_cmd(input_shp_path, output_dir, extents):
    """
    """
    main(input_shp_path, output_dir, extents)

if __name__ == "__main__":
    input_shp_path = "C:/DATA_SANDBOX/got_backup/mcherif/Documents/DATA_SANDBOX/lxFlowAlign/data/faults/raw/example5_TG12072023/example5_detailedmapping.shp"
    output_dir = "C:/DATA_SANDBOX/got_backup/mcherif/Documents/DATA_SANDBOX/lxFlowAlign/data/faults/train_data/example5_TG12072023/"
    main(input_shp_path, output_dir, None)