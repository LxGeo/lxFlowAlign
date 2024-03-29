import click
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio import mask
from shapely.geometry import box
from LxGeoPyLibs.geometry.utils_rio import extents_to_profile
from lxFlowAlign.dataset.fake_disalignment import disalign_dataset
from LxGeoPyLibs.geometry.rasterizers import rasterize_from_profile

from skimage import morphology
def binary_to_multiclass(x):
        """
        function used to transform building binary map to 3 classes (inner, countour, outer)
        """
        inner_map = morphology.erosion(x, morphology.square(5))
        contour_map = x-inner_map
        background_map = np.ones_like(contour_map); background_map-=contour_map;background_map-=inner_map;
        return np.stack([background_map, inner_map, contour_map])

def generate_fake_data(input_gdf, raster_extents, resolution=0.5):
    """
    Given input geodataframe apply fake noise to disalign it.
    Results in (
        * binary rasterization of disalgined gdf
        * Two band optical flow of aligned on input_gdf geometries
        ) each with its respective raster profile
    """

    rasterized_profile = extents_to_profile(raster_extents, crs=input_gdf.crs, count=3, dtype=rio.uint8)
    flow_profile = extents_to_profile(raster_extents, crs=input_gdf.crs, count=2, dtype=np.float32)
    
    gdf_disaligned = disalign_dataset(input_gdf)
    
    bin_raster_2_inner = rasterize_from_profile(gdf_disaligned.geometry, rasterized_profile, 1)
    bin_raster_2_contour = rasterize_from_profile(gdf_disaligned.geometry.boundary.buffer(0.5), rasterized_profile, 1)
    # clean inner map
    bin_raster_2_inner[bin_raster_2_contour>0]=0
    bin_raster_2_background = np.ones_like(bin_raster_2_inner)
    # clean background map
    bin_raster_2_background[bin_raster_2_contour>0]=0;bin_raster_2_background[bin_raster_2_inner>0]=0
    bin_raster_2= np.stack([bin_raster_2_background, bin_raster_2_inner, bin_raster_2_contour])
    
    dispx = rasterize_from_profile(gdf_disaligned.geometry, flow_profile, gdf_disaligned.disp_x.values / -resolution)
    dispy = rasterize_from_profile(gdf_disaligned.geometry, flow_profile, gdf_disaligned.disp_y.values/ resolution)
    
    flow = np.stack([dispx, dispy])
       
    return (bin_raster_2, rasterized_profile), (flow, flow_profile)


@click.command()
@click.argument('input_shp_path', type=click.Path(exists=True))
@click.argument('input_ortho_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option("--extents", type=click.STRING, help="Optional extents. Could be dataset path or comma separeted bounds!")
def main(input_shp_path, input_ortho_path, output_dir, extents):
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
    
    ortho_raster_1 = None
    with rio.open(input_ortho_path) as ortho_dst:
        if not raster_extents:
            raster_extents = ortho_dst.bounds
        
        mask_shape = box(*raster_extents)
        out_image, out_transform = rio.mask.mask(ortho_dst, [mask_shape], crop=True, filled=True)
        c_profile = ortho_dst.profile.copy(); c_profile.update({"transform": out_transform})
        c_profile.update({"height": out_image.shape[-2], "width":out_image.shape[-1] })
        out_image[:, (out_image[0]==0) & (out_image[1]==0) & (out_image[2]==0)] = np.array([255,0,0])[:,np.newaxis]
        ortho_raster_1 = (out_image, c_profile)

    bin_raster_2, flow = generate_fake_data(input_gdf, raster_extents)
    bin_raster_2=(bin_raster_2[0]*255, bin_raster_2[1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    rasters_paths = [os.path.join(output_dir, x) for x in ("image_1.tif", "image_2.tif", "flow.tif")]
    for (raster_data, c_profile), out_path in zip((ortho_raster_1, bin_raster_2, flow),rasters_paths ):
        with rio.open(out_path, "w", **c_profile) as tar:
            tar.write(raster_data.astype(c_profile["dtype"]))

if __name__ == "__main__":
    main()
    