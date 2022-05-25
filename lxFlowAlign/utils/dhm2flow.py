
import click
import os
import numpy as np
import rasterio as rio
from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites import formulas


@click.command()
@click.argument('input_dhm_path', type=click.Path(exists=True))
@click.option('--im1_imd', type=click.Path(exists=True))
@click.option("--im2_imd", type=click.Path(exists=True))
@click.option("--out_flow", type=click.Path(exists=False))
def main(input_dhm_path, im1_imd, im2_imd, out_flow):
    """
    """

    with rio.open(input_dhm_path) as in_dst:
        out_profile = in_dst.profile.copy()
        img_resolution = out_profile["transform"][0]
        img_array = in_dst.read()

    imd1 = IMetaData(im1_imd)
    imd2 = IMetaData(im2_imd)

    epipolarity_angle = formulas.compute_rotation_angle(imd1.satAzimuth(), imd1.satElevation(), imd2.satAzimuth(), imd2.satElevation())
    x_flow, y_flow = formulas.compute_axis_displacement_ratios(epipolarity_angle)

    out_image = np.concatenate( [img_array*x_flow/img_resolution, img_array*y_flow/img_resolution ] )

    out_image[out_image>100]=0
    out_image[out_image<-100]=0
    
    ## check output folder
    out_folder = os.path.dirname(out_flow)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    
    ## Update profile before saving
    out_profile["count"] = 2

    with rio.open(out_flow, "w", **out_profile) as out_dst:
        out_dst.write(out_image)
    

if __name__ == "__main__":
    main()