
import click
import numpy as np
import rasterio as rio
from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites import formulas
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.patchified_dataset import CallableModel
from math import radians


class DhmFlowModel(CallableModel):

    device = "cpu"

    def __init__(self, x_flow, y_flow, x_res, y_res, dhm2flow=True):
        super().__init__()
        self.x_flow = x_flow
        self.y_flow = y_flow
        self.x_res = x_res
        self.y_res = y_res
        self.dhm2flow = dhm2flow

    def forward(self, batch_image):
        batch_image = batch_image[0]
        batched_pred = []
        for idx in range(batch_image.shape[0]):
            batched_pred.append( self.forward_one(batch_image[idx]) )
        return np.stack(batched_pred, axis=0)
        
    
    def forward_one(self, image):
        if self.dhm2flow:
            assert image.shape[0]==1, f"Input image of shape {image.shape} has {image.shape[0]} bands when expected 1!"
            out_image = np.concatenate( [image*self.x_flow/self.x_res, image*self.y_flow/self.y_res ] )
            return out_image
        else:
            assert image.shape[0]==2, f"Input image of shape {image.shape} has {image.shape[0]} bands when expected 2!"
            #out_image = image[0]*self.x_res/self.x_flow + image[1]*self.y_res/self.y_flow
            out_image = self.x_res * image[0]/self.x_flow + self.y_res * image[1]/self.y_flow
            out_image = np.expand_dims(out_image, 0)
            return out_image
        
        
    

@click.command()
@click.option("-i", "--input_image", type=click.Path(exists=True))
@click.option('--imd1', type=click.Path(exists=True))
@click.option("--imd2", type=click.Path(exists=True))
@click.option("-o", "--out_image", type=click.Path(exists=False))
@click.option("-inv","--inverse", is_flag=True, help="Flag for inverse transformation (flow2dhm)!")
def main(input_image, imd1, imd2, out_image, inverse):
    """
    """

    with rio.open(input_image) as in_dst:
        out_profile = in_dst.profile.copy()
        x_resolution = out_profile["transform"][0]
        y_resolution = out_profile["transform"][4]

    imd1 = IMetaData(imd1)
    imd2 = IMetaData(imd2)

    x_flow, y_flow = formulas.compute_roof2roof_constants(radians(imd1.satAzimuth()), radians(imd1.satElevation()),
     radians(imd2.satAzimuth()), radians(imd2.satElevation())
     )

    transformer_model = DhmFlowModel(x_flow, y_flow, x_resolution, y_resolution, not inverse)

    in_dataset = RasterDataset(input_image)

    in_dataset.predict_to_file(out_image, transformer_model, tile_size=(256,256))
    
    

if __name__ == "__main__":
    main()