

from collections import defaultdict
from typing import OrderedDict
from lxFlowAlign.training.ptl.optflow_model import lightningOptFlowModel
from LxGeoPyLibs.dataset.patchified_dataset import CallableModel
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_with_reference import VectorWithRefDataset
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.dataset.specific_datasets.raster_with_references import RasterWithRefsDataset
from LxGeoPyLibs.dataset.specific_datasets.multi_vector_dataset import MultiVectorDataset 
from LxGeoPyLibs.dataset.vector_dataset import VectorDataset
from LxGeoPyLibs.dataset.hetero_dataset import HeteroDataset
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_dataset import RasterizedVectorDataset
from LxGeoPyLibs.geometry.rasterizers.polygons_rasterizer import polygons_to_multiclass
import click
import numpy as np
import ezflow
from functools import partial

class temp_model(lightningOptFlowModel, CallableModel):
    def __init__(self, **kwargs):
        CallableModel.__init__(self, bs=8)
        lightningOptFlowModel.__init__(self, **kwargs)

@click.command()
@click.argument('in_raster_path', type=click.Path(exists=True))
@click.argument('in_vector_path', type=click.Path(exists=True))
@click.option("--out_raster_path", type=click.Path(exists=False), help="Path to prediction raster!")
@click.option("--model_path", type=click.Path(exists=True), help="Path to optflow model")
def main(in_raster_path, in_vector_path, out_raster_path, model_path):
    mdl = temp_model.load_from_checkpoint(model_path)
    mdl = mdl.cuda()

    rasterization_method = polygons_to_multiclass
    in_dataset = VectorWithRefDataset(in_raster_path, in_vector_path, polygons_to_multiclass)
    in_dataset.predict_to_file(out_raster_path, mdl, (256,256))

def proba_to_oneHotLabel_callable(x):
        out = np.zeros_like(x)
        ai = np.expand_dims(np.argmax(x, axis=0), axis=0)
        np.put_along_axis(out, ai, 1, axis=0)
        return out

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__ == "__main__":
    #main()
    
    mdl_path = "./models/thesis/FlowNetC_disp15/epoch=197-step=43956.ckpt"
    #model_cfg = ezflow.config.get_cfg(cfg_path="../DATA_SANDBOX/lxFlowAlign/proba_data/configs/models/custom_flownet_c.yaml", custom=True)
    mdl = temp_model.load_from_checkpoint(mdl_path)
    mdl = mdl.cuda()

    #in_raster_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/DL/ortho1/rooftop/build-probas.tif"
    #in_vector_path = "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/external/bing/buildings.shp"

    #rasterization_method = lambda *args, **kwargs: 255 * polygons_to_multiclass(*args, **kwargs)
    #in_dataset = VectorWithRefDataset(in_raster_path, in_vector_path, rasterization_method)

    
    #in_raster_path2 = "../DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/Brazil/belford_east/prediction/a86ba6d5_4875_11ec_9ac4_4ccc6ad69a1c_e13e6e0a712d14afe1c02fc883101f4c/build_probas.tif"
    
    #in_dataset = RasterWithRefsDataset(in_raster_path, OrderedDict(im2=in_raster_path2),preprocessing=proba_to_oneHotLabel_callable, ref_preprocessing=defaultdict(im2=proba_to_oneHotLabel_callable))
    
    
    datasets_def = OrderedDict()
    datasets_def["vector1"]= {
            "dataset_type":RasterizedVectorDataset,
            "vector_path":"C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Ethiopia_Addis_Ababa_1_A_Neo/h_Ethiopia_Addis_Ababa_1_A_kaw.shp",
            "rasterization_method": partial(polygons_to_multiclass, contours_width=2),
            #"preprocessing":lambda x:x/255,
            "gsd":0.3, "pixel_patch_size": (512,512), "pixel_patch_overlap":100
            }
    datasets_def["vector2"]= {
            "dataset_type":RasterizedVectorDataset,
            "vector_path":"C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Ethiopia_Addis_Ababa_1_B_Neo/h_Ethiopia_Addis_Ababa_1_B_NET4.shp",
            "rasterization_method": partial(polygons_to_multiclass, contours_width=2),
            #"preprocessing":lambda x:x/255,
            "gsd":0.3, "pixel_patch_size": (512,512), "pixel_patch_overlap":100
            }    
    in_dataset = HeteroDataset(datasets_def, bounds_geom=VectorDataset(datasets_def["vector2"]["vector_path"]).bounds_geom)
    
    
    """datasets_def = OrderedDict()
    datasets_def["raster1"]= {
            "dataset_type":RasterDataset,
            "preprocessing":lambda x:x/255,
            "image_path":"//cherif/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_A_Neo/preds/probas.tif",
            }
    datasets_def["raster2"]= {
            "dataset_type":RasterDataset,
            "preprocessing":lambda x:x/255,
            "image_path":"//cherif/DATA_SANDBOX/Alignment_Project/PerfectGT/India_Mumbai_B_Neo/preds/probas.tif",
            }
    in_dataset = HeteroDataset(datasets_def, bounds_geom=RasterDataset(datasets_def["raster2"]["image_path"]).bounds_geom)"""

    out_file_path = "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxFlowAlign/flownet/Ethiopia_Addis_Ababa_gt_v1tov2/disparity_flowne_pred.tif"
    
    in_dataset.predict_to_file(out_file_path, mdl, (512,512))
    