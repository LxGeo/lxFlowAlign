
from lxFlowAlign.training.ptl.optflow_model import lightningOptFlowModel
from LxGeoPyLibs.dataset.patchified_dataset import test_model
from LxGeoPyLibs.dataset.specific_datasets.rasterized_vector_with_reference import VectorWithRefDataset
from LxGeoPyLibs.geometry.rasterizers.polygons_rasterizer import polygons_to_multiclass
import click

class temp_model(lightningOptFlowModel, test_model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


if __name__ == "__main__":
    main()
    """
    mdl_path = "./models/proba/FlowNetC/epoch=82-step=369350.ckpt"
    mdl = temp_model.load_from_checkpoint(mdl_path)
    mdl = mdl.cuda()

    in_raster_path = "../DATA_SANDBOX/lxFlowAlign/proba_data/raw/BRAGANCA/DL/ortho1/rooftop/build-probas.tif"
    #in_vector_path = "../DATA_SANDBOX/lxFlowAlign/proba_data/raw/BRAGANCA/DL/ortho2/rooftop/build-poly.shp"
    in_vector_path = "../DATA_SANDBOX/lxFlowAlign/proba_data/raw/BRAGANCA/DL/ortho2/rooftop/build-poly.shp"

    rasterization_method = polygons_to_multiclass
    in_dataset = VectorWithRefDataset(in_raster_path, in_vector_path, polygons_to_multiclass)

    in_dataset.predict_to_file("../DATA_SANDBOX/preds.tif", mdl, (256,256))"""
    