
import numpy as np
from skimage.segmentation import slic
from skimage.measure import label, regionprops
import click
import rasterio as rio
import tqdm
from fast_slic import Slic

import networkx as nx


def label_map2graph(grid):
    # get unique labels
    vertices = np.unique(grid)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
  
    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[int(x/num_vertices)]] for x in edges] 

    return vertices, edges



@click.command()
@click.option('--in_dsm', type=click.Path(exists=True))
@click.option("--out_dhm", type=click.Path())
def main(in_dsm, out_dhm):
    
    with rio.open(in_dsm) as dst:
        dsm_image = dst.read()[0]
        in_profile = dst.profile.copy()

    mask = dsm_image != in_profile["nodata"]
    
    #dsm_image = np.stack([dsm_image,dsm_image,dsm_image],-1)
    slic = Slic(num_components=1000, compactness=1e-7)
    dsm_seg = slic.iterate(np.stack([dsm_image, dsm_image, dsm_image],-1).astype(np.uint8))
    dsm_seg+=dsm_seg.min()+1
    #dsm_seg = slic(dsm_image,slic_zero=True)
    props = regionprops(dsm_seg, dsm_image)
    props={ p.label: p.intensity_min for p in props}

    V,E = label_map2graph(dsm_seg)

    G = nx.from_edgelist(E)

    dtm_image = dsm_image.copy()

    for v in tqdm.tqdm(V):
        neighbours = list(nx.all_neighbors(G, v))        
        min_val = min([props[n] for n in neighbours])
        if props[v] > min_val :
            dtm_image[dsm_seg==v] = min_val
    
    with rio.open(out_dhm, "w", **in_profile) as tar:
        tar.write(np.expand_dims(dtm_image,0))




if __name__ == "__main__":
    main()