from typing import Optional, Tuple, Union, Literal, List, Dict, Set, Callable
import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
from anndata import AnnData
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries


def cell_seg_on_ssdna_by_cellpose(
    img: Union[np.ndarray, str], # 2d array, uint8 或者tif文件路径
    batch_size: int = 64,
    model_name: str = 'cpsam_v2',
) -> np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img, 0)
    img = img[:,:,None]

    from cellpose import models
    
    model = models.CellposeModel(gpu=True, pretrained_model=model_name)
    cell_labels, flows, styles = model.eval(img, batch_size=batch_size)
    return cell_labels.astype(np.uint32)

def get_cell_shape(
    cell_labels: np.ndarray,
    thickness: int = 1, 
) -> np.ndarray:
    """
    获取cell_labels的轮廓，返回一个uint8的数组，255表示轮廓，0表示背景

    Args:
      cell_labels: 输入的cell_labels，每个像素表示一个细胞，不同的像素表示不同的细胞
      thickness: 轮廓的厚度，默认1
    Returns:
      cell_labels的轮廓，uint8的数组，255表示轮廓，0表示背景

    """

    bound = np.zeros_like(cell_labels, dtype=np.uint8)
    for i in range(thickness):
        cell_labels = np.where(bound == 0, cell_labels, 0)
        bound_one = find_boundaries(cell_labels, mode="inner").astype(np.uint8)
        bound += bound_one

    bound = bound * 255
    return bound

def get_cell_labels_from_mask(
    mask: Union[np.ndarray, str],
) -> np.ndarray:
    if isinstance(mask, str):
        mask = cv2.imread(mask, 0)
    cell_labels = cv2.connectedComponents(mask)[1]
    cell_labels = cell_labels.astype(np.uint32)
    return cell_labels



