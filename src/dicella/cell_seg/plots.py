from typing import Optional, Tuple, Union, Literal, List, Dict, Set, Callable
import sys
import numpy as np
from scipy import ndimage
import cv2
from cellpose import models, core, io, plot
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

from .cell_seg import get_cell_shape

def draw_rna_ssdna_and_cell_boundaries(
    rna: Union[np.ndarray, str],
    ssdna: Union[np.ndarray, str],
    cell_labels: List[Union[np.ndarray, str]],
    thickness: int = 1,
    colors: List[List[int]] = [[0, 0, 255]], #[红色]
):
    if isinstance(rna, str):
        rna = cv2.imread(rna, 0)
    if isinstance(ssdna, str):
        ssdna = cv2.imread(ssdna, 0)
    
    assert isinstance(cell_labels, list)
    assert len(cell_labels) == len(colors)
    
    all_bounds = []
    for cell_labels_one in cell_labels:
        if isinstance(cell_labels_one, str):
            cell_labels_one = cv2.imread(cell_labels_one, 2)
        bound_one = get_cell_shape(cell_labels_one, thickness)
        all_bounds.append(bound_one)
    
    bgr = np.zeros((*ssdna.shape, 3), dtype=np.uint8) # cv2 -> BGR 格式
    bgr[..., 0] = ssdna  # ssdna画成白色
    bgr[..., 1] = ssdna
    bgr[..., 2] = ssdna

    # # rna 画成绿色，覆盖ssdna。如果rna为0，则保留ssdna
    # rna_norm = (np.clip(rna.astype(np.float32) / 7, 0, 1.0) * 255).astype(np.uint8)
    # bgr[..., 0] = np.where(rna_norm > 0, 0, ssdna)
    # bgr[..., 1] = np.where(rna_norm > 0, rna_norm, ssdna)
    # bgr[..., 2] = np.where(rna_norm > 0, 0, ssdna)
    
    # rna画成绿色，和ssdna用加色模式叠加
    rna_norm = (np.clip(rna.astype(np.float32) / 7, 0, 1.0) * 255).astype(np.uint8)
    bgr[..., 1] = np.clip((bgr[..., 1].astype(np.uint16) + rna_norm.astype(np.uint16)), 0, 255).astype(np.uint8)

    for bound_one, color_one in zip(all_bounds, colors):
        bgr[bound_one == 255] = color_one

    return bgr


def draw_ssdna_and_cell_boundaries(
    ssdna: Union[np.ndarray, str],
    cell_labels: List[Union[np.ndarray, str]],
    thickness: int = 1,
    colors: List[List[int]] = [[0, 0, 255]], #[红色]
):
    if isinstance(ssdna, str):
        ssdna = cv2.imread(ssdna, 0)
    
    assert isinstance(cell_labels, list)

    all_bounds = []
    for cell_labels_one in cell_labels:
        if isinstance(cell_labels_one, str):
            cell_labels_one = cv2.imread(cell_labels_one, 2)
        bound_one = get_cell_shape(cell_labels_one, thickness)
        all_bounds.append(bound_one)


    # ssdna_norm = ssdna.astype(np.float32) / max(ssdna.max(), 1)

    bgr = np.zeros((*ssdna.shape, 3), dtype=np.uint8) # cv2 -> BGR 格式
    bgr[..., 0] = ssdna  # ssdna画成白色
    bgr[..., 1] = ssdna
    bgr[..., 2] = ssdna
    for bound_one, color_one in zip(all_bounds, colors):
        bgr[bound_one == 255] = color_one

    return bgr




    