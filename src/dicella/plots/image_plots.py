from typing import Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clr
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

def plot_ssdna_and_cell_boundaries(
    image: np.ndarray,
    cell_boundraies: np.ndarray,
    dpi: int = 600,
    ssnda_vmax: int = 100,
    cell_bound_vmax: int = 1,
    save: bool = True,
    save_path: str = "./",
    save_name: str = "ssdna_and_cell_boundaries.pdf"
) -> None:
    greenmap = clr.LinearSegmentedColormap.from_list('custom blue', [(0,'#000000FF'),(1,'#00FF00FF')], N=256)
    redmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#FFFFFF00','#FF0000FF'], N=256)
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image, cmap=greenmap, vmax=ssnda_vmax)
    ax.imshow(cell_boundraies, cmap=redmap, vmax=cell_bound_vmax)
    