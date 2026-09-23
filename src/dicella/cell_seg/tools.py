from typing import Optional, Tuple, Union, Literal, List, Dict, Set, Callable
import numpy as np
import pandas as pd
from scipy import ndimage
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy import ndimage
import cv2

def get_adata_from_gem_and_cell_labels(
    gem_file: str,
    labels: Optional[Union[np.ndarray, str]],
    obs_name: Literal['geneName', 'geneID'] = 'geneName',
) -> AnnData:
    data = read_gem_as_dataframe(gem_file)
    shape = (data["x"].max(), data["y"].max())
    y_min, x_min = data["y"].min(), data["x"].min()

    if isinstance(labels, str):
        labels = cv2.imread(labels, 2)

    label_coords = get_coords_labels(labels)
    label_coords["y"] += y_min
    label_coords["x"] += x_min
    
    data = pd.merge(data, label_coords, on=["y", "x"], how="inner")
    data['label'] = data['label'].astype(str)

    uniq_cell = sorted(data["label"].unique())
    if obs_name == 'geneName':
        uniq_gene = sorted(data["geneName"].unique())
    else:
        uniq_gene = sorted(data["geneID"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))
    x_ind = data["label"].map(cell_dict).astype(int).values
    if obs_name == 'geneName':
        y_ind = data["geneName"].map(gene_dict).astype(int).values
    else:
        y_ind = data["geneID"].map(gene_dict).astype(int).values

    X = csr_matrix((data["total"].values, (x_ind, y_ind)), shape=shape)
    layers = {}
    if "spliced" in data.columns:
        layers['splice'] = csr_matrix((data["spliced"].values, (x_ind, y_ind)), shape=shape)
    if "unspliced" in data.columns:
        layers['unspliced'] = csr_matrix((data["unspliced"].values, (x_ind, y_ind)), shape=shape)
    
    obs = pd.DataFrame(index=uniq_cell)
    if obs_name == 'geneName':
        var = pd.DataFrame(index=uniq_gene)
    else:
        if 'geneName' in data.columns:
            geneID_geneName = dict(zip(data['geneID'], data['geneName']))
            gene_names = [[geneID_geneName[gid]] for gid in uniq_gene]
            var = pd.DataFrame(gene_names, index=uniq_gene, columns=['geneName'])
        else:
            var = pd.DataFrame(index=uniq_gene)

    adata = AnnData(X=X, obs=obs, var=var, layers=layers)

    centroid_df = get_controid(labels)
    centroid_df['y'] += y_min
    centroid_df['x'] += x_min
    centroid_df.index = centroid_df.index.astype(str)
    
    spatial = np.array(centroid_df.loc[adata.obs_names,['y', 'x']])
    adata.obs['y'] = spatial[:,0]
    adata.obs['x'] = spatial[:,1]
    adata.obsm['spatial'] = spatial
    
    adata.layers['counts'] = adata.X.copy()

    return adata

def read_gem_agg(
    path: str,
    gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
) -> AnnData:
    """
    读取GEM文件并聚合基因
    Args:
        path: GEM文件路径
        gene_agg: 字典，键为层名，值为基因列表或函数，用于聚合基因
    Returns:
        AnnData对象，行是y坐标，列是x坐标。X是总UMI数。layers是聚合后的基因层。X[0,1]是y坐标为ymin，x坐标为xmin+1的UMI数。
    """
    data = read_gem_as_dataframe(path)
    x_min, y_min = data["x"].min(), data["y"].min()
    y, x = data["y"].values, data["x"].values
    y_max, x_max = y.max(), x.max()
    shape = (y_max + 1, x_max + 1)
    layers = {}

    # See read_gem_as_dataframe for standardized column names
    X = csr_matrix((data["total"].values, (y, x)), shape=shape, dtype=np.uint16)
    if "spliced" in data.columns:
        layers['spliced'] = csr_matrix((data["spliced"].values, (y, x)), shape=shape, dtype=np.uint16)
    if "unspliced" in data.columns:
        layers['unspliced'] = csr_matrix((data["unspliced"].values, (y, x)), shape=shape, dtype=np.uint16)

    adata = AnnData(X=X, layers=layers)[y_min:, x_min:].copy()

    return adata


def read_gem_as_dataframe(path: str) -> pd.DataFrame:
    """
    读取GEM文件为DataFrame
    Args:
        path: GEM文件路径
    Returns:
        Pandandas DataFrame
    """
    dtype = {
        "geneID": "category",  # geneID
        "geneName": "category",  # geneName
        "x": np.uint32,  # x
        "y": np.uint32,  # y
        # Multiple different names for total counts
        "MIDCounts": np.uint16,
        "MIDCount": np.uint16,
        "UMICount": np.uint16,
        "UMICounts": np.uint16,
        "EXONIC": np.uint16,  # spliced
        "ExonCount": np.uint16, # spliced
        "INTRONIC": np.uint16,  # unspliced,
    }
    rename = {
        "MIDCounts": "total",
        "MIDCount": "total",
        "UMICount": "total",
        "UMICounts": "total",
        "EXONIC": "spliced",
        "ExonCount": "spliced",
        "INTRONIC": "unspliced",
    }

    # Use first 10 rows for validation.
    df = pd.read_csv(path, sep="\t", dtype=dtype, comment="#", nrows=10)

    # If duplicate columns are provided, we don't know which to use!
    rename_inverse = {}
    for _from, _to in rename.items():
        rename_inverse.setdefault(_to, []).append(_from)
    for _to, _cols in rename_inverse.items():
        if sum(_from in df.columns for _from in _cols) > 1:
            raise IOError(f"Found multiple columns mapping to `{_to}`.")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=dtype,
        comment="#",
    ).rename(columns=rename)

def get_coords_labels(labels: np.ndarray) -> pd.DataFrame:
    """
    从标签图像中提取坐标和标签
    Args:
        labels: 标签矩阵
    Returns:
        DataFrame，包含x坐标、y坐标和标签
    """
    nz = labels.nonzero()
    x, y = nz
    data = labels[nz]
    values = np.vstack((x, y, data)).T
    return pd.DataFrame(values, columns=["x", "y", "label"])



def expand_cell_labesl(
    labels: np.ndarray,
    iteration: int = 10,
    max_cell_area: int = 1000,
):
    labels = labels.copy()
    for i in range(iteration):
        cell_area = np.bincount(labels.flatten())
        large_cells =np.where(cell_area>=max_cell_area)[0]
        new_pixels = is_next_to_only_one_cell(labels, ignore_cells=large_cells)
        dilated_labels = ndimage.grey_dilation(labels, size=3)
        labels = np.where(new_pixels==True, dilated_labels, labels)
    return labels


def is_next_to_only_one_cell(
    labels: np.ndarray, 
    ignore_cells: List[int] = [],
):
    """
    判断每个像素是否只与一个标签相邻（不包括背景0），用8连通来定义相邻
    :param labels: 标签图像
    :param ignore_cells: 要忽略的标签列表，如果某个像素点只与`ignore_cells`中的某个标签相邻，则该像素点是False
    :return: 逻辑数组，每个像素是否只与一个标签相邻（不包括背景0）

    """
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    
    local_max = ndimage.maximum_filter(labels, footprint=footprint, mode='constant', cval=0)
    
    # 将0替换为比所有label都大的值，这样min_filter就不会被背景0干扰
    max_possible = labels.max() + 1
    safe_labels = np.where(labels == 0, max_possible, labels)
    local_min_nonzero = ndimage.minimum_filter(safe_labels, footprint=footprint, mode='constant', cval=max_possible)
    
    # 唯一标签判定：非零min == max，且至少有一个非零邻居
    if len(ignore_cells) >= 1:
        is_single_label = (local_max == local_min_nonzero) & (local_max > 0) & (labels == 0) & ~np.isin(local_max, ignore_cells)
    else:
        is_single_label = (local_max == local_min_nonzero) & (local_max > 0) & (labels == 0)
    
    return is_single_label


def get_coords_labels(labels: np.ndarray) -> pd.DataFrame:
    """Convert labels into sparse-format dataframe.

    Args:
        labels: cell segmentation labels matrix.

    Returns:
        A DataFrame of columns "x", "y", and "label". The coordinates are
        relative to the labels matrix.
    """
    nz = labels.nonzero()
    y, x = nz
    data = labels[nz]
    values = np.vstack((y, x, data)).T
    return pd.DataFrame(values, columns=["y", "x", "label"])


def get_controid(
    cell_labels: np.ndarray,
):
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids>0]
    centroids = ndimage.center_of_mass(cell_labels.astype(np.bool), cell_labels, cell_ids)
    df = pd.DataFrame(centroids, columns=["y", "x"])
    df["cell_label"] = cell_ids
    df = df.set_index('cell_label')
    return df
    
