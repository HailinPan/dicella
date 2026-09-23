from typing import Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from skimage import feature, filters, measure, segmentation
from skimage.segmentation import find_boundaries
from sklearn.mixture import GaussianMixture
from scipy.ndimage import find_objects
from scipy import ndimage
from skimage.feature import peak_local_max
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass


def circle(
    k: int
) -> np.ndarray:
    """
    Generate a circle kernel.

    Args:
        k (int): The size of the kernel. Must be odd and greater than 0.

    Returns:
        np.ndarray: The circle kernel.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")
    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def conv2d(
    X: np.ndarray, 
    k: int, 
    mode: Literal["circle", "square"] = "circle",
) -> np.ndarray:
    """Convolve an array with the specified kernel size and mode.
    
    Args:
        X (np.ndarray): The input array.
        k (int): The size of the kernel. Must be odd and greater than 0.
        mode (Literal["circle", "square"]): The mode of the convolution.
            "circle" uses a circle kernel, and "square" uses a square kernel.

    Returns:
        np.ndarray: The convolved array.
    """
    X = X.astype(np.float32)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")
    if mode not in ("circle", "square"):
        raise ValueError(f'`mode` must be one of "circle", "square"')

    if k == 1:
        return X

    kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
    smoothed_X = signal.convolve2d(X, kernel, boundary="symm", mode="same")
    smoothed_X = smoothed_X / kernel.sum()
    return smoothed_X


def gaussian_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur an array.

    Args:
        X (np.ndarray): The input array.
        k (int): The size of the kernel. Must be odd and greater than 0.

    Returns:
        np.ndarray: The blurred array.
    """
    return cv2.GaussianBlur(src=X.astype(np.float32), ksize=(k, k), sigmaX=0, sigmaY=0)


def get_cell_boundaries(
    labels: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """Get the cell boundaries from the label image.

    Args:
        labels (np.ndarray): The label image.
        thickness (int, optional): The thickness of the boundaries. Defaults to 1.

    Returns:
        np.ndarray: The cell boundaries.
    """
    bound = np.zeros_like(labels, dtype=np.uint8)
    for i in range(thickness):
        labels = np.where(bound == 0, labels, 0)
        bound_one = find_boundaries(labels, mode="inner").astype(np.uint8)
        bound += bound_one

    bound = bound * 255
    return bound


def get_centroids_for_an_gray_image_by_mixgaussion(
    image: np.ndarray,
    n_components: int,
) -> np.ndarray:
    X = []
    assert image.dtype == np.uint8, "image must be uint8."
    for y in range(len(image)):
        for x in range(len(image[y])):
            for i in range(image[y,x]):
                X.append([y,x])
    X = np.array(X)
    gm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    return gm.means_


def filter_cell_labels_by_area(
    labels: np.ndarray,
    min_area: int,
    max_area: int = 10000,
):
    cell_area_dict = cal_cell_area(labels)
    need_cells = [k for k, v in cell_area_dict.items() if v >= min_area and v <= max_area]
    sub_labels = np.where(np.isin(labels, need_cells), 0, labels)
    return sub_labels


def get_boxes_for_all_cells(
    labels: np.ndarray,

):
    """
    Get the bounding boxes for all cells in the label image.

    Args:
        labels (np.ndarray): The label image.

    Returns:
        dict: A dictionary where keys are cell labels and values are dictionaries with keys 'y_min', 'y_max', 'x_min', 'x_max'.

    Input: np.array(
    [[0,0,1,1,0,0,0],
     [0,0,1,0,2,2,2],
     [3,0,5,0,0,0,2]])
    
    Output:
    {1: {'y_min': 0, 'y_max': 1, 'x_min': 2, 'x_max': 3},
    2: {'y_min': 1, 'y_max': 2, 'x_min': 4, 'x_max': 6},
    3: {'y_min': 2, 'y_max': 2, 'x_min': 0, 'x_max': 0},
    5: {'y_min': 2, 'y_max': 2, 'x_min': 2, 'x_max': 2}}
    """
    labels = labels.astype(np.int64)
    slices = find_objects(labels)

    bboxes = {}
    for label_idx, sl in enumerate(slices, start=1):
        if sl is not None:
            sy, sx = sl
            bboxes[label_idx] = {
                'y_min': sy.start, 'y_max': sy.stop - 1,
                'x_min': sx.start, 'x_max': sx.stop - 1
        }
    return bboxes


def split_cells_from_image_by_watershed(
    labels: np.ndarray,
    image: np.ndarray,
    area_per_segment: int,
    k_for_convolution: int = 9,
    min_distance: int = 7,
):
    out_labels = labels.copy()
    cell_area_dict = cal_cell_area(labels)
    bboxes_dict = get_boxes_for_all_cells(labels)
    the_largest_cell_id_now = np.max(labels).astype(np.int64)
    for cell_id, cell_area in cell_area_dict.items():
        if cell_area.astype(np.float32) / area_per_segment > 1:
            y_min, y_max, x_min, x_max = bboxes_dict[cell_id].values()
            sub_image = image[y_min:y_max+1, x_min:x_max+1]
            sub_label = labels[y_min:y_max+1, x_min:x_max+1]
            sub_label = np.where(sub_label == cell_id, cell_id, 0)
            sub_label_mask = np.where(sub_label > 0, 1, 0)
            num_peaks = int(cell_area / area_per_segment) + 1
            sub_watershed = watershed_image_after_otsu(sub_image, image_mask=sub_label_mask, min_label=the_largest_cell_id_now+1, num_peaks=num_peaks, k_for_convolution=k_for_convolution, min_distance=min_distance)
            if np.max(sub_watershed) == 0:
                print(cell_id)
            if np.max(sub_watershed) > 0:
                the_largest_cell_id_now = np.max(sub_watershed).astype(np.int64)
            tmp = out_labels[y_min:y_max+1, x_min:x_max+1]
            tmp[tmp==cell_id] = sub_watershed[tmp==cell_id]
    return out_labels

def split_cells_from_image_by_kmeans(
    labels: np.ndarray,
    image: np.ndarray,
    area_per_segment: int,
    k_for_convolution: int = 9,
    min_distance: int = 7,
    random_state: int = 42,
):
    out_labels = labels.copy()
    cell_area_dict = cal_cell_area(labels)
    bboxes_dict = get_boxes_for_all_cells(labels)
    the_largest_cell_id_now = np.max(labels).astype(np.int64)
    for cell_id, cell_area in cell_area_dict.items():
        if cell_area.astype(np.float32) / area_per_segment > 1:
            y_min, y_max, x_min, x_max = bboxes_dict[cell_id].values()
            sub_image = image[y_min:y_max+1, x_min:x_max+1]
            sub_label = labels[y_min:y_max+1, x_min:x_max+1]
            sub_label = np.where(sub_label == cell_id, cell_id, 0)
            sub_label_mask = np.where(sub_label > 0, 1, 0)
            max_cluster_num = int(cell_area / area_per_segment) + 1
            sub_kmeans = kmeans_image(
                image=sub_image, 
                min_cluster_num=2,
                max_cluster_num=max_cluster_num,
                image_mask=sub_label_mask,
                k_for_convolution=k_for_convolution,
                min_distance=min_distance,
                min_label=the_largest_cell_id_now+1,
                random_state=random_state,
                otsu_first=True
            )
            if np.max(sub_kmeans) > 0:
                the_largest_cell_id_now = np.max(sub_kmeans).astype(np.int64)
            tmp = out_labels[y_min:y_max+1, x_min:x_max+1]
            tmp[tmp==cell_id] = sub_kmeans[tmp==cell_id]
    return out_labels


            

def watershed_image(
    image: np.ndarray,
    image_mask: np.ndarray = None,
    k_for_convolution: int = 9,
    min_distance: int = 3,
    min_label: int = 1,
    return_markers: bool = False,
    return_smoothed_image: bool = False,
    num_peaks: int = np.inf,
) -> np.ndarray:
    if image_mask is not None:
        image = np.where(image_mask > 0, image, 0)
    smoothed_image = gaussian_blur(image, k=k_for_convolution)
    coordinates = peak_local_max(smoothed_image, min_distance=min_distance, num_peaks=num_peaks)
    markers = np.zeros_like(image, dtype=np.int64)
    if len(coordinates) > 0:
        markers[tuple(coordinates.T)] = np.arange(min_label, min_label + len(coordinates))
    else:
        markers = np.ones_like(image, dtype=np.int64) * min_label
    watershed = segmentation.watershed(-smoothed_image, mask=np.where(image>0, True, False), markers=markers)
    if return_markers and return_smoothed_image:
        return watershed, markers, smoothed_image
    elif return_markers:
        return watershed, markers
    elif return_smoothed_image:
        return watershed, smoothed_image
    else:
        return watershed

def otsu_thresholding(X):
    # 初始化类间方差和最佳阈值
    max_variance = 0
    best_threshold = 0

    for i in X:
        # 计算前景和背景的像素数
        w0 = np.sum(X<i)
        w1 = len(X) - w0

        # 如果前景或背景像素数为0，跳过
        if w0 == 0 or w1 == 0:
            continue

        # 计算前景和背景的平均灰度值
        m0 = np.mean(X[X<i])
        m1 = np.mean(X[X>=i])

        # 计算类间方差
        variance = w0 * w1 * (m0 - m1) ** 2

        # 如果当前方差大于最大方差，更新最大方差和最佳阈值
        if variance > max_variance:
            max_variance = variance
            best_threshold = i
    return best_threshold

# for one cell
def watershed_image_after_otsu(
    image: np.ndarray,
    image_mask: np.ndarray = None,
    k_for_convolution: int = 9,
    min_distance: int = 3,
    min_label: int = 1,
    num_peaks: int = np.inf,
) -> np.ndarray:
    threshold = otsu_thresholding(image[image>0])
    binary_mask = np.where(image > threshold, 1, 0)
    if image_mask is not None:
        binary_mask = np.where(image_mask > 0, binary_mask, 0)
    _, markers, smoothed_image = watershed_image(image, binary_mask, k_for_convolution, min_distance, min_label, num_peaks=num_peaks, return_markers=True, return_smoothed_image=True)
    watershed = segmentation.watershed(-smoothed_image, mask=image_mask, markers=markers)
    return watershed


def kmeans_image(
    image: np.ndarray,
    min_cluster_num: int = 2,
    max_cluster_num: int = 10,
    image_mask: np.ndarray = None,
    k_for_convolution: int = 9,
    min_distance: int = 7,
    min_label: int = 1,
    random_state: int = 42,
    otsu_first: bool = True,
):
    if otsu_first:
        threshold = otsu_thresholding(image[image>0])
    else:
        threshold = 0
    binary_mask = np.where(image > threshold, 1, 0)
        
    if image_mask is not None:
        binary_mask = np.where(image_mask > 0, binary_mask, 0)
    clean_image = np.where(binary_mask > 0, image, 0)
    smoothed_image = gaussian_blur(clean_image, k=k_for_convolution)
    coordinates = peak_local_max(smoothed_image, min_distance=min_distance)
    out_labels = np.zeros_like(image, dtype=np.int64)
    if len(coordinates) <= 1:
        out_labels = np.where(image>0, min_label, 0)
        return out_labels

    X = []
    for y in range(len(clean_image)):
        for x in range(len(clean_image[y])):
            for i in range(clean_image[y, x]):
                X.append([y, x])
    X = np.array(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)  # 推荐用新式随机数生成器，可复现
    n_sample = min(5000, X.shape[0])
    X = rng.choice(X, size=n_sample, replace=False, axis=0)
    highest_score = -np.inf
    best_n_cluster = min_cluster_num
    for n_cluster in range(min_cluster_num, max_cluster_num+1):
        kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X)
        avg_score = silhouette_score(X, labels, sample_size=5000, random_state=random_state)
        if avg_score > highest_score:
            highest_score = avg_score
            best_n_cluster = n_cluster
    kmeans = KMeans(n_clusters=best_n_cluster, n_init=10, random_state=random_state)
    kmeans.fit(X)

    X_all = []
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y, x] > 0:
                X_all.append([y, x])
    X_all = np.array(X_all, dtype=np.int64)
    labels = kmeans.predict(X_all)
    out_labels[tuple(X_all.T)] = labels + min_label
    return out_labels

    
def cal_cell_area(cell_labels: np.ndarray):
    """Calculate spot numbers for each cell.

    Args:
        cell_labels: cell labels.
    Returns:
        dict
    """
    cell_area_dict = {}

    t = np.bincount(cell_labels.flatten())
    for i in range(len(t)):
        if i > 0 and t[i] > 0:
            cell_area_dict[i] = t[i]
    return cell_area_dict


def get_dis_between_peak_and_cetroid_for_all_cells(
    labels: np.ndarray,
    image: np.ndarray,
    k_for_convolution: int = 9,
):
    dis_dict = {}
    bboxes_dict = get_boxes_for_all_cells(labels)
    for cell_id, bbox in bboxes_dict.items():
        one_dis = 0.0
        y_min, y_max, x_min, x_max = bbox['y_min'], bbox['y_max'], bbox['x_min'], bbox['x_max']
        sub_image = image[y_min:y_max+1, x_min:x_max+1]
        sub_label = labels[y_min:y_max+1, x_min:x_max+1]
        sub_label = np.where(sub_label == cell_id, cell_id, 0)
        sub_label_mask = np.where(sub_label > 0, 1, 0)
        sub_image = sub_image * sub_label_mask
        sub_smoothed_image = gaussian_blur(sub_image, k=k_for_convolution)
        coordinates = peak_local_max(sub_smoothed_image, num_peaks=1)
        if len(coordinates) == 1:
            cy, cx = center_of_mass(sub_label_mask)
            one_dis = np.linalg.norm(coordinates[0] - np.array([cy, cx]))
        dis_dict[cell_id] = one_dis
    return dis_dict

        