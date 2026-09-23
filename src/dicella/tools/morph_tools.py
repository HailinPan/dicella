from typing import Optional, Tuple, Union, Literal, List, Dict, Set
import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .common import create_dir_if_not_exists

def remove_columns_contain_strs(
    df: pd.DataFrame,
    strs: List[str],
):
    pattern = '|'.join(strs)
    mask = ~df.columns.str.contains(pattern, regex=True)
    df = df.loc[:,mask]
    return df

def extract_columns_contain_strs(
    df: pd.DataFrame,
    strs: List[str],
):
    pattern = '|'.join(strs)
    mask = df.columns.str.contains(pattern, regex=True)
    df = df.loc[:,mask]
    return df

def remove_location_columns(
    df: pd.DataFrame,
    strs: List[str] = ['BoundingBox', 'AreaShape_Center_', 'TrackObjects_', 'Location_'],
):
    return remove_columns_contain_strs(df, strs)


def agg_texture_columns(
    df: pd.DataFrame,
):
    df = df.copy()
    all_texture_columns = df.columns[df.columns.str.contains('Texture_', regex=True)]
    group_dict = {}
    for col in all_texture_columns:
        group_name = re.search(r'(\S+)_\d+_\d+_\d+$', col).group(1)
        if group_name not in group_dict:
            group_dict[group_name] = []
        group_dict[group_name].append(col)
    

    for group_name, cols in group_dict.items():
        group_name_prefix = re.search(r'(\S+exture_)', group_name).group(1)
        group_name_postfix = re.search(r'Texture_(\S+)', group_name).group(1)
        group_name_mean = group_name_prefix + "mean_" + group_name_postfix
        group_name_max = group_name_prefix + "max_" + group_name_postfix
        df[group_name_mean] = df[cols].mean(axis=1)
        df[group_name_max] = df[cols].max(axis=1)
    return df


def add_prefix_to_feature_columns(
    df: pd.DataFrame,
    prefix: str,
):
    assert df.columns[0] == 'ImageNumber'
    assert df.columns[1] == 'ObjectNumber'
    df.columns = list(df.columns[0:2]) + [f"{prefix}_{i}" for i in df.columns[2:]]
    return df

def extract_features(
    df: pd.DataFrame,
    affected_by_absolute_signal_strength: bool = False,
    affected_by_rotation: bool = False,
):
    feature_info_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "morph_feature.csv"),
        # "/Data/user/panhailin/code/git_hub/dicella/src/dicella/tools/morph_feature.csv", 
        comment='#',
        dtype={
            "feature_name": str,
            "affected_by_absolute_signal_strength": bool,
            "affected_by_rotation": bool
        }
    )
    need_feature_names = feature_info_df.loc[(feature_info_df['affected_by_absolute_signal_strength'] == affected_by_absolute_signal_strength) & (feature_info_df['affected_by_rotation'] == affected_by_rotation),'feature_name']
    if "ObjectNumber" in df.columns:
        need_feature_names = ['ObjectNumber'] + list(need_feature_names)
    if "ImageNumber" in df.columns:
        need_feature_names = ['ImageNumber'] + list(need_feature_names)
    df = extract_columns_contain_strs(df, need_feature_names)
    return df


def scale_features(
    df: pd.DataFrame,
    features: List[str] = None
):
    scaler = RobustScaler()
    exclude = {"ImageNumber", "ObjectNumber"} 
    if features is None:
        all_columns = df.columns
        features = [f for f in all_columns if f not in exclude]

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index
    )
    other_columns = list(set(all_columns) - set(features))
    df_scaled = pd.concat((df[other_columns], df_scaled), dim=1)
    return df_scaled


def drop_high_nan_columns(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    删除包含高缺失值的列

    Args:
        df: 输入的 pandas DataFrame
        threshold: 缺失值比例阈值，默认 0.05

    Returns:
        pandas DataFrame，包含缺失值比例低于阈值的列
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 pandas DataFrame")

    if df.empty:
        return df.copy()
    
    nan_ratio = df.isna().mean()
    keep_cols = nan_ratio[nan_ratio <= threshold].index.tolist()

    return df[keep_cols]

def drop_high_nan_rows(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    删除包含高缺失值的行

    Args:
        df: 输入的 pandas DataFrame
        threshold: 缺失值比例阈值，默认 0.05

    Returns:
        pandas DataFrame，包含缺失值比例低于阈值的行
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 pandas DataFrame")

    if df.empty:
        return df.copy()
    
    nan_ratio = df.isna().mean(axis=1)
    keep_rows = nan_ratio[nan_ratio <= threshold].index.tolist()

    return df.loc[keep_rows,:]


def fill_nan_with_column_mean(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    用列的均值填充缺失值

    Args:
        df: 输入的 pandas DataFrame

    Returns:
        pandas DataFrame，缺失值已用列的均值填充
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 pandas DataFrame")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def remove_redundant_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "spearman",
    ignore_cols: List[str] = ['ImageNumber', 'ObjectNumber']
) -> pd.DataFrame:
    """
    移除相关性高的特征

    Args:
        df: 输入的 pandas DataFrame
        threshold: 相关性阈值，默认 0.9
        method: 相关性计算方法，默认 "spearman"
        ignore_cols: 忽略的列，默认 ['ImageNumber', 'ObjectNumber']

    Returns:
        pandas DataFrame，去冗余后的特征
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 pandas DataFrame")

    # 1. 分离标识列与待分析列
    id_cols = [c for c in ignore_cols if c in df.columns]
    analysis_cols = [c for c in df.columns if c not in id_cols]

    if len(analysis_cols) <= 1:
        return df[id_cols + analysis_cols].copy()

    # 2. 从待分析列中分离数值列与非数值列
    numeric_cols = df[analysis_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in analysis_cols if c not in numeric_cols]

    if len(numeric_cols) <= 1:
        return df[id_cols + analysis_cols].copy()

    # 3. 计算绝对相关矩阵并转换为距离矩阵
    corr_matrix = df[numeric_cols].corr(method=method).abs()
    corr_matrix = corr_matrix.fillna(0.0)

    # 距离 = 1 - |r|，确保对角线为0且矩阵对称
    dist_matrix = 1.0 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0  # 强制对称，消除浮点误差

    # 4. 层次聚类 + 切割
    condensed_dist = squareform(dist_matrix, checks=False)
    link_matrix = linkage(condensed_dist, method="average")

    # fcluster 的 t 参数是距离阈值；threshold=0.95 → 距离阈值=0.05
    cluster_labels = fcluster(link_matrix, t=1.0 - threshold, criterion="distance")

    # 5. 按簇分组，每组选代表
    cluster_map: Dict[int, List[str]] = {}
    for feat, label in zip(numeric_cols, cluster_labels):
        cluster_map.setdefault(label, []).append(feat)

    selected_numeric: List[str] = []
    for members in cluster_map.values():
        if len(members) == 1:
            selected_numeric.append(members[0])
        else:
            # 选择与全局其他特征平均绝对相关性最高的作为代表（信息枢纽）
            scores = {
                feat: corr_matrix.loc[feat].drop(feat).mean()
                for feat in members
            }
            max_score = max(scores.values())
            best = sorted(
                f for f, s in scores.items() if abs(s - max_score) < 1e-12
            )[0]
            selected_numeric.append(best)

    # 6. 组装最终列顺序：标识列在前 + 保留的分析列按原始顺序
    selected_analysis_set = set(selected_numeric) | set(non_numeric_cols)
    final_analysis_cols = [c for c in analysis_cols if c in selected_analysis_set]
    final_cols = id_cols + final_analysis_cols

    return df[final_cols]


def filter_feature_csv(
    feature_csv_file: str,
    column_nan_threshold: float = 0.05,
    row_nan_threshold: float = 0.05,
    correlation_threshold: float = 0.9,
    save: bool = False,
    save_path: str = "./",
    save_name: str = "feature_filtered.csv",

):
    df = pd.read_csv(feature_csv_file)
    df = agg_texture_columns(df)
    df = drop_high_nan_columns(df, threshold=column_nan_threshold)
    df = drop_high_nan_rows(df, threshold=row_nan_threshold)
    df = fill_nan_with_column_mean(df)
    df = extract_features(df)
    df = remove_redundant_features(df, threshold=correlation_threshold)
    if save:
        create_dir_if_not_exists(save_path)
        df.to_csv(os.path.join(save_path, save_name), index=False)
        return df
    else:
        return df

def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """对DataFrame每一列计算Robust Z-Score"""
    median = df.median(axis=0)
    q75 = df.quantile(0.75, axis=0)
    q25 = df.quantile(0.25, axis=0)
    iqr = q75 - q25
    
    # 防止除零：IQR为0的列（如恒定值）设为0
    iqr = iqr.replace(0, np.nan)
    
    result = (df - median) / iqr
    return result.fillna(0.0)

