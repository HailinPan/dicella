from typing import Optional, Tuple, Union, Literal
import os
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.sparse import issparse
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clr
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

from ..tools import create_dir_if_not_exists

def plot_3d_gene_expression(
    adata: AnnData, 
    obsm_key: str, 
    gene_name: str, 
    layer: str = 'X',
    point_size: float = 3.0,
    save_html: bool = True,
    save_path: str = './',
    save_name: str = '3d_gene_expression.html',
):
    """
    使用 Plotly 绘制空转数据的三维散点图
    
    参数:
        adata (AnnData): 包含空转数据的 AnnData 对象
        obsm_key (str): obsm 中的坐标键名 (如 'X_umap', 'X_pca')
        gene_name (str): 目标基因名称
        layer (str): 要使用的表达量层 (默认 'X')
        point_size (float): 点的大小 (默认 3.0)
        save_html (bool): 是否保存为 HTML 文件 (默认 True)
        save_path (str): 保存路径 (默认 './')
        save_name (str): 保存文件名 (默认 '3d_gene_expression.html')
    """
    
    # 决定使用哪一层数据
    adata = adata.copy()
    if layer != 'X':
        adata.X = adata.layers[layer].copy()
    
    # 2. 提取 3D 坐标
    if obsm_key not in adata.obsm:
        raise ValueError(f"❌ 在 adata.obsm 中未找到 '{obsm_key}'。\n可用的键: {list(adata.obsm.keys())}")
    
    coords = adata.obsm[obsm_key]
    if coords.shape[1] < 3:
        raise ValueError(f"❌ '{obsm_key}' 只有 {coords.shape[1]} 个维度，绘制3D图至少需要3个维度。")
    
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # 3. 提取基因表达量 (兼容稀疏矩阵)
    if gene_name in adata.var_names:
        gene_idx = adata.var_names.get_loc(gene_name)
        expr = adata.X[:, gene_idx]
        
        # 处理 scipy 稀疏矩阵
        if issparse(expr):
            expr = expr.toarray()
        expr = np.asarray(expr).flatten()
        
    elif gene_name in adata.obs.columns:
        # 备用：如果基因表达量已经被计算并存放在 obs 中
        expr = adata.obs[gene_name].values
    else:
        raise ValueError(f"❌ 基因 '{gene_name}' 未在 var_names 或 obs 中找到。")
    
    # 4. 构建 DataFrame
    df = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': z,
        'Expression': expr,
        'Cell_ID': adata.obs_names.astype(str)  # 确保 barcode 是字符串
    })
    
    # 5. 绘图 (Plotly Express)
        
    fig = px.scatter_3d(
        df, 
        x='X', y='Y', z='Z',
        color='Expression',
        hover_data={'Cell_ID': True, 'X': ':.2f', 'Y': ':.2f', 'Z': ':.2f', 'Expression': ':.3f'},
        color_continuous_scale='Plasma',  # 推荐 Plasma 或 Viridis，对色弱友好
        opacity=0.7,
        labels={'color': 'Expression'}
    )
    
    # 6. 样式优化 (关键：让高表达细胞显示在最上层)
    # 按表达量排序，确保高表达的细胞渲染在顶层，不被低表达的0值遮挡
    df_sorted = df.sort_values(by='Expression')
    fig.data[0].x = df_sorted['X']
    fig.data[0].y = df_sorted['Y']
    fig.data[0].z = df_sorted['Z']
    fig.data[0].marker.color = df_sorted['Expression']
    fig.data[0].customdata = df_sorted[['Cell_ID', 'X', 'Y', 'Z', 'Expression']].values
    
    # 调整点的大小和背景
    fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title=f'{obsm_key}1',
            yaxis_title=f'{obsm_key}2',
            zaxis_title=f'{obsm_key}3',
            bgcolor='rgb(240, 240, 245)'  # 浅灰背景让细胞更突出
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    # 7. 显示或保存
    if save_html:
        create_dir_if_not_exists(save_path)
        html_whole_path = os.path.join(save_path, save_name)
        fig.write_html(html_whole_path)
        print(f"✅ 交互式图表已保存至: {html_whole_path}")
    