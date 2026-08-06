# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'DiCellA'
copyright = '2026, HailinPan'
author = 'HailinPan'

# ✅ 安全读取版本号：本地未安装时回退到 "unknown"
try:
    from importlib.metadata import version as get_version
    release = get_version("dicella")
except Exception:
    release = "unknown"

version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",          # ✅ 新增：支持 Markdown 编写文档
    "sphinx.ext.autodoc",   # ✅ 新增：自动从代码提取 API 文档
    "sphinx.ext.napoleon",  # ✅ 新增：支持 Google/NumPy 风格 docstring
    "sphinx.ext.intersphinx", # ✅ 新增：链接到 Python/第三方库官方文档
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = "en"  # ✅ 显式声明（你 quickstart 时选了 en，写出来更清晰）

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # ✅ 修改：alabaster → RTD 主题（Read the Docs 标配）
html_static_path = ['_static']

# ✅ 新增：intersphinx 映射（按需保留）
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# ✅ 新增：MyST 扩展配置（如果用 Markdown）
myst_enable_extensions = [
    "colon_fence",      # ::: 指令语法
    "deflist",          # 定义列表
    "fieldlist",        # 字段列表
    "substitution",     # 变量替换
]