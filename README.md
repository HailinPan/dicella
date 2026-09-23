# DiCellA

**Digital Cell Analysis Toolkit**

This package is a digital cell analysis toolkit that provides a set of tools for cell analysis.

## Repository
- [DiCellA in GitHub](https://github.com/HailinPan/dicella)

## Installation

### Core installation (without cell segmentation)
```bash
# create a new conda environment
conda create -n dicella_env python=3.11 -y
conda activate dicella_env

# install from PyPI (core only, no cellpose/GPU required)
pip install dicella

# or install from GitHub
pip install git+https://github.com/HailinPan/dicella.git
```

### Full installation (with cell segmentation)
Cell segmentation depends on [Cellpose](https://github.com/MouseLand/cellpose), which requires PyTorch and a CUDA-enabled GPU.
```bash
# option 1: install dicella with the [seg] extra
pip install "dicella[seg]"

# option 2: install cellpose separately, then dicella
pip install cellpose
pip install dicella

# option 3: install everything at once from GitHub
pip install "git+https://github.com/HailinPan/dicella.git#egg=dicella[seg]"
```

### Development installation
- Clone the repository from GitHub.
- Create a new conda environment.
- Activate the environment.
- Install the dependencies.

```bash
# clone the repository
git clone https://github.com/HailinPan/dicella.git

# create a new conda environment
conda create -n dicella_env python=3.11 -y
conda activate dicella_env

# install the dependencies
pip install -r requirements.txt

# install the package
cd dicella
pip install -e .
```

### Verify installation

```bash
# check core import works (no cellpose needed)
python -c "import dicella; print(dicella.__version__)"

# check CLI is available
dicella --help

# if you installed with [seg], check cell segmentation import
python -c "from dicella.cell_seg import cell_seg_on_ssdna_by_cellpose; print('cellpose OK')"
```

