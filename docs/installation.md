# Installation Guide

This guide will help you set up the runtime environment and install DiCellA. We recommend using Conda to manage your Python environment to avoid dependency conflicts.

## Prerequisites

Before installing DiCellA, create and activate a dedicated Conda environment:

```bash
# Create a new conda environment named dicella_env with Python 3.11
conda create -n dicella_env python=3.11 -y

# Activate the environment
conda activate dicella_env
```

## Installation Methods
Choose one of the following methods based on your use case.
### Method 1: Install from PyPI

```bash
pip install dicella
```
### Method 2: Install from GitHub Source
```bash
pip install git+https://github.com/HailinPan/dicella.git
```