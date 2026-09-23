from typing import Optional, Tuple, Union, Literal, List, Dict, Set
import numpy as np
import pandas as pd
import os

def create_dir_if_not_exists(
    dir_path: str,
):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)