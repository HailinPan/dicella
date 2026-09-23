"""DiCellA: Digital Cell Analysis Toolkit."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from . import tools as tl
from . import plots as pl
from . import cell_seg as cs

__all__ = ["__version__", "tl", "pl"]