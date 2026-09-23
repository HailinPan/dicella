"""DiCellA command-line interface.

Usage examples:
    dicella cell-seg --input image.tif --output labels.tif
    dicella mask-to-labels --input mask.png --output labels.tif
    dicella get-boundaries --input labels.tif --output boundaries.tif --thickness 2
    dicella filter-features --input features.csv --output filtered.csv
    dicella gem-to-adata --gem data.gem --labels mask.tif --output adata.h5ad
    dicella --version
    python -m dicella --help
"""

import argparse
import sys
import os
import cv2
import numpy as np
import pandas as pd

from dicella.tools.common import create_dir_if_not_exists

# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------
def cmd_gem_to_tif(args):
    """Convert a GEM file to a TIF file. 画出RNA图，为了配准ssdna图。"""
    from dicella.cell_seg import read_gem_agg

    print("Start to convert GEM file to TIF file.")

    adata = read_gem_agg(args.gem)
    rna = adata.X.toarray().astype(np.uint8)

    create_dir_if_not_exists(args.output)
    cv2.imwrite(os.path.join(args.output, f"{args.prefix}_rna_figure_from_gem.tif"), rna)
    print(f"RNA figure saved to {os.path.join(args.output, f'{args.prefix}_rna_figure_from_gem.tif')}")

def cmd_cell_seg_on_ssdna_by_cellpose(args):
    """Cell segmentation with ssdna image by CellPose."""
    from dicella.cell_seg import cell_seg_on_ssdna_by_cellpose
    kwargs = {}
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.model_name is not None:
        kwargs["model_name"] = args.model_name

    print("Start to cell segmentation with ssdna image by CellPose.")
    cell_labels = cell_seg_on_ssdna_by_cellpose(args.img, **kwargs)
    create_dir_if_not_exists(args.output)
    cv2.imwrite(os.path.join(args.output, f"{args.prefix}_cell_labels_cellpose.tif"), cell_labels)
    print(f"Cell segmentation with ssdna image by CellPose done. Result saved to {os.path.join(args.output, f'{args.prefix}_cell_labels_cellpose.tif')}")
    


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with sub-commands."""

    parser = argparse.ArgumentParser(
        prog="dicella",
        description="DiCellA: Digital Cell Analysis Toolkit",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # --- gem-to-tif ---
    p = subparsers.add_parser(
        "gem-to-tif",
        help="Convert a GEM file to a TIF file. 画出RNA图，为了配准ssdna图。",
        description="Convert a GEM file to a TIF file. 画出RNA图，为了配准ssdna图。",
    )
    p.add_argument("--gem", required=True, help="Path to the input GEM file.")
    p.add_argument("-o", "--output", required=True, help="Path to save the TIF file.")
    p.add_argument("-p", "--prefix", required=True, help="Prefix for the TIF file. Suggested: sample_id.")
    p.set_defaults(func=cmd_gem_to_tif)

    # --- cell-seg-with-ssdna-by-cellpose ---
    p = subparsers.add_parser(
        "cell-seg-on-ssdna-by-cellpose",
        help="Cell segmentation with ssdna image by CellPose.",
        description="Cell segmentation with ssdna image by CellPose.",
    )
    p.add_argument("--img", required=True, help="Path to the input ssdna image.")
    p.add_argument("-o", "--output", required=True, help="Path to save the TIF file.")
    p.add_argument("-p", "--prefix", required=True, help="Prefix for the TIF file. Suggested: sample_id.")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for CellPose.")
    p.add_argument("--model-name", type=str, default=None, help="Model name for CellPose.")
    p.set_defaults(func=cmd_cell_seg_on_ssdna_by_cellpose)

    return parser

    


def main(args=None):
    """Main entry point for the CLI."""
    parser = build_parser()

    if args is None:
        args = sys.argv[1:]

    # Handle --version before sub-command parsing
    if "--version" in args and len(args) == 1:
        from dicella import __version__
        print(f"dicella {__version__}")
        return

    parsed = parser.parse_args(args)

    if parsed.version:
        from dicella import __version__
        print(f"dicella {__version__}")
        return

    if not hasattr(parsed, "func"):
        parser.print_help()
        return

    parsed.func(parsed)


if __name__ == "__main__":
    main()
