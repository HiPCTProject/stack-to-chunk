"""
Utility functions for reading tiff files.

Copied and modified from brainglobe/cellfinder and brainglobe/brainglobe-utils.
"""

import os
from pathlib import Path

import dask.array as da
from dask.delayed import delayed
from numpy.typing import DTypeLike
from tifffile import TiffFile, imread


def load_env_var_as_path(env_var: str) -> Path:
    """Load an environment variable as a Path object."""
    path_str = os.getenv(env_var)

    if path_str is None:
        msg = f"Please set the environment variable {env_var}."
        raise ValueError(msg)

    path = Path(path_str)
    if not path.is_dir():
        msg = f"{path} is not a valid directory path."
        raise ValueError(msg)
    return path


def get_tiff_meta(
    path: Path,
) -> tuple[tuple[int, int], DTypeLike]:
    """
    Get the shape and dtype of the first page of a tiff file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the tiff file.

    Returns
    -------
    tuple[tuple[int, int], np.dtype]
        Shape and dtype of the first page.

    """
    with TiffFile(path) as tfile:
        nz = len(tfile.pages)
        if not nz:
            msg = f"tiff file {path} has no pages!"
            raise ValueError(msg)
        first_page = tfile.pages[0]

    return tfile.pages[0].shape, first_page.dtype


def read_tiff_stack_with_dask(path: Path) -> da.Array:
    """
    Read a stack of tiff files into a dask array.

    Based on https://github.com/tlambert03/napari-ndtiffs

    Parameters
    ----------
    path : pathlib.Path
        Path to the folder containing the tiff files, or
        a text file containing the paths to the tiff files.

    Returns
    -------
    da.Array
        Dask array of the tiff stack.

    """
    if path.suffix == ".txt":
        with path.open("r") as f:
            file_paths = [Path(line.rstrip()) for line in f.readlines()]
    else:
        file_paths = sorted(path.glob("*.tif"))

    shape, dtype = get_tiff_meta(file_paths[0])
    lazy_imread = delayed(imread)
    lazy_arrays = [lazy_imread(fn) for fn in sorted(file_paths)]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=shape, dtype=dtype)
        for delayed_reader in lazy_arrays
    ]
    return da.stack(dask_arrays, axis=0)
