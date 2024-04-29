"""
Utilities for downsampling images.

These are based on the ``ome_zarr.dask_utils.py`` module of the ome-zarr-py library,
originally contributed by by Andreas Eisenbarth @aeisenbarth.
See https://github.com/toloudis/ome-zarr-py/pull/1
"""

import numpy as np
import skimage.transform
from dask import array as da


def _rechunk_to_even(image: da.Array) -> da.Array:
    """
    Rechunk the input image so that chunk sizes are even in each dimension.

    This guarantees integer chunk sizes after downsampling by two.
    """
    factors = np.array([0.5] * image.ndim)
    even_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    return image.rechunk(even_chunksize)


def _half_shape(input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Calculate the output shape after downsampling by two in each dimension.

    Rounds up to the nearest integer after division.
    """
    return tuple(np.ceil(np.array(input_shape) / 2).astype(int))


def _resize_block(block: da.Array) -> da.Array:
    """
    Resize a block by a factor of 2 in each dimension using linear interpolation.
    """
    new_block_shape = _half_shape(block.shape)
    return skimage.transform.resize(
        block,
        new_block_shape,
        order=1,
        anti_aliasing=False,
    ).astype(block.dtype)


def downsample_by_two(image: da.Array) -> da.Array:
    """
    Downsample a dask array by two in each dimension.

    Parameters
    ----------
    image : da.Array
        The input image.

    Returns
    -------
    da.Array
        The downsampled image, which has half the size of the input image in each
        dimension.

    """
    new_image_shape = _half_shape(image.shape)
    new_image_slices = tuple(slice(0, d) for d in new_image_shape)

    # Rechunk the image so that chunk sizes will be whole numbers after downsampling
    image_rechunked = _rechunk_to_even(image)
    new_image_chunksize = _half_shape(image_rechunked.chunksize)

    new_image = da.map_blocks(
        _resize_block,
        image_rechunked,
        chunks=new_image_chunksize,
        dtype=image.dtype,
    )[new_image_slices]

    # restore the original chunking and type
    return new_image.rechunk(image.chunksize).astype(image.dtype)
