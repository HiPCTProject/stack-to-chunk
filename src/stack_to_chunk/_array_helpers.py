import dask.array as da
import numpy as np
import skimage.measure
import zarr
from joblib import delayed
from loguru import logger


@delayed  # type: ignore[misc]
def _copy_slab(arr_zarr: zarr.Array, slab: da.Array, zstart: int, zend: int) -> None:
    """
    Copy a single slab of data to a zarr array.

    Parameters
    ----------
    arr_zarr :
        Array to copy to.
    slab :
        Slab of data to copy.
    zstart, zend :
        Start and end indices to copy to in destination array.

    """
    logger.info(f"Reading z={zstart} -> {zend - 1}")
    data = np.empty(slab.shape, dtype=slab.dtype)
    for i in range(slab.shape[2]):
        logger.info(f"Reading z={zstart + i}")
        data[:, :, i] = np.array(slab[:, :, i])

    logger.info(f"Writing z={zstart} -> {zend - 1}")
    # Write out data
    arr_zarr[:, :, zstart:zend] = data
    logger.info(f"Finished copying z={zstart} -> {zend - 1}")


@delayed  # type: ignore[misc]
def _downsample_slab(arr_in: zarr.Array, arr_out: zarr.Array, block_idx_z: int) -> None:
    """
    Copy a single slab from one array to the next, downsampling by a factor of two.

    Data is copied from a slab starting at `block_idx` and ending at
    `block_idx + 2 * arr_in.chunks`, ie a cube of (2, 2, 2) chunks.
    Data is downsampled using local mean, and writen to a single chunk in `arr_out`.

    Parameters
    ----------
    arr_in :
        Input array.
    arr_out :
        Output array. Must have the same chunk shape as `arr_in`.
    block_idx :
        Index of block to copy. Must be a multiple of swice the chunk size of `arr_in`.

    """
    chunk_size = arr_in.shards[2] * 2
    if block_idx_z % chunk_size != 0:
        msg = f"Block index {block_idx_z} not aligned with chunks {chunk_size}"
        raise ValueError(msg)

    data = arr_in[
        :,
        :,
        block_idx_z : block_idx_z + chunk_size,
    ]
    # Pad to an even number
    pads = np.array(data.shape) % 2
    pad_width = [(0, p) for p in pads]
    data = np.pad(data, pad_width, mode="edge")
    data = skimage.measure.block_reduce(data, block_size=2, func=np.mean)

    block_idx_out = block_idx_z // 2
    chunk_size_out = chunk_size // 2
    arr_out[
        :,
        :,
        block_idx_out : block_idx_out + chunk_size_out,
    ] = data
