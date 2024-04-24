import dask.array as da
import numpy as np
import zarr
from loguru import logger


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
    logger.info(f"Reading z={zstart} -> {zend-1}")
    data = np.empty(slab.shape, dtype=slab.dtype)
    for i in range(slab.shape[2]):
        logger.info(f"Reading z={zstart + i}")
        data[:, :, i] = np.array(slab[:, :, i])

    logger.info(f"Writing z={zstart} -> {zend-1}")
    # Write out data
    arr_zarr[:, :, zstart:zend] = data
    logger.info(f"Finished copying z={zstart} -> {zend-1}")


def _downsample_block(
    arr_in: zarr.Array, arr_out: zarr.Array, block_idx: tuple[int, int, int]
) -> None:
    """
    Copy a single block from one array to the next, downsampling by a factor of two.
    """
    chunk_size = arr_in.chunks[0] * 2
    np.testing.assert_equal(
        np.array(block_idx) % chunk_size,
        np.array([0, 0, 0]),
        err_msg=f"Block index {block_idx} not aligned with chunks {chunk_size}",
    )

    data = arr_in[
        block_idx[0] : block_idx[0] + chunk_size,
        block_idx[1] : block_idx[1] + chunk_size,
        block_idx[2] : block_idx[2] + chunk_size,
    ]
    # Pad to an even number
    pads = np.array(data.shape) % 2
    pad_width = [(0, p) for p in pads]
    data = np.pad(data, pad_width, mode="edge")

    # Take mean
    data = (data[::2, ::2, ::2] + data[1::2, 1::2, 1::2]) / 2

    block_idx_out = np.array(block_idx) // 2
    chunk_size_out = chunk_size // 2
    arr_out[
        block_idx_out[0] : block_idx_out[0] + chunk_size_out,
        block_idx_out[1] : block_idx_out[1] + chunk_size_out,
        block_idx_out[2] : block_idx_out[2] + chunk_size_out,
    ] = data
