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
def _downsample_block(
    arr_in: zarr.Array, arr_out: zarr.Array, block_idx: tuple[int, int, int]
) -> None:
    """
    Copy a single block from one array to the next, downsampling by a factor of two.

    Data is copied from a block starting at `block_idx` and ending at
    `block_idx + 2 * arr_in.chunks`, ie a cube of (2, 2, 2) chunks.
    Data is downsampled using local mean, and writen to a single chunk in `arr_out`.

    Parameters
    ----------
    arr_in :
        Input array.
    arr_out :
        Output array. Must have the same chunk shape as `arr_in`.
    block_idx :
        Index of block to copy. Must be a multiple of the shard shape in `arr_out`.

    """
    shard_shape: tuple[int, int, int] = arr_out.shards
    np.testing.assert_equal(
        np.array(block_idx) % np.array(shard_shape),
        np.array([0, 0, 0]),
        err_msg=f"Block index {block_idx} not aligned with shards {shard_shape}",
    )

    in_slice = (
        slice(
            block_idx[0] * 2, min((block_idx[0] + shard_shape[0]) * 2, arr_in.shape[0])
        ),
        slice(
            block_idx[1] * 2, min((block_idx[1] + shard_shape[1]) * 2, arr_in.shape[1])
        ),
        slice(
            block_idx[2] * 2, min((block_idx[2] + shard_shape[2]) * 2, arr_in.shape[2])
        ),
    )
    data = arr_in[in_slice]

    # Pad to an even number
    pads = np.array(data.shape) % 2
    pad_width = [(0, p) for p in pads]
    data = np.pad(data, pad_width, mode="edge")
    data = skimage.measure.block_reduce(data, block_size=2, func=np.mean)

    out_slice = (
        slice(block_idx[0], min((block_idx[0] + shard_shape[0]), arr_out.shape[0])),
        slice(block_idx[1], min((block_idx[1] + shard_shape[1]), arr_out.shape[1])),
        slice(block_idx[2], min((block_idx[2] + shard_shape[2]), arr_out.shape[2])),
    )
    arr_out[out_slice] = data
