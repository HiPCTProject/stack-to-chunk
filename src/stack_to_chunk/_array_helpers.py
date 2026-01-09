from pathlib import Path

import dask.array as da
import numpy as np
import skimage.measure
import tensorstore as ts
from joblib import delayed
from loguru import logger


@delayed  # type: ignore[misc]
def _copy_slab(arr_path: Path, slab: da.Array, zstart: int, zend: int) -> None:
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
    arr_zarr = _open_with_tensorstore(arr_path)
    arr_zarr[:, :, zstart:zend].write(data).result()
    logger.info(f"Finished copying z={zstart} -> {zend - 1}")


@delayed  # type: ignore[misc]
def _downsample_block(
    arr_in_path: Path, arr_out_path: Path, block_idx: tuple[int, int, int]
) -> None:
    """
    Copy a single block from one array to the next, downsampling by a factor of two.

    Data is copied from a block starting at `block_idx` and ending at
    `block_idx + 2 * arr_in.chunks`, ie a cube of (2, 2, 2) chunks.
    Data is downsampled using local mean, and writen to a single chunk in `arr_out`.

    Parameters
    ----------
    arr_in_path :
        Path to input array.
    arr_out_path :
        Path to output array. Must have the same chunk shape as `arr_in`.
    block_idx :
        Index of block to copy. Must be a multiple of the shard shape in `arr_out`.

    """
    arr_in = _open_with_tensorstore(arr_in_path)
    arr_out = _open_with_tensorstore(arr_out_path)
    shard_shape: tuple[int, int, int] = arr_out.chunk_layout.write_chunk.shape
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
    data = arr_in[in_slice].read().result()

    # Pad to an even number
    pads = np.array(data.shape) % 2
    pad_width = [(0, p) for p in pads]
    data = np.pad(data, pad_width, mode="edge")
    data = skimage.measure.block_reduce(data, block_size=2, func=np.mean).astype(
        data.dtype
    )

    out_slice = (
        slice(block_idx[0], min((block_idx[0] + shard_shape[0]), arr_out.shape[0])),
        slice(block_idx[1], min((block_idx[1] + shard_shape[1]), arr_out.shape[1])),
        slice(block_idx[2], min((block_idx[2] + shard_shape[2]), arr_out.shape[2])),
    )
    arr_out[out_slice].write(data).result()


def _open_with_tensorstore(arr_path: Path) -> ts.TensorStore:
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(arr_path),
            },
            "open": True,
        }
    ).result()
