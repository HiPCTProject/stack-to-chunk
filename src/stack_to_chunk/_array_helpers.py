import ctypes
import ctypes.util
import gc
from collections.abc import Callable
from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import skimage.measure
import tensorstore as ts
from joblib import delayed
from loguru import logger


def _release_memory() -> None:
    """
    Return freed memory to the operating system.

    Large transient numpy and tensorstore buffers are freed at the Python level
    when a slab-processing function returns, but with the glibc allocator the pages
    are retained in the process arena rather than handed back to the OS. Without
    this, resident memory ratchets up slab-by-slab. ``malloc_trim`` is glibc-only,
    so it is guarded and becomes a no-op elsewhere.
    """
    gc.collect()
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    try:
        libc = ctypes.CDLL(libc_name)
        malloc_trim = libc.malloc_trim
    except (OSError, AttributeError):
        return
    malloc_trim(0)


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
        data[:, :, i] = np.array(slab[:, :, i], dtype=slab.dtype)

    logger.info(f"Writing z={zstart} -> {zend - 1}")
    # Write out data
    arr_zarr = _open_with_tensorstore(arr_path)
    arr_zarr[:, :, zstart:zend].write(data).result()
    logger.info(f"Finished copying z={zstart} -> {zend - 1}")

    # Hand the slab buffer back to the OS before returning so resident memory
    # does not ratchet up slab-by-slab.
    del data, arr_zarr
    _release_memory()


@delayed  # type: ignore[misc]
def _downsample_block(
    arr_in_path: Path,
    arr_out_path: Path,
    block_idx: tuple[int, int, int],
    downsample_func: Callable[[npt.ArrayLike], npt.NDArray] = np.mean,
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
    downsample_func :
        Function to use to downsample blocks of data.

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
    data = skimage.measure.block_reduce(
        data, block_size=2, func=downsample_func
    ).astype(data.dtype)

    out_slice = (
        slice(block_idx[0], min((block_idx[0] + shard_shape[0]), arr_out.shape[0])),
        slice(block_idx[1], min((block_idx[1] + shard_shape[1]), arr_out.shape[1])),
        slice(block_idx[2], min((block_idx[2] + shard_shape[2]), arr_out.shape[2])),
    )
    arr_out[out_slice].write(data).result()

    # Hand the block/shard buffers back to the OS before returning.
    del data, arr_in, arr_out
    _release_memory()


def _open_with_tensorstore(arr_path: Path) -> ts.TensorStore:
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(arr_path),
            },
            "open": True,
            # Bound the read cache so repeated opens (e.g. in the downsample read
            # path) cannot grow an unbounded in-memory cache.
            "cache_pool": {"total_bytes_limit": 100_000_000},
            "recheck_cached_data": "open",
        }
    ).result()
