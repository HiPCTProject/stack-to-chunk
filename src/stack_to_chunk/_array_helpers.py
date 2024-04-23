import dask.array as da
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
    logger.info(f"Reading z={zstart} -> {zend}")
    # Read in data
    slab = slab.compute(num_workers=1)
    logger.info(f"Writing z={zstart} -> {zend}")
    # Write out data
    arr_zarr[:, :, zstart:zend] = slab
    logger.info(f"Finished copying z={zstart} -> {zend}")
