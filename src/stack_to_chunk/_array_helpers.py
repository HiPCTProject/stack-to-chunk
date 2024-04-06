import numpy as np
import numpy.typing as npt
import zarr
from dask import delayed


@delayed  # type: ignore[misc]
def _copy_slab(
    arr_zarr: zarr.Array, slab: npt.NDArray[np.uint16], zstart: int, zend: int
) -> None:
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
    # Write out data
    arr_zarr[:, :, zstart:zend] = slab
