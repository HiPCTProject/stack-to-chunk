from pathlib import Path
from typing import Any, Literal

import dask
import dask.array as da
import dask.delayed
import zarr
from numcodecs.abc import Codec

from stack_to_chunk.ome_ngff import SPATIAL_UNIT

"""
Strategy:

- Divide data up into slabs of size 128.
- For each slab read data into memory.
  This requires 128 * nx * ny * 2 bytes of memory per slab.
- Write out data to zarr array.
- Successively downsample and write out data.
- Parallelise the above across slabs.


Assumes that:
- It's expensive to read a single slice of original data into memory, and
  the whole slice must be read in at once (both of these are true for JPEG2000)
"""


def create_group(
    path: Path,
    *,
    name: str,
    spatial_unit: SPATIAL_UNIT,
    voxel_sizes: tuple[float, float, float],
    n_downsample_levels: int,
) -> zarr.Group:
    """
    Create a zarr group suitable for storing multiscale data.

    This is specialised to work with 3D spatial data.

    Parameters
    ----------
    path :
      Directory to create group under on local filesystem.
    name :
      Name of dataset.
    spatial_unit :
      Spatial unit for the dataset.
    voxel_sizes :
      Voxel sizes (in units of spatial unit).
    n_downsample_levels :
      Number of dowmsampling levels to include in the data.
    """
    root = zarr.open_group(store=path, mode="w")

    multiscales: dict[str, Any] = {}
    multiscales["version"] = "0.4"
    multiscales["name"] = name
    multiscales["axes"] = [
        {"name": "z", "type": "space", "unit": spatial_unit},
        {"name": "y", "type": "space", "unit": spatial_unit},
        {"name": "x", "type": "space", "unit": spatial_unit},
    ]
    multiscales["type"] = "linear"
    multiscales["metadata"] = {
        "description": "Downscaled using linear resampling",
    }

    datasets = []
    for level in range(n_downsample_levels + 1):
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [vs / 2**level for vs in voxel_sizes]},
                ],
            },
        )
    multiscales["datasets"] = datasets
    root.attrs["multiscales"] = multiscales

    return root


def setup_copy_to_zarr(
    arr: da.Array,
    group: zarr.Group,
    *,
    chunk_size: int = 64,
    compressor: Literal["default"] | Codec = "default",
):
    """
    Copy a 3D Dask array that is sliced (ie. has chunks of shape (nx, ny, 1))
    to a zarr array on disk that has isometric chunks (ie. shape (n, n, n)).

    Parameters
    ----------
    arr :
        Dask array to copy to zarr.
    group :
        zarr Group already set up for writing multiscale data with `create_group()`.
        The array is written to an array named "0" within the group.
    chunk_size :
        (isometric) chunk size to use for zarr chunking.
    compressor :
        Compressor to use to compress the zarr data.
    """
    assert arr.ndim == 3, "Input array is not 3-dimensional"
    assert arr.chunksize[2] == 1, "Input array is not chunked in slices"

    print("Setting up copy to zarr...")
    slice_size_bytes = arr.nbytes // arr.size * arr.chunksize[0] * arr.chunksize[1]
    slab_size_bytes = slice_size_bytes * chunk_size
    print(f"Each dask task will read ~{slab_size_bytes / 1e6:.02f} MB into memory")

    group["0"] = zarr.create(
        arr.shape, chunks=chunk_size, dtype=arr.dtype, compressor=compressor
    )

    @dask.delayed
    def copy_slab(slab: da.Array, zstart: int, zend: int) -> None:
        # Read in data
        data_in = slab.persist()
        # Write out data
        group["0"][:, :, zstart:zend] = data_in

    jobs = []
    nz = arr.shape[2]
    for z in range(0, nz, chunk_size):
        zmin = z
        zmax = min(z + chunk_size, nz)
        jobs.append(copy_slab(arr[:, :, zmin:zmax], zmin, zmax))

    return dask.delayed(jobs)
