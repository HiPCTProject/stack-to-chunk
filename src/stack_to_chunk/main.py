from pathlib import Path

import zarr

from stack_to_chunk.ome_ngff import SPATIAL_UNIT

"""
Strategy:

- Divide data up into slabs of size 128.
- For each slab read data into memory.
  This requires 128 * nx * ny * 2 bytes of memory per slab.
- Write out data to zarr array.
- Successively downsample and write out data.
- Parallelise the above across slabs.
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
    multiscales = root.create_group("multiscales")

    multiscales.attrs["version"] = "0.4"
    multiscales.attrs["name"] = name
    multiscales.attrs["axes"] = [
        {"name": "z", "type": "space", "unit": spatial_unit},
        {"name": "y", "type": "space", "unit": spatial_unit},
        {"name": "x", "type": "space", "unit": spatial_unit},
    ]
    multiscales.attrs["type"] = "linear"
    multiscales.attrs["metadata"] = {
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
    multiscales.attrs["datasets"] = datasets
