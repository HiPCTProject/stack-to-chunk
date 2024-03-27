"""
Main code for converting stacks to chunks.

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

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from numcodecs import blosc
from numcodecs.abc import Codec

from stack_to_chunk._array_helpers import _copy_slab
from stack_to_chunk.ome_ngff import SPATIAL_UNIT

__all__ = ["MultiScaleGroup"]


class MultiScaleGroup:
    """A class for creating and interacting with a OME-zarr multi-scale group."""

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        spatial_unit: SPATIAL_UNIT,
        voxel_sizes: tuple[float, float, float],
    ):
        if path.exists():
            msg = f"{path} already exists"
            raise FileExistsError(msg)
        self._path = path
        self._name = name
        self._spatial_unit = spatial_unit
        self._voxel_sizes = voxel_sizes

        self._create_zarr_group()

    def _create_zarr_group(self) -> None:
        """
        Create the zarr group.

        Saves a reference to the group on the ._group attribute.
        """
        self._group = zarr.open_group(store=self._path, mode="w")
        multiscales: dict[str, Any] = {}
        multiscales["version"] = "0.4"
        multiscales["name"] = self._name
        multiscales["axes"] = [
            {"name": "z", "type": "space", "unit": self._spatial_unit},
            {"name": "y", "type": "space", "unit": self._spatial_unit},
            {"name": "x", "type": "space", "unit": self._spatial_unit},
        ]
        multiscales["type"] = "linear"
        multiscales["metadata"] = {
            "description": "Downscaled using linear resampling",
        }

        multiscales["datasets"] = []
        self._group.attrs["multiscales"] = multiscales

    @property
    def levels(self) -> list[int]:
        """
        List of downsample levels currently stored.

        Level 0 corresponds to full resolution data, and level `i` to
        data downsampled by a factor of `2**i`.
        """
        return [int(k) for k in self._group]

    def add_full_res_data(
        self,
        data: da.Array,
        *,
        chunk_size: int,
        compressor: Literal["default"] | Codec,
        n_processes: int,
    ) -> None:
        """
        Add the 'original' full resolution data to this group.

        Raises
        ------
        RuntimeError :
            If full resolution data have already been added.

        """
        if "0" in self._group:
            msg = "Full resolution data already added to this zarr group."
            raise RuntimeError(msg)

        assert data.ndim == 3, "Input array is not 3-dimensional"
        assert data.chunksize[2] == 1, "Input array is not chunked in slices"

        print("Setting up copy to zarr...")
        slice_size_bytes = (
            data.nbytes // data.size * data.chunksize[0] * data.chunksize[1]
        )
        slab_size_bytes = slice_size_bytes * chunk_size
        print(f"Each dask task will read ~{slab_size_bytes / 1e6:.02f} MB into memory")

        self._group["0"] = zarr.create(
            data.shape,
            chunks=chunk_size,
            dtype=data.dtype,
            compressor=compressor,
        )

        nz = data.shape[2]
        slab_idxs: list[tuple[int, int]] = [
            (z, min(z + chunk_size, nz)) for z in range(0, nz, chunk_size)
        ]
        args = [
            (self._group["0"], data[:, :, zmin:zmax], zmin, zmax)
            for (zmin, zmax) in slab_idxs
        ]

        print("Starting full resolution copy to zarr...")
        blosc.use_threads = False
        with Pool(n_processes) as p:
            p.starmap(_copy_slab, args)

        print("Finished full resolution copy to zarr.")

    def add_downsample_level(self, level: int) -> None:
        """
        Add a level of downsampling.

        Level `i` corresponds to a downsampling factor of `2**i`.
        """
        if not level >= 1 and int(level) == level:
            msg = "level must be an integer >= 1"
            raise ValueError(msg)

        level_str = str(int(level))
        if level_str in self._group:
            msg = f"Level {level_str} already found in zarr group"
            raise RuntimeError(msg)

        if (level_minus_one := str(int(level) - 1)) not in self._group:
            raise RuntimeError(
                f"Level below (level={level_minus_one}) not present in group.",
            )

        source_data = self._group[level_minus_one]
        new_shape = np.array(source_data.shape) // 2

        self._group[level_str] = zarr.create(
            new_shape,
            chunks=source_data.chunks,
            dtype=source_data.dtype,
            compressor=source_data.compressor,
        )
