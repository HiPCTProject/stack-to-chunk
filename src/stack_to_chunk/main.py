"""
Main code for converting stacks to chunks.
"""

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from dask.array.core import Array
from loguru import logger
from numcodecs import blosc
from numcodecs.abc import Codec

from stack_to_chunk._array_helpers import _copy_slab
from stack_to_chunk.ome_ngff import SPATIAL_UNIT


def memory_per_process(input_data: Array, *, chunk_size: int) -> int:
    """
    The amount of memory each stack-to-chunk process will use (in bytes).

    This is a lower bound on memory use, equal to the size of a slab of data with size
    (nx, ny, chunk_size), where (nx, ny) is the shape of a single input
    slice and chunk_size is the chunk size of the output zarr store.
    """
    itemsize = np.dtype(input_data.dtype).itemsize
    return int(input_data.shape[0] * input_data.shape[1] * itemsize * chunk_size)


class MultiScaleGroup:
    """
    A class for creating and interacting with a OME-zarr multi-scale group.

    Parameters
    ----------
    path :
        Path to zarr group on disk.
    name :
        Name to save to zarr group.
    voxel_size :
        Size of a single voxel, in units of spatial_units.
    spatial_units :
        Units of the voxel size.

    """

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        voxel_size: tuple[float, float, float],
        spatial_unit: SPATIAL_UNIT,
    ) -> None:
        if path.exists():
            msg = f"{path} already exists"
            raise FileExistsError(msg)
        self._path = path
        self._name = name
        self._spatial_unit = spatial_unit
        self._voxel_size = voxel_size

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

        Level 0 corresponds to full resolution data, and level ``i`` to
        data downsampled by a factor of ``2**i``.
        """
        return [int(k) for k in self._group]

    def add_full_res_data(
        self,
        data: Array,
        *,
        chunk_size: int,
        compressor: Literal["default"] | Codec,
        n_processes: int,
    ) -> None:
        """
        Add the 'original' full resolution data to this group.

        Parameters
        ----------
        data :
            Input data. Must be 3D, and have a chunksize of ``(nx, ny, 1)``, where
            ``(nx, ny)`` is the shape of the input 2D slices.

        chunk_size :
            Size of chunks in output zarr dataset.
        compressor :
            Compressor to use when writing data to zarr dataset.
        n_processes :
            Number of parallel processes to use to read/write data.

        Raises
        ------
        RuntimeError :
            If full resolution data have already been added.

        """
        if "0" in self._group:
            msg = "Full resolution data already added to this zarr group."
            raise RuntimeError(msg)

        assert data.ndim == 3, "Input array is not 3-dimensional"
        if data.chunksize[2] != 1:
            msg = (
                f"Input array is must have a chunk size of 1 in the third dimension. "
                f"Got chunks: {data.chunksize}"
            )
            raise ValueError(msg)

        logger.info("Setting up copy to zarr...")
        slice_size_bytes = (
            data.nbytes // data.size * data.chunksize[0] * data.chunksize[1]
        )
        slab_size_bytes = slice_size_bytes * chunk_size
        logger.info(
            f"Each process will read ~{slab_size_bytes / 1e6:.02f} MB into memory"
        )

        self._group.create_dataset(
            name="0",
            shape=data.shape,
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype=data.dtype,
            compressor=compressor,
        )

        nz = data.shape[2]
        slab_idxs: list[tuple[int, int]] = [
            (z, min(z + chunk_size, nz)) for z in range(0, nz, chunk_size)
        ]
        all_args = [
            (self._group["0"], data[:, :, zmin:zmax], zmin, zmax)
            for (zmin, zmax) in slab_idxs
        ]

        logger.info("Starting full resolution copy to zarr...")
        blosc_use_threads = blosc.use_threads
        blosc.use_threads = 0

        # Use try/finally pattern to allow code coverage to be collected
        p = Pool(n_processes)
        try:
            p.starmap(_copy_slab, all_args)
        finally:
            p.close()
            p.join()

        blosc.use_threads = blosc_use_threads
        self._add_level_metadata(0)
        logger.info("Finished full resolution copy to zarr.")

    def add_downsample_level(self, level: int) -> None:
        """
        Add a level of downsampling.

        Parameters
        ----------
        level :
            Level of downsampling. Level ``i`` corresponds to a downsampling factor
            of ``2**i``.

        Notes
        -----
        To add level ``i`` to the zarr group, level ``i - 1`` must first have been
        added.

        """
        if not (level >= 1 and int(level) == level):
            msg = "level must be an integer >= 1"
            raise ValueError(msg)

        level_str = str(int(level))
        if level_str in self._group:
            msg = f"Level {level_str} already found in zarr group"
            raise RuntimeError(msg)

        if (level_minus_one := str(int(level) - 1)) not in self._group:
            msg = f"Level below (level={level_minus_one}) not present in group."
            raise RuntimeError(
                msg,
            )

        source_data = self._group[level_minus_one]
        new_shape = np.ceil(np.array(source_data.shape) / 2)

        self._group[level_str] = zarr.create(
            new_shape,
            chunks=source_data.chunks,
            dtype=source_data.dtype,
            compressor=source_data.compressor,
        )

    def _add_level_metadata(self, level: int = 0) -> None:
        """
        Add the required multiscale metadata for the corresponding level.

        Parameters
        ----------
        level :
            Level of downsampling. Level 0 corresponds to full resolution data.

        """
        # we assume that the scale factor is always 2 in each dimension
        scale_factors = [float(s * 2**level) for s in self._voxel_size]
        new_dataset = {
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": scale_factors,
                }
            ],
        }

        multiscales = self._group.attrs["multiscales"]
        existing_dataset_paths = [d["path"] for d in multiscales["datasets"]]
        if new_dataset["path"] not in existing_dataset_paths:
            multiscales["datasets"].append(new_dataset)
        self._group.attrs["multiscales"] = multiscales
