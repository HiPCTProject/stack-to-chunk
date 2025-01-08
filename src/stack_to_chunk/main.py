"""
Main code for converting stacks to chunks.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from dask.array.core import Array
from joblib import Parallel
from loguru import logger
from numcodecs import blosc
from numcodecs.abc import Codec

from stack_to_chunk._array_helpers import _copy_slab, _downsample_block
from stack_to_chunk.ome_ngff import SPATIAL_UNIT, DatasetDict


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
        self._path = path
        self._name = name
        self._spatial_unit = spatial_unit
        self._voxel_size = voxel_size

        if isinstance(path, Path) and not path.exists():
            self._create_zarr_group()

        self._group = zarr.open_group(store=self._path, mode="r+")

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
            {"name": "x", "type": "space", "unit": self._spatial_unit},
            {"name": "y", "type": "space", "unit": self._spatial_unit},
            {"name": "z", "type": "space", "unit": self._spatial_unit},
        ]
        multiscales["type"] = "local mean"
        multiscales["metadata"] = {
            "description": "Downscaled using local mean in 2x2x2 blocks.",
            "method": "skimage.measure.block_reduce",
            "version": "0.24.0",
            "kwargs": {"block_size": 2, "func": "np.mean"},
        }

        multiscales["datasets"] = []
        self._group.attrs["multiscales"] = [multiscales]

    @property
    def levels(self) -> list[int]:
        """
        List of downsample levels currently stored.

        Level 0 corresponds to full resolution data, and level ``i`` to
        data downsampled by a factor of ``2**i``.
        """
        return [int(k) for k in self._group]

    def __getitem__(self, level: int) -> zarr.Array:
        """
        Get zarr Array for a  given level.
        """
        if level not in self.levels:
            msg = f"Given level {level} not in added levels {self.levels}"
            raise ValueError(msg)

        return self._group[str(level)]

    def create_initial_dataset(
        self, data: Array, *, chunk_size: int, compressor: Literal["default"] | Codec
    ) -> None:
        """
        Set up the inital full-resolution dataset.

        Parameters
        ----------
        data :
            Full input data. Must be 3D, and have a chunksize of ``(nx, ny, 1)``, where
            ``(nx, ny)`` is the shape of the input 2D slices.
        chunk_size :
            Size of chunks in output zarr dataset.
        compressor :
            Compressor to use when writing data to zarr dataset.

        Raises
        ------
        RuntimeError :
            If full resolution dataset has already been created.

        """
        if "0" in self._group:
            msg = "Full resolution dataset already set up."
            raise RuntimeError(msg)
        self._group.create_dataset(
            name="0",
            shape=data.shape,
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype=data.dtype,
            compressor=compressor,
            dimension_separator="/",
        )
        self._add_level_metadata(0)

    def add_full_res_data(
        self,
        data: Array,
        *,
        n_processes: int,
        start_z_idx: int = 0,
    ) -> None:
        """
        Add the 'original' full resolution data to this group.

        Parameters
        ----------
        data :
            Input data. Must be 3D, and have a chunksize of ``(nx, ny, 1)``, where
            ``(nx, ny)`` is the shape of the input 2D slices.
        n_processes :
            Number of parallel processes to use to read/write data.
        start_z_idx :
            z-index at which this stack of input data starts. Can be useful to write
            multiple slabs in parallel using a compute cluster where the job wants
            to be split into many small individual Python processes.

        Notes
        -----
        Make sure create_initial_dataset has been run first to set up the
        zarr dataset.

        """
        if "0" not in self._group:
            msg = (
                "Full resolution dataset not present. "
                "Run `create_initial_datset()` first."
            )
            raise RuntimeError(msg)

        chunk_size: int = self._group["0"].chunks[0]

        assert data.ndim == 3, "Input array is not 3-dimensional"
        if start_z_idx % chunk_size != 0:
            msg = (
                f"start_z_idx ({start_z_idx}) is not a multiple "
                f"of chunk_size ({chunk_size})"
            )
            raise ValueError(msg)

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

        nz = data.shape[2]
        slab_idxs: list[tuple[int, int]] = [
            (z, min(z + chunk_size, nz)) for z in range(0, nz, chunk_size)
        ]
        all_args = [
            (
                self._group["0"],
                data[:, :, zmin:zmax],
                zmin + start_z_idx,
                zmax + start_z_idx,
            )
            for (zmin, zmax) in slab_idxs
        ]

        logger.info("Starting full resolution copy to zarr...")
        blosc_use_threads = blosc.use_threads
        blosc.use_threads = 0

        jobs = [_copy_slab(*args) for args in all_args]
        Parallel(n_jobs=n_processes)(jobs)

        blosc.use_threads = blosc_use_threads
        logger.info("Finished full resolution copy to zarr.")

    def add_downsample_level(self, level: int, *, n_processes: int) -> None:
        """
        Add a level of downsampling.

        Parameters
        ----------
        level :
            Level of downsampling. Level ``i`` corresponds to a downsampling factor
            of ``2**i``.
        n_processes :
            Number of parallel processes to use to read/write data. See the
            joblib.Parallel documentation for more info of allowed values.
            In particluar, you can set ``n_processes=-1`` to get joblib to use
            all available CPUs.

        Notes
        -----
        To add level ``i`` to the zarr group, level ``i - 1`` must first have been
        added.

        """
        logger.info(f"Downsampling to level {level} with {n_processes=}")
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

        source_arr = self._group[level_minus_one]
        new_shape = np.ceil(np.array(source_arr.shape) / 2)
        chunk_size = source_arr.chunks[0]

        sink_arr = self._group.create_dataset(
            name=level_str,
            shape=new_shape,
            chunks=source_arr.chunks,
            dtype=source_arr.dtype,
            compressor=source_arr.compressor,
            dimension_separator="/",
        )

        block_indices = [
            (x, y, z)
            for x in range(0, source_arr.shape[0], chunk_size * 2)
            for y in range(0, source_arr.shape[1], chunk_size * 2)
            for z in range(0, source_arr.shape[2], chunk_size * 2)
        ]

        all_args = [(source_arr, sink_arr, idxs) for idxs in block_indices]

        logger.info(f"Starting downsampling from level {level_minus_one} > {level}...")
        blosc_use_threads = blosc.use_threads
        blosc.use_threads = 0

        jobs = [_downsample_block(*args) for args in all_args]
        logger.info(f"Launching {len(jobs)} jobs")
        Parallel(n_jobs=n_processes, verbose=10)(jobs)

        self._add_level_metadata(level)
        blosc.use_threads = blosc_use_threads
        logger.info(f"Finished downsampling from level {level_minus_one} > {level}")

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
                },
                {
                    "type": "translation",
                    "translation": (np.array(scale_factors) * 0.5).tolist(),
                },
            ],
        }

        multiscales = self._group.attrs["multiscales"][0]
        existing_dataset_paths = [d["path"] for d in multiscales["datasets"]]
        if new_dataset["path"] in existing_dataset_paths:
            return

        multiscales["datasets"].append(new_dataset)

        def get_level(dataset_meta: DatasetDict) -> int:
            return int(dataset_meta["path"])

        multiscales["datasets"] = sorted(multiscales["datasets"], key=get_level)
        self._group.attrs["multiscales"] = [multiscales]


def open_multiscale_group(path: Path) -> MultiScaleGroup:
    """
    Open a previously created multiscale zarr group.

    Parameters
    ----------
    path :
        Path to existing group.

    """
    group = zarr.open_group(store=path, mode="r")
    attrs = group.attrs.asdict()
    multiscales = attrs["multiscales"][0]
    name = multiscales["name"]
    transforms = multiscales["datasets"][0]["coordinateTransformations"]

    if transforms[1]["type"] == "scale":
        logger.warning(
            "Dataset is invalid, because the scale transform is not first. "
            "Will attempt to fix metadata...",
            stacklevel=1,
        )
        group = zarr.open_group(store=path, mode="a")
        _fix_transform_order(group)
        return open_multiscale_group(path)

    voxel_size = transforms[0]["scale"]
    spatial_unit = multiscales["axes"][0]["unit"]

    return MultiScaleGroup(
        path, name=name, voxel_size=voxel_size, spatial_unit=spatial_unit
    )


def _fix_transform_order(group: zarr.Group) -> None:
    attrs = group.attrs.asdict()
    multiscales = deepcopy(attrs["multiscales"])
    for i in range(len(multiscales[0]["datasets"])):
        transforms = multiscales[0]["datasets"][i]["coordinateTransformations"]
        if not (
            len(transforms) == 2
            and transforms[0]["type"] == "translation"
            and transforms[1]["type"] == "scale"
        ):
            logger.error(
                "Did not find a translation and scale transform (in that order)"
            )

        # Flip order
        translation = transforms[0]["translation"]
        scale = transforms[1]["scale"]
        translation = (np.array(translation) * np.array(scale)).tolist()
        multiscales[0]["datasets"][i]["coordinateTransformations"] = [
            {"type": "scale", "scale": scale},
            {"type": "translation", "translation": translation},
        ]

    group.attrs.put({"multiscales": multiscales})
