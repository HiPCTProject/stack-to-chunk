"""Main functionality tests."""

import json
import re
from pathlib import Path
from typing import Any

import dask.array as da
import numcodecs
import numpy as np
import pytest
import zarr

from stack_to_chunk import MultiScaleGroup, memory_per_process, open_multiscale_group


def check_zattrs(zarr_path: Path, expected: dict[str, Any]) -> None:
    with (zarr_path / ".zattrs").open() as f:
        data = json.load(f)
    assert data == expected


def check_full_res_copy(zarr_path: Path, group: zarr.Group, arr: da.Array) -> None:
    """
    Check various array properties after a full resolution copy.
    """
    assert group.levels == [0]
    zarr_arr = zarr.open(zarr_path / "0")
    # Check that data is equal in dask array and zarr array
    np.testing.assert_equal(arr[:], zarr_arr[:])
    # Check metadata
    check_zattrs(
        zarr_path,
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space", "unit": "centimeter"},
                        {"name": "y", "type": "space", "unit": "centimeter"},
                        {"name": "z", "type": "space", "unit": "centimeter"},
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {"translation": [0.5, 0.5, 0.5], "type": "translation"},
                                {"scale": [3.0, 4.0, 5.0], "type": "scale"},
                            ],
                            "path": "0",
                        }
                    ],
                    "metadata": {
                        "description": "Downscaled using local mean in 2x2x2 blocks.",
                        "kwargs": {"block_size": 2, "func": "np.mean"},
                        "method": "skimage.measure.block_reduce",
                        "version": "0.24.0",
                    },
                    "name": "my_zarr_group",
                    "type": "local mean",
                    "version": "0.4",
                }
            ]
        },
    )

    with (zarr_path / ".zgroup").open() as f:
        data = json.load(f)
    assert data == {"zarr_format": 2}


@pytest.fixture
def arr() -> da.Array:
    shape = (583, 245, 156)
    arr = da.random.randint(low=0, high=2**16, dtype=np.uint16, size=shape)
    return arr.rechunk(chunks=(shape[0], shape[1], 1))


def test_workflow(tmp_path: Path, arr: da.Array) -> None:
    """Basic smoke test of the workflow as a user would use it."""
    zarr_path = tmp_path / "group.ome.zarr"
    group = MultiScaleGroup(
        tmp_path / zarr_path,
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )

    assert zarr_path.exists()
    assert group.levels == []

    compressor = numcodecs.blosc.Blosc(cname="zstd", clevel=2, shuffle=2)
    chunk_size = 64

    check_zattrs(
        zarr_path,
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space", "unit": "centimeter"},
                        {"name": "y", "type": "space", "unit": "centimeter"},
                        {"name": "z", "type": "space", "unit": "centimeter"},
                    ],
                    "datasets": [],
                    "metadata": {
                        "description": "Downscaled using local mean in 2x2x2 blocks.",
                        "kwargs": {"block_size": 2, "func": "np.mean"},
                        "method": "skimage.measure.block_reduce",
                        "version": "0.24.0",
                    },
                    "name": "my_zarr_group",
                    "type": "local mean",
                    "version": "0.4",
                }
            ]
        },
    )

    assert memory_per_process(arr, chunk_size=chunk_size) == 18282880
    group.create_initial_dataset(
        data=arr,
        chunk_size=chunk_size,
        compressor=compressor,
    )
    group.add_full_res_data(
        arr,
        n_processes=1,
    )

    zarr_arr = zarr.open(zarr_path / "0")
    assert zarr_arr.chunks == (chunk_size, chunk_size, chunk_size)
    assert zarr_arr.shape == arr.shape
    assert zarr_arr.dtype == np.uint16
    assert zarr_arr.compressor == compressor
    check_full_res_copy(zarr_path, group, arr)

    # Check that re-loading works
    del group
    group = open_multiscale_group(zarr_path)
    assert group.levels == [0]

    group.add_downsample_level(1, n_processes=2)
    assert group.levels == [0, 1]

    check_zattrs(
        zarr_path,
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space", "unit": "centimeter"},
                        {"name": "y", "type": "space", "unit": "centimeter"},
                        {"name": "z", "type": "space", "unit": "centimeter"},
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {"translation": [0.5, 0.5, 0.5], "type": "translation"},
                                {"scale": [3.0, 4.0, 5.0], "type": "scale"},
                            ],
                            "path": "0",
                        },
                        {
                            "coordinateTransformations": [
                                {"translation": [0.5, 0.5, 0.5], "type": "translation"},
                                {"scale": [6.0, 8.0, 10.0], "type": "scale"},
                            ],
                            "path": "1",
                        },
                    ],
                    "metadata": {
                        "description": "Downscaled using local mean in 2x2x2 blocks.",
                        "kwargs": {"block_size": 2, "func": "np.mean"},
                        "method": "skimage.measure.block_reduce",
                        "version": "0.24.0",
                    },
                    "name": "my_zarr_group",
                    "type": "local mean",
                    "version": "0.4",
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="Level 1 already found in zarr group"):
        group.add_downsample_level(1, n_processes=2)
    with pytest.raises(
        RuntimeError, match=r"Level below \(level=2\) not present in group."
    ):
        group.add_downsample_level(3, n_processes=2)
    with pytest.raises(ValueError, match="level must be an integer >= 1"):
        group.add_downsample_level(0.1, n_processes=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level must be an integer >= 1"):
        group.add_downsample_level(-2, n_processes=2)


def test_parallel_copy(tmp_path: Path, arr: da.Array) -> None:
    """
    Test running several slab copies one after another.

    Simulates what happens on a compute cluster.
    """
    zarr_path = tmp_path / "group.ome.zarr"
    group = MultiScaleGroup(
        tmp_path / zarr_path,
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )
    compressor = numcodecs.blosc.Blosc(cname="zstd", clevel=2, shuffle=2)
    chunk_size = 64
    group.create_initial_dataset(
        data=arr,
        chunk_size=chunk_size,
        compressor=compressor,
    )
    # Add first slab
    group.add_full_res_data(
        arr[:, :, :64],
        n_processes=1,
        start_z_idx=0,
    )
    # Add second slab
    group.add_full_res_data(
        arr[:, :, 64:],
        n_processes=1,
        start_z_idx=64,
    )
    with pytest.raises(
        ValueError,
        match=re.escape("start_z_idx (2) is not a multiple of chunk_size (64)"),
    ):
        group.add_full_res_data(
            arr[:64],
            n_processes=1,
            start_z_idx=2,
        )

    check_full_res_copy(zarr_path, group, arr)


def test_wrong_chunksize(tmp_path: Path, arr: da.Array) -> None:
    zarr_path = tmp_path / "group.ome.zarr"
    group = MultiScaleGroup(
        tmp_path / zarr_path,
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )
    group.create_initial_dataset(
        data=arr,
        chunk_size=64,
        compressor="default",
    )

    with pytest.raises(
        ValueError,
        match="Input array is must have a chunk size of 1 in the third dimension.",
    ):
        group.add_full_res_data(
            arr.rechunk(chunks=(arr.shape[0], arr.shape[1], 2)),
            n_processes=1,
        )


def test_known_data(tmp_path: Path) -> None:
    # Test downsampling on some simple data that gives an exact known result
    arr = da.from_array(np.arange(8).reshape((2, 2, 2)))
    arr = arr.rechunk(chunks=(2, 2, 1))

    group = MultiScaleGroup(
        tmp_path / "group.ome.zarr",
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )
    group.create_initial_dataset(
        data=arr,
        chunk_size=1,
        compressor="default",
    )
    group.add_full_res_data(arr, n_processes=1)
    group.add_downsample_level(1, n_processes=1)
    arr_downsammpled = group[1]
    np.testing.assert_equal(arr_downsammpled[:], [[[3]]])


def test_padding(tmp_path: Path) -> None:
    # Test data that doesn't fit exactly into (2, 2, 2) shaped chunks
    arr_npy = np.arange(8).reshape((2, 2, 2))
    arr_npy = np.concatenate([arr_npy, [[[10, 10], [12, 16]]]], axis=0)
    arr = da.from_array(arr_npy)
    arr = arr.rechunk(chunks=(2, 2, 1))

    group = MultiScaleGroup(
        tmp_path / "group.ome.zarr",
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )
    group.create_initial_dataset(
        data=arr,
        chunk_size=1,
        compressor="default",
    )
    group.add_full_res_data(arr, n_processes=1)
    group.add_downsample_level(1, n_processes=1)
    arr_downsammpled = group[1]
    np.testing.assert_equal(arr_downsammpled[:], [[[3]], [[12]]])
