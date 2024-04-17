"""Main functionality tests."""

import json
from pathlib import Path

import dask.array as da
import numcodecs
import numpy as np
import pytest
import zarr

from stack_to_chunk import MultiScaleGroup, memory_per_process, open_multiscale_group


@pytest.fixture()
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

    expected_multiscales_keys = [
        "name",
        "axes",
        "version",
        "datasets",
        "type",
        "metadata",
    ]
    multiscales = group._group.attrs["multiscales"]
    assert all(k in multiscales for k in expected_multiscales_keys)
    assert len(multiscales["datasets"]) == 0

    assert memory_per_process(arr, chunk_size=chunk_size) == 18282880
    group.add_full_res_data(
        arr,
        n_processes=2,
        chunk_size=chunk_size,
        compressor=compressor,
    )

    assert len(multiscales["datasets"]) == 1
    level_0 = multiscales["datasets"][0]
    assert all(k in level_0 for k in ["path", "coordinateTransformations"])

    assert group.levels == [0]
    zarr_arr = zarr.open(zarr_path / "0")
    assert zarr_arr.chunks == (chunk_size, chunk_size, chunk_size)
    assert zarr_arr.shape == arr.shape
    assert zarr_arr.dtype == np.uint16
    assert zarr_arr.compressor == compressor
    # Check that data is equal in dask array and zarr array
    np.testing.assert_equal(arr[:], zarr_arr[:])
    # Check metadata
    with (zarr_path / ".zattrs").open() as f:
        data = json.load(f)
    assert data == {
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
                            {"scale": [3, 4, 5], "type": "scale"}
                        ],
                        "path": "0",
                    }
                ],
                "metadata": {"description": "Downscaled using linear resampling"},
                "name": "my_zarr_group",
                "type": "linear",
                "version": "0.4",
            }
        ]
    }

    with (zarr_path / ".zgroup").open() as f:
        data = json.load(f)
    assert data == {"zarr_format": 2}

    # Check that trying to add data again fails
    with pytest.raises(
        RuntimeError, match="Full resolution data already added to this zarr group."
    ):
        group.add_full_res_data(
            arr,
            n_processes=2,
            chunk_size=64,
            compressor="default",
        )

    # Check that re-loading works
    del group
    group = open_multiscale_group(zarr_path)
    assert group.levels == [0]

    group.add_downsample_level(1)
    assert group.levels == [0, 1]

    with pytest.raises(RuntimeError, match="Level 1 already found in zarr group"):
        group.add_downsample_level(1)
    with pytest.raises(
        RuntimeError, match=r"Level below \(level=2\) not present in group."
    ):
        group.add_downsample_level(3)
    with pytest.raises(ValueError, match="level must be an integer >= 1"):
        group.add_downsample_level(0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level must be an integer >= 1"):
        group.add_downsample_level(-2)


def test_wrong_chunksize(tmp_path: Path, arr: da.Array) -> None:
    zarr_path = tmp_path / "group.ome.zarr"
    group = MultiScaleGroup(
        tmp_path / zarr_path,
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )

    with pytest.raises(
        ValueError,
        match="Input array is must have a chunk size of 1 in the third dimension.",
    ):
        group.add_full_res_data(
            arr.rechunk(chunks=(arr.shape[0], arr.shape[1], 2)),
            n_processes=2,
            chunk_size=64,
            compressor="default",
        )
