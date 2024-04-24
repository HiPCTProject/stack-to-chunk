"""Main functionality tests."""

from pathlib import Path

import dask.array as da
import numcodecs
import numpy as np
import pytest
import zarr

from stack_to_chunk import MultiScaleGroup, memory_per_process


def test_workflow(tmp_path: Path) -> None:
    """Basic smoke test of the workflow as a user would use it."""
    group = MultiScaleGroup(
        tmp_path / "group.zarr",
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_size=(3, 4, 5),
    )

    assert (tmp_path / "group.zarr").exists()
    assert group.levels == []

    chunk_size = 64
    shape = (583, 245, 156)
    arr = da.random.randint(low=0, high=2**16, dtype=np.uint16, size=shape)
    arr = arr.rechunk(chunks=(shape[0], shape[1], 1))

    compressor = numcodecs.blosc.Blosc(cname="zstd", clevel=2, shuffle=2)

    with pytest.raises(
        ValueError,
        match="Input array is must have a chunk size of 1 in the third dimension.",
    ):
        group.add_full_res_data(
            arr.rechunk(chunks=(shape[0], shape[1], 2)),
            n_processes=2,
            chunk_size=chunk_size,
            compressor=compressor,
        )

    assert memory_per_process(arr, chunk_size=chunk_size) == 18282880
    group.add_full_res_data(
        arr,
        n_processes=2,
        chunk_size=chunk_size,
        compressor=compressor,
    )
    with pytest.raises(
        RuntimeError, match="Full resolution data already added to this zarr group."
    ):
        group.add_full_res_data(
            arr,
            n_processes=2,
            chunk_size=chunk_size,
            compressor="default",
        )

    assert group.levels == [0]
    zarr_arr = zarr.open(tmp_path / "group.zarr" / "0")
    assert zarr_arr.chunks == (chunk_size, chunk_size, chunk_size)
    assert zarr_arr.shape == shape
    assert zarr_arr.dtype == np.uint16
    assert zarr_arr.compressor == compressor

    # Check that data is equal in dask array and zarr array
    np.testing.assert_equal(arr[:], zarr_arr[:])

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


def test_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "group.zarr"
    path.mkdir()
    with pytest.raises(FileExistsError, match=r".group\.zarr already exists"):
        MultiScaleGroup(
            tmp_path / "group.zarr",
            name="my_zarr_group",
            spatial_unit="centimeter",
            voxel_size=(3, 4, 5),
        )
