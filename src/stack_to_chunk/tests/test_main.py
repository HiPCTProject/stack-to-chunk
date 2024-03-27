"""Main functionality tests."""

from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from stack_to_chunk import MultiScaleGroup


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

    # Check that data is equal in dask array and zarr array
    np.testing.assert_equal(arr[:], zarr_arr[:])

    group.add_downsample_level(1)
    assert group.levels == [0, 1]
