"""Main functionality tests."""
from pathlib import Path

import dask.array as da
import numpy as np

from stack_to_chunk import copy_to_zarr, create_group


def test_workflow(tmp_path: Path) -> None:
    """Basic smoke test of the workflow as a user would use it."""
    group = create_group(
        tmp_path / "group.zarr",
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_sizes=(3, 4, 5),
        n_downsample_levels=2,
    )

    assert (tmp_path / "group.zarr").exists()

    chunk_size = 64
    shape = (583, 245, 156)
    arr = da.random.randint(low=0, high=2**16, dtype=np.uint16, size=shape)
    arr = arr.rechunk(chunks=(shape[0], shape[1], 1))
    copy_to_zarr(arr, group, n_processes=2, chunk_size=chunk_size, compressor="default")

    assert "0" in group
    assert group["0"].chunks == (chunk_size, chunk_size, chunk_size)
    assert group["0"].shape == shape
    assert group["0"].dtype == np.uint16
