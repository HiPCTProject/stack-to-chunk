"""Main functionality tests."""
from pathlib import Path

from stack_to_chunk import create_group


def test_workflow(tmp_path: Path) -> None:
    """Basic smoke test of the workflow as a user would use it."""
    create_group(
        tmp_path / "group.zarr",
        name="my_zarr_group",
        spatial_unit="centimeter",
        voxel_sizes=(3, 4, 5),
        n_downsample_levels=2,
    )

    assert (tmp_path / "group.zarr").exists()
