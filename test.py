from pathlib import Path

from stack_to_chunk.main import create_group

create_group(
    Path("arr.zarr"),
    name="arrayyyyyy",
    spatial_unit="micrometer",
    voxel_sizes=[1, 1, 1],
    n_downsample_levels=1,
)
