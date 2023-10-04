from pathlib import Path

import dask.array as da

from stack_to_chunk.main import create_group, setup_copy_to_zarr

group = create_group(
    Path("arr.zarr"),
    name="arrayyyyyy",
    spatial_unit="micrometer",
    voxel_sizes=[1, 1, 1],
    n_downsample_levels=1,
)

nslice = 8
"""
jp2_path = Path("/Users/dstansby/Dropbox (UCL)/A98/A98_jp2")
jp2s = [glymur.Jp2k(p) for p in sorted(jp2_path.glob("*.jp2"))]
stack = da.stack([da.from_array(jp2) for jp2 in jp2s[:nslice]], axis=-1)
assert stack.chunksize[2] == 1
assert stack.chunksize[1] > 1
assert stack.chunksize[0] > 1
"""


stack = da.zeros(shape=(43, 23, 26), chunks=(43, 23, 1))  # type: ignore[attr-defined]
slice_size_bytes = stack.nbytes // stack.size * stack.chunksize[0] * stack.chunksize[1]
print(slice_size_bytes / 1e6 * 64)

delayed = setup_copy_to_zarr(stack, group=group, chunk_size=4)
delayed.visualize(filename="graph.svg", engine="graphviz")
