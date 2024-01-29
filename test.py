"""Quick test script."""

from pathlib import Path

import dask.array as da
import glymur
import zarr

from stack_to_chunk.main import downsample_group

if __name__ == "__main__":
    arr_path = Path("/Users/dstansby/Dropbox (UCL)/A98/A98_zarr/A98.zarr")
    """group = create_group(
        Path("/Users/dstansby/Dropbox (UCL)/A98/A98_zarr/A98.zarr"),
        name="A98",
        spatial_unit="micrometer",
        voxel_sizes=[1, 1, 1],
        n_downsample_levels=0,
    )"""
    group = zarr.Group(arr_path)

    jp2_path = Path("/Users/dstansby/Dropbox (UCL)/A98/A98_jp2")
    jp2s = [glymur.Jp2k(p) for p in sorted(jp2_path.glob("*.jp2"))]
    stack = da.stack([da.from_array(jp2) for jp2 in jp2s], axis=-1)

    slice_size_bytes = (
        stack.nbytes // stack.size * stack.chunksize[0] * stack.chunksize[1]
    )

    downsample_group(group, level=1)
