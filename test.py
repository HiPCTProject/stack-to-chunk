"""Quick test script."""
from pathlib import Path

import dask.array as da
import glymur

from stack_to_chunk.main import copy_to_zarr, create_group

if __name__ == "__main__":
    group = create_group(
        Path("arr.zarr"),
        name="arrayyyyyy",
        spatial_unit="micrometer",
        voxel_sizes=[1, 1, 1],
        n_downsample_levels=1,
    )

    nslice = 256
    jp2_path = Path("/Users/dstansby/Dropbox (UCL)/A98/A98_jp2")
    jp2s = [glymur.Jp2k(p) for p in sorted(jp2_path.glob("*.jp2"))]
    stack = da.stack([da.from_array(jp2) for jp2 in jp2s], axis=-1)  # type: ignore[attr-defined]

    slice_size_bytes = (
        stack.nbytes // stack.size * stack.chunksize[0] * stack.chunksize[1]
    )

    copy_to_zarr(stack, group=group, chunk_size=64, compressor="default", n_processes=1)
