"""
Benchmark
"""

import pathlib
import shutil
import time

import dask_image.imread
import numpy as np
import tifffile

import stack_to_chunk

if __name__ == "__main__":
    plane = np.random.randint(low=0, high=2**16, size=(2000, 2000), dtype=np.uint16)
    image_dir = pathlib.Path(__file__).parent / "data"
    if not image_dir.exists():
        image_dir.mkdir()
        for i in range(64):
            tifffile.imwrite(image_dir / f"{str(i).zfill(3)}.tif", plane)

    images = dask_image.imread.imread(str(image_dir / "*.tif")).T
    print(f"Volume size: {images.nbytes / 1e6} MB")

    for n_processes in [1, 2, 3, 4]:
        shutil.rmtree(image_dir / "chunked.zarr")
        group = stack_to_chunk.MultiScaleGroup(
            image_dir / "chunked.zarr",
            name="my_zarr_group",
            spatial_unit="centimeter",
            voxel_size=(3, 4, 5),
        )
        t_start = time.time()
        group.add_full_res_data(
            images, chunk_size=32, compressor="default", n_processes=n_processes
        )
        t_end = time.time()
        print(f"{n_processes=}, t={t_end - t_start} seconds")
