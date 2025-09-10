"""
Tutorial
========

This page steps through going from a set of 2D image files to a
3D chunked zarr dataset.
"""

import pathlib
import sys
import tempfile

import dask_image.imread
import matplotlib.pyplot as plt
import skimage.color
import skimage.data
import tifffile
import zarr
from loguru import logger
from pydantic_zarr.v3 import ArraySpec

import stack_to_chunk

# %%
# Generating sample data
# ----------------------
#
# We'll start by generating a set of sample data to downsample.
# To do this we'll just save 35 copies of a grayscale cat to a temporary directory.
data_2d = skimage.color.rgb2gray(skimage.data.cat())
temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = pathlib.Path(temp_dir.name)
slice_dir = temp_dir_path / "slices"
slice_dir.mkdir()

for i in range(35):
    tifffile.imwrite(slice_dir / f"{str(i).zfill(3)}.tif", data_2d.T)

plt.imshow(data_2d, cmap="gray")

# %%
# Setting up input
# ----------------
# stack-to-chunk takes a 3D dask array as input. dask provides an interface to lazily
# load each slice as and when it's needed. So although the dask array we create looks
# and behaves like an array, no data is actually read in from the TIFF files at this
# point.
#
# This also makes stack-to-chunk flexible - as long as you can put your 2D images into
# a 3D dask array, they can be used with stack-to-chunk.
#
# For this tutorial, ``dask_image`` provides a convenient way for us to read in all our
# TIFF files:

images = dask_image.imread.imread(str(slice_dir / "*.tif")).T
print(images)

# %%
#
# A few things to note here:
#   - We have a single 3D dask array, with the array axes being the x, y, z axes of the
#     image.
#   - The chunk size of the dask array is ``(nx, ny, 1)`` ie each individual slice
#     (corresponding) to each individual file on disk) is a chunk in the dask array.
#
# Running stack-to-chunk
# ----------------------
# The starting point for running ``stack-to-chunk`` is creating a `MultiScaleGroup`.
# This represents a local zarr group that will contain the output multi-scale dataset.
#
# Once we've created it, the ``levels`` property shows that no levels have been added
# to the group yet.
#
# We'll also enable logging here, so we can see that stack-to-chunk provides some
# useful progress messages:

logger.enable("stack_to_chunk")
logger.add(sys.stdout, level="INFO")

group = stack_to_chunk.MultiScaleGroup(
    temp_dir_path / "chunked.ome.zarr",
    name="my_zarr_group",
    spatial_unit="centimeter",
    voxel_size=(3, 4, 5),
    array_spec=ArraySpec.from_zarr(
        zarr.empty(images.shape, chunks=(16, 16, 16), dimension_names=("z", "y", "x"))
    ),
)
print(group.levels)

# %%
# The first step in creating new data in the group is to make a copy of the data slices
# without any downsampling. Before doing this lets do a quick check of how much memory
# each process will take up when we run stack-to-chunk:

bytes_per_process = stack_to_chunk.memory_per_process(images, chunk_size=16)
print(f"Each process will use {bytes_per_process / 1e6:.1f} MB")


# %%
# And finally, lets create our first data copy:

group.add_full_res_data(images, n_processes=1)

# %%
# The levels property can be inspected to show we've added the first level. Ekach level
# is downsampled by a factor of ``2**level``, so level 0 is downsampled by a factor of
# 1, which is just a copy of the original data (as expected).
print(group.levels)

# %%
# Now lets add some downsampling levels:
group.add_downsample_level(1, n_processes=1)
group.add_downsample_level(2, n_processes=1)
group.add_downsample_level(3, n_processes=1)
print(group.levels)

# %%
# The downsampled data can be accessed as `zarr.Array` objects by indexing
# ``group``. As an example, lets plot the third downsampled level:

plt.imshow(group[3][:, :, 0], cmap="gray")

# %%
# Cleanup
# -------
#
# Finally we need to clean up the temporary directory we made earlier.
temp_dir.cleanup()
