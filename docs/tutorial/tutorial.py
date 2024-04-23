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
from loguru import logger

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
    tifffile.imwrite(slice_dir / f"{str(i).zfill(3)}.tif", data_2d)

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

group = stack_to_chunk.MultiScaleGroup(
    temp_dir_path / "chunked.zarr",
    name="my_zarr_group",
    spatial_unit="centimeter",
    voxel_size=(3, 4, 5),
)
print(group.levels)

# %%
# The first step in creating new data in the group is to make a copy of the data slices
# without any downsampling. We'll also enable logging here, so we can see that
# stack-to-chunk provides some useful progress messages:

logger.enable("stack_to_chunk")
logger.add(sys.stdout, level="INFO")

group.add_full_res_data(images, chunk_size=16, compressor="default", n_processes=1)
print(group.levels)

# %%
# The levels property shows that we have added a level. Each level is downsampled by a
# factor of ``2**level``, so level 0 is downsampled by a factor of 1, which is just
# a copy of the original data.

# %%
# Cleanup
# -------
#
# Finally we need to clean up the temporary directory we made earlier.
temp_dir.cleanup()
