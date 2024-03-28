"""
Tutorial
========

This page steps through going from a set of 2D image files to a
3D chunked zarr dataset.
"""

import pathlib
import tempfile

import dask_image.imread
import matplotlib.pyplot as plt
import skimage.color
import skimage.data
import tifffile

# %%
# Generating sample data
# ----------------------
#
# We'll start by generating a set of sample data to downsample.
# To do this we'll just save 35 copies of a grayscale cat to a temporary directory.
data_2d = skimage.color.rgb2gray(skimage.data.cat())
temp_dir = tempfile.TemporaryDirectory()
slice_dir = pathlib.Path(temp_dir.name) / "slices"
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

images = dask_image.imread.imread(str(slice_dir / "*.tif"))
print(images)

# %%
#
# A few things to note here:
#   - We have a single 3D dask array, with the array axes being the z, y, x axes of the
#     image.
#   - The chunk size of the dask array is ``(1, ny, nx)`` ie each individual slice
#     (corresponding) to each individual file on disk) is a chunk in the dask array.
#
# Running stack-to-chunk
# ----------------------

# %%
# Cleanup
# -------
#
# Finally we need to clean up the temporary directory we made earlier.
temp_dir.cleanup()
