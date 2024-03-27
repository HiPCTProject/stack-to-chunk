.. stack-to-chunk documentation master file, created by
   sphinx-quickstart on Wed Mar 27 18:55:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

stack-to-chunk
==============


Convert stacks of images to a chunked zarr dataset. Specifically this code is written to:

- Take stacks of 2D images (e.g., TIFF, JPEG files) that represent a 3D spatial volume as input.
- Convert them to an `OME Next Generation File Format (NGFF) <https://ngff.openmicroscopy.org/0.4/index.html>`_ zarr dataset.

Internals
---------

The code is designed based on the following assumptions:

1. Input data are stored in individual 2D slices. Reading part of a single slice requires reading the whole slice into memory, and this is an expensive operation.
2. Writing a single chunk of output data is an expensive operation.
3. Reading a single chunk of output data is a cheap operation.

If we have input slices of shape ``(nx, ny)``, and an output chunk shape of ``(nc, nc, nc)`` it makes sense to split the conversion into individual 'slabs' that have shape ``(nx, ny, nc)``. This means there is a one-to-one mapping from slices to slabs, and slabs to chunks, allowing each slab to be processed in parallel without interfering with the other slabs.

.. toctree::
   :maxdepth: 2

   auto_examples/tutorial.rst
   api
