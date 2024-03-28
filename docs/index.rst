.. stack-to-chunk documentation master file, created by
   sphinx-quickstart on Wed Mar 27 18:55:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

stack-to-chunk
==============


Convert stacks of images to a chunked zarr dataset. Specifically this code is written to:

- Take stacks of 2D images (e.g., TIFF, JPEG files) that represent a 3D spatial volume as input.
- Convert them to an `OME Next Generation File Format (NGFF) <https://ngff.openmicroscopy.org/0.4/index.html>`_ zarr dataset.

.. toctree::
   :maxdepth: 2

   auto_examples/tutorial.rst
   guide
   api
