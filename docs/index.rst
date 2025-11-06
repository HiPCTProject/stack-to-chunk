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

Installing
----------

``stack-to-chunk`` is designed to be used as a tool, which means its dependencies are pinned to exact versions.
For this reason, it's recommended to install ``stack-to-chunk`` in its own virtual environment.
This is easy to do with `virtual environments <https://docs.astral.sh/uv/pip/environments/>`_ using the `uv <https://docs.astral.sh/uv/>`_ Python package and project manager.

``stack-to-chunk`` can be installed using pip:

.. code:: bash

    pip install stack-to-chunk

Changelog
---------
See https://github.com/HiPCTProject/stack-to-chunk/releases for the list of tags and a changelog for each one.

Version 2
~~~~~~~~~

Version 2 of ``stack-to-chunk`` is a major breaking release to add support for OME-Zarr version 0.5 (and therefore Zarr version 3).
The following are major changes to the library:

- ``stack-to-chunk`` automatically adds sharding to the resulting data.
   See the guide page for more information on how this works.
- ``memory_per_process`` has been renamed ``memory_per_slab_process``, as it calculates the memory required to copy a single slab of data.
- A new function, ``memory_per_downsample_process`` has also been added to calculate the minimum memory required in a downsampling process.
