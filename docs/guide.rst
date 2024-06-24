Guide
=====

Parallelisation strategy
------------------------

The code is designed based on the following assumptions:

1. Input data are stored in individual 2D slices. Reading part of a single slice requires reading the whole slice into memory, and this is an expensive operation.
2. Writing a single chunk of output data is an expensive operation.
3. Reading a single chunk of output data is a cheap operation.

If we have input slices of shape ``(nx, ny)``, and an output chunk shape of ``(nc, nc, nc)`` it makes sense to split the conversion into individual 'slabs' that have shape ``(nx, ny, nc)``.
This means there is a one-to-one mapping from slices to slabs, and slabs to chunks, allowing each slab to be processed in parallel without interfering with the other slabs.

Scheduling with dask
--------------------
``stack-to-chunk`` uses `dask` run the data copying and downsample tasks in parallel.
If you want to schedule the tasks yourself, specifying ``n_processes=None`` to `add_full_res_data` will return a `dask.delayed.Delayed` object instead of carrying out the tasks immediately.
This can then be used to schedule the tasks manually using `dask`.
One example where this is useuful is scheduling on a compute cluster using `dask-jobqueue`.
