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
