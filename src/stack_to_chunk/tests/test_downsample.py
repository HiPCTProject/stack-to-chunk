"""Tests for the downsample.py module."""

import dask.array as da
import numpy as np
import pytest
from skimage.transform import resize

from stack_to_chunk.downsample import (
    _half_shape,
    _rechunk_to_even,
    _resize_block,
    downsample_by_two,
)


class TestDownsample:
    """Tests for the downsample.py module."""

    shape_3d = tuple[int, int, int]
    image_shape: shape_3d = (583, 245, 156)
    image_chunksize: shape_3d = (64, 64, 64)
    image_darray: da.Array = da.random.randint(
        low=0, high=2**16, dtype=np.uint16, size=image_shape
    )

    @pytest.mark.parametrize(  # type: ignore[misc]
        "chunksize",
        [
            (64, 64, 64),
            (64, 63, 63),
            (63, 63, 63),
        ],
    )
    def test_rechunk_to_even(self, chunksize: shape_3d) -> None:
        """Test rechunking to even chunk sizes."""
        chunked_array = self.image_darray.rechunk(chunksize)
        even_chunksize = _rechunk_to_even(chunked_array).chunksize
        assert even_chunksize == self.image_chunksize

    def test_half_shape(self) -> None:
        """Test calculating the half shape of an input shape."""
        assert _half_shape(self.image_shape) == (292, 123, 78)
        assert _half_shape(self.image_chunksize) == (32, 32, 32)

    def test_resize_block(self) -> None:
        """Test resizing a single block by a factor of 2 in each dimension."""
        block = self.image_darray[:64, :64, :64].compute()
        resized_block = _resize_block(block)
        assert resized_block.shape == (32, 32, 32)
        assert resized_block.dtype == block.dtype

    def test_downsample_by_two(self) -> None:
        """Test downsampling a chunked image by a factor of 2 in each dimension."""
        input_array = self.image_darray.rechunk(self.image_chunksize)
        downsampled = downsample_by_two(input_array)
        assert downsampled.chunksize == input_array.chunksize
        assert downsampled.dtype == input_array.dtype
        assert downsampled.ndim == input_array.ndim
        assert downsampled.shape == _half_shape(input_array.shape)

        # directly downsample image (without parallelization)
        directly_downsampled = resize(
            input_array,
            output_shape=_half_shape(input_array.shape),
            order=1,
            anti_aliasing=False,
        ).astype(input_array.dtype)

        np.testing.assert_equal(downsampled.compute(), directly_downsampled)
