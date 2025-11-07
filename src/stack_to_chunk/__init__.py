"""Convert stacks of images to chunked datasets."""

__all__ = [
    "SPATIAL_UNIT",
    "MultiScaleGroup",
    "__version__",
    "memory_per_downsample_process",
    "memory_per_slab_process",
    "open_multiscale_group",
]

from loguru import logger

from ._version import __version__
from .main import (
    MultiScaleGroup,
    memory_per_downsample_process,
    memory_per_slab_process,
    open_multiscale_group,
)
from .ome_ngff import SPATIAL_UNIT

logger.disable("stack_to_chunk")
