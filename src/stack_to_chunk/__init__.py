"""Convert stacks of images to chunked datasets."""

__all__ = [
    "MultiScaleGroup",
    "memory_per_process",
    "open_multiscale_group",
    "SPATIAL_UNIT",
    "__version__",
]

from loguru import logger

from ._version import __version__
from .main import MultiScaleGroup, memory_per_process, open_multiscale_group
from .ome_ngff import SPATIAL_UNIT

logger.disable("stack_to_chunk")
