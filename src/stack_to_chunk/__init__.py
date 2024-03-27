"""Convert stacks of images to chunked datasets."""

__all__ = ["MultiScaleGroup", "SPATIAL_UNIT"]

from ._version import __version__  # noqa: F401
from .main import MultiScaleGroup
from .ome_ngff import SPATIAL_UNIT
