"""Convert stacks of images to chunked datasets."""

__all__ = ["MultiScaleGroup", "SPATIAL_UNIT", "__version__"]

from ._version import __version__
from .main import MultiScaleGroup
from .ome_ngff import SPATIAL_UNIT
