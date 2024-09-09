"""Info for working with OME-NGFF."""

from typing import Literal, TypedDict

SPATIAL_UNIT = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]


class DatasetDict(TypedDict):
    # TODO: add a coordinate transform typed dict
    coordinateTransformations: list[dict[str, str | list[int]]]
    path: str
