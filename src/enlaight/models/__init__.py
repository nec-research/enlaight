"""Module with prototype-based classifier models."""

from .CBC import CBC, RobustStableCBC, StableCBC
from .GLVQ import GLVQ
from .GTLVQ import GTLVQ
from .RBF import RBF, RobustRBF

__all__ = [
    "GLVQ",
    "GTLVQ",
    "CBC",
    "StableCBC",
    "RBF",
    "RobustRBF",
    "RobustStableCBC",
]
