"""
model
~~~~~

Common types used by :mod:`pjimg`.
"""
from typing import Union

import numpy as np
from numpy.typing import NDArray


# Exported names.
__all__ = ['ImgAry', 'Loc', 'NumAry', 'Numeric', 'Size',]


# Basic types.
ImgAry = NDArray[np.float_]
Loc = tuple[int, int, int]
Numeric = Union[np.bool_, np.integer, np.inexact]
Size = tuple[int, int, int]

# Compound types.
NumAry = NDArray[Numeric]
