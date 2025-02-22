"""
model
~~~~~

Common types used by :mod:`pjimg`.
"""
from typing import Callable, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Exported names.
__all__ = [
    'ArrayLike', 'ImgAry', 'IntAry', 'IntAry64', 'Interpolator', 'Loc',
    'NumAry', 'Numeric', 'RatioAry', 'Size', 'T'
]


# Basic types.
ImgAry = NDArray[np.float64]
IntAry = NDArray[np.uint8]
IntAry64 = NDArray[np.int64]
Loc = Sequence[int]
Numeric = Union[np.bool_, np.integer, np.inexact]
RatioAry = NDArray[np.float64]
Size = Sequence[int]

# Compound types.
T = TypeVar('T', bound=Numeric)
NumAry = NDArray[T]

# Function types.
Interpolator = Callable[[NumAry, NumAry, RatioAry], NumAry]
