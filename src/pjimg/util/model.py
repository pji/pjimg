"""
model
~~~~~

Common types used by :mod:`pjimg`.
"""
from typing import Any, Callable, Sequence, TypeVar, Union

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray


# Exported names.
__all__ = [
    'AnyAry', 'ArrayLike', 'ImgAry',
    'IntAry', 'IntAry64', 'ImgTnsr',
    'Interpolator','Loc', 'NumAry',
    'Numeric', 'RatioAry', 'Size', 'T'
]


# Basic types.
AnyAry = NDArray[Any]
ImgAry = NDArray[np.float64]
IntAry = NDArray[np.uint8]
IntAry64 = NDArray[np.int64]
Loc = Sequence[int]
Numeric = Union[np.bool_, np.integer, np.inexact]
RatioAry = NDArray[np.float64]
Size = Sequence[int]

# Tensor types.
ImgTnsr = torch.Tensor

# Compound types.
T = TypeVar('T', bound=Numeric)
NumAry = NDArray[T]

# Function types.
Interpolator = Callable[[NumAry, NumAry, RatioAry], NumAry]
