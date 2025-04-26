"""
tensor
~~~~~

Common functions for working with :class:`torch.Tensor` objects.
"""
from collections.abc import Sequence

import torch

from pjimg.util.constants import X, Y, Z
from pjimg.util.model import Loc, Size


# Types.
IdxMesh = torch.Tensor
IdxMeshes = Sequence[IdxMesh]


# Functions.
def hypot_3d(t: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Perform a three-dimensional hypotenuse."""
    return torch.sqrt(t[Z] ** 2 + t[Y] ** 2 + t[X] ** 2)


def index_space(
    size: Size,
    loc: Loc = (0, 0, 0),
    device: str = 'cpu',
    dtype: torch.dtype = torch.long,
    center: bool = False
) -> IdxMeshes:
    """Create tensors that index the given volume of space."""
    adjust: list[float] = [float(n) for n in loc]
    if center:
        adjust = [n - ((s - 1) / 2) for s, n in zip(size, adjust)]
    rulers = [
        torch.arange(n, dtype=dtype, device=device) + m
        for n, m in zip(size, adjust)
    ]
    meshes = torch.meshgrid(*rulers, indexing='ij')
    return meshes
