"""
model
-----

Types used for :mod:`pjimg.sources`.
"""
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Union

import numpy as np

from pjimg.util import ImgAry, Loc, Size


# Typing.
Seed = Union[None, int, str, bytes]


# Base classes.
class Serializable(ABC):
    def __eq__(self, other):
        """Determine the equality of this and another object."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()

    def __repr__(self):
        """Return a string representation of the object."""
        cls = self.__class__.__name__
        attrs = self.asdict()
        for attr in attrs:
            if isinstance(attrs[attr], str):
                attrs[attr] = f"'{attrs[attr]}'"
        args = [f'{k}={attrs[k]}' for k in attrs]
        args_str = ", ".join(args)
        if len(args_str) > 30:
            args_str = args_str[:10] + '...' + args_str[-10:]
        return f'{cls}({args_str})'

    def asargs(self) -> tuple[Any, ...]:
        """Serialize the object to a tuple."""
        sig = signature(self.__init__)                      # type: ignore
        params = sig.parameters
        return tuple(getattr(self, p) for p in params)

    def asdict(self) -> dict[str, Any]:
        """Serialize the object to a dictionary."""
        sig = signature(self.__init__)                      # type: ignore
        params = sig.parameters
        return {k: getattr(self, k) for k in params}


class Source(Serializable):
    """A source of image data."""
    @abstractmethod
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
