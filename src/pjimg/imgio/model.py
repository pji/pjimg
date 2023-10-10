"""
model
~~~~~

Types used in :mod:`pjimg.imgio`.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

from pjimg.util import ArrayLike, IntAry


# Typing.
Saver = Union[
    Callable[[Union[str, Path], IntAry, bool], None],
    Callable[[Union[str, Path], IntAry, float, str], None]
]
WrappedSaver = Union[
    Callable[[Union[str, Path], ArrayLike, bool], None],
    Callable[[Union[str, Path], ArrayLike, float, str], None]
]


# Dataclasses.
@dataclass
class Image:
    ext: str
    description: str = ''


@dataclass
class Video:
    ext: str
    description: str = ''
    codecs: tuple[str, ...] = tuple()


# Exceptions.
class UnsupportedFileType(TypeError):
    """The given file type isn't supported."""
